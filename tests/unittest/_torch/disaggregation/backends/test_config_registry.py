# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""U0 (integration plan §11): the KV transfer config file and the backend registry.

YAML -> ``KVTransferConfig`` validation, ``build_backends`` over a fake type, rollback on a failed
build, ``close_backends`` tolerance, and the built-in ``mooncake`` entry: imported lazily, refusing
a ``hint_key``, and built over a fake blob store. Import-light like the ``kv_transfer`` suite:
nothing here needs ``tensorrt_llm``.
"""

import importlib
import logging
import os
import subprocess
import sys
import textwrap

import pytest

__extra_import_path__ = ["~/tensorrt_llm/_torch", "blob"]
from disaggregation.backends import registry as registry_module  # noqa: E402
from disaggregation.backends.blob.backend import BlobStoreBackend  # noqa: E402
from disaggregation.backends.blob.host_landing import HostLandingBlobBackend  # noqa: E402
from disaggregation.backends.config import (  # noqa: E402
    BACKEND_ROLES,
    KV_TRANSFER_CONFIG_ENV,
    BackendEntry,
    KVTransferConfig,
    load_kv_transfer_config,
)
from disaggregation.backends.registry import (  # noqa: E402
    BackendBuildContext,
    BackendHandle,
    BackendRegistry,
    build_backends,
    close_backends,
)
from store_fakes import (  # noqa: E402
    FINGERPRINT,
    ArenaResolver,
    FakeBlobStore,
    MemoryArena,
    fake_open_staging,
)

pytestmark = pytest.mark.cpu_only

PLAN_SECTION_8_YAML = textwrap.dedent(
    """
    fetch_timeout_s: 30
    publish_timeout_s: 60
    probe_timeout_s: 0.05
    backends:
      - name: shared-store
        type: mooncake
        roles: [fetch, publish]
        master_server_address: 127.0.0.1:50051
        protocol: tcp
        local_hostname: 127.0.0.1
        global_segment_size: 0
        landing: host
        namespace: tinyllama-e2e
    """
)


def write_yaml(tmp_path, text: str) -> str:
    path = tmp_path / "kv_transfer.yaml"
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return str(path)


def load(tmp_path, text: str) -> KVTransferConfig:
    return load_kv_transfer_config(write_yaml(tmp_path, text))


# ---------------------------------------------------------------------------------------------
# YAML -> KVTransferConfig
# ---------------------------------------------------------------------------------------------


def test_env_var_name_is_the_documented_one():
    assert KV_TRANSFER_CONFIG_ENV == "TRTLLM_KV_TRANSFER_CONFIG"
    assert BACKEND_ROLES == ("fetch", "publish")


def test_plan_section_8_example_loads(tmp_path):
    config = load(tmp_path, PLAN_SECTION_8_YAML)
    assert config.fetch_timeout_s == 30
    assert config.publish_timeout_s == 60
    assert config.probe_timeout_s == 0.05
    assert config.close_timeout_s == 30.0  # default
    assert len(config.backends) == 1
    entry = config.backends[0]
    assert entry.name == "shared-store"
    assert entry.type == "mooncake"
    assert entry.hint_key is None
    assert entry.roles == frozenset({"fetch", "publish"})
    assert entry.serves_fetch and entry.serves_publish
    # Type-specific keys pass through unread; entry keys do not.
    assert entry.options == {
        "master_server_address": "127.0.0.1:50051",
        "protocol": "tcp",
        "local_hostname": "127.0.0.1",
        "global_segment_size": 0,
        "landing": "host",
        "namespace": "tinyllama-e2e",
    }
    assert not set(entry.options) & {"name", "type", "hint_key", "roles"}


def test_minimal_entry_defaults_to_both_roles_and_finite_timeouts(tmp_path):
    config = load(
        tmp_path,
        """
        backends:
          - name: a
            type: fake
        """,
    )
    # Every wait on a peer or a store is bounded by default; ``null`` must be asked for.
    assert config.fetch_timeout_s == 30.0 and config.publish_timeout_s == 60.0
    assert config.unlaunched_timeout_s == 30.0
    assert config.landing_wait_timeout_s == 30.0
    assert config.probe_timeout_s == 1.0
    assert config.backends[0].roles == frozenset(BACKEND_ROLES)
    assert config.backends[0].options == {}


def test_publish_only_role(tmp_path):
    config = load(
        tmp_path,
        """
        backends:
          - name: a
            type: fake
            roles: [publish]
        """,
    )
    entry = config.backends[0]
    assert entry.serves_publish and not entry.serves_fetch


def test_backend_order_is_kept(tmp_path):
    config = load(
        tmp_path,
        """
        backends:
          - {name: first, type: fake}
          - {name: second, type: fake, hint_key: ctx}
        """,
    )
    assert [e.name for e in config.backends] == ["first", "second"]
    assert config.backends[1].hint_key == "ctx"


def test_unknown_top_level_key_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown kv transfer config keys.*probe_timeout"):
        load(
            tmp_path,
            """
            probe_timeout: 1
            backends:
              - {name: a, type: fake}
            """,
        )


@pytest.mark.parametrize("roles", ["[fetch, foo]", "[]", "[publish, publish, nope]"])
def test_bad_roles_are_refused(tmp_path, roles):
    with pytest.raises(ValueError, match="roles must be a non-empty subset"):
        load(
            tmp_path,
            f"""
            backends:
              - name: a
                type: fake
                roles: {roles}
            """,
        )


@pytest.mark.parametrize(
    "entry, missing",
    [("- {name: a}", "type"), ("- {type: fake}", "name"), ("- {name: '', type: fake}", "name")],
)
def test_entry_without_name_or_type_is_refused(tmp_path, entry, missing):
    with pytest.raises(ValueError, match=f"non-empty '{missing}'"):
        load(tmp_path, f"backends:\n  {entry}\n")


@pytest.mark.parametrize("text", ["fetch_timeout_s: 1\n", "backends: {a: 1}\n", "backends: 3\n"])
def test_backends_must_be_a_list(tmp_path, text):
    with pytest.raises(ValueError, match="needs a 'backends' list"):
        load(tmp_path, text)


def test_empty_backends_list_is_refused(tmp_path):
    with pytest.raises(ValueError, match="at least one backend"):
        load(tmp_path, "backends: []\n")


@pytest.mark.parametrize("text", ["- a\n- b\n", "just a string\n", ""])
def test_non_mapping_file_is_refused(tmp_path, text):
    with pytest.raises(ValueError, match="expected a mapping"):
        load(tmp_path, text)


def test_duplicate_backend_names_are_refused(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        load(
            tmp_path,
            """
            backends:
              - {name: a, type: fake}
              - {name: a, type: other}
            """,
        )


@pytest.mark.parametrize(
    "line, message",
    [
        ("fetch_timeout_s: 0", "fetch_timeout_s must be > 0 or null"),
        ("publish_timeout_s: -1", "publish_timeout_s must be > 0 or null"),
        ("unlaunched_timeout_s: 0", "unlaunched_timeout_s must be > 0 or null"),
        ("landing_wait_timeout_s: 0", "landing_wait_timeout_s must be > 0 or null"),
        ("probe_timeout_s: -0.1", "probe_timeout_s must be >= 0"),
        ("close_timeout_s: 0", "close_timeout_s must be > 0"),
    ],
)
def test_limits_out_of_range_are_refused(tmp_path, line, message):
    with pytest.raises(ValueError, match=message):
        load(tmp_path, f"{line}\nbackends:\n  - {{name: a, type: fake}}\n")


def test_null_timeouts_and_zero_probe_are_allowed(tmp_path):
    config = load(
        tmp_path,
        """
        fetch_timeout_s: null
        publish_timeout_s: null
        unlaunched_timeout_s: null
        landing_wait_timeout_s: null
        probe_timeout_s: 0
        backends:
          - {name: a, type: fake}
        """,
    )
    assert config.fetch_timeout_s is None and config.probe_timeout_s == 0
    assert config.unlaunched_timeout_s is None and config.landing_wait_timeout_s is None


def test_missing_file_raises_os_error(tmp_path):
    with pytest.raises(OSError):
        load_kv_transfer_config(str(tmp_path / "nope.yaml"))


# ---------------------------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------------------------


def make_context(**overrides) -> BackendBuildContext:
    arena = MemoryArena(1 << 12)
    kwargs = dict(
        resolver=ArenaResolver(arena),
        layout_fingerprint=FINGERPRINT,
        max_unit_bytes=256,
        device_index=None,
        unit_bytes_of=lambda name: 256,
    )
    kwargs.update(overrides)
    return BackendBuildContext(**kwargs)


class FakeBuilt:
    """What a fake factory builds; records closes so rollback can be asserted."""

    instances: list = []

    def __init__(self, entry, fail_close: bool = False) -> None:
        self.entry = entry
        self.closed = 0
        self.fail_close = fail_close
        FakeBuilt.instances.append(self)

    def close(self) -> None:
        self.closed += 1
        if self.fail_close:
            raise RuntimeError(f"{self.entry.name}: close failed")


def fake_factory(entry: BackendEntry, context: BackendBuildContext) -> BackendHandle:
    built = FakeBuilt(entry, fail_close=entry.options.get("fail_close", False))
    if entry.options.get("fail_build"):
        raise RuntimeError(f"{entry.name}: cannot build")
    return BackendHandle(
        name=entry.name,
        hint_key=entry.hint_key,
        fetcher=built if entry.serves_fetch else None,
        publisher=built if entry.serves_publish else None,
        pool_registrar=None,
        close=built.close,
        counters=lambda: {"built": 1},
    )


@pytest.fixture(autouse=True)
def _reset_fake_built():
    FakeBuilt.instances = []
    yield
    FakeBuilt.instances = []


def entry(name: str, type_: str = "fake", **options) -> BackendEntry:
    return BackendEntry.from_dict({"name": name, "type": type_, **options})


def test_build_backends_returns_one_handle_per_entry_in_order():
    registry = BackendRegistry()
    registry.register_backend_type("fake", fake_factory)
    config = KVTransferConfig(
        backends=(entry("a"), entry("b", roles=["publish"]), entry("c", hint_key="ctx"))
    )
    handles = build_backends(config, make_context(), registry)
    assert [h.name for h in handles] == ["a", "b", "c"]
    assert handles[0].fetcher is not None and handles[0].publisher is not None
    assert handles[1].fetcher is None and handles[1].publisher is not None
    assert handles[2].hint_key == "ctx"
    close_backends(handles)
    assert [b.closed for b in FakeBuilt.instances] == [1, 1, 1]


def test_handle_counters_default_to_an_empty_mapping():
    handle = BackendHandle("x", None, None, None, None, lambda: None)
    assert dict(handle.counters()) == {}


def test_duplicate_registration_is_refused():
    registry = BackendRegistry()
    registry.register_backend_type("fake", fake_factory)
    with pytest.raises(ValueError, match="already registered"):
        registry.register_backend_type("fake", fake_factory)
    with pytest.raises(ValueError, match="needs a name"):
        registry.register_backend_type("", fake_factory)


def test_unknown_type_names_the_known_types():
    registry = BackendRegistry()
    registry.register_backend_type("fake", fake_factory)
    with pytest.raises(
        ValueError, match=r"unknown kv transfer backend type 'nope'.*'fake'.*'memory'.*'mooncake'"
    ):
        registry.factory_for("nope")


def test_failed_build_closes_what_was_built_and_reraises():
    registry = BackendRegistry()
    registry.register_backend_type("fake", fake_factory)
    config = KVTransferConfig(
        backends=(entry("first"), entry("second", fail_build=True), entry("third"))
    )
    with pytest.raises(RuntimeError, match="second: cannot build"):
        build_backends(config, make_context(), registry)
    by_name = {b.entry.name: b for b in FakeBuilt.instances}
    assert by_name["first"].closed == 1
    assert "third" not in by_name  # never reached


def test_close_backends_reaches_every_handle_when_one_fails():
    registry = BackendRegistry()
    registry.register_backend_type("fake", fake_factory)
    config = KVTransferConfig(backends=(entry("a", fail_close=True), entry("b")))
    handles = build_backends(config, make_context(), registry)
    close_backends(handles)  # must not raise
    assert [b.closed for b in FakeBuilt.instances] == [1, 1]


def test_explicit_registration_shadows_the_builtin_table():
    registry = BackendRegistry()
    registry.register_backend_type("mooncake", fake_factory)
    assert registry.factory_for("mooncake") is fake_factory


def test_module_level_register_backend_type_uses_the_default_registry():
    name = f"test-only-{os.getpid()}"
    registry_module.register_backend_type(name, fake_factory)
    try:
        assert registry_module.DEFAULT_REGISTRY.factory_for(name) is fake_factory
    finally:
        registry_module.DEFAULT_REGISTRY._factories.pop(name, None)


# ---------------------------------------------------------------------------------------------
# The built-in ``mooncake`` entry
# ---------------------------------------------------------------------------------------------


def test_importing_the_registry_does_not_import_the_mooncake_driver():
    """A fresh interpreter: the built-in table is strings, imported on first use only."""
    torch_dir = os.path.abspath(os.path.join(registry_module.__file__, "..", "..", ".."))
    code = textwrap.dedent(
        """
        import sys
        import disaggregation.backends.registry as registry
        assert "disaggregation.backends.blob.drivers.mooncake" not in sys.modules, "eager import"
        assert "disaggregation.backends.blob.backend" not in sys.modules, "eager import"
        factory = registry.BackendRegistry().factory_for("mooncake")
        assert factory.__module__ == "disaggregation.backends.blob.drivers.mooncake"
        assert factory.__name__ == "build_mooncake_backend"
        assert "disaggregation.backends.blob.drivers.mooncake" in sys.modules
        """
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env={**os.environ, "PYTHONPATH": torch_dir},
        timeout=120,
    )


@pytest.fixture
def mooncake_module(monkeypatch):
    """The driver module with ``MooncakeBlobStore.open`` replaced by a fake; yields
    ``(factory_module, opened)``: the shared factory module, whose ``open_pinned_staging_pool`` the
    host-landing tests replace, and the fake stores opened so far."""
    driver = importlib.import_module("disaggregation.backends.blob.drivers.mooncake")
    factory = importlib.import_module("disaggregation.backends.blob.factory")
    opened: list[FakeBlobStore] = []

    def open_fake(config):
        store = FakeBlobStore()
        store.opened_with = config
        store.memcpy_env_at_open = os.environ.get("MC_STORE_MEMCPY")
        opened.append(store)
        return store

    monkeypatch.setattr(driver.MooncakeBlobStore, "open", open_fake)
    return factory, opened


@pytest.fixture(autouse=True)
def _memcpy_env_unset(monkeypatch):
    """Every test starts with ``MC_STORE_MEMCPY`` unset and ends with it as it was: building a
    device-landing mooncake entry sets the variable when it is unset, and that must not leak
    between tests. (A bare ``delenv`` of an absent variable records nothing to restore, hence the
    set-then-delete.)"""
    monkeypatch.setenv("MC_STORE_MEMCPY", "placeholder")
    monkeypatch.delenv("MC_STORE_MEMCPY")


MOONCAKE_OPTIONS = dict(
    master_server_address="127.0.0.1:50051",
    protocol="rdma",
    local_hostname="127.0.0.1",
    global_segment_size=0,
    namespace="u0",
)
"""An RDMA entry: ``landing`` defaults to ``device`` and the pools are registered."""

TCP_OPTIONS = {**MOONCAKE_OPTIONS, "protocol": "tcp"}
"""A TCP entry: ``landing`` defaults to ``host`` and two host pools are opened."""


@pytest.fixture
def fake_pools(mooncake_module, monkeypatch):
    """Host pools over host arenas instead of pinned torch buffers; yields the geometries the
    factory asked for, ``(slot_bytes, num_slots, device_index)`` per pool in opening order."""
    factory, _ = mooncake_module
    geometry = []

    def open_pool(store, *, slot_bytes, num_slots, device_index=None):
        geometry.append((slot_bytes, num_slots, device_index))
        return fake_open_staging(store, slot_bytes=slot_bytes, num_slots=num_slots)

    monkeypatch.setattr(factory, "open_pinned_staging_pool", open_pool)
    return geometry


def test_mooncake_factory_builds_a_blob_store_backend_over_the_opened_store(mooncake_module):
    _, opened = mooncake_module
    config = KVTransferConfig(backends=(entry("store", "mooncake", **MOONCAKE_OPTIONS),))
    handles = build_backends(config, make_context())
    try:
        assert len(opened) == 1
        assert opened[0].opened_with.master_server_address == "127.0.0.1:50051"
        assert opened[0].opened_with.protocol == "rdma"
        handle = handles[0]
        assert handle.name == "store" and handle.hint_key is None
        assert handle.landing == "device"
        assert isinstance(handle.fetcher, BlobStoreBackend)
        assert handle.publisher is handle.fetcher
        # No host pools: the KV pools must be registered with the transport, by the backend itself.
        assert handle.pool_registrar is handle.fetcher
        assert set(handle.counters()) >= {"fetch_hits", "fetch_misses", "publish_stored"}
        assert "landings_held" not in handle.counters()
    finally:
        close_backends(handles)
    assert opened[0].closed == 1
    close_backends(handles)  # idempotent
    assert opened[0].closed == 1


def test_mooncake_publish_only_entry_has_no_fetches(mooncake_module):
    config = KVTransferConfig(
        backends=(entry("store", "mooncake", roles=["publish"], **MOONCAKE_OPTIONS),)
    )
    handles = build_backends(config, make_context())
    try:
        assert handles[0].fetcher is None
        assert isinstance(handles[0].publisher, BlobStoreBackend)
    finally:
        close_backends(handles)


# ---- landing: the TCP default and the two shapes ----


def test_mooncake_over_tcp_lands_on_host_by_default(mooncake_module, fake_pools, caplog):
    """No ``landing`` key and ``protocol: tcp``: the factory injects ``host`` (and says so), the
    fetcher is the ``LandsOnHost`` shape, the publisher the inner backend, no pool is registered,
    and the handle reports the resolved landing for the status dump."""
    _, opened = mooncake_module
    config = KVTransferConfig(backends=(entry("store", "mooncake", **TCP_OPTIONS),))
    with caplog.at_level(logging.INFO):
        handles = build_backends(config, make_context())
    try:
        handle = handles[0]
        assert handle.landing == "host"
        assert isinstance(handle.fetcher, HostLandingBlobBackend)
        assert isinstance(handle.publisher, BlobStoreBackend)
        assert handle.pool_registrar is None
        assert handle.counters()["landings_held"] == 0
        assert len(fake_pools) == 2  # publish pool, landing pool
        assert opened[0].count("register_span") == 2
        messages = [r.getMessage() for r in caplog.records]
        assert any("landing: host" in m for m in messages), messages
    finally:
        close_backends(handles)
    assert opened[0].closed == 1


def test_mooncake_over_tcp_with_explicit_device_landing_is_allowed_with_a_warning(
    mooncake_module, caplog
):
    config = KVTransferConfig(
        backends=(entry("store", "mooncake", landing="device", **TCP_OPTIONS),)
    )
    with caplog.at_level(logging.WARNING):
        handles = build_backends(config, make_context())
    try:
        assert handles[0].landing == "device"
        assert isinstance(handles[0].fetcher, BlobStoreBackend)
        assert handles[0].pool_registrar is handles[0].fetcher
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("landing 'device' over tcp" in r.getMessage() for r in warnings)
    finally:
        close_backends(handles)


def test_mooncake_over_rdma_with_explicit_host_landing_builds_the_host_shape(
    mooncake_module, fake_pools
):
    config = KVTransferConfig(
        backends=(entry("store", "mooncake", landing="host", **MOONCAKE_OPTIONS),)
    )
    handles = build_backends(config, make_context())
    try:
        assert handles[0].landing == "host"
        assert isinstance(handles[0].fetcher, HostLandingBlobBackend)
    finally:
        close_backends(handles)


def test_mooncake_host_landing_opens_two_pools_with_their_own_geometry(
    mooncake_module, fake_pools, caplog
):
    """The publish pool is one slot per unit of a batch within its budget; the landing pool is
    every slot its budget affords, capped by ``max_landed_units`` when given. A landing pool
    short of two requests of ``max_seq_len`` is warned about."""
    _, opened = mooncake_module
    config = KVTransferConfig(
        backends=(
            entry(
                "store",
                "mooncake",
                transfer_batch_size=4,
                staging_buffer_bytes=4096,
                landing_buffer_bytes=8192,
                **TCP_OPTIONS,
            ),
        )
    )
    context = make_context(max_unit_bytes=512, device_index=3, max_request_blocks=16)
    with caplog.at_level(logging.INFO):
        handles = build_backends(config, context)
    try:
        assert fake_pools == [(512, 4, 3), (512, 16, 3)]
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("16 slots" in m and "32 blocks" not in m for m in warnings), warnings
    finally:
        close_backends(handles)
    # An explicit cap only lowers the count.
    capped = KVTransferConfig(
        backends=(
            entry(
                "store",
                "mooncake",
                landing_buffer_bytes=8192,
                max_landed_units=3,
                **TCP_OPTIONS,
            ),
        )
    )
    del fake_pools[:]
    caplog.clear()
    with caplog.at_level(logging.INFO):
        handles = build_backends(capped, make_context(max_unit_bytes=512, max_request_blocks=1))
    try:
        assert fake_pools[1] == (512, 3, None)
        assert not [r for r in caplog.records if r.levelno == logging.WARNING]
    finally:
        close_backends(handles)


@pytest.mark.parametrize("bound", [5.0, None])
def test_mooncake_host_landing_takes_the_wait_bound_from_the_context(
    mooncake_module, fake_pools, bound
):
    """The queue-wait bound is the coordinator's ``landing_wait_timeout_s``, carried by the
    build context; ``None`` means a landing waits without bound."""
    config = KVTransferConfig(backends=(entry("store", "mooncake", **TCP_OPTIONS),))
    handles = build_backends(config, make_context(landing_wait_timeout_s=bound))
    try:
        assert handles[0].fetcher.landing_wait_timeout_s == bound
    finally:
        close_backends(handles)


def test_mooncake_host_landing_needs_the_assembly_to_size_units(mooncake_module):
    _, opened = mooncake_module
    config = KVTransferConfig(backends=(entry("store", "mooncake", **TCP_OPTIONS),))
    with pytest.raises(ValueError, match="size units by name"):
        build_backends(config, make_context(unit_bytes_of=None))
    assert opened == []  # refused before a store is opened


# ---- MC_STORE_MEMCPY ----


@pytest.mark.parametrize("value", ["1", "true", " ON ", "surprise", " off "])
def test_mooncake_refuses_the_memcpy_bypass_when_it_would_register_gpu_memory(
    mooncake_module, monkeypatch, value
):
    """A ``landing: device`` backend registers GPU pools and keeps the client's memcpy bypass off
    for them (older clients memcpy'd such spans). The client reads any value but the spellings
    of "off" as on, and so does the guard. Refused before a store is opened."""
    _, opened = mooncake_module
    monkeypatch.setenv("MC_STORE_MEMCPY", value)
    config = KVTransferConfig(backends=(entry("store", "mooncake", **MOONCAKE_OPTIONS),))
    with pytest.raises(ValueError, match=f"MC_STORE_MEMCPY={value!r}.*landing: host"):
        build_backends(config, make_context())
    assert opened == []


def test_mooncake_refuses_the_memcpy_bypass_for_explicit_device_over_tcp(
    mooncake_module, monkeypatch
):
    _, opened = mooncake_module
    monkeypatch.setenv("MC_STORE_MEMCPY", "1")
    config = KVTransferConfig(
        backends=(entry("store", "mooncake", landing="device", **TCP_OPTIONS),)
    )
    with pytest.raises(ValueError, match="MC_STORE_MEMCPY='1'.*set MC_STORE_MEMCPY=0"):
        build_backends(config, make_context())
    assert opened == []


@pytest.mark.parametrize(
    "options",
    [MOONCAKE_OPTIONS, {**TCP_OPTIONS, "landing": "device"}],
    ids=["rdma-default-device", "tcp-explicit-device"],
)
def test_mooncake_device_landing_with_env_unset_pins_the_bypass_off(
    mooncake_module, caplog, options
):
    """Unset, ``MC_STORE_MEMCPY`` is decided by the client from the transports it loaded, not by
    the entry's ``protocol`` (an RDMA entry on a host without a reachable HCA comes up TCP-only
    and would turn the bypass on). A device-landing entry therefore sets the variable to off
    itself, before the store is opened, says so, and the setting outlives the build."""
    _, opened = mooncake_module
    assert "MC_STORE_MEMCPY" not in os.environ
    config = KVTransferConfig(backends=(entry("store", "mooncake", **options),))
    with caplog.at_level(logging.INFO):
        handles = build_backends(config, make_context())
    try:
        assert opened[0].memcpy_env_at_open == "0"
        assert os.environ["MC_STORE_MEMCPY"] == "0"
        assert handles[0].landing == "device"
        messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        assert any("MC_STORE_MEMCPY unset; set to 0" in m for m in messages), messages
    finally:
        close_backends(handles)


@pytest.mark.parametrize("value", ["0", "false", "OFF", "no"])
def test_mooncake_leaves_an_explicit_off_value_as_it_is(mooncake_module, monkeypatch, value):
    _, opened = mooncake_module
    monkeypatch.setenv("MC_STORE_MEMCPY", value)
    config = KVTransferConfig(backends=(entry("store", "mooncake", **MOONCAKE_OPTIONS),))
    close_backends(build_backends(config, make_context()))
    assert opened[0].memcpy_env_at_open == value and os.environ["MC_STORE_MEMCPY"] == value


@pytest.mark.parametrize(
    "value, options",
    [
        ("1", TCP_OPTIONS),
        ("1", {**MOONCAKE_OPTIONS, "landing": "host"}),
        (None, TCP_OPTIONS),
        (None, {**MOONCAKE_OPTIONS, "landing": "host"}),
    ],
    ids=["tcp-on", "rdma-on", "tcp-unset", "rdma-unset"],
)
def test_mooncake_host_landing_leaves_the_bypass_to_the_client(
    mooncake_module, fake_pools, monkeypatch, value, options
):
    """The guard reads the resolved landing: an unset ``landing`` over TCP is ``host``, and host
    landing registers pinned host memory only, which the bypass copies correctly. There the
    variable is neither refused nor set: left on, or left unset for the client to decide."""
    _, opened = mooncake_module
    if value is not None:
        monkeypatch.setenv("MC_STORE_MEMCPY", value)
    config = KVTransferConfig(backends=(entry("store", "mooncake", **options),))
    close_backends(build_backends(config, make_context()))
    assert opened[0].memcpy_env_at_open == value
    assert os.environ.get("MC_STORE_MEMCPY") == value


# ---- refusals before the store is opened ----


def test_mooncake_refuses_a_hint_key_before_opening_a_store(mooncake_module):
    _, opened = mooncake_module
    config = KVTransferConfig(
        backends=(entry("store", "mooncake", hint_key="ctx", **MOONCAKE_OPTIONS),)
    )
    with pytest.raises(ValueError, match="takes no hint_key"):
        build_backends(config, make_context())
    assert opened == []


def test_mooncake_refuses_unknown_options_before_opening_a_store(mooncake_module):
    _, opened = mooncake_module
    config = KVTransferConfig(
        backends=(entry("store", "mooncake", bogus_option=1, **MOONCAKE_OPTIONS),)
    )
    with pytest.raises(
        ValueError, match=r"backend 'store' \(type mooncake\): unknown keys.*bogus_option"
    ):
        build_backends(config, make_context())
    assert opened == []


def test_mooncake_reports_a_misspelt_backend_key_against_the_entry_not_the_driver(mooncake_module):
    # ``namespce`` is neither side's key. The error names the entry and its type, not
    # ``MooncakeStoreConfig``: the user wrote one flat entry and need not know the split.
    _, opened = mooncake_module
    config = KVTransferConfig(
        backends=(entry("store", "mooncake", namespce="typo", **MOONCAKE_OPTIONS),)
    )
    with pytest.raises(ValueError, match=r"\(type mooncake\): unknown keys \['namespce'\]") as info:
        build_backends(config, make_context())
    assert "MooncakeStoreConfig" not in str(info.value)
    assert opened == []


def test_mooncake_pool_failure_closes_the_store(mooncake_module, monkeypatch):
    module, opened = mooncake_module

    def fail_pool(*args, **kwargs):
        raise RuntimeError("no pinned memory")

    monkeypatch.setattr(module, "open_pinned_staging_pool", fail_pool)
    config = KVTransferConfig(backends=(entry("store", "mooncake", **TCP_OPTIONS),))
    with pytest.raises(RuntimeError, match="no pinned memory"):
        build_backends(config, make_context())
    assert len(opened) == 1 and opened[0].closed == 1

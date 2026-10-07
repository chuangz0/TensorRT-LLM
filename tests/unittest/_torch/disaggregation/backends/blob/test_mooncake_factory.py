# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The built-in ``mooncake`` entry's factory over a fake blob store: landing resolution (the TCP
default, the two shapes), the host pools' geometry and wait bound, the ``MC_STORE_MEMCPY``
guard, option validation before a store is opened, and rollback when a pool fails to open.
"""

import importlib
import logging
import os

import pytest

__extra_import_path__ = ["~/tensorrt_llm/_torch"]
from disaggregation.backends.blob.backend import BlobStoreBackend  # noqa: E402
from disaggregation.backends.blob.host_landing import HostLandingBlobBackend  # noqa: E402
from disaggregation.backends.config import BackendEntry, KVTransferConfig  # noqa: E402
from disaggregation.backends.registry import (  # noqa: E402
    BackendBuildContext,
    build_backends,
    close_backends,
)
from store_fakes import (  # noqa: E402
    FINGERPRINT,
    ArenaResolver,
    FakeBlobStore,
    MemoryArena,
    fake_open_slot_pool,
)

pytestmark = pytest.mark.cpu_only


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


def entry(name: str, type_: str = "fake", **options) -> BackendEntry:
    return BackendEntry.from_dict({"name": name, "type": type_, **options})


@pytest.fixture
def mooncake_module(monkeypatch):
    """The driver module with ``MooncakeBlobStore.open`` replaced by a fake; yields
    ``(factory_module, opened)``: the shared factory module, whose ``open_pinned_slot_pool`` the
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
        return fake_open_slot_pool(store, slot_bytes=slot_bytes, num_slots=num_slots)

    monkeypatch.setattr(factory, "open_pinned_slot_pool", open_pool)
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
        assert set(handle.read_counters()) >= {"fetch_hits", "fetch_misses", "publish_stored"}
        assert "landings_held" not in handle.read_counters()
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
        assert handle.read_counters()["landings_held"] == 0
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
                publish_buffer_bytes=4096,
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
    """The queue-wait bound is the coordinator's ``fetch_wait_timeout_s``, carried by the
    build context; ``None`` means a landing waits without bound."""
    config = KVTransferConfig(backends=(entry("store", "mooncake", **TCP_OPTIONS),))
    handles = build_backends(config, make_context(fetch_wait_timeout_s=bound))
    try:
        assert handles[0].fetcher.fetch_wait_timeout_s == bound
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
    for them. The client reads any value but the spellings of "off" as on, and so does the
    guard. Refused before a store is opened."""
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

    monkeypatch.setattr(module, "open_pinned_slot_pool", fail_pool)
    config = KVTransferConfig(backends=(entry("store", "mooncake", **TCP_OPTIONS),))
    with pytest.raises(RuntimeError, match="no pinned memory"):
        build_backends(config, make_context())
    assert len(opened) == 1 and opened[0].closed == 1

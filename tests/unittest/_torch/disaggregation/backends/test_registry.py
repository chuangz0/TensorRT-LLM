# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The backend registry: ``build_backends`` over a fake type, rollback on a failed build,
``close_backends`` tolerance, registration rules, and the built-in table imported lazily.
Import-light like the ``kv_transfer`` suite: nothing here needs ``tensorrt_llm``.
"""

import os
import subprocess
import sys
import textwrap

import pytest

__extra_import_path__ = ["~/tensorrt_llm/_torch"]
from disaggregation.backends import registry as registry_module  # noqa: E402
from disaggregation.backends.config import BackendEntry, KVTransferConfig  # noqa: E402
from disaggregation.backends.registry import (  # noqa: E402
    BackendBuildContext,
    BackendHandle,
    BackendRegistry,
    build_backends,
    close_backends,
)

pytestmark = pytest.mark.cpu_only


def make_context(**overrides) -> BackendBuildContext:
    """A build context the fake factory never reads (the mooncake factory's tests, which do
    read it, build theirs over ``store_fakes``)."""
    kwargs = dict(
        resolver=lambda local_group, local: (),
        layout_fingerprint=b"\x01layout",
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
        fetcher=built if entry.has_fetch_role else None,
        publisher=built if entry.has_publish_role else None,
        pool_registrar=None,
        close=built.close,
        read_counters=lambda: {"built": 1},
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
    assert dict(handle.read_counters()) == {}


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

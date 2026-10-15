# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Store config, opening the Mooncake SDK and driver, and the blob package's layering.

The SDK is a fake module put into ``sys.modules``, the master probe and the route probe are
replaced, and the InfiniBand sysfs tree is a temporary directory, so nothing here needs a network,
a Mooncake install or a GPU.
"""

from __future__ import annotations

import ast
import pathlib
import sys
import types

import pytest
import yaml
from fake_mooncake import FakeMooncakeStore, FakePool

import tensorrt_llm._torch.disaggregation.blob as blob_package
from tensorrt_llm._torch.disaggregation.blob import mooncake_setup
from tensorrt_llm._torch.disaggregation.blob.backend import BlobConfig
from tensorrt_llm._torch.disaggregation.blob.mooncake import MooncakeDriver
from tensorrt_llm._torch.disaggregation.blob.mooncake_setup import (
    MooncakeConfig,
    load_store_config,
    open_mooncake_driver,
    open_sdk,
)

pytestmark = pytest.mark.cpu_only

MASTER = "10.0.0.1:50051"
ROUTED_ADDRESS = "10.0.0.7"


def store_config(protocol: str = "tcp", **mooncake) -> dict:
    return {
        "blob": {"key_prefix": "t", "rows_per_batch": 4},
        "mooncake": {
            "master_server_address": MASTER,
            "global_segment_size": 0,
            "protocol": protocol,
            **mooncake,
        },
    }


class Sdk:
    """The fake ``mooncake.store`` module's state: every store it built, and the masters probed."""

    def __init__(self) -> None:
        self.pool = FakePool()
        self.stores: list[FakeMooncakeStore] = []
        self.masters: list[str] = []
        self.setup_answer = 0

    def build(self) -> FakeMooncakeStore:
        store = FakeMooncakeStore(self.pool)
        store.setup_answer = self.setup_answer
        self.stores.append(store)
        return store


@pytest.fixture
def sdk(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> Sdk:
    sdk = Sdk()
    package = types.ModuleType("mooncake")
    package.__path__ = []
    module = types.ModuleType("mooncake.store")
    module.MooncakeDistributedStore = sdk.build
    package.store = module
    monkeypatch.setitem(sys.modules, "mooncake", package)
    monkeypatch.setitem(sys.modules, "mooncake.store", module)
    monkeypatch.setattr(mooncake_setup, "_wait_for_master", sdk.masters.append)
    monkeypatch.setattr(mooncake_setup, "_local_address", lambda master: ROUTED_ADDRESS)
    monkeypatch.setattr(mooncake_setup, "IB_SYSFS_ROOT", str(tmp_path / "no-infiniband"))
    monkeypatch.delenv("MC_TCP_PROTO", raising=False)
    monkeypatch.delenv("MC_STORE_LOCAL_HOT_CACHE_SIZE", raising=False)
    return sdk


def ib_device(root: pathlib.Path, name: str, link_layer: str, state: str, rate: str) -> None:
    port = root / name / "ports" / "1"
    port.mkdir(parents=True)
    (port / "link_layer").write_text(f"{link_layer}\n")
    (port / "state").write_text(f"{state}\n")
    (port / "rate").write_text(f"{rate}\n")


# Store config.


def test_store_config_refuses_missing_and_unknown_sections_and_keys() -> None:
    config = store_config()
    del config["mooncake"]["global_segment_size"]
    with pytest.raises(ValueError, match="'mooncake'.*global_segment_size"):
        load_store_config(config)
    with pytest.raises(ValueError, match="unknown sections"):
        load_store_config({**store_config(), "mooncake_store": {}})
    with pytest.raises(ValueError, match="'mooncake' is a required mapping"):
        load_store_config({"blob": {}})


BAD_VALUES = {
    "blob.rows_per_bach=4": ("blob", "rows_per_bach", 4),  # an unknown key
    "mooncake.devce_name=mlx5_0": ("mooncake", "devce_name", "mlx5_0"),
    "mooncake.protocol=rdmaa": ("mooncake", "protocol", "rdmaa"),
    "mooncake.global_segment_size=True": ("mooncake", "global_segment_size", True),
    "blob.rows_per_batch=64.0": ("blob", "rows_per_batch", 64.0),
    "blob.bytes_per_batch=1.28e8": ("blob", "bytes_per_batch", 1.28e8),
    "blob.key_prefix=7": ("blob", "key_prefix", 7),
    "blob.workers=2.5": ("blob", "workers", 2.5),
    "blob.workers=True": ("blob", "workers", True),
    "mooncake.global_segment_size=1.0": ("mooncake", "global_segment_size", 1.0),
    "mooncake.local_buffer_size=-1": ("mooncake", "local_buffer_size", -1),
    "mooncake.master_server_address=5": ("mooncake", "master_server_address", 5),
    "mooncake.device_name=None": ("mooncake", "device_name", None),
}


@pytest.mark.parametrize("case", BAD_VALUES)
def test_unknown_keys_and_values_of_the_wrong_type_are_rejected(sdk: Sdk, case: str) -> None:
    section, name, value = BAD_VALUES[case]
    config = store_config()
    config[section][name] = value
    if section == "mooncake":
        with pytest.raises(ValueError, match=f"section 'mooncake'.*{name}"):
            open_mooncake_driver(config)
        assert sdk.stores == []  # refused before the SDK was set up
        return
    blob_section, _ = load_store_config(config)  # handed back as given, for BlobConfig to check
    with pytest.raises((TypeError, ValueError), match=name):
        BlobConfig(**blob_section)


def test_store_config_reads_a_yaml_file(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "store.yaml"
    path.write_text(yaml.safe_dump(store_config(local_buffer_size=0)))
    blob, mooncake = load_store_config(path)
    assert blob == {"key_prefix": "t", "rows_per_batch": 4}
    assert mooncake == MooncakeConfig(MASTER, 0, protocol="tcp", local_buffer_size=0)


# Opening the SDK.


def test_open_passes_setup_arguments_in_order(
    sdk: Sdk, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sysfs = tmp_path / "infiniband"
    ib_device(sysfs, "mlx5_0", "InfiniBand", "4: ACTIVE", "400 Gb/sec (4X NDR)")
    ib_device(sysfs, "mlx5_1", "InfiniBand", "4: ACTIVE", "400 Gb/sec (4X NDR)")
    monkeypatch.setattr(mooncake_setup, "IB_SYSFS_ROOT", str(sysfs))
    driver, blob = open_mooncake_driver(store_config("rdma"))
    (store,) = sdk.stores
    (setup,) = store.calls_to("setup")
    assert setup.args == (
        ROUTED_ADDRESS,  # local_hostname: where the route to the master leaves from
        "P2PHANDSHAKE",  # metadata_server
        0,  # global_segment_size
        16 << 20,  # local_buffer_size
        "rdma",  # protocol
        "mlx5_0,mlx5_1",  # device_name: the fastest active InfiniBand ports
        MASTER,  # master_server_address
    )
    assert setup.kwargs == {}
    assert sdk.masters == [MASTER]
    assert isinstance(driver, MooncakeDriver)
    assert blob == {"key_prefix": "t", "rows_per_batch": 4}

    open_sdk(MooncakeConfig(MASTER, 1 << 30, "tcp", "eth9", "10.9.9.9", "etcd://m:2379", 0))
    assert sdk.stores[1].calls_to("setup")[0].args == (
        "10.9.9.9",
        "etcd://m:2379",
        1 << 30,
        0,
        "tcp",
        "eth9",
        MASTER,
    )


def test_device_detection_picks_the_fastest_active_ib_ports(tmp_path: pathlib.Path) -> None:
    ib_device(tmp_path, "mlx5_0", "InfiniBand", "4: ACTIVE", "400 Gb/sec (4X NDR)")
    ib_device(tmp_path, "mlx5_1", "InfiniBand", "4: ACTIVE", "200 Gb/sec (4X HDR)")
    ib_device(tmp_path, "mlx5_2", "Ethernet", "4: ACTIVE", "400 Gb/sec (4X NDR)")
    ib_device(tmp_path, "mlx5_3", "InfiniBand", "1: DOWN", "400 Gb/sec (4X NDR)")
    ib_device(tmp_path, "mlx5_4", "InfiniBand", "4: ACTIVE", "400 Gb/sec (4X NDR)")
    assert mooncake_setup._fastest_active_ib_devices(str(tmp_path)) == ["mlx5_0", "mlx5_4"]
    assert mooncake_setup._fastest_active_ib_devices(str(tmp_path / "missing")) == []


def test_missing_sdk_reports_install_and_shadow_hint(sdk: Sdk, monkeypatch) -> None:
    shadow = types.ModuleType("mooncake")
    shadow.__file__ = "/opt/nixl/lib/python3/site-packages/mooncake/__init__.py"
    monkeypatch.setitem(sys.modules, "mooncake", shadow)
    monkeypatch.setitem(sys.modules, "mooncake.store", None)
    with pytest.raises(ImportError) as missing:
        open_sdk(MooncakeConfig(MASTER, 0, "tcp"))
    message = str(missing.value)
    assert "mooncake-transfer-engine-cuda13==0.3.13" in message
    assert "/opt/nixl/" in message and "shadows" in message
    with pytest.raises(ImportError, match="mooncake-transfer-engine-cuda13"):
        open_mooncake_driver(store_config())


def test_tcp_legacy_framing_is_refused(sdk: Sdk, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MC_TCP_PROTO", "1")
    with pytest.raises(ValueError, match="MC_TCP_PROTO"):
        open_mooncake_driver(store_config("tcp"))
    assert sdk.stores == []
    open_sdk(MooncakeConfig(MASTER, 0, "rdma", device_name="mlx5_0"))  # RDMA does not frame
    assert len(sdk.stores) == 1


def test_local_hot_cache_is_refused(sdk: Sdk, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MC_STORE_LOCAL_HOT_CACHE_SIZE", "64")
    with pytest.raises(ValueError, match="MC_STORE_LOCAL_HOT_CACHE_SIZE"):
        open_mooncake_driver(store_config())
    assert sdk.stores == []
    monkeypatch.setenv("MC_STORE_LOCAL_HOT_CACHE_SIZE", "0")  # off
    open_sdk(MooncakeConfig(MASTER, 0, "tcp"))
    assert len(sdk.stores) == 1


def test_setup_failure_closes_the_half_open_client(sdk: Sdk) -> None:
    sdk.setup_answer = -1
    with pytest.raises(RuntimeError, match="setup failed with -1"):
        open_mooncake_driver(store_config())
    (store,) = sdk.stores
    assert len(store.calls_to("close")) == 1


# Layering.


BLOB_PACKAGE = blob_package.__name__
DISAGGREGATION = BLOB_PACKAGE.rsplit(".", 1)[0]
THIRD_PARTY_ALLOWED = {"numpy", "yaml"}


def _imports(path: pathlib.Path) -> list[tuple[str, str | None]]:
    """Every module ``path`` imports, absolute, with the function it is imported in (or None)."""
    found: list[tuple[str, str | None]] = []

    class Visitor(ast.NodeVisitor):
        function: str | None = None

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            outer, self.function = self.function, node.name
            self.generic_visit(node)
            self.function = outer

        def visit_Import(self, node: ast.Import) -> None:
            found.extend((alias.name, self.function) for alias in node.names)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            module = node.module or ""
            if node.level:  # relative to the blob package
                anchor = BLOB_PACKAGE.rsplit(".", node.level - 1)[0]
                module = f"{anchor}.{module}" if module else anchor
            found.append((module, self.function))

    Visitor().visit(ast.parse(path.read_text()))
    return found


def test_blob_modules_import_only_their_layer() -> None:
    root = pathlib.Path(blob_package.__file__).parent
    modules = {path.stem: _imports(path) for path in sorted(root.glob("*.py"))}
    assert {"backend", "mooncake", "mooncake_setup", "store", "keys"} <= set(modules)
    sdk_imports = []
    for name, imports in modules.items():
        for module, function in imports:
            top = module.split(".")[0]
            where = f"blob/{name}.py imports {module}"
            if top == "mooncake":
                sdk_imports.append((name, function))
                continue
            assert top != "torch" and not module.startswith("tensorrt_llm._torch.pyexecutor"), where
            if top == "tensorrt_llm":
                assert module == "tensorrt_llm.logger" or module.startswith(DISAGGREGATION), where
            else:
                assert top in sys.stdlib_module_names or top in THIRD_PARTY_ALLOWED, where
    assert sdk_imports == [("mooncake_setup", "open_sdk")]

    def package_imports(name: str) -> set[str]:
        return {m for m, _ in modules[name] if m.startswith(BLOB_PACKAGE + ".")}

    assert package_imports("mooncake") == {f"{BLOB_PACKAGE}.store"}
    assert package_imports("mooncake_setup") == {f"{BLOB_PACKAGE}.mooncake"}
    memory_kind = f"{DISAGGREGATION}.base.memory"  # all the boundary shares with the contract
    for name in ("store", "keys"):
        for module, _ in modules[name]:
            assert module.split(".")[0] in sys.stdlib_module_names or module == memory_kind, name

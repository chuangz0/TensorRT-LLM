# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``MooncakeStoreConfig`` validation, the ``KeyScheme`` re-encoding, and the lazy import in
``open_mooncake_client``. No backend, no threads."""

import dataclasses
import sys
import types

import pytest

__extra_import_path__ = ["~/tensorrt_llm/_torch"]
from disaggregation.backends.store import client as client_module  # noqa: E402
from disaggregation.backends.store.client import open_mooncake_client  # noqa: E402
from disaggregation.backends.store.config import (  # noqa: E402
    DEFAULT_METADATA_SERVER,
    MooncakeStoreConfig,
)
from disaggregation.backends.store.keys import KeyScheme  # noqa: E402

# ---- config ----


def test_config_defaults_are_the_documented_ones():
    cfg = MooncakeStoreConfig("master:50051")
    assert cfg.metadata_server == DEFAULT_METADATA_SERVER == "P2PHANDSHAKE"
    assert cfg.protocol == "rdma" and cfg.local_hostname is None
    assert cfg.namespace == "trtllm" and cfg.stage_through_host is False
    assert cfg.transfer_batch_size > 0 and cfg.max_inflight_ops > 0 and cfg.num_workers > 0
    assert cfg.probe_ttl_s > 0


def test_config_is_frozen():
    cfg = MooncakeStoreConfig("master:50051")
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.namespace = "other"


@pytest.mark.parametrize(
    "overrides, needle",
    [
        ({"master_server_address": ""}, "master_server_address"),
        ({"metadata_server": ""}, "metadata_server"),
        ({"namespace": ""}, "namespace"),
        ({"global_segment_size": -1}, "global_segment_size"),
        ({"local_buffer_size": 0}, "local_buffer_size"),
        ({"transfer_batch_size": 0}, "transfer_batch_size"),
        ({"max_inflight_ops": 0}, "max_inflight_ops"),
        ({"num_workers": -2}, "num_workers"),
        ({"probe_ttl_s": 0.0}, "probe_ttl_s"),
        ({"protocol": "ucx"}, "protocol"),
        ({"protocol": ""}, "protocol"),
        ({"protocol": "RDMA"}, "protocol"),
        ({"stage_through_host": True, "staging_buffer_bytes": 0}, "staging_buffer_bytes"),
    ],
)
def test_config_rejects_bad_values_naming_the_field(overrides, needle):
    kwargs = {"master_server_address": "master:50051", **overrides}
    with pytest.raises(ValueError, match=needle):
        MooncakeStoreConfig(**kwargs)


@pytest.mark.parametrize("protocol", ["tcp", "rdma"])
def test_config_accepts_the_two_transport_protocols(protocol):
    assert MooncakeStoreConfig("m:1", protocol=protocol).protocol == protocol


def test_config_allows_zero_global_segment_and_unchecked_staging_bytes_when_not_staging():
    cfg = MooncakeStoreConfig("m:1", global_segment_size=0, staging_buffer_bytes=0)
    assert cfg.global_segment_size == 0 and cfg.stage_through_host is False


def test_from_dict_round_trips_known_keys():
    raw = {
        "master_server_address": "10.0.0.1:50051",
        "protocol": "tcp",
        "namespace": "ns",
        "num_workers": 3,
        "stage_through_host": True,
        "staging_buffer_bytes": 4096,
    }
    cfg = MooncakeStoreConfig.from_dict(raw)
    assert cfg == MooncakeStoreConfig(**raw)
    assert cfg.protocol == "tcp" and cfg.num_workers == 3 and cfg.staging_buffer_bytes == 4096


def test_from_dict_rejects_unknown_keys_and_names_them_all():
    raw = {"master_server_address": "m:1", "zzz_typo": 1, "another": 2}
    with pytest.raises(ValueError) as info:
        MooncakeStoreConfig.from_dict(raw)
    assert "zzz_typo" in str(info.value) and "another" in str(info.value)
    assert "master_server_address" not in str(info.value).split(":", 1)[1]


def test_from_dict_still_validates_values():
    with pytest.raises(ValueError, match="num_workers"):
        MooncakeStoreConfig.from_dict({"master_server_address": "m:1", "num_workers": 0})


def test_from_dict_requires_master_address():
    with pytest.raises(TypeError):
        MooncakeStoreConfig.from_dict({"protocol": "tcp"})


# ---- keys ----


class _OpaqueName(bytes):
    """A name that objects to being interpreted: any decode is a contract breach."""

    def decode(self, *args, **kwargs):  # noqa: D401
        raise AssertionError("keys.py decoded a unit name")

    def __str__(self):
        raise AssertionError("keys.py stringified a unit name")


@pytest.mark.parametrize(
    "name",
    [b"", b"\x00", b"\x00\xff/\\ \n", bytes(range(256)), b"/" * 7, "unicode-é".encode()],
)
def test_key_is_three_slash_separated_hex_components_and_inverts(name):
    scheme = KeyScheme("ns", b"\xde\xad")
    key = scheme.key(_OpaqueName(name))
    namespace, fingerprint_hex, name_hex = key.split("/")
    assert namespace == "ns" and fingerprint_hex == "dead" and name_hex == name.hex()
    assert key.startswith(scheme.prefix + "/")
    assert scheme.name(key) == name
    # Two distinct names never collide, whatever bytes they carry.
    assert scheme.key(name + b"\x00") != key


def test_name_rejects_a_key_of_another_scheme():
    ours, theirs = KeyScheme("ns", b"\x01"), KeyScheme("ns", b"\x02")
    key = theirs.key(b"abc")
    with pytest.raises(ValueError):
        ours.name(key)
    with pytest.raises(ValueError):
        KeyScheme("other", b"\x01").name(ours.key(b"abc"))


def test_same_name_under_different_fingerprints_or_namespaces_are_different_keys():
    a, b, c = KeyScheme("ns", b"\x01"), KeyScheme("ns", b"\x02"), KeyScheme("ns2", b"\x01")
    keys = {s.key(b"same") for s in (a, b, c)}
    assert len(keys) == 3


def test_scheme_rejects_empty_namespace_or_fingerprint():
    with pytest.raises(ValueError):
        KeyScheme("", b"\x01")
    with pytest.raises(ValueError):
        KeyScheme("ns", b"")


# ---- open_mooncake_client ----


def test_open_client_reports_missing_bindings_with_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "mooncake", None)
    monkeypatch.setitem(sys.modules, "mooncake.store", None)
    with pytest.raises(ImportError, match="mooncake-transfer-engine") as info:
        open_mooncake_client(MooncakeStoreConfig("m:1"))
    assert isinstance(info.value.__cause__, ImportError)


def _install_fake_bindings(monkeypatch, status: int):
    calls = []

    class MooncakeDistributedStore:
        def setup(self, *args):
            calls.append(args)
            return status

    pkg = types.ModuleType("mooncake")
    store = types.ModuleType("mooncake.store")
    store.MooncakeDistributedStore = MooncakeDistributedStore
    pkg.store = store
    monkeypatch.setitem(sys.modules, "mooncake", pkg)
    monkeypatch.setitem(sys.modules, "mooncake.store", store)
    return calls, MooncakeDistributedStore


def test_open_client_passes_config_fields_in_setup_order(monkeypatch):
    calls, cls = _install_fake_bindings(monkeypatch, status=0)
    cfg = MooncakeStoreConfig(
        "10.0.0.2:50051",
        local_hostname="10.0.0.9",
        metadata_server="etcd://x",
        protocol="tcp",
        device_name="mlx5_0",
        global_segment_size=123,
        local_buffer_size=45,
    )
    store = open_mooncake_client(cfg)
    assert isinstance(store, cls)
    assert calls == [("10.0.0.9", "etcd://x", 123, 45, "tcp", "mlx5_0", "10.0.0.2:50051")]


def test_open_client_defaults_hostname_when_unset(monkeypatch):
    calls, _ = _install_fake_bindings(monkeypatch, status=0)
    monkeypatch.setattr(client_module, "_default_hostname", lambda: "127.0.0.42")
    open_mooncake_client(MooncakeStoreConfig("m:1"))
    assert calls[0][0] == "127.0.0.42"


def test_open_client_turns_nonzero_setup_status_into_runtime_error(monkeypatch):
    _install_fake_bindings(monkeypatch, status=-13)
    with pytest.raises(RuntimeError, match=r"status -13") as info:
        open_mooncake_client(MooncakeStoreConfig("m:1", local_hostname="h", protocol="tcp"))
    assert "m:1" in str(info.value) and "tcp" in str(info.value)

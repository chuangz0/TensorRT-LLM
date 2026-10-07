# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``BlobStoreConfig`` (the backend's options) and ``MooncakeStoreConfig`` (the driver's) as
two disjoint halves of one flat entry, and the ``KeyScheme`` re-encoding. No backend, no
threads."""

import dataclasses

import pytest

__extra_import_path__ = ["~/tensorrt_llm/_torch"]
from disaggregation.backends.blob.backend import BlobStoreConfig  # noqa: E402
from disaggregation.backends.blob.drivers.mooncake import (  # noqa: E402
    DEFAULT_METADATA_SERVER,
    MooncakeStoreConfig,
)
from disaggregation.backends.blob.keys import KeyScheme  # noqa: E402

# ---- the two halves ----


def test_backend_and_driver_fields_do_not_overlap():
    # One flat entry is split by key name, so a key must belong to exactly one side.
    assert not BlobStoreConfig.fields() & MooncakeStoreConfig.fields()


def test_backend_fields_are_the_documented_nine():
    assert BlobStoreConfig.fields() == {
        "namespace",
        "transfer_batch_size",
        "landing",
        "publish_buffer_bytes",
        "landing_buffer_bytes",
        "max_landed_units",
        "max_inflight_ops",
        "num_workers",
        "probe_ttl_s",
    }


# ---- BlobStoreConfig ----


def test_backend_config_defaults_are_the_documented_ones():
    cfg = BlobStoreConfig()
    assert cfg.namespace == "trtllm" and cfg.landing == "device" and not cfg.lands_on_host
    assert cfg.transfer_batch_size > 0 and cfg.max_inflight_ops > 0 and cfg.num_workers > 0
    assert cfg.probe_ttl_s > 0
    assert cfg.publish_buffer_bytes == 512 << 20 and cfg.landing_buffer_bytes == 2 << 30
    assert cfg.max_landed_units is None  # every slot the landing budget affords


def test_configs_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        BlobStoreConfig().namespace = "other"
    with pytest.raises(dataclasses.FrozenInstanceError):
        MooncakeStoreConfig("master:50051").protocol = "tcp"


@pytest.mark.parametrize(
    "overrides, needle",
    [
        ({"namespace": ""}, "namespace"),
        ({"transfer_batch_size": 0}, "transfer_batch_size"),
        ({"max_inflight_ops": 0}, "max_inflight_ops"),
        ({"num_workers": -2}, "num_workers"),
        ({"probe_ttl_s": 0.0}, "probe_ttl_s"),
        ({"landing": "gpu"}, "landing"),
        ({"landing": True}, "landing"),
        ({"landing": "host", "publish_buffer_bytes": 0}, "publish_buffer_bytes"),
        ({"landing": "host", "landing_buffer_bytes": 0}, "landing_buffer_bytes"),
        ({"max_landed_units": 0}, "max_landed_units"),
        # The wrong type is a bad value too, not a TypeError from the first comparison.
        ({"namespace": 7}, "namespace must be a non-empty string"),
        ({"transfer_batch_size": "4"}, "transfer_batch_size must be an integer"),
        ({"num_workers": 2.0}, "num_workers must be an integer"),
        ({"num_workers": True}, "num_workers must be an integer"),
        ({"publish_buffer_bytes": "512M"}, "publish_buffer_bytes must be an integer"),
        ({"max_landed_units": "3"}, "max_landed_units must be an integer"),
        ({"probe_ttl_s": "1"}, "probe_ttl_s must be a number"),
    ],
)
def test_backend_config_rejects_bad_values_naming_the_field(overrides, needle):
    with pytest.raises(ValueError, match=needle):
        BlobStoreConfig(**overrides)


def test_backend_from_dict_reports_a_wrongly_typed_value_as_a_value_error():
    with pytest.raises(ValueError, match="num_workers must be an integer, got '2'"):
        BlobStoreConfig.from_dict({"num_workers": "2"})


def test_backend_config_leaves_pool_budgets_unchecked_when_landing_on_device():
    cfg = BlobStoreConfig(publish_buffer_bytes=0, landing_buffer_bytes=0)
    assert cfg.publish_buffer_bytes == 0 and cfg.landing_buffer_bytes == 0
    assert cfg.landing == "device"


def test_backend_from_dict_round_trips_known_keys():
    raw = {
        "namespace": "ns",
        "num_workers": 3,
        "landing": "host",
        "publish_buffer_bytes": 4096,
        "landing_buffer_bytes": 8192,
        "max_landed_units": 16,
    }
    cfg = BlobStoreConfig.from_dict(raw)
    assert cfg == BlobStoreConfig(**raw)
    assert cfg.num_workers == 3 and cfg.lands_on_host and cfg.max_landed_units == 16


@pytest.mark.parametrize("old_key", ["stage_through_host", "staging_buffer_bytes"])
def test_backend_from_dict_refuses_a_retired_key_by_name(old_key):
    with pytest.raises(ValueError, match=old_key):
        BlobStoreConfig.from_dict({old_key: 4096})


# ---- MooncakeStoreConfig ----


def test_driver_config_defaults_are_the_documented_ones():
    cfg = MooncakeStoreConfig("master:50051")
    assert cfg.metadata_server == DEFAULT_METADATA_SERVER == "P2PHANDSHAKE"
    assert cfg.protocol == "rdma" and cfg.local_hostname is None and cfg.device_name == ""
    assert cfg.global_segment_size > 0 and cfg.local_buffer_size > 0


@pytest.mark.parametrize(
    "overrides, needle",
    [
        ({"master_server_address": ""}, "master_server_address"),
        ({"metadata_server": ""}, "metadata_server"),
        ({"global_segment_size": -1}, "global_segment_size"),
        ({"local_buffer_size": 0}, "local_buffer_size"),
        ({"protocol": "ucx"}, "protocol"),
        ({"protocol": ""}, "protocol"),
        ({"protocol": "RDMA"}, "protocol"),
    ],
)
def test_driver_config_rejects_bad_values_naming_the_field(overrides, needle):
    kwargs = {"master_server_address": "master:50051", **overrides}
    with pytest.raises(ValueError, match=needle):
        MooncakeStoreConfig(**kwargs)


@pytest.mark.parametrize("protocol", ["tcp", "rdma"])
def test_driver_config_accepts_the_two_transport_protocols(protocol):
    assert MooncakeStoreConfig("m:1", protocol=protocol).protocol == protocol


def test_driver_config_allows_zero_global_segment():
    assert MooncakeStoreConfig("m:1", global_segment_size=0).global_segment_size == 0


def test_driver_from_dict_round_trips_known_keys():
    raw = {"master_server_address": "10.0.0.1:50051", "protocol": "tcp", "local_hostname": "h"}
    cfg = MooncakeStoreConfig.from_dict(raw)
    assert cfg == MooncakeStoreConfig(**raw)
    assert cfg.protocol == "tcp" and cfg.local_hostname == "h"


def test_driver_from_dict_rejects_unknown_keys_and_names_them_all():
    raw = {"master_server_address": "m:1", "zzz_typo": 1, "another": 2}
    with pytest.raises(ValueError) as info:
        MooncakeStoreConfig.from_dict(raw)
    assert "zzz_typo" in str(info.value) and "another" in str(info.value)
    assert "master_server_address" not in str(info.value).split(":", 1)[1]


def test_driver_from_dict_still_validates_values():
    with pytest.raises(ValueError, match="local_buffer_size"):
        MooncakeStoreConfig.from_dict({"master_server_address": "m:1", "local_buffer_size": 0})


def test_driver_from_dict_requires_master_address():
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

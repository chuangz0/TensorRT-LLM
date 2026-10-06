# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``MooncakeBlobStore`` over a fake ``MooncakeDistributedStore``: the translation of the
bindings' status codes into the ``BlobStore`` protocol, and ``open`` over the lazily imported
bindings. No backend, no threads, no master."""

import logging
import sys
import types
from collections import deque

import pytest

__extra_import_path__ = ["~/tensorrt_llm/_torch"]
from disaggregation.backends.blob.drivers import mooncake as driver_module  # noqa: E402
from disaggregation.backends.blob.drivers.mooncake import (  # noqa: E402
    LEASE_EXPIRED,
    NO_AVAILABLE_HANDLE,
    OBJECT_ALREADY_EXISTS,
    OBJECT_NOT_FOUND,
    MooncakeBlobStore,
    MooncakeStoreConfig,
)
from disaggregation.backends.blob.store import BlobStoreError, GetStatus, PutStatus  # noqa: E402

KEYS = ["k0", "k1", "k2"]
BUFFERS = [[(0x1000, 64)], [(0x2000, 40), (0x2100, 24)], [(0x3000, 16)]]


class _FakeBindings:
    """Shaped like ``MooncakeDistributedStore``; every method answers what the test set: a
    fixed value, or, given a ``deque``, one entry per call."""

    def __init__(self, **answers):
        self.answers = answers
        self.calls = []

    def __getattr__(self, name):
        if name not in self.answers:
            raise AttributeError(name)

        def method(*args):
            self.calls.append((name, args))
            answer = self.answers[name]
            return answer.popleft() if isinstance(answer, deque) else answer

        return method


def _store(**answers) -> MooncakeBlobStore:
    return MooncakeBlobStore(_FakeBindings(**answers), MooncakeStoreConfig("m:1", protocol="tcp"))


# ---- translation ----


def test_holds_translates_one_and_zero_to_booleans():
    store = _store(batch_is_exist=[1, 0, 1])
    assert store.holds(KEYS) == [True, False, True]
    assert store.raw.calls == [("batch_is_exist", (KEYS,))]


def test_holds_raises_on_a_negative_status_or_the_wrong_count():
    for answer in ([1, -1, 0], [1, 0], []):
        with pytest.raises(BlobStoreError, match="batch_is_exist"):
            _store(batch_is_exist=answer).holds(KEYS)


def test_get_full_read_is_hit_and_not_found_is_miss():
    store = _store(batch_get_into_multi_buffers=[64, OBJECT_NOT_FOUND, 16])
    assert store.get(KEYS, BUFFERS) == [GetStatus.HIT, GetStatus.MISS, GetStatus.HIT]
    (call,) = store.raw.calls
    assert call == (
        "batch_get_into_multi_buffers",
        (KEYS, [[0x1000], [0x2000, 0x2100], [0x3000]], [[64], [40, 24], [16]]),
    )


def test_get_other_codes_and_short_reads_are_failed_and_a_wrong_count_raises():
    for code in (-600, -800, 0, 1, 63, 65):
        assert _store(batch_get_into_multi_buffers=[code]).get(KEYS[:1], BUFFERS[:1]) == [
            GetStatus.FAILED
        ]
    # An answer of the wrong length is a failed call: none of it can be trusted.
    with pytest.raises(BlobStoreError, match="batch_get_into_multi_buffers answered 1 of 3"):
        _store(batch_get_into_multi_buffers=[64]).get(KEYS, BUFFERS)


def test_get_asks_once_more_for_the_keys_whose_lease_expired(caplog):
    """``-707`` means the bytes landed but the lease the get itself took ran out before the check
    after the transfer; a second get takes a fresh lease. Only the expired keys are asked again,
    with their own buffers, and the retry's answer stands whatever it is."""
    store = _store(
        batch_get_into_multi_buffers=deque(
            [[LEASE_EXPIRED, 64, LEASE_EXPIRED], [64, OBJECT_NOT_FOUND]]
        )
    )
    with caplog.at_level(logging.INFO, logger=driver_module.__name__):
        assert store.get(KEYS, BUFFERS) == [GetStatus.HIT, GetStatus.HIT, GetStatus.MISS]
    first, second = store.raw.calls
    assert first[1][0] == KEYS
    assert second == (
        "batch_get_into_multi_buffers",
        ([KEYS[0], KEYS[2]], [[0x1000], [0x3000]], [[64], [16]]),
    )
    (record,) = [r for r in caplog.records if "lease expired" in r.getMessage()]
    assert record.levelno == logging.INFO and "2 of 3" in record.getMessage()


def test_get_retries_the_lease_once_only():
    store = _store(batch_get_into_multi_buffers=deque([[LEASE_EXPIRED], [LEASE_EXPIRED]]))
    assert store.get(KEYS[:1], BUFFERS[:1]) == [GetStatus.FAILED]
    assert len(store.raw.calls) == 2


def test_put_translates_stored_declined_and_failed_codes(caplog):
    """``0`` is stored. ``OBJECT_ALREADY_EXISTS`` and ``NO_AVAILABLE_HANDLE`` mean the store chose
    not to hold the key: declined, logged at debug. Any other code is a key that could not be
    written: failed, logged at warning. A wrong count raises."""
    store = _store(batch_put_from_multi_buffers=[0, OBJECT_ALREADY_EXISTS, NO_AVAILABLE_HANDLE])
    with caplog.at_level(logging.DEBUG, logger=driver_module.__name__):
        assert store.put(KEYS, BUFFERS) == [
            PutStatus.STORED,
            PutStatus.DECLINED,
            PutStatus.DECLINED,
        ]
    (call,) = store.raw.calls
    assert call[1][0] == KEYS and call[1][1] == [[0x1000], [0x2000, 0x2100], [0x3000]]
    declined = [r for r in caplog.records if "declined" in r.getMessage()]
    assert [r.levelno for r in declined] == [logging.DEBUG, logging.DEBUG]
    assert "put of k1 declined with status -705" in declined[0].getMessage()
    assert "put of k2 declined with status -200" in declined[1].getMessage()
    for code in (-1, -600, -704, -800, -900, 1):
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger=driver_module.__name__):
            assert _store(batch_put_from_multi_buffers=[code]).put(KEYS[:1], BUFFERS[:1]) == [
                PutStatus.FAILED
            ]
        (record,) = [r for r in caplog.records if "put of k0 failed" in r.getMessage()]
        assert record.levelno == logging.WARNING and f"status {code}" in record.getMessage()
    with pytest.raises(BlobStoreError, match="batch_put_from_multi_buffers answered 1 of 3"):
        _store(batch_put_from_multi_buffers=[0]).put(KEYS, BUFFERS)


def test_register_and_unregister_raise_on_a_nonzero_status_and_close_does_not():
    ok = _store(register_buffer=0, unregister_buffer=0, close=0)
    ok.register_span(0x1000, 64)
    ok.unregister_span(0x1000, 64)
    ok.close()
    assert [name for name, _ in ok.raw.calls] == ["register_buffer", "unregister_buffer", "close"]
    assert ok.raw.calls[1] == ("unregister_buffer", (0x1000,))  # the bindings unregister by address
    bad = _store(register_buffer=-5, unregister_buffer=-6, close=-7)
    with pytest.raises(BlobStoreError, match="register_buffer failed with status -5"):
        bad.register_span(0x1000, 64)
    with pytest.raises(BlobStoreError, match="unregister_buffer failed with status -6"):
        bad.unregister_span(0x1000, 64)
    bad.close()  # a failed close is logged, not raised
    assert "mooncake" in bad.describe() and "m:1" in bad.describe() and "tcp" in bad.describe()


# ---- open ----


def test_open_reports_missing_bindings_with_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "mooncake", None)
    monkeypatch.setitem(sys.modules, "mooncake.store", None)
    with pytest.raises(ImportError, match="mooncake-transfer-engine") as info:
        MooncakeBlobStore.open(MooncakeStoreConfig("m:1"))
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


def test_open_passes_config_fields_in_setup_order(monkeypatch):
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
    store = MooncakeBlobStore.open(cfg)
    assert isinstance(store, MooncakeBlobStore) and isinstance(store.raw, cls)
    assert calls == [("10.0.0.9", "etcd://x", 123, 45, "tcp", "mlx5_0", "10.0.0.2:50051")]


def test_open_defaults_hostname_when_unset(monkeypatch):
    calls, _ = _install_fake_bindings(monkeypatch, status=0)
    monkeypatch.setattr(driver_module, "_default_hostname", lambda: "127.0.0.42")
    MooncakeBlobStore.open(MooncakeStoreConfig("m:1"))
    assert calls[0][0] == "127.0.0.42"


def test_open_turns_nonzero_setup_status_into_blob_store_error(monkeypatch):
    _install_fake_bindings(monkeypatch, status=-13)
    with pytest.raises(BlobStoreError, match=r"status -13") as info:
        MooncakeBlobStore.open(MooncakeStoreConfig("m:1", local_hostname="h", protocol="tcp"))
    assert "m:1" in str(info.value) and "tcp" in str(info.value)

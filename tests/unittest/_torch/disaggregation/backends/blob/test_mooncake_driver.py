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


def test_put_zero_is_stored_and_any_other_status_is_declined_with_the_codes_logged(caplog):
    store = _store(batch_put_from_multi_buffers=[0, -1, -704])
    with caplog.at_level(logging.DEBUG, logger=driver_module.__name__):
        assert store.put(KEYS, BUFFERS) == [
            PutStatus.STORED,
            PutStatus.DECLINED,
            PutStatus.DECLINED,
        ]
    (call,) = store.raw.calls
    assert call[1][0] == KEYS and call[1][1] == [[0x1000], [0x2000, 0x2100], [0x3000]]
    (record,) = [r for r in caplog.records if "put declined" in r.getMessage()]
    assert record.levelno == logging.DEBUG and "[-704, -1]" in record.getMessage()
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


def test_a_span_over_several_cuda_allocations_registers_and_unregisters_each(monkeypatch):
    pieces = [(0x1000, 0x100), (0x1100, 0x100), (0x1200, 0x80)]
    monkeypatch.setattr(driver_module, "_allocation_spans", lambda address, size: pieces)
    store = _store(register_buffer=0, unregister_buffer=0)
    store.register_span(0x1000, 0x280)
    store.unregister_span(0x1000, 0x280)
    assert store.raw.calls == [("register_buffer", piece) for piece in pieces] + [
        ("unregister_buffer", (start,)) for start, _ in pieces
    ]


def test_a_piece_that_fails_to_register_unregisters_the_pieces_before_it(monkeypatch):
    pieces = [(0x1000, 0x100), (0x1100, 0x100), (0x1200, 0x80)]
    monkeypatch.setattr(driver_module, "_allocation_spans", lambda address, size: pieces)
    store = _store(register_buffer=deque([0, 0, -600]), unregister_buffer=0)
    with pytest.raises(BlobStoreError, match=r"status -600 \(piece 3 of 3\)"):
        store.register_span(0x1000, 0x280)
    assert store.raw.calls[3:] == [
        ("unregister_buffer", (0x1000,)),
        ("unregister_buffer", (0x1100,)),
    ]


def _map_chunks(cuda, device: int, count: int):
    """``count`` chunks mapped back to back into one reserved range, with the properties KV cache
    manager V2 prefers (an exportable handle, GPU-direct RDMA capable). Returns (base, chunk,
    handles), or None when no such property is supported."""
    handle_types = [
        cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC,
        cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
        cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE,
    ]
    for handle_type in handle_types:
        prop = cuda.CUmemAllocationProp()
        prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = device
        prop.requestedHandleTypes = handle_type
        prop.allocFlags.gpuDirectRDMACapable = 1
        err, granularity = cuda.cuMemGetAllocationGranularity(
            prop, cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
        )
        if err != cuda.CUresult.CUDA_SUCCESS:
            continue
        err, handle = cuda.cuMemCreate(granularity, prop, 0)
        if err != cuda.CUresult.CUDA_SUCCESS:
            continue
        handles = [handle]
        for _ in range(count - 1):
            err, handle = cuda.cuMemCreate(granularity, prop, 0)
            assert err == cuda.CUresult.CUDA_SUCCESS
            handles.append(handle)
        err, base = cuda.cuMemAddressReserve(granularity * count, 0, 0, 0)
        assert err == cuda.CUresult.CUDA_SUCCESS
        for i, handle in enumerate(handles):
            (err,) = cuda.cuMemMap(int(base) + i * granularity, granularity, 0, handle, 0)
            assert err == cuda.CUresult.CUDA_SUCCESS
        access = cuda.CUmemAccessDesc()
        access.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = device
        access.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        (err,) = cuda.cuMemSetAccess(base, granularity * count, [access], 1)
        assert err == cuda.CUresult.CUDA_SUCCESS
        return int(base), granularity, handles
    return None


def test_allocation_spans_tile_a_vmm_range_one_allocation_per_piece():
    """Over a range of ``cuMemCreate`` chunks mapped back to back, as KV cache manager V2 lays out
    a pool, the pieces tile the span exactly and none crosses an allocation the driver reports.
    (Whether neighbouring chunks are reported as one allocation is the driver's business.)"""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    cuda = pytest.importorskip("cuda.bindings.driver")
    torch.zeros(1, device="cuda")
    count = 4
    mapped = _map_chunks(cuda, torch.cuda.current_device(), count)
    if mapped is None:
        pytest.skip("no supported VMM allocation property")
    base, chunk, handles = mapped
    try:
        for start, size in ((base, chunk * count), (base + chunk // 2, chunk * 2)):
            pieces = driver_module._allocation_spans(start, size)
            assert len(pieces) > 1, pieces  # a whole span back means the lookup failed
            assert pieces[0][0] == start
            assert sum(length for _, length in pieces) == size
            for (a, la), (b, _) in zip(pieces, pieces[1:]):
                assert a + la == b
            for piece_start, length in pieces:
                err, alloc_base, alloc_size = cuda.cuMemGetAddressRange(piece_start)
                assert err == cuda.CUresult.CUDA_SUCCESS
                assert int(alloc_base) <= piece_start
                assert piece_start + length <= int(alloc_base) + int(alloc_size)
        host = torch.empty(4096, dtype=torch.uint8)
        assert driver_module._allocation_spans(host.data_ptr(), 4096) == [(host.data_ptr(), 4096)]
    finally:
        for i, handle in enumerate(handles):
            cuda.cuMemUnmap(base + i * chunk, chunk)
            cuda.cuMemRelease(handle)
        cuda.cuMemAddressFree(base, chunk * count)


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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``MooncakeBlobStore`` over a fake ``MooncakeDistributedStore``: the translation of the
bindings' status codes into the ``BlobStore`` protocol, and ``open`` over the lazily imported
bindings. No backend, no threads, no master."""

import contextlib
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
    assert store.contains(KEYS) == [True, False, True]
    assert store.raw.calls == [("batch_is_exist", (KEYS,))]


def test_holds_raises_on_a_negative_status_or_the_wrong_count():
    for answer in ([1, -1, 0], [1, 0], []):
        with pytest.raises(BlobStoreError, match="batch_is_exist"):
            _store(batch_is_exist=answer).contains(KEYS)


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
    with pytest.raises(
        BlobStoreError, match=r"status -600 for \[0x1200, 0x1280\) \(piece 3 of 3\)"
    ):
        store.register_span(0x1000, 0x280)
    assert store.raw.calls[3:] == [
        ("unregister_buffer", (0x1000,)),
        ("unregister_buffer", (0x1100,)),
    ]


def test_a_piece_that_fails_to_unregister_is_kept_for_a_retry_and_the_rest_are_released(
    monkeypatch,
):
    """A failed piece does not stop the pieces after it from being released, and only it stays on
    the books: the backend keeps the handle live after the error, and the retry it allows asks
    the bindings for that piece alone."""
    pieces = [(0x1000, 0x100), (0x1100, 0x100), (0x1200, 0x80)]
    monkeypatch.setattr(driver_module, "_allocation_spans", lambda address, size: pieces)
    store = _store(register_buffer=0, unregister_buffer=deque([0, -6, 0, 0]))
    store.register_span(0x1000, 0x280)
    with pytest.raises(BlobStoreError, match=r"status -6 for \[0x1100, 0x1200\) \(1 of 3 pieces\)"):
        store.unregister_span(0x1000, 0x280)
    assert store.raw.calls[3:] == [("unregister_buffer", (start,)) for start, _ in pieces]
    store.unregister_span(0x1000, 0x280)
    assert store.raw.calls[6:] == [("unregister_buffer", (0x1100,))]


_HANDLE_TYPES = (
    "CU_MEM_HANDLE_TYPE_FABRIC",
    "CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR",
    "CU_MEM_HANDLE_TYPE_NONE",
)
"""The exportable handle types KV cache manager V2 asks for, most preferred first."""


def _chunk_property(cuda, device: int, handle_type):
    prop = cuda.CUmemAllocationProp()
    prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device
    prop.requestedHandleTypes = handle_type
    prop.allocFlags.gpuDirectRDMACapable = 1
    return prop


@contextlib.contextmanager
def _mapped_chunks(cuda, device: int, count: int):
    """``count`` chunks of one allocation granularity each, mapped back to back into one reserved
    range, with the properties KV cache manager V2 prefers (an exportable handle, GPU-direct RDMA
    capable). Yields ``(base, chunk)``, or ``None`` when the device supports no such property.
    The mappings, the reservation and the handles are released on exit however the body ended."""
    ok = cuda.CUresult.CUDA_SUCCESS
    minimum = cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
    with contextlib.ExitStack() as release:
        for name in _HANDLE_TYPES:
            prop = _chunk_property(cuda, device, getattr(cuda.CUmemAllocationHandleType, name))
            err, chunk = cuda.cuMemGetAllocationGranularity(prop, minimum)
            if err != ok:
                continue
            err, handle = cuda.cuMemCreate(chunk, prop, 0)
            if err == ok:
                break
        else:
            yield None
            return
        release.callback(cuda.cuMemRelease, handle)
        handles = [handle]
        for _ in range(count - 1):
            err, handle = cuda.cuMemCreate(chunk, prop, 0)
            assert err == ok
            release.callback(cuda.cuMemRelease, handle)
            handles.append(handle)
        err, reserved = cuda.cuMemAddressReserve(chunk * count, 0, 0, 0)
        assert err == ok
        release.callback(cuda.cuMemAddressFree, reserved, chunk * count)
        base = int(reserved)
        for i, handle in enumerate(handles):
            (err,) = cuda.cuMemMap(base + i * chunk, chunk, 0, handle, 0)
            assert err == ok
            release.callback(cuda.cuMemUnmap, base + i * chunk, chunk)
        access = cuda.CUmemAccessDesc()
        access.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = device
        access.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        (err,) = cuda.cuMemSetAccess(base, chunk * count, [access], 1)
        assert err == ok
        yield base, chunk


def _reported_allocations(cuda, base: int, chunk: int, count: int) -> list[tuple[int, int]]:
    """The allocations ``cuMemGetAddressRange`` reports for the chunks at ``base``, each once, in
    address order."""
    ranges = []
    for i in range(count):
        err, start, length = cuda.cuMemGetAddressRange(base + i * chunk)
        assert err == cuda.CUresult.CUDA_SUCCESS
        if (int(start), int(length)) not in ranges:
            ranges.append((int(start), int(length)))
    return ranges


def _cut(start: int, size: int, ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """``[start, start + size)`` cut at the boundaries of ``ranges``: the part inside each."""
    pieces = []
    for base, length in ranges:
        lo, hi = max(start, base), min(start + size, base + length)
        if lo < hi:
            pieces.append((lo, hi - lo))
    return pieces


def test_allocation_spans_cut_a_vmm_range_at_the_allocations_the_driver_reports():
    """Over a range of ``cuMemCreate`` chunks mapped back to back, as KV cache manager V2 lays out
    a pool, the pieces are exactly the span cut at the boundaries ``cuMemGetAddressRange`` reports
    for the chunks, whether the driver reports every chunk as its own allocation or merges some.
    A driver that reports the whole range as one allocation leaves nothing to cut, so the test
    skips. Pinned host memory, as the staging pool uses, comes back whole."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    cuda = pytest.importorskip("cuda.bindings.driver")
    torch.zeros(1, device="cuda")
    count = 4
    with _mapped_chunks(cuda, torch.cuda.current_device(), count) as mapped:
        if mapped is None:
            pytest.skip("no supported VMM allocation property")
        base, chunk = mapped
        ranges = _reported_allocations(cuda, base, chunk, count)
        if len(ranges) == 1:
            pytest.skip("the CUDA driver reports the mapped chunks as one allocation")
        for start, size in ((base, chunk * count), (base + chunk // 2, chunk * 2)):
            assert driver_module._allocation_spans(start, size) == _cut(start, size, ranges)
        host = torch.empty(4096, dtype=torch.uint8, pin_memory=True)
        assert driver_module._allocation_spans(host.data_ptr(), 4096) == [(host.data_ptr(), 4096)]


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

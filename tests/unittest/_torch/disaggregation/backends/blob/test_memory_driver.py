# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``type: memory`` through the registry and ``BlobStoreBackend`` over a bare ``MemoryBlobStore``:
the second driver exercises ``factory.py`` and the backend without any knobs in between."""

import importlib

__extra_import_path__ = ["~/tensorrt_llm/_torch"]
from disaggregation.backends.blob.backend import BlobStoreBackend, BlobStoreConfig  # noqa: E402
from disaggregation.backends.blob.drivers.memory import MemoryBlobStore  # noqa: E402
from disaggregation.backends.blob.store import GetStatus  # noqa: E402
from disaggregation.backends.config import BackendEntry, KVTransferConfig  # noqa: E402
from disaggregation.backends.registry import (  # noqa: E402
    BackendBuildContext,
    BackendHandle,
    build_backends,
    close_backends,
)
from disaggregation.base.cache_backend import Delivered, Unit  # noqa: E402
from store_fakes import (  # noqa: E402
    FINGERPRINT,
    ArenaResolver,
    MemoryArena,
    extent,
    pattern,
    read,
    wait_until,
    write,
)


def test_memory_type_builds_a_blob_store_backend_that_registers_pools():
    # The registry imports the driver by name when it builds; the file-scoped import path (see
    # ``conftest.py``) resolves that name only for imports made from this file, so make it here.
    importlib.import_module("disaggregation.backends.blob.drivers.memory")
    entry = BackendEntry.from_dict({"name": "local", "type": "memory", "namespace": "t"})
    context = BackendBuildContext(
        resolver=ArenaResolver(MemoryArena(1 << 10)),
        layout_fingerprint=FINGERPRINT,
        max_unit_bytes=64,
    )
    handles = build_backends(KVTransferConfig(backends=(entry,)), context)
    try:
        (handle,) = handles
        assert isinstance(handle, BackendHandle) and handle.name == "local"
        assert isinstance(handle.fetcher, BlobStoreBackend)
        assert handle.publisher is handle.fetcher and handle.pool_registrar is handle.fetcher
        assert set(handle.counters()) >= {"fetch_hits", "publish_stored"}
    finally:
        close_backends(handles)


def test_backend_over_the_memory_store_round_trips_and_probes():
    store = MemoryBlobStore()
    arena = MemoryArena(1 << 12)
    resolver = ArenaResolver(arena)
    backend = BlobStoreBackend(store, BlobStoreConfig(namespace="t"), resolver, FINGERPRINT)
    try:
        backend.register_pool(arena.address, arena.size)
        assert store.registered == {arena.address: arena.size}
        src = resolver.add(0, 0, 40, 24)
        write(src, pattern(1, 64))
        unit = Unit(name=b"u", local_group=0, local=0)
        attempt = backend.publish(extent([unit]))
        backend.settle([attempt])
        assert attempt.poll() == Delivered(frozenset({b"u"}))
        assert store.objects[backend.key_for(b"u")] == pattern(1, 64)

        dst = resolver.add(1, 0, 64)
        write(dst, bytes([0xEE]) * 64)
        attempt = backend.fetch(extent([Unit(name=b"u", local_group=1, local=0)]))
        backend.settle([attempt])
        assert attempt.poll() == Delivered(frozenset({b"u"}))
        assert read(dst) == pattern(1, 64)

        assert backend.probe(b"n", [b"u", b"never"]) is None
        wait_until(lambda: backend.counters.probe_hits + backend.counters.probe_misses == 2)
        assert backend.probe(b"n", [b"u", b"never"]) == frozenset({b"u"})
    finally:
        backend.close()


def test_get_of_an_unknown_key_is_a_miss_that_leaves_the_destination_alone():
    store = MemoryBlobStore()
    arena = MemoryArena(128)
    store.register_span(arena.address, arena.size)
    dst = arena.carve(64)
    write([dst], bytes([0xEE]) * 64)
    assert store.holds(["nope"]) == [False]
    assert store.get(["nope"], [[dst]]) == [GetStatus.MISS]
    assert read([dst]) == bytes([0xEE]) * 64

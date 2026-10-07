# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``type: memory`` through the registry and ``BlobStoreBackend`` over a bare ``MemoryBlobStore``:
the second driver exercises ``factory.py`` and the backend without any knobs in between."""

import importlib

import pytest

__extra_import_path__ = ["~/tensorrt_llm/_torch"]
from disaggregation.backends.blob.backend import BlobStoreBackend, BlobStoreConfig  # noqa: E402
from disaggregation.backends.blob.drivers.memory import MemoryBlobStore  # noqa: E402
from disaggregation.backends.blob.host_landing import HostLandingBlobBackend  # noqa: E402
from disaggregation.backends.blob.store import GetStatus, PutStatus  # noqa: E402
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
    fake_open_slot_pool,
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


def test_put_of_a_held_key_keeps_the_first_object_and_answers_stored():
    """As a Mooncake master does, so the fakes built on this store race like the real one."""
    store = MemoryBlobStore()
    arena = MemoryArena(128)
    store.register_span(arena.address, arena.size)
    first, second = arena.carve(32), arena.carve(32)
    write([first], pattern(1, 32))
    write([second], pattern(2, 32))
    assert store.put(["k"], [[first]]) == [PutStatus.STORED]
    assert store.put(["k"], [[second]]) == [PutStatus.STORED]
    assert store.objects["k"] == pattern(1, 32)


def test_get_of_an_unknown_key_is_a_miss_that_leaves_the_destination_alone():
    store = MemoryBlobStore()
    arena = MemoryArena(128)
    store.register_span(arena.address, arena.size)
    dst = arena.carve(64)
    write([dst], bytes([0xEE]) * 64)
    assert store.contains(["nope"]) == [False]
    assert store.get(["nope"], [[dst]]) == [GetStatus.MISS]
    assert read([dst]) == bytes([0xEE]) * 64


def test_memory_type_with_host_landing_builds_the_lands_on_host_shape(monkeypatch):
    """``landing: host`` over the memory driver: the factory opens two host pools (here faked
    over host arenas, so no torch), the fetcher is a ``HostLandingBlobBackend``, the publisher is
    the inner backend, no pool is registered, and a unit round-trips publish -> land -> place."""
    importlib.import_module("disaggregation.backends.blob.drivers.memory")
    factory = importlib.import_module("disaggregation.backends.blob.factory")
    monkeypatch.setattr(factory, "open_pinned_slot_pool", fake_open_slot_pool)
    arena = MemoryArena(1 << 10)
    resolver = ArenaResolver(arena)
    src = resolver.add(0, 0, 64)
    dst = resolver.add(0, 1, 64)
    entry = BackendEntry.from_dict(
        {
            "name": "local",
            "type": "memory",
            "namespace": "t",
            "landing": "host",
            "publish_buffer_bytes": 256,
            "landing_buffer_bytes": 512,
        }
    )
    context = BackendBuildContext(
        resolver=resolver,
        layout_fingerprint=FINGERPRINT,
        max_unit_bytes=64,
        unit_bytes_of=lambda name: 64,
    )
    handles = build_backends(KVTransferConfig(backends=(entry,)), context)
    try:
        (handle,) = handles
        assert handle.landing == "host" and handle.pool_registrar is None
        assert isinstance(handle.fetcher, HostLandingBlobBackend)
        assert isinstance(handle.publisher, BlobStoreBackend)
        assert handle.counters()["landings_held"] == 0
        write(src, pattern(3, 64))
        attempt = handle.publisher.publish(extent([Unit(name=b"u", local_group=0, local=0)]))
        handle.publisher.settle([attempt])
        assert attempt.poll() == Delivered(frozenset({b"u"}))
        landing = handle.fetcher.fetch_to_host([b"u"])
        wait_until(lambda: landing.poll() is not None)
        assert landing.poll() == Delivered(frozenset({b"u"}))
        assert handle.counters()["landings_held"] == 1
        write(dst, bytes([0xEE]) * 64)
        placed = landing.place(extent([Unit(name=b"u", local_group=0, local=1)]))
        handle.fetcher.settle([placed])
        assert placed.poll() == Delivered(frozenset({b"u"}))
        assert read(dst) == pattern(3, 64)
        landing.close()
        assert handle.counters()["landings_held"] == 0
    finally:
        close_backends(handles)


def test_host_landing_needs_the_assembly_to_size_units(monkeypatch):
    importlib.import_module("disaggregation.backends.blob.drivers.memory")
    entry = BackendEntry.from_dict({"name": "local", "type": "memory", "landing": "host"})
    context = BackendBuildContext(
        resolver=ArenaResolver(MemoryArena(64)), layout_fingerprint=FINGERPRINT, max_unit_bytes=64
    )
    with pytest.raises(ValueError, match="size units by name"):
        build_backends(KVTransferConfig(backends=(entry,)), context)

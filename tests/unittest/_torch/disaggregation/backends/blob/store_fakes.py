# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fakes for the Mooncake store backend: an in-memory ``StoreClient``, host memory standing in
for KV pages, and a ``Copier`` that moves bytes with ``memmove``.

``FakeStoreClient`` follows the return conventions of ``mooncake.store.MooncakeDistributedStore``
as ``client.StoreClient`` documents them: statuses, not exceptions; ``-704`` for a key that is not
there; one key is one object assembled from a buffer list. Every buffer handed to a put or a get
must lie inside a span ``register_buffer`` accepted, otherwise that key answers a negative status
and nothing is written (see ``UNREGISTERED`` for how this compares with the real client).

This module is named ``store_fakes`` rather than ``fakes`` because pytest's default import mode
puts each test directory on ``sys.path``, and ``orchestration/kv_transfer/fakes.py`` already owns the name
``fakes`` in a session that collects both suites.
"""

from __future__ import annotations

import ctypes
import threading
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

__extra_import_path__ = ["~/tensorrt_llm/_torch"]
from disaggregation.backends.blob.backend import BlobStoreBackend  # noqa: E402
from disaggregation.backends.blob.mooncake import MooncakeStoreConfig  # noqa: E402
from disaggregation.backends.blob.staging import HostStagingPool  # noqa: E402
from disaggregation.base.cache_backend import CacheExtent, Unit  # noqa: E402
from disaggregation.base.region import Segment  # noqa: E402

MISSING = -704
"""The status Mooncake answers for a key that is not in the store."""

UNREGISTERED = -1
"""The status this fake answers for a buffer outside every registered span.

Stricter than the real client over TCP, which accepts unregistered buffers for both put and get
(it goes through its own local buffer); over RDMA they cannot be reached. The backend does not
rely on either: it refuses unregistered destinations itself, before the client is called.
"""

TOO_SMALL = -600
TOO_LARGE = -800
"""What the real client answers when the buffers of a get do not add up to the object's size."""

FINGERPRINT = b"\x01layout"


# ---------------------------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------------------------


class MemoryArena:
    """One ``ctypes`` buffer standing in for a rank's KV pool; segments are carved from it in
    order. ``address``/``size`` are what ``register_pool`` takes."""

    def __init__(self, size: int) -> None:
        self._buf = (ctypes.c_char * size)()
        self.address = ctypes.addressof(self._buf)
        self.size = size
        self._next = 0

    def carve(self, size: int) -> Segment:
        if self._next + size > self.size:
            raise ValueError(f"arena of {self.size} B cannot carve {size} more")
        seg = (self.address + self._next, size)
        self._next += size
        return seg

    def contains(self, address: int, size: int) -> bool:
        return self.address <= address and address + size <= self.address + self.size


def write(segments: Iterable[Segment], data: bytes) -> None:
    """Spread ``data`` over ``segments`` in order; ``data`` must be exactly their total."""
    segments = tuple(segments)
    total = sum(s for _, s in segments)
    if len(data) != total:
        raise ValueError(f"{len(data)} B of data for {total} B of segments")
    offset = 0
    for address, size in segments:
        ctypes.memmove(address, data[offset : offset + size], size)
        offset += size


def read(segments: Iterable[Segment]) -> bytes:
    """The segments' bytes, concatenated in order."""
    return b"".join(ctypes.string_at(address, size) for address, size in segments)


def fill(segments: Iterable[Segment], byte: int) -> None:
    for address, size in segments:
        ctypes.memset(address, byte, size)


def pattern(seed: int, size: int) -> bytes:
    """Deterministic, non-repeating-looking bytes for ``seed``."""
    return bytes((seed * 131 + i * 7 + (i >> 8)) & 0xFF for i in range(size))


class ArenaResolver:
    """A ``RegionResolver`` over a table: ``(local_group, local) -> segments``.

    ``add`` carves fresh segments from the arena; ``sizes`` has one entry per segment, so a unit
    of two segments is ``add(g, l, 64, 64)``.
    """

    def __init__(self, arena: MemoryArena) -> None:
        self.arena = arena
        self._table: dict[tuple[int, int], tuple[Segment, ...]] = {}

    def add(self, local_group: int, local: int, *sizes: int) -> tuple[Segment, ...]:
        if (local_group, local) in self._table:
            raise ValueError(f"({local_group}, {local}) already resolved")
        segments = tuple(self.arena.carve(size) for size in sizes)
        self._table[(local_group, local)] = segments
        return segments

    def segments(self, local_group: int, local: int) -> tuple[Segment, ...]:
        return self._table[(local_group, local)]

    def __call__(self, local_group: int, local: int) -> Sequence[Segment]:
        return self._table[(local_group, local)]


# ---------------------------------------------------------------------------------------------
# Blocking / failing knobs shared by the client and the copier
# ---------------------------------------------------------------------------------------------


class _Knobs:
    """``block(*methods)`` holds every listed call (all calls when none are listed) at its entry
    until ``unblock``; ``entered`` counts calls that reached the gate so a test can wait for a
    worker to be inside. ``fail_next(method)`` makes that method's next call raise."""

    def __init__(self) -> None:
        self._gate = threading.Event()
        self._gate.set()
        self._blocked: set[str] | None = None
        self._fail: dict[str, list[BaseException]] = {}
        self._fail_at: dict[str, tuple[int, BaseException]] = {}
        self._seen: dict[str, int] = {}
        self._lock = threading.Lock()
        self.entered = 0
        self._entered_cond = threading.Condition(self._lock)
        self.calls: list[tuple[str, tuple]] = []
        self.trace: Trace | None = None
        """When set, every call is also appended to this shared, thread-stamped event log."""

    def block(self, *methods: str) -> None:
        self._blocked = set(methods) or None
        self._gate.clear()

    def unblock(self) -> None:
        self._gate.set()

    def fail_next(self, method: str, exc: BaseException | None = None) -> None:
        with self._lock:
            self._fail.setdefault(method, []).append(exc or RuntimeError(f"{method} failed"))

    def fail_at(self, method: str, nth: int, exc: BaseException | None = None) -> None:
        """Make the ``nth`` call (1-based, counted from now) of ``method`` raise."""
        with self._lock:
            self._fail_at[method] = (
                self._seen.get(method, 0) + nth,
                exc or RuntimeError(f"{method} failed on call {nth}"),
            )

    def wait_entered(self, count: int, timeout: float = 5.0) -> None:
        """Block until ``count`` calls have reached the gate."""
        with self._entered_cond:
            if not self._entered_cond.wait_for(lambda: self.entered >= count, timeout):
                raise AssertionError(f"only {self.entered} of {count} calls reached the gate")

    def count(self, method: str) -> int:
        with self._lock:
            return sum(1 for m, _ in self.calls if m == method)

    def _enter(self, method: str, *args) -> None:
        with self._lock:
            self.calls.append((method, args))
            self.entered += 1
            self._entered_cond.notify_all()
            self._seen[method] = self._seen.get(method, 0) + 1
            pending = self._fail.get(method)
            exc = pending.pop(0) if pending else None
            armed = self._fail_at.get(method)
            if armed is not None and self._seen[method] == armed[0]:
                del self._fail_at[method]
                exc = armed[1]
        if self.trace is not None:
            self.trace.add(method, args)
        if exc is not None:
            raise exc
        if not self._gate.is_set() and (self._blocked is None or method in self._blocked):
            self._gate.wait()


# ---------------------------------------------------------------------------------------------
# Store client
# ---------------------------------------------------------------------------------------------


class FakeStoreClient(_Knobs):
    """An in-memory ``StoreClient``; see the module docstring for the conventions it follows."""

    def __init__(self) -> None:
        super().__init__()
        self.objects: dict[str, bytes] = {}
        self.registered: dict[int, int] = {}
        """address -> size of every live registration."""
        self.setup_args: tuple | None = None
        self.closed = 0

    # -- StoreClient --

    def setup(self, *args) -> int:
        self._enter("setup", *args)
        self.setup_args = args
        return 0

    def register_buffer(self, buffer_ptr: int, size: int) -> int:
        self._enter("register_buffer", buffer_ptr, size)
        with self._lock:
            self.registered[buffer_ptr] = size
        return 0

    def unregister_buffer(self, buffer_ptr: int) -> int:
        self._enter("unregister_buffer", buffer_ptr)
        with self._lock:
            return 0 if self.registered.pop(buffer_ptr, None) is not None else UNREGISTERED

    def batch_is_exist(self, keys: Sequence[str]) -> list[int]:
        self._enter("batch_is_exist", tuple(keys))
        with self._lock:
            return [1 if key in self.objects else 0 for key in keys]

    def batch_put_from_multi_buffers(
        self,
        keys: Sequence[str],
        all_buffer_ptrs: Sequence[Sequence[int]],
        all_sizes: Sequence[Sequence[int]],
    ) -> list[int]:
        self._enter("batch_put_from_multi_buffers", tuple(keys))
        results = []
        for key, ptrs, sizes in zip(keys, all_buffer_ptrs, all_sizes):
            if not self._all_registered(ptrs, sizes):
                results.append(UNREGISTERED)
                continue
            data = b"".join(ctypes.string_at(p, s) for p, s in zip(ptrs, sizes))
            with self._lock:
                self.objects[key] = data
            results.append(0)
        return results

    def batch_get_into_multi_buffers(
        self,
        keys: Sequence[str],
        all_buffer_ptrs: Sequence[Sequence[int]],
        all_sizes: Sequence[Sequence[int]],
    ) -> list[int]:
        self._enter("batch_get_into_multi_buffers", tuple(keys))
        results = []
        for key, ptrs, sizes in zip(keys, all_buffer_ptrs, all_sizes):
            with self._lock:
                data = self.objects.get(key)
            if data is None:
                results.append(MISSING)
                continue
            if not self._all_registered(ptrs, sizes):
                results.append(UNREGISTERED)
                continue
            total = sum(sizes)
            if total != len(data):
                # As the real client: the buffers must add up to the object exactly; nothing is
                # written otherwise (-600 when they are too small, -800 when too large).
                results.append(TOO_SMALL if total < len(data) else TOO_LARGE)
                continue
            offset = 0
            for p, s in zip(ptrs, sizes):
                ctypes.memmove(p, data[offset : offset + s], s)
                offset += s
            results.append(len(data))
        return results

    def get_size(self, key: str) -> int:
        self._enter("get_size", key)
        with self._lock:
            data = self.objects.get(key)
        return MISSING if data is None else len(data)

    def close(self) -> int:
        self._enter("close")
        self.closed += 1
        return 0

    # -- knobs --

    def evict(self, *keys: str) -> None:
        with self._lock:
            for key in keys:
                self.objects.pop(key, None)

    def evict_all(self) -> None:
        with self._lock:
            self.objects.clear()

    def _all_registered(self, ptrs: Sequence[int], sizes: Sequence[int]) -> bool:
        with self._lock:
            spans = tuple(self.registered.items())
        return all(
            any(base <= p and p + s <= base + size for base, size in spans)
            for p, s in zip(ptrs, sizes)
        )


# ---------------------------------------------------------------------------------------------
# Copier
# ---------------------------------------------------------------------------------------------


class FakeCopier(_Knobs):
    """A ``Copier`` over ``memmove``; ``copies`` lists ``(kind, dst, src, size)``."""

    def __init__(self) -> None:
        super().__init__()
        self.copies: list[tuple[str, int, int, int]] = []
        self.syncs = 0

    def copy(self, dst: int, src: int, size: int, kind: str) -> None:
        self._enter("copy", kind, dst, src, size)
        ctypes.memmove(dst, src, size)
        with self._lock:
            self.copies.append((kind, dst, src, size))

    def sync(self) -> None:
        self._enter("sync")
        with self._lock:
            self.syncs += 1

    def kinds(self) -> list[str]:
        with self._lock:
            return [kind for kind, *_ in self.copies]


class Trace:
    """A thread-stamped event log shared by a copier and a staging pool: ``(event, thread, args)``
    in real-time order, so tests can check that a ``sync`` precedes a ``release`` on the same
    thread and that no slot is copied into or out of by a thread that does not hold it."""

    def __init__(self) -> None:
        self.events: list[tuple[str, int, tuple]] = []
        self._lock = threading.Lock()

    def add(self, event: str, args: tuple) -> None:
        with self._lock:
            self.events.append((event, threading.get_ident(), tuple(args)))

    def by_thread(self, ident: int) -> list[tuple[str, tuple]]:
        with self._lock:
            return [(e, a) for e, t, a in self.events if t == ident]

    def check_slot_exclusivity(self, pool: HostStagingPool) -> None:
        """Replay: every copy touching a slot must be issued by the thread holding that slot, and
        a slot must never be held by two threads at once."""
        owner: dict[int, int] = {}
        with self._lock:
            events = list(self.events)
        for event, thread, args in events:
            if event == "acquire":
                for slot in args[0]:
                    assert slot not in owner, (
                        f"slot {slot} acquired by {thread} while held by {owner[slot]}"
                    )
                    owner[slot] = thread
            elif event == "release":
                for slot in args[0]:
                    assert owner.pop(slot, None) == thread, f"slot {slot} released by a non-owner"
            elif event == "copy":
                kind, dst, src, size = args
                host = dst if kind == "d2h" else src
                slot = (host - pool.slot_address(0)) // pool.slot_bytes
                assert 0 <= slot < pool.num_slots, f"copy outside the staging buffer: {args}"
                assert owner.get(slot) == thread, (
                    f"thread {thread} copied through slot {slot} it does not hold"
                )
        assert not owner, f"slots still held at the end of the trace: {owner}"


class TracingStagingPool(HostStagingPool):
    """A ``HostStagingPool`` that logs ``acquire``/``release`` to the same ``Trace`` as its copier."""

    def __init__(self, *args, trace: Trace, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.trace = trace

    def acquire(self, count: int) -> list[int]:
        slots = super().acquire(count)
        self.trace.add("acquire", (tuple(slots),))
        return slots

    def release(self, slots: Sequence[int]) -> None:
        self.trace.add("release", (tuple(slots),))
        super().release(slots)


def make_staging(
    client: FakeStoreClient, *, slots: int, slot_bytes: int
) -> tuple[TracingStagingPool, FakeCopier, MemoryArena, Trace]:
    """A registered host arena cut into ``slots`` traced slots over a traced ``FakeCopier``."""
    host = MemoryArena(slot_bytes * slots)
    assert client.register_buffer(host.address, host.size) == 0
    trace = Trace()
    copier = FakeCopier()
    copier.trace = trace
    pool = TracingStagingPool(host.address, slot_bytes, slots, copier, keepalive=host, trace=trace)
    return pool, copier, host, trace


# ---------------------------------------------------------------------------------------------
# Extents and a wired backend
# ---------------------------------------------------------------------------------------------


def unit_name(local_group: int, local: int, seed: str = "u") -> bytes:
    return f"{seed}:{local_group}:{local}".encode()


def extent(units: Iterable[Unit], name: bytes = b"ext", is_last: bool = True) -> CacheExtent:
    return CacheExtent(name=name, units=tuple(units), is_last=is_last)


def config(**overrides) -> MooncakeStoreConfig:
    base = dict(master_server_address="fake:0", num_workers=2, probe_ttl_s=30.0)
    base.update(overrides)
    return MooncakeStoreConfig(**base)


@dataclass
class Rank:
    """One process's view: its arena, resolver, units and a backend over a (possibly shared)
    client. ``unit(g, l, *sizes)`` carves memory and returns the ``Unit`` naming it."""

    client: FakeStoreClient
    backend: BlobStoreBackend
    arena: MemoryArena
    resolver: ArenaResolver
    registration: object | None = None
    units: dict[tuple[int, int], Unit] = field(default_factory=dict)

    def unit(self, local_group: int, local: int, *sizes: int, seed: str = "u") -> Unit:
        self.resolver.add(local_group, local, *sizes)
        unit = Unit(name=unit_name(local_group, local, seed), local_group=local_group, local=local)
        self.units[(local_group, local)] = unit
        return unit

    def segments(self, unit: Unit) -> tuple[Segment, ...]:
        return self.resolver.segments(unit.local_group, unit.local)

    def write(self, unit: Unit, data: bytes) -> None:
        write(self.segments(unit), data)

    def read(self, unit: Unit) -> bytes:
        return read(self.segments(unit))

    def fill(self, unit: Unit, byte: int) -> None:
        fill(self.segments(unit), byte)

    def key(self, unit: Unit) -> str:
        return self.backend.key_for(unit.name)

    def finish(self, attempt):
        """Settle one attempt and return its outcome."""
        self.backend.settle([attempt])
        return attempt.poll()

    # ``with make_rank() as rank:`` closes the backend inside the test body. The repository's
    # ``threadleak`` check runs before fixture teardown, so a backend closed there would count
    # its ``mooncake-store_N`` workers as leaked; closing here also asserts ``close`` joins them.
    def __enter__(self) -> Rank:
        return self

    def __exit__(self, *exc) -> None:
        unblock = getattr(self.client, "unblock", None)  # a real client has no gate
        if unblock is not None:
            unblock()
        self.backend.close()


def make_rank(
    client: FakeStoreClient | None = None,
    *,
    arena_bytes: int = 1 << 16,
    register: bool = True,
    staging=None,
    fingerprint: bytes = FINGERPRINT,
    **config_overrides,
) -> Rank:
    """A backend over ``client`` (a fresh one when ``None``) with its pool registered."""
    client = client if client is not None else FakeStoreClient()
    arena = MemoryArena(arena_bytes)
    resolver = ArenaResolver(arena)
    cfg = config(**config_overrides)
    backend = BlobStoreBackend(client, cfg, resolver, fingerprint, staging=staging)
    rank = Rank(client, backend, arena, resolver)
    if register and not cfg.stage_through_host:
        rank.registration = backend.register_pool(arena.address, arena.size)
    return rank


def wait_until(
    predicate: Callable[[], bool], timeout: float = 5.0, what: str = "condition"
) -> None:
    """Spin on ``predicate`` with a deadline; the backend's threads have no other seam."""
    import time

    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.002)

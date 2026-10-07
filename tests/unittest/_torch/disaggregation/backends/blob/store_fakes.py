# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fakes for the blob store backend: an in-process ``BlobStore`` with blocking and failing knobs,
host memory standing in for KV pages, and a ``Copier`` that moves bytes with ``memmove``.

``FakeBlobStore`` is the knobs (``block`` / ``fail_next`` / ``fail_at``, a call log) in front of
a ``MemoryBlobStore``, the in-process driver, so the store logic lives in one place. That store
answers ``FAILED`` for a buffer outside every registered span; the real store over TCP is laxer,
but the backend refuses unregistered destinations itself, before the store is called, so neither
behaviour is relied on.

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
from disaggregation.backends.blob.backend import BlobStoreBackend, BlobStoreConfig  # noqa: E402
from disaggregation.backends.blob.drivers.memory import MemoryBlobStore  # noqa: E402
from disaggregation.backends.blob.host_landing import HostLandingBlobBackend  # noqa: E402
from disaggregation.backends.blob.slot_pool import HostSlotPool  # noqa: E402
from disaggregation.backends.blob.store import GetStatus, PutStatus  # noqa: E402
from disaggregation.backends.config import DEFAULT_LANDING_WAIT_TIMEOUT_S  # noqa: E402
from disaggregation.base.cache_backend import CacheExtent, Unit  # noqa: E402
from disaggregation.base.region import Segment  # noqa: E402

FINGERPRINT = b"\x01layout"
SLOT = 128
"""Default slot width of the host pools the fakes build; every test unit is smaller."""


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
# Blocking / failing knobs shared by the store and the copier
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
# Blob store
# ---------------------------------------------------------------------------------------------


class FakeBlobStore(_Knobs):
    """The knobs in front of an in-memory ``BlobStore``: every protocol method passes the gate
    (``block`` / ``fail_next`` / ``fail_at``, recorded in ``calls``) and then forwards to
    ``inner``. ``objects`` / ``registered`` read through to it."""

    def __init__(self) -> None:
        super().__init__()
        self.inner = MemoryBlobStore()
        self.closed = 0

    @property
    def objects(self) -> dict[str, bytes]:
        return self.inner.objects

    @property
    def registered(self) -> dict[int, int]:
        return self.inner.registered

    # -- BlobStore --

    def register_span(self, address: int, size: int) -> None:
        self._enter("register_span", address, size)
        self.inner.register_span(address, size)

    def unregister_span(self, address: int, size: int) -> None:
        self._enter("unregister_span", address, size)
        self.inner.unregister_span(address, size)

    def contains(self, keys: Sequence[str]) -> list[bool]:
        self._enter("contains", tuple(keys))
        return self.inner.contains(keys)

    def put(self, keys: Sequence[str], buffers: Sequence[Sequence[Segment]]) -> list[PutStatus]:
        self._enter("put", tuple(keys))
        return self.inner.put(keys, buffers)

    def get(self, keys: Sequence[str], buffers: Sequence[Sequence[Segment]]) -> list[GetStatus]:
        self._enter("get", tuple(keys))
        return self.inner.get(keys, buffers)

    def close(self) -> None:
        self._enter("close")
        self.closed += 1
        self.inner.close()

    def describe(self) -> str:
        return "fake"

    # -- knobs --

    def evict(self, *keys: str) -> None:
        for key in keys:
            self.objects.pop(key, None)

    def evict_all(self) -> None:
        self.objects.clear()


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
    """A thread-stamped event log shared by a copier and a slot pool: ``(event, thread, args)``
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

    def check_slot_exclusivity(self, pool: HostSlotPool) -> None:
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
                assert 0 <= slot < pool.num_slots, f"copy outside the slot pool buffer: {args}"
                assert owner.get(slot) == thread, (
                    f"thread {thread} copied through slot {slot} it does not hold"
                )
        assert not owner, f"slots still held at the end of the trace: {owner}"


class TracingSlotPool(HostSlotPool):
    """A ``HostSlotPool`` that logs ``acquire``/``release`` to the same ``Trace`` as its copier."""

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


def make_publish_pool(
    store: FakeBlobStore, *, slots: int, slot_bytes: int
) -> tuple[TracingSlotPool, FakeCopier, MemoryArena, Trace]:
    """A registered host arena cut into ``slots`` traced slots over a traced ``FakeCopier``."""
    host = MemoryArena(slot_bytes * slots)
    store.register_span(host.address, host.size)
    trace = Trace()
    copier = FakeCopier()
    copier.trace = trace
    pool = TracingSlotPool(host.address, slot_bytes, slots, copier, keepalive=host, trace=trace)
    return pool, copier, host, trace


def fake_open_slot_pool(
    store, *, slot_bytes: int, num_slots: int, device_index=None
) -> HostSlotPool:
    """Stand-in for ``factory.open_pinned_slot_pool``: a registered host arena over a
    ``FakeCopier`` instead of a pinned torch buffer over CUDA. Same signature, so a test patches
    the factory's name with it."""
    host = MemoryArena(slot_bytes * num_slots)
    store.register_span(host.address, host.size)
    return HostSlotPool(host.address, slot_bytes, num_slots, FakeCopier(), keepalive=host)


# ---------------------------------------------------------------------------------------------
# Extents and a wired backend
# ---------------------------------------------------------------------------------------------


def unit_name(local_group: int, local: int, seed: str = "u") -> bytes:
    return f"{seed}:{local_group}:{local}".encode()


def extent(units: Iterable[Unit], name: bytes = b"ext", is_last: bool = True) -> CacheExtent:
    return CacheExtent(name=name, units=tuple(units), is_last=is_last)


def config(**overrides) -> BlobStoreConfig:
    base = dict(num_workers=2, probe_ttl_s=30.0)
    base.update(overrides)
    return BlobStoreConfig(**base)


@dataclass
class Rank:
    """One process's view: its arena, resolver, units and a backend over a (possibly shared)
    store. ``unit(g, l, *sizes)`` carves memory and returns the ``Unit`` naming it.

    ``backend`` is the ``BlobStoreBackend`` for the ``device`` landing and the
    ``HostLandingBlobBackend`` for ``host``; ``inner`` is the ``BlobStoreBackend`` either way.
    The host shape also carries its two pools and their copiers; the publish pool is traced.
    """

    store: FakeBlobStore
    backend: BlobStoreBackend | HostLandingBlobBackend
    arena: MemoryArena
    resolver: ArenaResolver
    inner: BlobStoreBackend | None = None
    registration: object | None = None
    units: dict[tuple[int, int], Unit] = field(default_factory=dict)
    publish_pool: TracingSlotPool | None = None
    publish_copier: FakeCopier | None = None
    trace: Trace | None = None
    landing_pool: HostSlotPool | None = None
    landing_copier: FakeCopier | None = None

    def unit(self, local_group: int, local: int, *sizes: int, seed: str = "u") -> Unit:
        self.resolver.add(local_group, local, *sizes)
        unit = Unit(name=unit_name(local_group, local, seed), local_group=local_group, local=local)
        self.units[(local_group, local)] = unit
        return unit

    def unit_bytes(self, name: bytes) -> int:
        """Size of the unit called ``name`` on this rank; what the host shape is built with."""
        for unit in self.units.values():
            if unit.name == name:
                return sum(size for _, size in self.segments(unit))
        raise KeyError(f"no unit called {name!r} on this rank")

    def segments(self, unit: Unit) -> tuple[Segment, ...]:
        return self.resolver.segments(unit.local_group, unit.local)

    def land(self, units: Iterable[Unit], timeout: float = 5.0):
        """Host shape: ``fetch_to_host`` the units and wait for the landing's outcome."""
        landing = self.backend.fetch_to_host([u.name for u in units])
        wait_until(lambda: landing.poll() is not None, timeout, what="landing outcome")
        return landing

    def place(self, landing, units: Iterable[Unit], name: bytes = b"ext"):
        """Host shape: place the units out of ``landing`` and return the placement's outcome."""
        return self.finish(landing.place(extent(units, name=name)))

    def free_landing_slots(self) -> int:
        """The pool keeps no public free count; its ``_free`` list is the one place to read it."""
        assert self.landing_pool is not None
        return len(self.landing_pool._free)

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
    # its ``blob-store-{i}`` workers as leaked; closing here also asserts ``close`` joins them.
    def __enter__(self) -> Rank:
        return self

    def __exit__(self, *exc) -> None:
        # Every gate is opened first, so a test that fails with a worker parked behind one ends
        # in a failure report rather than a ``close`` that waits for that worker forever.
        for gated in (self.store, self.publish_copier, self.landing_copier):
            unblock = getattr(gated, "unblock", None)  # a real store has no gate
            if unblock is not None:
                unblock()
        self.backend.close()


def make_rank(
    store: FakeBlobStore | None = None,
    *,
    arena_bytes: int = 1 << 16,
    register: bool = True,
    publish_pool=None,
    fingerprint: bytes = FINGERPRINT,
    **config_overrides,
) -> Rank:
    """A ``landing: device`` backend over ``store`` (a fresh one when ``None``) with its pool
    registered. ``publish_pool`` builds the inner backend of a host shape; ``make_host_rank`` is the
    usual way there."""
    store = store if store is not None else FakeBlobStore()
    arena = MemoryArena(arena_bytes)
    resolver = ArenaResolver(arena)
    cfg = config(**config_overrides)
    backend = BlobStoreBackend(store, cfg, resolver, fingerprint, publish_pool=publish_pool)
    rank = Rank(store, backend, arena, resolver, inner=backend)
    if register and not cfg.lands_on_host:
        rank.registration = backend.register_pool(arena.address, arena.size)
    return rank


def make_host_rank(
    store: FakeBlobStore | None = None,
    *,
    publish_slots: int = 4,
    landing_slots: int = 4,
    slot_bytes: int = SLOT,
    landing_wait_timeout_s: float | None = DEFAULT_LANDING_WAIT_TIMEOUT_S,
    arena_bytes: int = 1 << 16,
    unit_bytes: Callable[[bytes], int] | None = None,
    **config_overrides,
) -> Rank:
    """A ``landing: host`` backend: a traced publish pool, a landing pool over its own
    ``FakeCopier``, and a ``HostLandingBlobBackend`` over the inner backend. The caller's pool is
    deliberately not registered, as it would not be on a machine without GPUDirect. Units are
    sized by ``unit_bytes``, by default from the rank's own table (``Rank.unit``)."""
    store = store if store is not None else FakeBlobStore()
    publish_pool, publish_copier, _, trace = make_publish_pool(
        store, slots=publish_slots, slot_bytes=slot_bytes
    )
    landing_pool = fake_open_slot_pool(store, slot_bytes=slot_bytes, num_slots=landing_slots)
    rank = make_rank(
        store,
        arena_bytes=arena_bytes,
        publish_pool=publish_pool,
        landing="host",
        **config_overrides,
    )
    rank.backend = HostLandingBlobBackend(
        rank.inner,
        landing_pool,
        unit_bytes or rank.unit_bytes,
        landing_wait_timeout_s=landing_wait_timeout_s,
    )
    rank.publish_pool, rank.publish_copier, rank.trace = publish_pool, publish_copier, trace
    # The pool does not expose its copier; the fake one is read back here so tests can gate it.
    rank.landing_pool, rank.landing_copier = landing_pool, landing_pool._copier
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

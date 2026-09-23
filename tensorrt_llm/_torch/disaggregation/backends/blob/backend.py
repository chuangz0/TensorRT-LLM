# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A cache backend over a blob store: ``Fetches``, ``Publishes``, ``RegistersPools``.

The class depends only on the ``BlobStore`` protocol (``store.py``); a driver under ``drivers/``
opens the store and ``factory.py`` builds the backend over it. One store object per unit. A
unit's segments go in as that object's buffer list, so it is stored whole or not at all, which is
what per-unit atomic visibility needs (contract §6.3). Every delivery runs on the backend's own
threads; they call the store and, when staging, the copier, and nothing else. ``probe`` is
answered the same way on a thread of its own: the first call queues the lookup and answers
``None``, a later call returns the answer once.

Reading a unit is two round trips: ``holds`` then ``get``. The first tells a miss from a failure
(a lookup that cannot be answered raises, it never answers "absent"); the contract forbids
reporting either as the other (§5.2). A unit that was present at the lookup and gone by the get is
a miss (the store wrote nothing); any other trouble reading a present unit fails the whole
attempt, because earlier batches of the same attempt may already have written their destinations
and the contract says that content is then undefined (§5.2 invariant 3).
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence, TypeVar

from ...base.cache_backend import (
    Attempt,
    CacheExtent,
    Delivered,
    Failed,
    Outcome,
    Registration,
    Route,
    SubmissionRejected,
)
from ...base.region import RegionResolver, Segment
from .keys import KeyScheme
from .staging import HostStagingPool
from .store import BlobStore, BlobStoreError, GetStatus, PutStatus
from .worker_pool import DaemonWorkerPool

__all__ = ["MAX_PROBES", "BlobStoreBackend", "BlobStoreConfig", "StoreCounters"]

MAX_PROBES = 1024
"""Bound on remembered lookups, pending or answered. A lookup thread that stopped answering must
not let the table grow without end; past the bound a new ``probe`` raises and the caller plans
without the store."""

# The stdlib logger keeps this module import-light (no ``tensorrt_llm`` import); note that
# ``TLLM_LOG_LEVEL_BY_MODULE`` does not route it, so configure ``logging`` for this name directly.
logger = logging.getLogger(__name__)

_T = TypeVar("_T")

_DEFAULT_STAGING_BUFFER_BYTES = 536870912


@dataclass(frozen=True)
class BlobStoreConfig:
    """What the backend reads from a backend entry; the driver's connection options are separate.

    Attributes:
        namespace: Leading component of every key; two deployments share cache only when they agree.
        transfer_batch_size: Units per store call. Bounds one call, not one delivery.
        stage_through_host: Pass units through a pinned host buffer instead of registering the
            caller's pools. Costs a copy each way; works without GPUDirect RDMA.
        staging_buffer_bytes: Ceiling on the pinned staging allocation when staging.
        max_inflight_ops: Deliveries that may be queued or running at once. A submission past
            this bound is refused with ``SubmissionRejected``.
        num_workers: Threads that drive store calls.
        probe_ttl_s: Seconds an unconsumed probe answer is kept before it is dropped.
    """

    namespace: str = "trtllm"
    transfer_batch_size: int = 64
    stage_through_host: bool = False
    staging_buffer_bytes: int = _DEFAULT_STAGING_BUFFER_BYTES
    max_inflight_ops: int = 256
    num_workers: int = 2
    probe_ttl_s: float = 30.0

    def __post_init__(self) -> None:
        if not self.namespace:
            raise ValueError("namespace must not be empty")
        for name in ("transfer_batch_size", "max_inflight_ops", "num_workers"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.stage_through_host and self.staging_buffer_bytes <= 0:
            raise ValueError("staging_buffer_bytes must be > 0 when stage_through_host is set")
        if self.probe_ttl_s <= 0:
            raise ValueError("probe_ttl_s must be > 0")

    @classmethod
    def fields(cls) -> frozenset[str]:
        """The option keys this class reads; a driver's keys must not overlap them."""
        return frozenset(f.name for f in dataclasses.fields(cls))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> BlobStoreConfig:
        """Build from a plain mapping, refusing keys this class does not know."""
        unknown = sorted(set(raw) - cls.fields())
        if unknown:
            raise ValueError(f"unknown BlobStoreConfig keys: {unknown}")
        return cls(**raw)


@dataclass
class StoreCounters:
    """Operational counters (contract §6.2 implementation requirement 4). Read-only for callers."""

    fetch_hits: int = 0
    fetch_misses: int = 0
    publish_stored: int = 0
    publish_raced: int = 0
    """Units the store declined and then turned out to hold: another publisher got there first."""
    publish_present: int = 0
    probe_hits: int = 0
    probe_misses: int = 0
    failed_attempts: int = 0


class _StoreAttempt:
    """``quiet`` is set once the caller's memory is no longer touched, ``done`` once an outcome is."""

    __slots__ = ("_outcome", "done", "quiet")

    def __init__(self) -> None:
        self._outcome: Optional[Outcome] = None
        self.quiet = threading.Event()
        self.done = threading.Event()

    def poll(self) -> Optional[Outcome]:
        return self._outcome

    def finish(self, outcome: Outcome) -> None:
        if self._outcome is None:
            self._outcome = outcome
        self.quiet.set()
        self.done.set()


@dataclass
class _Task:
    """One unit to move: its key, where it lives, and how big it is."""

    name: bytes
    key: str
    segments: tuple[Segment, ...]
    total: int = field(init=False)

    def __post_init__(self) -> None:
        self.total = sum(size for _, size in self.segments)


@dataclass
class _Probe:
    created: float
    done: bool = False
    answer: frozenset[bytes] = frozenset()
    error: Optional[BaseException] = None


class _Registration:
    def __init__(self, backend: BlobStoreBackend, address: int, size: int) -> None:
        self.address = address
        self.size = size
        self.live = True
        self.closing = False
        self._backend = backend

    def covers(self, address: int, size: int) -> bool:
        return self.address <= address and address + size <= self.address + self.size

    def close(self) -> None:
        self._backend._unregister(self)


def _batched(items: Sequence[_T], size: int) -> Iterator[Sequence[_T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class BlobStoreBackend:
    """Fetch from and publish to one blob store.

    Args:
        store: An opened blob store. Owned from here on; ``close`` closes it.
        config: Batch sizes, bounds and whether to stage through host memory.
        resolver: Maps a unit's local coordinates to its memory segments.
        layout_fingerprint: Folded into every key; see ``KeyScheme``.
        staging: Required when ``config.stage_through_host``; ignored otherwise. Owned from here
            on: ``close`` shuts it down so that no worker stays parked waiting for a slot.
    """

    def __init__(
        self,
        store: BlobStore,
        config: BlobStoreConfig,
        resolver: RegionResolver,
        layout_fingerprint: bytes,
        *,
        staging: HostStagingPool | None = None,
    ) -> None:
        if config.stage_through_host and staging is None:
            raise ValueError("stage_through_host needs a HostStagingPool")
        self._store = store
        self._config = config
        self._resolve = resolver
        self._keys = KeyScheme(config.namespace, layout_fingerprint)
        self._staging = staging if config.stage_through_host else None
        self._batch = config.transfer_batch_size
        if self._staging is not None:
            self._batch = min(self._batch, self._staging.num_slots)
        self._lock = threading.Lock()
        self._registrations: list[_Registration] = []
        self._pending: set[tuple[int, int]] = set()
        """Spans whose ``register_span`` call is in progress; they refuse overlaps like live ones."""
        self._probes: dict[tuple[bytes, tuple[bytes, ...]], _Probe] = {}
        self._inflight = threading.BoundedSemaphore(config.max_inflight_ops)
        # Daemon workers: a store call that never returns must not keep the process alive once
        # the engine has given the backend up.
        self._pool = DaemonWorkerPool(config.num_workers, thread_name_prefix="blob-store")
        # Lookups have their own thread so a probe is not queued behind every delivery in flight.
        self._lookups = DaemonWorkerPool(1, thread_name_prefix="blob-store-probe")
        self._closed = False
        self.counters = StoreCounters()

    def key_for(self, name: bytes) -> str:
        """The store key this backend uses for the unit called ``name``."""
        return self._keys.key(name)

    # ---- RegistersPools ----

    def register_pool(self, address: int, size: int) -> Registration:
        if size <= 0:
            raise ValueError(f"size must be > 0, got {size}")
        span = (address, size)
        with self._lock:
            # The store call runs outside the lock, so the span is reserved across it: a
            # concurrent duplicate must be refused before it reaches the store, where its own
            # registration or cleanup could tear down the winner's.
            self._check_overlap(address, size)
            self._pending.add(span)
        try:
            self._store.register_span(address, size)
            reg = _Registration(self, address, size)
            with self._lock:
                self._registrations.append(reg)
        finally:
            with self._lock:
                self._pending.discard(span)
        return reg

    def _check_overlap(self, address: int, size: int) -> None:
        """Caller holds the lock."""
        spans = [(reg.address, reg.size) for reg in self._registrations] + sorted(self._pending)
        for start, length in spans:
            if address < start + length and start < address + size:
                raise ValueError(
                    f"[{address:#x}, {address + size:#x}) overlaps registered "
                    f"[{start:#x}, {start + length:#x})"
                )

    def _unregister(self, reg: _Registration) -> None:
        with self._lock:
            if not reg.live or reg.closing or self._closed:
                return
            reg.closing = True
        try:
            self._store.unregister_span(reg.address, reg.size)
        finally:
            # A close that raised has not closed: the handle stays live and may be retried.
            reg.closing = False
        with self._lock:
            reg.live = False
            self._registrations.remove(reg)

    def _unregistered(self, segments: Sequence[Segment]) -> Optional[Segment]:
        """The first segment not inside one live registration, if any. Caller holds the lock."""
        for address, size in segments:
            if not any(reg.covers(address, size) for reg in self._registrations):
                return address, size
        return None

    # ---- Fetches / Publishes ----

    def fetch(self, extent: CacheExtent, *, route: Optional[Route] = None) -> Attempt:
        if route is not None:
            raise SubmissionRejected("a store has one source and takes no route")
        return self._start(extent, self._do_fetch)

    def publish(self, extent: CacheExtent) -> Attempt:
        return self._start(extent, self._do_publish)

    def probe(self, name: bytes, units: Sequence[bytes]) -> Optional[frozenset[bytes]]:
        """Queue a lookup on first sight and answer ``None``; hand out the answer once it is in.

        The answer is consumed by the call that receives it, and an unclaimed one expires after
        ``probe_ttl_s``, as does a lookup still pending after that long (it is asked again on the
        next call). A lookup that failed raises here, once, and is then forgotten so the next
        call asks again. With ``MAX_PROBES`` lookups remembered, a new one raises instead.
        """
        if not units:
            return frozenset()
        key = (name, tuple(units))
        now = time.monotonic()
        with self._lock:
            self._expire_probes(now)
            entry = self._probes.get(key)
            if entry is None:
                if self._closed:
                    raise RuntimeError("store backend is closed")
                if len(self._probes) >= MAX_PROBES:
                    raise RuntimeError(f"{MAX_PROBES} store lookups already remembered")
                entry = _Probe(created=now)
                self._probes[key] = entry
                try:
                    self._lookups.submit(self._lookup, entry, key[1])
                except RuntimeError:
                    del self._probes[key]
                    raise
                return None
            if not entry.done:
                return None
            del self._probes[key]
        if entry.error is not None:
            raise RuntimeError("store lookup failed") from entry.error
        return entry.answer

    def open_route(self, hint: Mapping[str, object]) -> Route:
        raise NotImplementedError("a store has one source and nothing to route")

    def quiesce(self, attempts: Iterable[Attempt]) -> bool:
        for attempt in attempts:
            self._own(attempt).quiet.wait()
        return True

    def settle(self, attempts: Iterable[Attempt]) -> None:
        for attempt in attempts:
            self._own(attempt).done.wait()

    def close(self) -> None:
        """Finish the work in flight, release registrations, then close the store. Idempotent.

        A worker parked for a staging slot is woken and its delivery fails, so this returns.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
        if self._staging is not None:
            self._staging.shutdown()
        self._pool.shutdown(wait=True)
        self._lookups.shutdown(wait=True)
        with self._lock:
            live, self._registrations = self._registrations, []
        for reg in live:
            reg.live = False
            try:
                self._store.unregister_span(reg.address, reg.size)
            except BlobStoreError as exc:
                logger.warning(
                    "blob store [%s]: unregistering [%#x, %#x) failed during close: %s",
                    self._store.describe(),
                    reg.address,
                    reg.address + reg.size,
                    exc,
                )
        self._store.close()

    # ---- submission ----

    @staticmethod
    def _own(attempt: Attempt) -> _StoreAttempt:
        if not isinstance(attempt, _StoreAttempt):
            raise TypeError(f"attempt {attempt!r} was not made by this backend")
        return attempt

    def _start(
        self,
        extent: CacheExtent,
        run: Callable[[_StoreAttempt, Sequence[_Task]], Outcome],
    ) -> Attempt:
        attempt = _StoreAttempt()
        if not extent.units:
            attempt.finish(Delivered(frozenset()))
            return attempt
        with self._lock:
            if self._closed:
                raise SubmissionRejected("store backend is closed")
            tasks, problem = self._prepare(extent)
        if problem is not None:
            self._fail(attempt, problem)
            return attempt
        if not self._inflight.acquire(blocking=False):
            raise SubmissionRejected(
                f"{self._config.max_inflight_ops} deliveries already in flight"
            )
        try:
            self._pool.submit(self._run, attempt, run, tasks)
        except RuntimeError as exc:
            self._inflight.release()
            raise SubmissionRejected(str(exc)) from exc
        return attempt

    def _prepare(self, extent: CacheExtent) -> tuple[list[_Task], Optional[str]]:
        """Resolve every unit. A unit the backend cannot reach makes the whole delivery fail
        (§6.4 invariant 3d); nothing has escaped yet, but a wiring error is not back-pressure,
        so it is reported through the attempt rather than as ``SubmissionRejected``."""
        tasks: list[_Task] = []
        for unit in extent.units:
            try:
                segments = tuple(self._resolve(unit.local_group, unit.local))
            except (KeyError, ValueError) as exc:
                return tasks, f"unit ({unit.local_group}, {unit.local}) does not resolve: {exc}"
            task = _Task(unit.name, self._keys.key(unit.name), segments)
            if task.total <= 0:
                return tasks, f"unit ({unit.local_group}, {unit.local}) resolves to no memory"
            if self._staging is not None:
                if not self._staging.fits(task.total):
                    return tasks, f"unit of {task.total} B exceeds the staging slot"
            else:
                bad = self._unregistered(segments)
                if bad is not None:
                    return tasks, f"[{bad[0]:#x}, {bad[0] + bad[1]:#x}) is not registered"
            tasks.append(task)
        return tasks, None

    def _run(
        self,
        attempt: _StoreAttempt,
        run: Callable[[_StoreAttempt, Sequence[_Task]], Outcome],
        tasks: Sequence[_Task],
    ) -> None:
        outcome: Outcome = Failed("delivery did not run")
        try:
            outcome = run(attempt, tasks)
        except Exception as exc:  # noqa: BLE001 - thread boundary; the outcome carries the error
            outcome = Failed(f"{type(exc).__name__}: {exc}")
        except BaseException as exc:
            outcome = Failed(f"{type(exc).__name__}: {exc}")
            raise
        finally:
            # Whatever happened, the attempt reaches an outcome so no wait on it hangs.
            self._inflight.release()
            if isinstance(outcome, Failed):
                self._fail(attempt, outcome.reason)
            else:
                attempt.finish(outcome)

    def _fail(self, attempt: _StoreAttempt, reason: str) -> None:
        with self._lock:
            self.counters.failed_attempts += 1
        logger.warning("blob store [%s]: delivery failed: %s", self._store.describe(), reason)
        attempt.finish(Failed(reason))

    def _holds(self, keys: Sequence[str]) -> Sequence[bool]:
        """Ask the store for ``keys``; an answer of the wrong length is a failed lookup too."""
        present = self._store.holds(keys)
        if len(present) != len(keys):
            raise BlobStoreError(f"holds answered {len(present)} of {len(keys)} keys")
        return present

    def _staged(self, count: int, body: Callable[[list[int]], _T]) -> _T:
        """Run ``body`` with ``count`` staging slots, which go back when it returns or raises.

        Drain contract: ``body`` waits for its own copies (``staging.sync``) before it returns,
        so on the normal path the slots are quiet when released. When ``body`` raises, the copies
        are drained here instead: a copy still in flight into a slot someone else then reuses
        would corrupt their delivery, and one out of the caller's memory would break quiescence."""
        assert self._staging is not None
        slots = self._staging.acquire(count)
        try:
            return body(slots)
        except BaseException:
            try:
                self._staging.sync()
            except Exception:  # noqa: BLE001 - best effort; the original error is what matters
                logger.warning("staging sync failed while unwinding a failed delivery")
            raise
        finally:
            self._staging.release(slots)

    # ---- the work ----

    def _do_fetch(self, attempt: _StoreAttempt, tasks: Sequence[_Task]) -> Outcome:
        served: set[bytes] = set()
        for batch in _batched(tasks, self._batch):
            try:
                present = self._holds([task.key for task in batch])
            except BlobStoreError as exc:
                return Failed(f"store lookup failed: {exc}")
            hits = [task for task, held in zip(batch, present) if held]
            misses = len(batch) - len(hits)
            if hits:
                if self._staging is None:
                    results = self._store.get([t.key for t in hits], [t.segments for t in hits])
                    got, bad = _reads(hits, results)
                else:
                    got, bad = self._staged(
                        len(hits), lambda slots, hits=hits: self._staged_get(hits, slots)
                    )
                if bad:
                    return Failed(f"{bad} of {len(hits)} present units could not be read")
                misses += len(hits) - len(got)
                served.update(task.name for task in got)
            with self._lock:
                self.counters.fetch_hits += len(batch) - misses
                self.counters.fetch_misses += misses
        return Delivered(frozenset(served))

    def _staged_get(self, hits: Sequence[_Task], slots: Sequence[int]) -> tuple[list[_Task], int]:
        assert self._staging is not None
        results = self._store.get(
            [t.key for t in hits],
            [[(self._staging.slot_address(slot), t.total)] for slot, t in zip(slots, hits)],
        )
        got, bad = _reads(hits, results)
        if not bad:
            for slot, task in zip(slots, hits):
                if task in got:
                    self._staging.scatter(slot, task.segments)
            self._staging.sync()
        return got, bad

    def _do_publish(self, attempt: _StoreAttempt, tasks: Sequence[_Task]) -> Outcome:
        served: set[bytes] = set()
        pending: list[_Task] = []
        for batch in _batched(tasks, self._batch):
            try:
                present = self._holds([task.key for task in batch])
            except BlobStoreError as exc:
                # A store that cannot answer whether it holds a unit is out of reach; treating the
                # answer as "absent" would turn an outage into writes against it (and a fetch on
                # the same answer fails, so the two directions agree).
                return Failed(f"store lookup failed: {exc}")
            # Present units are merged, not rewritten (§6.3 requirement 2).
            pending.extend(task for task, held in zip(batch, present) if not held)
            served.update(task.name for task, held in zip(batch, present) if held)
        with self._lock:
            self.counters.publish_present += len(tasks) - len(pending)
        if self._staging is None:
            for batch in _batched(pending, self._batch):
                taken, problem = self._put(batch, [t.segments for t in batch])
                if problem is not None:
                    return Failed(problem)
                served.update(task.name for task in taken)
            return Delivered(frozenset(served))
        # Staging goes in rounds of the whole slot pool, gathering a round before writing any of
        # it, so that an extent no larger than the pool is quiet before its first remote write
        # (design §10.1). A larger extent is quiet only after its last round's gather.
        rounds = list(_batched(pending, self._staging.num_slots))
        for index, group in enumerate(rounds):
            last = index == len(rounds) - 1
            problem = self._staged(
                len(group),
                lambda slots, group=group, last=last: self._staged_put(
                    attempt, group, slots, last, served
                ),
            )
            if problem is not None:
                return Failed(problem)
        return Delivered(frozenset(served))

    def _staged_put(
        self,
        attempt: _StoreAttempt,
        group: Sequence[_Task],
        slots: Sequence[int],
        last: bool,
        served: set[bytes],
    ) -> Optional[str]:
        assert self._staging is not None
        for slot, task in zip(slots, group):
            self._staging.gather(slot, task.segments)
        self._staging.sync()
        if last:
            attempt.quiet.set()
        for batch, batch_slots in zip(_batched(group, self._batch), _batched(slots, self._batch)):
            taken, problem = self._put(
                batch,
                [
                    [(self._staging.slot_address(slot), task.total)]
                    for slot, task in zip(batch_slots, batch)
                ],
            )
            if problem is not None:
                return problem
            served.update(task.name for task in taken)
        return None

    def _put(
        self, tasks: Sequence[_Task], buffers: Sequence[Sequence[Segment]]
    ) -> tuple[list[_Task], Optional[str]]:
        """Write one batch. Returns the units the store now holds, and a reason if the call failed.

        A unit the store declined but holds anyway lost a race with another publisher and counts
        as taken; one it declined and does not hold is simply not served. A call that misbehaves
        (exception, wrong count, a unit that failed to write, a failed lookup afterwards) is a
        failure. Units stored before such a failure are still returned and counted in
        ``publish_stored`` (they are in the store), but the caller's attempt ends ``Failed`` and
        reports no served set.
        """
        results = self._store.put([t.key for t in tasks], buffers)
        if len(results) != len(tasks):
            return [], f"put answered {len(results)} of {len(tasks)} keys"
        stored = [task for task, status in zip(tasks, results) if status is PutStatus.STORED]
        with self._lock:
            self.counters.publish_stored += len(stored)
        failed = sum(1 for status in results if status is PutStatus.FAILED)
        if failed:
            return stored, f"{failed} of {len(tasks)} units could not be written"
        declined = [task for task, status in zip(tasks, results) if status is PutStatus.DECLINED]
        raced: list[_Task] = []
        if declined:
            try:
                present = self._holds([task.key for task in declined])
            except BlobStoreError as exc:
                return stored, f"store lookup failed after a declined put: {exc}"
            raced = [task for task, held in zip(declined, present) if held]
            if len(raced) < len(declined):
                logger.warning(
                    "blob store [%s]: did not take %d of %d units",
                    self._store.describe(),
                    len(declined) - len(raced),
                    len(tasks),
                )
        with self._lock:
            self.counters.publish_raced += len(raced)
        return stored + raced, None

    def _lookup(self, entry: _Probe, units: Sequence[bytes]) -> None:
        held: set[bytes] = set()
        try:
            for batch in _batched(units, self._batch):
                present = self._holds([self._keys.key(u) for u in batch])
                held.update(u for u, is_held in zip(batch, present) if is_held)
            with self._lock:
                self.counters.probe_hits += len(held)
                self.counters.probe_misses += len(units) - len(held)
                entry.answer = frozenset(held)
        except BaseException as exc:  # noqa: BLE001 - thread boundary; re-raised from probe
            logger.warning(
                "blob store [%s]: lookup failed: %s: %s",
                self._store.describe(),
                type(exc).__name__,
                exc,
            )
            with self._lock:
                entry.error = exc
            if not isinstance(exc, Exception):
                raise
        finally:
            # Whatever happened, the probe stops pending so the caller is never deferred forever.
            with self._lock:
                entry.done = True

    def _expire_probes(self, now: float) -> None:
        """Drop answered and still-pending lookups older than the TTL. Caller holds the lock.

        A pending entry that expires is detached, not cancelled: the lookup thread writes its
        answer into an object nobody reads any more, which is harmless.
        """
        ttl = self._config.probe_ttl_s
        stale = [key for key, entry in self._probes.items() if now - entry.created > ttl]
        for key in stale:
            del self._probes[key]


def _reads(tasks: Sequence[_Task], results: Sequence[GetStatus]) -> tuple[list[_Task], int]:
    """Split the answer of one get into the units fully delivered and how many went wrong.

    A unit the store no longer holds (gone since the lookup) is neither: nothing was written and
    it is simply not served. An answer of the wrong length counts every unit as wrong.
    """
    if len(results) != len(tasks):
        return [], len(tasks)
    got = [task for task, status in zip(tasks, results) if status is GetStatus.HIT]
    bad = sum(1 for status in results if status is GetStatus.FAILED)
    return got, bad

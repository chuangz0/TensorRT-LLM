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
"""A cache backend over a blob store, in two shapes chosen by ``BlobStoreConfig.landing``.

``landing: device`` is ``BlobStoreBackend`` alone: ``Fetches``, ``Publishes``, ``RegistersPools``;
the store reads and writes the caller's pages directly. ``landing: host`` wraps it in
``HostLandingBlobBackend`` (``LandsOnHost``): a fetch lands in the backend's own pinned host memory
first and is copied into the caller's pages once the scheduler has reserved them, and a publish
is gathered into host memory before the store is asked to take it. Two host pools serve the two
directions (see ``HostLandingBlobBackend``).

The classes depend only on the ``BlobStore`` protocol (``store.py``); a driver under ``drivers/``
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

__all__ = [
    "DEFAULT_LANDING_WAIT_TIMEOUT_S",
    "LOOKUP_RETRIES",
    "LOOKUP_RETRY_DELAY_S",
    "MAX_PROBES",
    "BlobStoreBackend",
    "BlobStoreConfig",
    "HostLandingBlobBackend",
    "StoreCounters",
    "UnitBytes",
]

MAX_PROBES = 1024
"""Bound on remembered lookups, pending or answered. A lookup thread that stopped answering must
not let the table grow without end; past the bound a new ``probe`` raises and the caller plans
without the store."""

LOOKUP_RETRIES = 2
"""Times a probe's store lookup is asked again after a ``BlobStoreError`` (so up to three tries),
``LOOKUP_RETRY_DELAY_S`` apart: one RPC hiccup must not consume the planner's probe budget;
anything longer is a real failure and shows up as ``probe_failed``."""

LOOKUP_RETRY_DELAY_S = 0.005

DEFAULT_LANDING_WAIT_TIMEOUT_S = 30.0
"""Longest a landing waits in the pool's queue for its slots before it fails, when the assembly
carries no ``landing_wait_timeout_s`` of its own. Short capacity is a wait, not an error
(``LandsOnHost.fetch_to_host``); the assembly normally passes the coordinator's configured value,
so queue waits and page waits share one bound. ``None`` means no bound."""

# The stdlib logger keeps this module import-light (no ``tensorrt_llm`` import); note that
# ``TLLM_LOG_LEVEL_BY_MODULE`` does not route it, so configure ``logging`` for this name directly.
logger = logging.getLogger(__name__)

_T = TypeVar("_T")

_DEFAULT_STAGING_BUFFER_BYTES = 512 * 1024 * 1024
_DEFAULT_LANDING_BUFFER_BYTES = 2 * 1024 * 1024 * 1024

UnitBytes = Callable[[bytes], int]
"""Byte size of the unit called ``name``; what a host-first fetch needs to size a get whose
destination is a host slot rather than the unit's own pages."""


@dataclass(frozen=True)
class BlobStoreConfig:
    """What the backend reads from a backend entry; the driver's connection options are separate.

    Attributes:
        namespace: Leading component of every key; two deployments share cache only when they agree.
        transfer_batch_size: Units per store call. Bounds one call, not one delivery.
        landing: Where a fetch lands first. ``device``: the store writes the caller's pages, which
            are registered with it (needs GPUDirect for device memory). ``host``: the store only
            touches pinned host memory of the backend's own; a fetch lands there and is copied
            into pages afterwards (``LandsOnHost``), a publish is gathered there first. The
            caller's pools stay unregistered.
        staging_buffer_bytes: Ceiling on the pinned publish pool when ``landing`` is ``host``.
        landing_buffer_bytes: Ceiling on the pinned landing pool when ``landing`` is ``host``.
        max_landed_units: Cap on the landing pool's slot count; ``None`` takes every slot the
            budget affords. A landing holds one slot per unit until it is released.
        max_inflight_ops: Deliveries that may be queued or running at once. A submission past
            this bound is refused with ``SubmissionRejected``. Landings are bounded by their
            slots instead and do not count.
        num_workers: Threads that drive store calls.
        probe_ttl_s: Seconds an unconsumed probe answer is kept before it is dropped.
    """

    namespace: str = "trtllm"
    transfer_batch_size: int = 64
    landing: str = "device"
    staging_buffer_bytes: int = _DEFAULT_STAGING_BUFFER_BYTES
    landing_buffer_bytes: int = _DEFAULT_LANDING_BUFFER_BYTES
    max_landed_units: Optional[int] = None
    max_inflight_ops: int = 256
    num_workers: int = 2
    probe_ttl_s: float = 30.0

    def __post_init__(self) -> None:
        if not isinstance(self.namespace, str) or not self.namespace:
            raise ValueError(f"namespace must be a non-empty string, got {self.namespace!r}")
        for name in (
            "transfer_batch_size",
            "max_inflight_ops",
            "num_workers",
            "staging_buffer_bytes",
            "landing_buffer_bytes",
        ):
            _require_int(name, getattr(self, name))
        for name in ("transfer_batch_size", "max_inflight_ops", "num_workers"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.landing not in ("device", "host"):
            raise ValueError(f"landing must be 'device' or 'host', got {self.landing!r}")
        if self.lands_on_host:
            for name in ("staging_buffer_bytes", "landing_buffer_bytes"):
                if getattr(self, name) <= 0:
                    raise ValueError(f"{name} must be > 0 when landing is 'host'")
        if self.max_landed_units is not None:
            _require_int("max_landed_units", self.max_landed_units)
            if self.max_landed_units <= 0:
                raise ValueError("max_landed_units must be > 0")
        if isinstance(self.probe_ttl_s, bool) or not isinstance(self.probe_ttl_s, (int, float)):
            raise ValueError(f"probe_ttl_s must be a number, got {self.probe_ttl_s!r}")
        if self.probe_ttl_s <= 0:
            raise ValueError("probe_ttl_s must be > 0")

    @property
    def lands_on_host(self) -> bool:
        return self.landing == "host"

    @classmethod
    def fields(cls) -> frozenset[str]:
        """The option keys this class reads; a driver's keys must not overlap them."""
        return frozenset(f.name for f in dataclasses.fields(cls))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> BlobStoreConfig:
        """Build from a plain mapping, refusing keys this class does not know. A value of the
        wrong type is a ``ValueError`` like any other bad value."""
        unknown = sorted(set(raw) - cls.fields())
        if unknown:
            raise ValueError(f"unknown BlobStoreConfig keys: {unknown}")
        return cls(**raw)


def _require_int(name: str, value: Any) -> None:
    """A YAML ``true`` or ``"4"`` where a count or byte size belongs is a config error, not a
    ``TypeError`` from the first comparison that meets it."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")


@dataclass
class StoreCounters:
    """Operational counters (contract §6.2 implementation requirement 4). Read-only for callers."""

    fetch_hits: int = 0
    fetch_misses: int = 0
    publish_stored: int = 0
    publish_raced: int = 0
    """Units the store declined and then turned out to hold: another publisher got there first."""
    publish_declined: int = 0
    """Units the store declined and does not hold: it chose not to take them (no room, say)."""
    publish_present: int = 0
    probe_hits: int = 0
    probe_misses: int = 0
    probe_failed: int = 0
    """Lookups the store could not answer; the planner then decides as if the probe were unanswered."""
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


_Work = Callable[[_StoreAttempt, Sequence[_Task]], Outcome]


class BlobStoreBackend:
    """Fetch from and publish to one blob store.

    Args:
        store: An opened blob store. Owned from here on; ``close`` closes it.
        config: Batch sizes, bounds and the landing shape.
        resolver: Maps a unit's local coordinates to its memory segments.
        layout_fingerprint: Folded into every key; see ``KeyScheme``.
        staging: The publish pool, required when ``config.landing`` is ``host`` and ignored
            otherwise. Owned from here on: ``close`` shuts it down so that no worker stays parked
            waiting for a slot. With a publish pool this backend serves ``publish`` only: a fetch
            of that shape goes through ``HostLandingBlobBackend.fetch_to_host``.
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
        if config.lands_on_host and staging is None:
            raise ValueError("landing 'host' needs a HostStagingPool for publishes")
        self._store = store
        self._config = config
        self._resolve = resolver
        self._keys = KeyScheme(config.namespace, layout_fingerprint)
        self._staging = staging if config.lands_on_host else None
        self._put_batch = config.transfer_batch_size
        """Units per put call. Each unit of a staged put holds a publish-pool slot for the call,
        so the pool's slot count bounds it too; lookups have no slot and use
        ``transfer_batch_size`` as is."""
        if self._staging is not None:
            self._put_batch = min(self._put_batch, self._staging.num_slots)
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

    def describe(self) -> str:
        """The store's one-line description, for log lines."""
        return self._store.describe()

    # ---- RegistersPools ----

    def register_pool(self, address: int, size: int) -> Registration:
        if size <= 0:
            raise ValueError(f"size must be > 0, got {size}")
        span = (address, size)
        with self._lock:
            if self._closed:
                raise RuntimeError("store backend is closed")
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
        except BaseException:
            # A close that raised has not closed: the handle stays live and may be retried.
            with self._lock:
                reg.closing = False
            raise
        with self._lock:
            reg.closing = False
            reg.live = False
            # ``close`` may have taken the list meanwhile; it leaves a closing handle to us.
            if reg in self._registrations:
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
        if self._staging is not None:
            raise RuntimeError("a host-landing backend fetches through fetch_to_host")
        return self._start(extent, self._do_fetch)

    def publish(self, extent: CacheExtent) -> Attempt:
        return self._start(extent, self._do_publish)

    def probe(self, name: bytes, units: Sequence[bytes]) -> Optional[frozenset[bytes]]:
        """Queue a lookup on first sight and answer ``None``; hand out the answer once it is in.

        The answer is consumed by the call that receives it, and an unclaimed one expires after
        ``probe_ttl_s``, as does a lookup still pending after that long (it is asked again on the
        next call). A lookup whose store call fails is retried (``LOOKUP_RETRIES``); one that
        still failed raises here, once, and is then forgotten so the next call asks again. With
        ``MAX_PROBES`` lookups remembered, a new one raises instead.
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
            # The cause's text rides in the message: the caller logs ``str(exc)`` only.
            raise RuntimeError(f"store lookup failed: {entry.error}") from entry.error
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
            # A handle another thread is closing right now is left to that thread, so the store
            # is not asked to unregister the same span twice.
            live = [reg for reg in self._registrations if not reg.closing]
            self._registrations = []
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

    def _start(self, extent: CacheExtent, run: _Work) -> Attempt:
        return self.submit_delivery(extent, run, self._check_destinations)

    # ---- the surface the host-landing shape builds on ----

    @property
    def workers(self) -> DaemonWorkerPool:
        """The delivery threads; the host shape runs its landings on them too."""
        return self._pool

    def submit_delivery(
        self,
        extent: CacheExtent,
        run: _Work,
        check: Callable[[Sequence[_Task]], Optional[str]],
    ) -> Attempt:
        """``admit_delivery`` then ``launch_delivery``: resolve ``extent``'s units, let ``check``
        refuse them, and hand ``run`` to a worker when they pass."""
        attempt, tasks = self.admit_delivery(extent, check)
        if tasks is not None:
            self.launch_delivery(attempt, run, tasks)
        return attempt

    def admit_delivery(
        self, extent: CacheExtent, check: Callable[[Sequence[_Task]], Optional[str]]
    ) -> tuple[_StoreAttempt, Optional[list[_Task]]]:
        """Resolve ``extent``'s units and let ``check`` refuse them. Returns the attempt and the
        tasks to launch, or the attempt already finished and ``None`` when nothing will run: an
        empty extent is delivered at once, a refused one fails before anything moves. A closed
        backend refuses with ``SubmissionRejected``."""
        attempt = _StoreAttempt()
        if not extent.units:
            attempt.finish(Delivered(frozenset()))
            return attempt, None
        with self._lock:
            if self._closed:
                raise SubmissionRejected("store backend is closed")
            tasks, problem = self._resolve_units(extent)
            if problem is None:
                problem = check(tasks)
        if problem is not None:
            self._fail(attempt, problem)
            return attempt, None
        return attempt, tasks

    def launch_delivery(self, attempt: _StoreAttempt, run: _Work, tasks: Sequence[_Task]) -> None:
        """Take an in-flight slot and hand ``run`` to a worker; refuse with ``SubmissionRejected``
        when either is unavailable, in which case ``run`` never runs. The attempt reaches an
        outcome on the worker, through ``_run``."""
        if not self._inflight.acquire(blocking=False):
            raise SubmissionRejected(
                f"{self._config.max_inflight_ops} deliveries already in flight"
            )
        try:
            self._pool.submit(self._run, attempt, run, tasks)
        except RuntimeError as exc:
            self._inflight.release()
            raise SubmissionRejected(str(exc)) from exc

    def read_present_batches(
        self,
        tasks: Sequence[_Task],
        destinations: Callable[[Sequence[_Task]], Sequence[Sequence[Segment]]],
    ) -> tuple[list[_Task], Optional[str]]:
        """A whole fetch: ``holds`` then ``get`` per batch of ``transfer_batch_size``, into the
        buffers ``destinations`` names. Returns the units read whole, or the first batch's
        reason for failing; counts hits and misses."""
        got: list[_Task] = []
        for batch in _batched(tasks, self._config.transfer_batch_size):
            read, problem = self._read_present(batch, destinations)
            if problem is not None:
                return got, problem
            got.extend(read)
        return got, None

    def record_failure(self, reason: str) -> None:
        """Count and log a delivery that failed without an attempt of this backend's own."""
        with self._lock:
            self.counters.failed_attempts += 1
        logger.warning("blob store [%s]: delivery failed: %s", self._store.describe(), reason)

    def _check_destinations(self, tasks: Sequence[_Task]) -> Optional[str]:
        """Whether the store can reach every unit: inside a registration on the device path, or
        within a publish-pool slot when staging. A unit it cannot reach makes the whole delivery
        fail (§6.4 invariant 3d); nothing has escaped yet, but a wiring error is not
        back-pressure, so it is reported through the attempt rather than as
        ``SubmissionRejected``. Caller holds the lock."""
        for task in tasks:
            if self._staging is not None:
                if not self._staging.fits(task.total):
                    return f"unit of {task.total} B exceeds the staging slot"
            else:
                bad = self._unregistered(task.segments)
                if bad is not None:
                    return f"[{bad[0]:#x}, {bad[0] + bad[1]:#x}) is not registered"
        return None

    def _resolve_units(self, extent: CacheExtent) -> tuple[list[_Task], Optional[str]]:
        """One task per unit with its segments, or the first unit that does not resolve."""
        tasks: list[_Task] = []
        for unit in extent.units:
            try:
                segments = tuple(self._resolve(unit.local_group, unit.local))
            except (KeyError, ValueError) as exc:
                return tasks, f"unit ({unit.local_group}, {unit.local}) does not resolve: {exc}"
            task = _Task(unit.name, self._keys.key(unit.name), segments)
            if task.total <= 0:
                return tasks, f"unit ({unit.local_group}, {unit.local}) resolves to no memory"
            tasks.append(task)
        return tasks, None

    def _run(self, attempt: _StoreAttempt, run: _Work, tasks: Sequence[_Task]) -> None:
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
        self.record_failure(reason)
        attempt.finish(Failed(reason))

    def _holds(self, keys: Sequence[str]) -> Sequence[bool]:
        """Ask the store for ``keys``; an answer of the wrong length is a failed lookup too."""
        present = self._store.holds(keys)
        if len(present) != len(keys):
            raise BlobStoreError(f"holds answered {len(present)} of {len(keys)} keys")
        return present

    def _staged(self, count: int, body: Callable[[list[int]], _T]) -> _T:
        """Run ``body`` with ``count`` publish-pool slots, which go back when it returns or raises.

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
        got, problem = self.read_present_batches(tasks, lambda hits: [t.segments for t in hits])
        if problem is not None:
            return Failed(problem)
        return Delivered(frozenset(task.name for task in got))

    def _read_present(
        self,
        batch: Sequence[_Task],
        destinations: Callable[[Sequence[_Task]], Sequence[Sequence[Segment]]],
    ) -> tuple[list[_Task], Optional[str]]:
        """One batch of a fetch: ``holds``, then ``get`` the present units into the buffers
        ``destinations`` names for them. Returns the units read whole, or a reason the batch
        failed; counts hits and misses."""
        try:
            present = self._holds([task.key for task in batch])
        except BlobStoreError as exc:
            return [], f"store lookup failed: {exc}"
        hits = [task for task, held in zip(batch, present) if held]
        got: list[_Task] = []
        if hits:
            results = self._store.get([t.key for t in hits], destinations(hits))
            got, bad = _reads(hits, results)
            if bad:
                return got, f"{bad} of {len(hits)} present units could not be read"
        with self._lock:
            self.counters.fetch_hits += len(got)
            self.counters.fetch_misses += len(batch) - len(got)
        return got, None

    def _do_publish(self, attempt: _StoreAttempt, tasks: Sequence[_Task]) -> Outcome:
        served: set[bytes] = set()
        pending: list[_Task] = []
        for batch in _batched(tasks, self._config.transfer_batch_size):
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
            for batch in _batched(pending, self._put_batch):
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
        for batch, batch_slots in zip(
            _batched(group, self._put_batch), _batched(slots, self._put_batch)
        ):
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
            self.counters.publish_declined += len(declined) - len(raced)
        return stored + raced, None

    def _holds_with_retry(self, keys: Sequence[str]) -> Sequence[bool]:
        """``_holds``, asked again up to ``LOOKUP_RETRIES`` times after a ``BlobStoreError``.

        The last try's error propagates; any other exception propagates at once.
        """
        for tried in range(1, LOOKUP_RETRIES + 1):
            try:
                return self._holds(keys)
            except BlobStoreError as exc:
                logger.debug(
                    "blob store [%s]: lookup try %d of %d failed, retrying: %s",
                    self._store.describe(),
                    tried,
                    LOOKUP_RETRIES + 1,
                    exc,
                )
                time.sleep(LOOKUP_RETRY_DELAY_S)
        return self._holds(keys)

    def _lookup(self, entry: _Probe, units: Sequence[bytes]) -> None:
        held: set[bytes] = set()
        try:
            for batch in _batched(units, self._config.transfer_batch_size):
                present = self._holds_with_retry([self._keys.key(u) for u in batch])
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
                self.counters.probe_failed += 1
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


# ---------------------------------------------------------------------------------------------
# landing: host
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _LandingUnit:
    """One unit a landing asks for: its name, key and byte size; no local coordinates yet."""

    name: bytes
    key: str
    total: int


@dataclass(frozen=True)
class _SlotContent:
    """Where a unit the get delivered whole now sits: its slot, and how many bytes of it."""

    slot: int
    total: int


class _Landing:
    """One host-first landing: the ``Landing`` a ``HostLandingBlobBackend`` hands out.

    Its life is four states, moved under ``_lock``. ``QUEUED``: waiting in the landing pool for
    one slot per unit. ``LANDING``: slots granted, a worker runs ``holds`` then ``get`` into them.
    ``LANDED``: ``poll`` has an outcome; on ``Delivered`` the slots hold the served units until
    ``release``. ``RELEASED``: holds nothing, does nothing. ``release`` is non-blocking and runs
    on the caller's thread: it dequeues a queued landing, asks a landing one to give its slots
    back when its get returns, gives a landed one's slots back at once unless a placement is
    still copying out of them (then the last placement to finish gives them back), and does
    nothing after the backend has closed (the pool is being torn down and the workers are
    stopping).
    """

    QUEUED, LANDING, LANDED, RELEASED = "queued", "landing", "landed", "released"

    def __init__(
        self, backend: HostLandingBlobBackend, name: bytes, units: Sequence[_LandingUnit]
    ) -> None:
        self._backend = backend
        self.name = name
        self.units = tuple(units)
        self.enqueued_at = time.monotonic()
        self._lock = threading.Lock()
        self._state = self.QUEUED
        self._outcome: Optional[Outcome] = None
        self._slots: list[int] = []
        self._served: dict[bytes, _SlotContent] = {}
        """Unit name -> where its bytes sit, for units the get delivered whole."""
        self._placing = 0
        """Placements that may still read the slots; counted up in ``place``, down when the copy
        has landed or the placement was refused at submission."""
        self._release_requested = False

    # -- Landing --

    def poll(self) -> Optional[Outcome]:
        """The outcome, or ``None`` while queued or landing. A landing queued for longer than the
        backend's wait bound leaves the queue here and fails: the bound is checked on the
        caller's clock, so there is no timing thread."""
        with self._lock:
            if self._state is self.QUEUED and self._waited_too_long():
                if self._backend.landing_pool.dequeue(self):
                    self._finish_locked(
                        Failed(
                            f"waited {self._backend.landing_wait_timeout_s:g} s for "
                            f"{len(self.units)} landing slots"
                        ),
                        self.RELEASED,
                    )
            return self._outcome

    def place(self, extent: CacheExtent) -> Attempt:
        """Copy the units of ``extent`` out of their slots into the units' own segments, on a
        worker and its copy stream; ``Delivered`` after the copies have landed. Only units this
        landing delivered may be asked for, each at the size it landed with; nothing is touched
        otherwise. The slots stay held until the copy is done, whatever ``release`` says
        meanwhile."""
        with self._lock:
            placeable = (
                self._state is self.LANDED
                and isinstance(self._outcome, Delivered)
                and not self._release_requested
            )
            if not placeable:
                raise SubmissionRejected("the landing has no content to place")
            content = dict(self._served)
            self._placing += 1
        try:
            return self._backend._place(extent, content, self._placement_done)
        except BaseException:
            self._placement_done()
            raise

    def release(self) -> None:
        with self._lock:
            if self._state is self.RELEASED or self._backend.is_closed:
                self._state = self.RELEASED
                return
            if self._state is self.QUEUED:
                if self._backend.landing_pool.dequeue(self):
                    self._finish_locked(Failed("released before landing"), self.RELEASED)
                else:
                    # Popped by a granting thread that has yet to call ``slots_granted``.
                    self._release_requested = True
                return
            if self._state is self.LANDING or self._placing:
                self._release_requested = True
                return
            slots = self._give_up_slots_locked()
        self._give_back(slots)

    def release_for_close(self) -> None:
        """``release`` as the closing backend calls it, after the pool refused every queued
        landing: a get or a placement still in flight returns the slots when it ends, held slots
        go back now."""
        with self._lock:
            if self._state is self.LANDING or self._placing:
                self._release_requested = True
                return
            if self._state is self.QUEUED:
                self._finish_locked(Failed("released before landing"), self.RELEASED)
                return
            slots = self._give_up_slots_locked()
        self._give_back(slots)

    def deliver_empty(self) -> None:
        """A landing of no units: landed at once, holding nothing."""
        with self._lock:
            self._finish_locked(Delivered(frozenset()), self.LANDED)

    # -- SlotWaiter --

    def slots_granted(self, slots: list[int]) -> None:
        with self._lock:
            if self._release_requested or self._state is not self.QUEUED:
                self._finish_locked(Failed("released before landing"), self.RELEASED)
                give_back = slots
            else:
                self._slots = slots
                self._state = self.LANDING
                give_back = []
        if give_back:
            self._give_back(give_back)
            return
        try:
            self._backend.workers.submit(self._land)
        except RuntimeError as exc:
            self._end(Failed(f"could not start the landing: {exc}"))

    def slots_refused(self, reason: str) -> None:
        with self._lock:
            self._finish_locked(Failed(reason), self.RELEASED)

    # -- the work --

    def _land(self) -> None:
        """Worker: ``holds`` then ``get`` into the slots, batch by batch, back to back."""
        outcome: Outcome = Failed("landing did not run")
        try:
            outcome = self._backend._land(self.units, self._slots, self._served)
        except Exception as exc:  # noqa: BLE001 - thread boundary; the outcome carries the error
            outcome = Failed(f"{type(exc).__name__}: {exc}")
        except BaseException as exc:
            outcome = Failed(f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self._end(outcome)

    def _end(self, outcome: Outcome) -> None:
        """Record the outcome. The slots go back at once when nothing landed to place, or when
        a release was asked for while the get was in flight."""
        with self._lock:
            self._finish_locked(outcome, self.LANDED)
            done_with_slots = self._release_requested or isinstance(outcome, Failed)
            slots = self._give_up_slots_locked() if done_with_slots else []
        if isinstance(outcome, Failed):
            self._backend._count_failed(outcome.reason)
        self._give_back(slots)

    def _placement_done(self) -> None:
        """One placement no longer reads the slots (its copy landed, or it was refused before a
        worker took it). The last one out honours a release asked for while it ran."""
        with self._lock:
            self._placing -= 1
            give_back = self._placing == 0 and self._release_requested
            slots = self._give_up_slots_locked() if give_back else []
        self._give_back(slots)

    def _give_back(self, slots: list[int]) -> None:
        if slots:
            self._backend.landing_pool.release(slots)

    # -- helpers, caller holds the lock --

    def _waited_too_long(self) -> bool:
        bound = self._backend.landing_wait_timeout_s
        return bound is not None and time.monotonic() - self.enqueued_at > bound

    def _finish_locked(self, outcome: Outcome, state: str) -> None:
        if self._outcome is None:
            self._outcome = outcome
        self._state = state
        if state is self.RELEASED:
            self._backend._forget(self)

    def _give_up_slots_locked(self) -> list[int]:
        slots, self._slots = self._slots, []
        self._served = {}
        self._state = self.RELEASED
        self._backend._forget(self)
        return slots

    @property
    def holds_slots(self) -> bool:
        with self._lock:
            return bool(self._slots)


class HostLandingBlobBackend:
    """The ``landing: host`` shape of the blob backend: ``LandsOnHost`` over a ``BlobStoreBackend``.

    Composition rather than inheritance, so this class has no ``fetch``: a backend is a
    ``Fetches`` or a ``LandsOnHost``, never both. ``probe``, ``quiesce``, ``settle`` and the
    counters are the inner backend's; a publish goes to the inner backend directly, through its
    publish pool.

    Two host pools, because their slots live differently. The inner backend's publish pool hands
    slots to a worker for the length of one task, so a worker may block for them. The landing pool
    here hands slots to a ``_Landing`` that keeps them across scheduler rounds, until the
    coordinator has placed the content and released the landing; a worker blocked for one of
    those could be blocked for as long as the scheduler takes to find pages, and with every
    worker so blocked nothing would ever return a slot. So landings never block a worker: they
    wait in the pool's queue, and the thread that returns slots hands the head of the queue to a
    worker (``HostStagingPool.enqueue``). Landings are bounded by their slots and do not take the
    inner backend's in-flight semaphore; placements, which write pages, do.

    Args:
        inner: The backend over the store, built with its publish pool.
        landing_pool: The landing pool; every slot is at least ``max_unit_bytes`` wide.
        unit_bytes: Byte size of a unit by its name, to size the get into a slot.
        landing_wait_timeout_s: Longest a landing waits for slots before it fails; ``None`` for
            no bound. The assembly passes the coordinator's ``landing_wait_timeout_s``.
    """

    def __init__(
        self,
        inner: BlobStoreBackend,
        landing_pool: HostStagingPool,
        unit_bytes: UnitBytes,
        *,
        landing_wait_timeout_s: float | None = DEFAULT_LANDING_WAIT_TIMEOUT_S,
    ) -> None:
        if landing_wait_timeout_s is not None and landing_wait_timeout_s <= 0:
            raise ValueError("landing_wait_timeout_s must be > 0 or None")
        self._inner = inner
        self.landing_pool = landing_pool
        self._unit_bytes = unit_bytes
        self.landing_wait_timeout_s = landing_wait_timeout_s
        self._lock = threading.Lock()
        self._landings: set[_Landing] = set()
        """Landings that are queued or hold slots; ``close`` releases them."""
        self._closed = False

    # ---- what is forwarded ----

    @property
    def counters(self) -> StoreCounters:
        return self._inner.counters

    @property
    def workers(self) -> DaemonWorkerPool:
        return self._inner.workers

    @property
    def is_closed(self) -> bool:
        return self._closed

    def key_for(self, name: bytes) -> str:
        return self._inner.key_for(name)

    def describe(self) -> str:
        return self._inner.describe()

    def probe(self, name: bytes, units: Sequence[bytes]) -> Optional[frozenset[bytes]]:
        return self._inner.probe(name, units)

    def quiesce(self, attempts: Iterable[Attempt]) -> bool:
        return self._inner.quiesce(attempts)

    def settle(self, attempts: Iterable[Attempt]) -> None:
        self._inner.settle(attempts)

    def publish(self, extent: CacheExtent) -> Attempt:
        return self._inner.publish(extent)

    def landings_held(self) -> int:
        """Landings currently holding slots, for the status dump."""
        with self._lock:
            landings = list(self._landings)
        return sum(1 for landing in landings if landing.holds_slots)

    # ---- LandsOnHost ----

    def fetch_to_host(self, name: bytes, units: Sequence[bytes]) -> _Landing:
        """Queue a landing of ``units``: one slot each, then ``holds`` and ``get`` on a worker.
        Non-blocking. A unit no slot can hold, or more units than the pool has slots, fails the
        landing at once (a wiring or sizing error, not back-pressure)."""
        landing_units, problem = self._size_units(units)
        landing = _Landing(self, name, landing_units)
        with self._lock:
            if self._closed:
                raise SubmissionRejected("store backend is closed")
            # Tracked under the same lock as the closed check: ``close`` releases the landings it
            # finds here, so none may slip in behind it.
            if units and problem is None:
                self._landings.add(landing)
        if not units:
            landing.deliver_empty()
        elif problem is not None:
            self._count_failed(problem)
            landing.slots_refused(problem)
        else:
            self.landing_pool.enqueue(landing, len(landing_units))
        return landing

    def _size_units(self, units: Sequence[bytes]) -> tuple[list[_LandingUnit], Optional[str]]:
        """Key and size every unit; or the first reason the landing can never be served."""
        if len(units) > self.landing_pool.num_slots:
            return [], f"{len(units)} units exceed the {self.landing_pool.num_slots} landing slots"
        sized: list[_LandingUnit] = []
        for unit in units:
            try:
                total = self._unit_bytes(unit)
            except (KeyError, ValueError) as exc:
                return sized, f"unit {unit.hex()} has no known size: {exc}"
            if not self.landing_pool.fits(total):
                return sized, f"unit of {total} B exceeds the landing slot"
            sized.append(_LandingUnit(unit, self._inner.key_for(unit), total))
        return sized, None

    def close(self) -> None:
        """Refuse queued landings, give every held slot back, then close the inner backend
        (which shuts the publish pool and joins the workers). Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.landing_pool.shutdown()
        with self._lock:
            landings = list(self._landings)
        for landing in landings:
            landing.release_for_close()
        self._inner.close()

    # ---- for _Landing ----

    def _land(
        self,
        units: Sequence[_LandingUnit],
        slots: Sequence[int],
        served: dict[bytes, _SlotContent],
    ) -> Outcome:
        """Worker: fetch ``units`` into ``slots``, one batch at a time, recording in ``served``
        where each unit that arrived whole now sits."""
        by_key = {unit.key: slot for unit, slot in zip(units, slots)}
        tasks = [
            _Task(unit.name, unit.key, ((self.landing_pool.slot_address(slot), unit.total),))
            for unit, slot in zip(units, slots)
        ]
        got, problem = self._inner.read_present_batches(
            tasks, lambda hits: [t.segments for t in hits]
        )
        if problem is not None:
            return Failed(problem)
        for task in got:
            served[task.name] = _SlotContent(by_key[task.key], task.total)
        return Delivered(frozenset(served))

    def _place(
        self,
        extent: CacheExtent,
        content: Mapping[bytes, _SlotContent],
        done: Callable[[], None],
    ) -> Attempt:
        """Engine thread: resolve the extent's units, check each is among ``content`` at the size
        it landed with, then hand the scatter to a worker under the inner backend's in-flight
        bound. ``done`` is called exactly once, when the scatter has run or will never run; a
        refusal raised from here leaves that to the caller."""
        with self._lock:
            if self._closed:
                raise SubmissionRejected("store backend is closed")

        def all_landed(tasks: Sequence[_Task]) -> Optional[str]:
            missing = sum(1 for task in tasks if task.name not in content)
            if missing:
                return f"{missing} of {len(tasks)} units were not landed"
            for task in tasks:
                landed = content[task.name].total
                if task.total != landed:
                    return (
                        f"unit {task.name.hex()} landed as {landed} B but resolves to "
                        f"{task.total} B"
                    )
            return None

        def scatter(attempt: _StoreAttempt, tasks: Sequence[_Task]) -> Outcome:
            try:
                return self._scatter([(content[task.name].slot, task) for task in tasks])
            finally:
                done()

        attempt, tasks = self._inner.admit_delivery(extent, all_landed)
        if tasks is None:
            done()
            return attempt
        self._inner.launch_delivery(attempt, scatter, tasks)
        return attempt

    def _scatter(self, pairs: Sequence[tuple[int, _Task]]) -> Outcome:
        """Worker: copy each unit out of its slot into its segments, then wait for the copies.
        On an error the copies issued so far are drained too, so the slots are quiet before
        they can be released."""
        pool = self.landing_pool
        try:
            for slot, task in pairs:
                pool.scatter(slot, task.segments)
            pool.sync()
        except BaseException:
            try:
                pool.sync()
            except Exception:  # noqa: BLE001 - best effort; the original error is what matters
                logger.warning("landing sync failed while unwinding a failed placement")
            raise
        return Delivered(frozenset(task.name for _, task in pairs))

    def _count_failed(self, reason: str) -> None:
        self._inner.record_failure(f"landing: {reason}")

    def _forget(self, landing: _Landing) -> None:
        with self._lock:
            self._landings.discard(landing)

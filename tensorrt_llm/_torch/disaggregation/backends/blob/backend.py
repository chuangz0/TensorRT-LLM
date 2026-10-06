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
``HostLandingBlobBackend`` (``host_landing.py``; ``LandsOnHost``): a fetch lands in the backend's
own pinned host memory first and is copied into the caller's pages once the scheduler has
reserved them, and a publish is gathered into host memory before the store is asked to take it.
Two host pools serve the two directions (see ``HostLandingBlobBackend``).

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
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    TypeVar,
)

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
from ..config import strict_from_dict
from .keys import KeyScheme
from .staging import HostStagingPool
from .store import BlobStore, BlobStoreError, GetStatus, PutStatus
from .worker_pool import DaemonWorkerPool

__all__ = [
    "LOOKUP_RETRIES",
    "MAX_PROBES",
    "BlobStoreBackend",
    "BlobStoreConfig",
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
        return strict_from_dict(cls, raw)


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


def _report_outcome_of(work: Callable[[], Outcome], report: Callable[[Outcome], None]) -> None:
    """The thread boundary of a worker: run ``work`` and hand its outcome to ``report``, whatever
    happened, so that no wait on the outcome hangs. An ``Exception`` becomes a ``Failed`` outcome
    and goes no further; a ``BaseException`` becomes one too and then propagates."""
    outcome: Outcome = Failed("did not run")
    try:
        outcome = work()
    except Exception as exc:  # noqa: BLE001 - thread boundary; the outcome carries the error
        outcome = Failed(f"{type(exc).__name__}: {exc}")
    except BaseException as exc:
        outcome = Failed(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report(outcome)


@contextmanager
def _drain_copies_on_error(pool: HostStagingPool, warning: str) -> Iterator[None]:
    """Around copies into or out of ``pool``'s slots: when the body raises, wait for the copies
    issued so far before the error goes on, so that a slot someone else then reuses is not still
    being written and the caller's memory is not still being read. A sync that fails itself is
    logged as ``warning``; the original error is what matters."""
    try:
        yield
    except BaseException:
        try:
            pool.sync()
        except Exception:  # noqa: BLE001 - best effort; the original error is what matters
            logger.warning(warning)
        raise


_Work = Callable[[_StoreAttempt, Sequence[_Task]], Outcome]


class _PutResult(NamedTuple):
    """What one put call achieved: the units the store holds afterwards (written now, or found
    present after it declined them because another publisher got there first), and why the call
    failed, if it did. Units in ``held`` are in the store even when ``problem`` is set."""

    held: list[_Task]
    problem: Optional[str]


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
        self._put_batch_size = config.transfer_batch_size
        """Units per put call. Each unit of a staged put holds a publish-pool slot for the call,
        so the pool's slot count bounds it too; lookups have no slot and use
        ``transfer_batch_size`` as is."""
        if self._staging is not None:
            self._put_batch_size = min(self._put_batch_size, self._staging.num_slots)
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
            reg = self._adopt_registration(address, size)
        finally:
            with self._lock:
                self._pending.discard(span)
        return reg

    def _adopt_registration(self, address: int, size: int) -> _Registration:
        """Put a span the store just registered on the table. When ``close`` ran meanwhile, the
        span is taken back from the store and refused as the pre-check would have, so that a
        closed backend leaves nothing registered."""
        with self._lock:
            if not self._closed:
                reg = _Registration(self, address, size)
                self._registrations.append(reg)
                return reg
        try:
            self._store.unregister_span(address, size)
        except BlobStoreError as exc:
            logger.warning(
                "blob store [%s]: unregistering [%#x, %#x) failed after close: %s",
                self._store.describe(),
                address,
                address + size,
                exc,
            )
        raise RuntimeError("store backend is closed")

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

    def _first_unregistered_segment(self, segments: Sequence[Segment]) -> Optional[Segment]:
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
        return self.submit_delivery(extent, self._fetch_work, self._check_destinations)

    def publish(self, extent: CacheExtent) -> Attempt:
        return self.submit_delivery(extent, self._publish_work, self._check_destinations)

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
                self._remember_lookup(key, now)
                return None
            if not entry.done:
                return None
            del self._probes[key]
        if entry.error is not None:
            # The cause's text rides in the message: the caller logs ``str(exc)`` only.
            raise RuntimeError(f"store lookup failed: {entry.error}") from entry.error
        return entry.answer

    def _remember_lookup(self, key: tuple[bytes, tuple[bytes, ...]], now: float) -> None:
        """Start the lookup for ``key`` on the lookup thread and remember it, so that a later
        ``probe`` finds its answer. Refused on a closed backend or a full table. Caller holds
        the lock."""
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

    def note_failed_delivery(self, reason: str) -> None:
        """Count and log a failed delivery."""
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
                bad = self._first_unregistered_segment(task.segments)
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
        """Worker: run one delivery, then free its in-flight slot and finish its attempt."""

        def finish(outcome: Outcome) -> None:
            self._inflight.release()
            if isinstance(outcome, Failed):
                self._fail(attempt, outcome.reason)
            else:
                attempt.finish(outcome)

        _report_outcome_of(lambda: run(attempt, tasks), finish)

    def _fail(self, attempt: _StoreAttempt, reason: str) -> None:
        self.note_failed_delivery(reason)
        attempt.finish(Failed(reason))

    def _ask_holds(self, keys: Sequence[str]) -> Sequence[bool]:
        """Ask the store for ``keys``; an answer of the wrong length is a failed lookup too."""
        present = self._store.holds(keys)
        if len(present) != len(keys):
            raise BlobStoreError(f"holds answered {len(present)} of {len(keys)} keys")
        return present

    def _with_publish_slots(self, count: int, body: Callable[[list[int]], _T]) -> _T:
        """Run ``body`` with ``count`` publish-pool slots, which go back when it returns or raises.

        Drain contract: ``body`` waits for its own copies (``staging.sync``) before it returns,
        so on the normal path the slots are quiet when released. When ``body`` raises, the copies
        are drained here instead: a copy still in flight into a slot someone else then reuses
        would corrupt their delivery, and one out of the caller's memory would break quiescence."""
        assert self._staging is not None
        slots = self._staging.acquire(count)
        try:
            with _drain_copies_on_error(
                self._staging, "staging sync failed while unwinding a failed delivery"
            ):
                return body(slots)
        finally:
            self._staging.release(slots)

    # ---- the work ----

    def _fetch_work(self, attempt: _StoreAttempt, tasks: Sequence[_Task]) -> Outcome:
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
            present = self._ask_holds([task.key for task in batch])
        except BlobStoreError as exc:
            return [], f"store lookup failed: {exc}"
        hits = [task for task, held in zip(batch, present) if held]
        got: list[_Task] = []
        if hits:
            results = self._store.get([t.key for t in hits], destinations(hits))
            got, bad = _split_get_results(hits, results)
            if bad:
                return got, f"{bad} of {len(hits)} present units could not be read"
        with self._lock:
            self.counters.fetch_hits += len(got)
            self.counters.fetch_misses += len(batch) - len(got)
        return got, None

    def _publish_work(self, attempt: _StoreAttempt, tasks: Sequence[_Task]) -> Outcome:
        """A whole publish: skip the units the store holds, then write the rest into it directly
        or through the publish pool. ``served`` collects every unit the store holds at the end."""
        try:
            absent, present = self._partition_present(tasks)
        except BlobStoreError as exc:
            # A store that cannot answer whether it holds a unit is out of reach; treating the
            # answer as "absent" would turn an outage into writes against it (and a fetch on
            # the same answer fails, so the two directions agree).
            return Failed(f"store lookup failed: {exc}")
        served = {task.name for task in present}
        if self._staging is None:
            problem = self._publish_direct(absent, served)
        else:
            problem = self._publish_staged(attempt, absent, served)
        if problem is not None:
            return Failed(problem)
        return Delivered(frozenset(served))

    def _partition_present(self, tasks: Sequence[_Task]) -> tuple[list[_Task], list[_Task]]:
        """Split ``tasks`` into the units the store lacks and the ones it already holds, asking
        ``holds`` one batch at a time; counts the present ones. Present units are merged, not
        rewritten (§6.3 requirement 2). Raises ``BlobStoreError`` when the store cannot answer."""
        absent: list[_Task] = []
        present: list[_Task] = []
        for batch in _batched(tasks, self._config.transfer_batch_size):
            held = self._ask_holds([task.key for task in batch])
            for task, is_held in zip(batch, held):
                (present if is_held else absent).append(task)
        with self._lock:
            self.counters.publish_present += len(present)
        return absent, present

    def _publish_direct(self, absent: Sequence[_Task], served: set[bytes]) -> Optional[str]:
        """Write ``absent`` straight from the caller's pages, one put batch at a time, adding to
        ``served`` what the store took. Returns the first batch's reason for failing, if any."""
        for batch in _batched(absent, self._put_batch_size):
            result = self._put_one_batch(batch, [t.segments for t in batch])
            if result.problem is not None:
                return result.problem
            served.update(task.name for task in result.held)
        return None

    def _publish_staged(
        self, attempt: _StoreAttempt, absent: Sequence[_Task], served: set[bytes]
    ) -> Optional[str]:
        """Write ``absent`` through the publish pool in rounds of the whole slot pool, gathering
        a round before writing any of it, so that an extent no larger than the pool is quiet
        before its first remote write (design §10.1); a larger extent is quiet only after its
        last round's gather. Adds to ``served`` what the store took; returns the first round's
        reason for failing, if any."""
        assert self._staging is not None
        rounds = list(_batched(absent, self._staging.num_slots))
        for index, group in enumerate(rounds):
            last = index == len(rounds) - 1
            problem = self._with_publish_slots(
                len(group),
                lambda slots, group=group, last=last: self._staged_put(
                    attempt, group, slots, last, served
                ),
            )
            if problem is not None:
                return problem
        return None

    def _staged_put(
        self,
        attempt: _StoreAttempt,
        group: Sequence[_Task],
        slots: Sequence[int],
        last: bool,
        served: set[bytes],
    ) -> Optional[str]:
        """One round of a staged publish: gather ``group`` into ``slots``, wait for the copies
        (the caller's pages are quiet after the last round), then put from the slots batch by
        batch, adding to ``served`` what the store took."""
        assert self._staging is not None
        for slot, task in zip(slots, group):
            self._staging.gather(slot, task.segments)
        self._staging.sync()
        if last:
            attempt.quiet.set()
        for batch, batch_slots in zip(
            _batched(group, self._put_batch_size), _batched(slots, self._put_batch_size)
        ):
            result = self._put_one_batch(
                batch,
                [
                    [(self._staging.slot_address(slot), task.total)]
                    for slot, task in zip(batch_slots, batch)
                ],
            )
            if result.problem is not None:
                return result.problem
            served.update(task.name for task in result.held)
        return None

    def _put_one_batch(
        self, tasks: Sequence[_Task], buffers: Sequence[Sequence[Segment]]
    ) -> _PutResult:
        """Write one batch from ``buffers``.

        A unit the store declined but holds anyway lost a race with another publisher and counts
        as held; one it declined and does not hold is simply not served. A call that misbehaves
        (exception, wrong count, a unit that failed to write, a failed lookup afterwards) is a
        failure. Units stored before such a failure are still returned and counted in
        ``publish_stored`` (they are in the store), but the caller's attempt ends ``Failed`` and
        reports no served set.
        """
        results = self._store.put([t.key for t in tasks], buffers)
        if len(results) != len(tasks):
            return _PutResult([], f"put answered {len(results)} of {len(tasks)} keys")
        stored = [task for task, status in zip(tasks, results) if status is PutStatus.STORED]
        with self._lock:
            self.counters.publish_stored += len(stored)
        failed = sum(1 for status in results if status is PutStatus.FAILED)
        if failed:
            return _PutResult(stored, f"{failed} of {len(tasks)} units could not be written")
        declined = [task for task, status in zip(tasks, results) if status is PutStatus.DECLINED]
        raced: list[_Task] = []
        if declined:
            try:
                present = self._ask_holds([task.key for task in declined])
            except BlobStoreError as exc:
                return _PutResult(stored, f"store lookup failed after a declined put: {exc}")
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
        return _PutResult(stored + raced, None)

    def _ask_holds_with_retry(self, keys: Sequence[str]) -> Sequence[bool]:
        """``_ask_holds``, asked again up to ``LOOKUP_RETRIES`` times after a ``BlobStoreError``.

        The last try's error propagates; any other exception propagates at once.
        """
        for tried in range(1, LOOKUP_RETRIES + 1):
            try:
                return self._ask_holds(keys)
            except BlobStoreError as exc:
                logger.debug(
                    "blob store [%s]: lookup try %d of %d failed, retrying: %s",
                    self._store.describe(),
                    tried,
                    LOOKUP_RETRIES + 1,
                    exc,
                )
                time.sleep(LOOKUP_RETRY_DELAY_S)
        return self._ask_holds(keys)

    def _lookup(self, entry: _Probe, units: Sequence[bytes]) -> None:
        held: set[bytes] = set()
        try:
            for batch in _batched(units, self._config.transfer_batch_size):
                present = self._ask_holds_with_retry([self._keys.key(u) for u in batch])
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


def _split_get_results(
    tasks: Sequence[_Task], results: Sequence[GetStatus]
) -> tuple[list[_Task], int]:
    """Split the answer of one get into the units fully delivered and how many went wrong.

    A unit the store no longer holds (gone since the lookup) is neither: nothing was written and
    it is simply not served. An answer of the wrong length counts every unit as wrong.
    """
    if len(results) != len(tasks):
        return [], len(tasks)
    got = [task for task, status in zip(tasks, results) if status is GetStatus.HIT]
    bad = sum(1 for status in results if status is GetStatus.FAILED)
    return got, bad

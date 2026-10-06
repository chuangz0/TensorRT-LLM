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
"""The ``landing: host`` shape of the blob backend: ``HostLandingBlobBackend`` and its ``_Landing``.

A fetch lands in the backend's own pinned host memory first (``LandsOnHost.fetch_to_host``) and
is copied into the caller's pages once the scheduler has reserved them (``Landing.place``). The
store only ever touches pinned host memory, so the caller's pools stay unregistered. Publishes go
to the inner ``BlobStoreBackend`` directly, through its publish pool (``backend.py``).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Optional, Sequence

from ...base.cache_backend import (
    Attempt,
    CacheExtent,
    Delivered,
    Failed,
    Outcome,
    SubmissionRejected,
)

# The underscore names are the inner backend's own building blocks, shared with this module only.
from .backend import (
    BlobStoreBackend,
    StoreCounters,
    UnitBytes,
    _drain_copies_on_error,
    _report_outcome_of,
    _StoreAttempt,
    _Task,
)
from .staging import HostStagingPool
from .worker_pool import DaemonWorkerPool

__all__ = ["HostLandingBlobBackend"]


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
                    self._set_outcome_locked(
                        Failed(
                            f"waited {self._backend.landing_wait_timeout_s:g} s for "
                            f"{len(self.units)} landing slots"
                        )
                    )
                    self._leave_locked()
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
                    self._set_outcome_locked(Failed("released before landing"))
                    self._leave_locked()
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
                self._set_outcome_locked(Failed("released before landing"))
                self._leave_locked()
                return
            slots = self._give_up_slots_locked()
        self._give_back(slots)

    def deliver_empty(self) -> None:
        """A landing of no units: landed at once, holding nothing."""
        with self._lock:
            self._set_outcome_locked(Delivered(frozenset()))
            self._state = self.LANDED

    # -- SlotWaiter --

    def slots_granted(self, slots: list[int]) -> None:
        with self._lock:
            if self._release_requested or self._state is not self.QUEUED:
                self._set_outcome_locked(Failed("released before landing"))
                self._leave_locked()
                give_back = slots
            else:
                self._slots = slots
                self._state = self.LANDING
                give_back = []
        if give_back:
            self._give_back(give_back)
            return
        try:
            self._backend.workers.submit(self._run_on_worker)
        except RuntimeError as exc:
            self._end(Failed(f"could not start the landing: {exc}"))

    def slots_refused(self, reason: str) -> None:
        with self._lock:
            self._set_outcome_locked(Failed(reason))
            self._leave_locked()

    # -- the work --

    def _run_on_worker(self) -> None:
        """Worker: ``holds`` then ``get`` into the slots, batch by batch, back to back."""
        _report_outcome_of(
            lambda: self._backend._fetch_into_slots(self.units, self._slots, self._served),
            self._end,
        )

    def _end(self, outcome: Outcome) -> None:
        """Record the outcome. The slots go back at once when nothing landed to place, or when
        a release was asked for while the get was in flight."""
        with self._lock:
            self._set_outcome_locked(outcome)
            self._state = self.LANDED
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

    def _set_outcome_locked(self, outcome: Outcome) -> None:
        """The first outcome stands; ``poll`` answers it from now on."""
        if self._outcome is None:
            self._outcome = outcome

    def _leave_locked(self) -> None:
        """``RELEASED``: the backend forgets this landing."""
        self._state = self.RELEASED
        self._backend._forget(self)

    def _give_up_slots_locked(self) -> list[int]:
        slots, self._slots = self._slots, []
        self._served = {}
        self._leave_locked()
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
            no bound. The assembly passes the coordinator's ``landing_wait_timeout_s``
            (``KVTransferConfig.landing_wait_timeout_s``), so queue waits and page waits share
            one bound.
    """

    def __init__(
        self,
        inner: BlobStoreBackend,
        landing_pool: HostStagingPool,
        unit_bytes: UnitBytes,
        *,
        landing_wait_timeout_s: float | None,
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

    def _fetch_into_slots(
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
        with _drain_copies_on_error(pool, "landing sync failed while unwinding a failed placement"):
            for slot, task in pairs:
                pool.scatter(slot, task.segments)
            pool.sync()
        return Delivered(frozenset(task.name for _, task in pairs))

    def _count_failed(self, reason: str) -> None:
        self._inner.note_failed_delivery(f"landing: {reason}")

    def _forget(self, landing: _Landing) -> None:
        with self._lock:
            self._landings.discard(landing)

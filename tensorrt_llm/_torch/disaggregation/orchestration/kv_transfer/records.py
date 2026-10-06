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
"""One request, one direction, one record (design §4.1).

The record table is the single source of truth for a request's transfer state. Only
``KVTransferCoordinator`` writes it, and only on the engine thread. Request state on the request
object itself is a projection of this table, written through effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Literal

from ...base.cache_backend import Attempt, CacheExtent, Cancelled, Delivered, Failed, Outcome, Route

if TYPE_CHECKING:
    from ...remote_cache import FetchPlan
    from .interfaces import Landing

__all__ = [
    "AttemptRecord",
    "Direction",
    "RecordKey",
    "RecordState",
    "TransferRecord",
    "is_failure",
]

Direction = Literal["fetch", "publish"]
RecordKey = tuple[int, str]
"""``(request_id, direction)``: the record table's key, and the id carried in the allgather."""


def is_failure(outcome: Outcome | None) -> bool:
    """``Failed``, or ``Cancelled`` by the peer. A local cancel is an ordinary end that served
    nothing; ``None`` is no outcome yet."""
    if isinstance(outcome, Failed):
        return True
    return isinstance(outcome, Cancelled) and outcome.by_peer


class RecordState(Enum):
    """The observable states of a record. Release is an event, not a state: a released record
    leaves the table.

    ``STAGING`` and ``STAGED`` belong to a fetch from a ``LandsOnHost`` source only: the units
    are on their way to the backend's host memory, then landed there and waiting for the
    scheduler to reserve pages. Neither holds a page; the request stays schedulable. The copy
    into the pages that follows is an ordinary ``IN_FLIGHT``.
    """

    PLANNED = "PLANNED"
    STAGING = "STAGING"
    STAGED = "STAGED"
    IN_FLIGHT = "IN_FLIGHT"
    LANDED = "LANDED"
    FAILED = "FAILED"


@dataclass
class AttemptRecord:
    """One delivery within a record.

    Attributes:
        attempt: The backend's handle.
        outcome: What ``poll`` last returned; ``None`` while in flight.
        try_index: Which retry this attempt belongs to (fetch only; publish is always 0).
        source: ``FetchSource.name`` for a fetch, the publisher's assembly name for a publish.
            The coordinator maps it back to the backend for ``quiesce``.
        route: Worker backend only: opened for this attempt, closed at ``LANDED`` or after
            ``quiesce``.
    """

    attempt: Attempt
    outcome: Outcome | None = None
    try_index: int = 0
    source: str | None = None
    route: Route | None = None


@dataclass
class TransferRecord:
    """The transfer state of one request in one direction.

    Attributes:
        request_id: ``RequestView.py_request_id``.
        direction: ``"fetch"`` or ``"publish"``.
        state: See :class:`RecordState` and the state diagram in design §4.1.
        plan: Fetch only; written by the plan phase of ``advance``.
        extent: Known from launch (fetch) or publish time, since the units depend on the
            scheduler's allocation.
        attempts: Every attempt so far; several per publish when sent in pieces, accumulating
            across tries on retry.
        retries_left: Fetch only; how many more times the plan may be dropped and the request
            planned again. Every failed verdict spends one, whatever its cause: a failed or short
            delivery, a launch given up, or a wait that ran out (``landing_wait_timeout_s``,
            ``unlaunched_timeout_s``). At zero the next failure releases the record and the
            request computes locally.
        deadline: Monotonic time after which the record expires; ``None`` means no timeout. Set
            when a fetch is launched or starts landing on the host (cleared again once the
            landing is complete; the placement sets its own), when a publish's first submission
            is accepted, and at the request's end for a record kept to vote without one.
        expired: The deadline passed. The record no longer waits for the ranks' agreement: it is
            settled on this rank as soon as its own attempts are over.
        quiesce_refused: The backend could not vouch for the pages at the release point. The
            engine is fatal; the record stays so the pages are never handed out again.
        retry_hint: Fetch only; the block boundary the next try may aim for at most: the ranks'
            agreed MIN of the merged B (``remote_cache.merge``) of the try that came up short.
        rejected: Publish only; some submission of this record raised ``SubmissionRejected``.
            The units it carried were never offered, so the publish as a whole has failed even if
            the other pieces land.
        consecutive_launch_failures: Fetch only; launches in a row that never started, whether
            the backend refused the submission or the route could not be opened, since the last
            launch that went through. At ``MAX_CONSECUTIVE_LAUNCH_FAILURES`` the record gives up.
        launch_gave_up: Fetch only; this rank will not launch the current plan. The record votes
            ``FAILED`` every round until the ranks agree, and the scheduler is answered ``DEFER``
            meanwhile. Cleared when the plan is dropped for a retry.
        peer_launched_at: Fetch only; when this rank, voting ``UNLAUNCHED`` for the record (no
            attempt of its own in the pages: pages not reserved, launch not started, or landed
            on the host and waiting for pages), first saw another rank's vote that was not
            ``UNLAUNCHED``. Bounds how long this rank may hold the others up
            (``unlaunched_timeout_s``). Cleared when this rank starts its own attempt, when its
            landing is accepted or marked complete, and when the plan is dropped; it starts
            again if the peers move on while this rank still waits.
        landing: Host-first fetch only; the current try's ``Landing``, from ``fetch_to_host``
            until the coordinator releases it. Never among ``attempts``.
        committed_names: Fetch only; units the plan asked for that ``fetch_extent`` left out of
            the launched extent because the local cache had committed them meanwhile. They count
            as served when the delivery is merged.
        waiting_since: Fetch only; when this rank started waiting for something the scheduler
            or the backend has yet to give: the pages (from the plan's decision for a
            device-direct fetch, from ``STAGED`` for a host-first one) or the landing memory
            (``fetch_to_host`` refused). Bounded by ``landing_wait_timeout_s``; cleared when
            the wait ends.
    """

    request_id: int
    direction: Direction
    state: RecordState
    plan: FetchPlan | None = None
    extent: CacheExtent | None = None
    attempts: list[AttemptRecord] = field(default_factory=list)
    retries_left: int = 1
    deadline: float | None = None
    expired: bool = False
    quiesce_refused: bool = False
    retry_hint: int | None = None
    rejected: bool = False
    consecutive_launch_failures: int = 0
    launch_gave_up: bool = False
    peer_launched_at: float | None = None
    landing: Landing | None = None
    committed_names: frozenset[bytes] = frozenset()
    waiting_since: float | None = None

    @property
    def key(self) -> RecordKey:
        return (self.request_id, self.direction)

    @property
    def try_index(self) -> int:
        """The current try: the highest ``try_index`` among attempts, or 0 before any."""
        return max((a.try_index for a in self.attempts), default=0)

    def current_try_attempts(self) -> list[AttemptRecord]:
        """Attempts of the current try only; earlier tries were quiesced when they failed."""
        current = self.try_index
        return [a for a in self.attempts if a.try_index == current]

    def is_terminal(self) -> bool:
        """Every attempt of the current try has an outcome. False with no attempts at all."""
        current = self.current_try_attempts()
        return bool(current) and all(a.outcome is not None for a in current)

    def any_failed(self) -> bool:
        """Some attempt of the current try ended ``Failed`` or ``Cancelled(by_peer=True)``."""
        return any(is_failure(a.outcome) for a in self.current_try_attempts())

    def merged_served(self) -> frozenset[bytes]:
        """Union of ``Delivered.served`` over the current try; a local cancel contributes
        nothing."""
        served: set[bytes] = set()
        for a in self.current_try_attempts():
            if isinstance(a.outcome, Delivered):
                served |= a.outcome.served
        return frozenset(served)

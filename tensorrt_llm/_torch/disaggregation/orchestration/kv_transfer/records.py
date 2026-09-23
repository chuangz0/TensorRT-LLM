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

__all__ = ["AttemptRecord", "Direction", "RecordKey", "RecordState", "TransferRecord"]

Direction = Literal["fetch", "publish"]
RecordKey = tuple[int, str]
"""``(request_id, direction)``: the record table's key, and the id carried in the allgather."""


def _is_failure(outcome: Outcome | None) -> bool:
    if isinstance(outcome, Failed):
        return True
    return isinstance(outcome, Cancelled) and outcome.by_peer


class RecordState(Enum):
    PLANNED = "PLANNED"
    IN_FLIGHT = "IN_FLIGHT"
    LANDED = "LANDED"
    FAILED = "FAILED"
    RELEASED = "RELEASED"


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
        retries_left: Fetch only; a retry is allowed once when ``served`` came up short.
        deadline: Monotonic time after which the record is expired; ``None`` means no timeout.
        abandoned: Expired or cancelled. The transfer itself cannot be stopped; the record is
            still polled to its outcome and then released. Fetch: suppresses further expiry;
            publish: informational only, the deadline still fires.
        retry_hint: Fetch only; the block boundary the next try should aim for, from
            ``planner.retry_hint_from`` after a short ``served``.
        rejected: Publish only; some submission of this record raised ``SubmissionRejected``.
            The units it carried were never offered, so the publish as a whole has failed even if
            the other pieces land.
        consecutive_rejections: Fetch only; ``SubmissionRejected`` answers in a row since the
            last launch that went through. Past ``MAX_CONSECUTIVE_REJECTIONS`` the source counts
            as unavailable for this record.
    """

    request_id: int
    direction: Direction
    state: RecordState
    plan: FetchPlan | None = None
    extent: CacheExtent | None = None
    attempts: list[AttemptRecord] = field(default_factory=list)
    retries_left: int = 1
    deadline: float | None = None
    abandoned: bool = False
    retry_hint: int | None = None
    rejected: bool = False
    consecutive_rejections: int = 0

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

    def all_delivered(self) -> bool:
        """Terminal, and no attempt of the current try counts as a failure.

        A local cancel (``Cancelled(by_peer=False)``) is an ordinary end that delivered nothing,
        so it counts as ``Delivered`` with an empty ``served``.
        """
        return self.is_terminal() and not self.any_failed()

    def any_failed(self) -> bool:
        """Some attempt of the current try ended ``Failed`` or ``Cancelled(by_peer=True)``."""
        return any(_is_failure(a.outcome) for a in self.current_try_attempts())

    def merged_served(self) -> frozenset[bytes]:
        """Union of ``Delivered.served`` over the current try; a local cancel contributes
        nothing."""
        served: set[bytes] = set()
        for a in self.current_try_attempts():
            if isinstance(a.outcome, Delivered):
                served |= a.outcome.served
        return frozenset(served)

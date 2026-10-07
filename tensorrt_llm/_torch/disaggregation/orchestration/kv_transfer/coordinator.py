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
"""The one engine-side component that manages KV transfer (design §7.1).

It owns every ``TransferRecord``, is the only writer of a request's transfer state, and translates
the executor loop's timing into calls on the backend contract and on the engine's effects. It does
not decide capacity (scheduler), does not decide how far to fetch (planner), does not move bytes
(backends) and never sees an address (contract). The one distinction it draws between backends is
the optional ``LandsOnHost`` capability: a fetch from such a source lands in the backend's own
host memory before any page is reserved (the "host-first" section below); everything else reaches
a backend through the assembly table and the contract alone.

Everything runs on the engine thread. There are no threads here.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Callable, Mapping, Sequence, TypeVar

from ...base.cache_backend import (
    Attempt,
    CacheExtent,
    Delivered,
    Fetches,
    Publishes,
    Route,
    SubmissionRejected,
)
from ...base.capabilities import CarriesAux, LandsOnHost, PlacesPieces
from ...base.views import RequestView, ResourceView
from ...remote_cache import DEFER, FetchPlan, FetchSource, Planner, served_token_end, unit_names
from .consensus import (
    EncodedPlanAnswer,
    PlanAnswer,
    PlanAnswers,
    RoundMessage,
    Verdict,
    Vote,
    VoteKind,
    encode_plan_answer,
    reduce_plan_answers,
    reduce_votes,
    votes_by_key,
)
from .engine_protocols import Collective, EngineQueue, KVTransferEffects, PlanAuthority
from .records import AttemptRecord, RecordKey, RecordState, TransferRecord, is_failure

if TYPE_CHECKING:
    # ``backend``: Chunk/CacheKind only; ``cache_backend``: the contract.
    from ...base.backend import Chunk

__all__ = [
    "DEFER",
    "MAX_CONSECUTIVE_LAUNCH_FAILURES",
    "KVTransferCoordinator",
]
# ``DEFER`` is re-exported: it is the coordinator's ``fetch_answer`` the engine compares against,
# and the engine side takes everything it needs from this package.

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_LAUNCH_FAILURES = 3
"""Launches in a row that never started (refused submission or failed route) after which a fetch
record gives up on its plan and lets the ranks agree on a failure."""

T = TypeVar("T")


class KVTransferCoordinator:
    """Holds the record table and drives it from the executor loop.

    Args:
        sources: Fetch backends in priority order; the assembly table.
        publishers: Publish backends.
        planner: Source policy.
        reader: This rank's resource view, for extents and chunks.
        effects: The engine's side effects.
        queue: Work backends post for the engine thread.
        dist: The collective over the ranks that plan together; one ``allgather`` per ``advance``.
        fetch_timeout_s: Deadline for a fetch, from launch (or from landing on the host), and
            for a fetch record kept at its request's end to vote. Past it the request fails and
            this rank stops waiting for the others. ``None`` disables.
        publish_timeout_s: Deadline for a publish, from its first accepted submission, and for
            a publish record kept at its request's end to vote. Past it this rank warns and
            stops waiting for the others; the record still ends on this rank's own outcome.
            ``None`` disables.
        unlaunched_timeout_s: Longest this rank may stay unlaunched on a fetch another rank has
            already launched before it votes the fetch failed. ``None`` disables. Never starts
            counting on a single rank. Required: the default lives in ``KVTransferConfig``.
        landing_wait_timeout_s: Longest this rank waits for the scheduler's pages (from the
            plan's decision, or from landing on the host) or, host-first, for the backend's
            landing memory, before it votes the fetch failed. Counts on a single rank too, so a
            fetch the scheduler can never find pages for is given up and the request computes
            locally instead of stalling. ``None`` disables. Required, as above.
        plan_authority: Who decides plan answers on this rank. ``ALL_RANKS``: planned here and
            reduced in the collective. ``OWNER``: planned here alone, not carried in the
            payload, handed out with ``export_plan_answers``. ``FOLLOWER``: never planned or
            probed here; taken with ``adopt_plan_answers``. Votes and expiries travel through
            the collective in every mode.
        queue_budget: How many posted callables one ``advance`` runs.

    The engine must call ``notify_request_finished`` for every request it ever passed in, so that
    fetch records reach their release point and per-request state is dropped. Landings still
    held when the engine shuts down (``hooks.close`` frees parked and held requests only) are
    closed by the backend's own ``close``.

    A record of a finished request keeps taking part in the round until the ranks agree on it,
    so that every rank terminates the request in the same round; a record past its deadline is
    rank-local instead and ends here on its own outcome as soon as its own attempts are over. A
    publish past its deadline whose attempt is still live stays ``IN_FLIGHT`` (it is never
    quiesced under a live attempt), so ``inflight_request_ids`` stays non-empty and the
    scheduler's deadlock detector is off for that time: the intended trade for never blocking the
    engine thread.
    """

    def __init__(
        self,
        sources: Sequence[FetchSource],
        publishers: Sequence[Publishes],
        planner: Planner,
        reader: ResourceView,
        effects: KVTransferEffects,
        queue: EngineQueue,
        dist: Collective,
        *,
        unlaunched_timeout_s: float | None,
        landing_wait_timeout_s: float | None,
        fetch_timeout_s: float | None = None,
        publish_timeout_s: float | None = None,
        plan_authority: PlanAuthority = PlanAuthority.ALL_RANKS,
        queue_budget: int = 64,
    ) -> None:
        self._sources: dict[str, FetchSource] = {source.name: source for source in sources}
        self._publishers: dict[str, Publishes] = {
            f"publish:{index}": publisher for index, publisher in enumerate(publishers)
        }
        self._backends: dict[str, Fetches | LandsOnHost | Publishes] = {
            **{name: source.backend for name, source in self._sources.items()},
            **self._publishers,
        }
        self._host_first_sources: frozenset[str] = frozenset(
            name
            for name, source in self._sources.items()
            if isinstance(source.backend, LandsOnHost)
        )
        """Sources whose fetches land on the host first; decided once, read on every plan."""
        self._planner = planner
        self._reader = reader
        self._effects = effects
        self._queue = queue
        self._dist = dist
        self._fetch_timeout_s = fetch_timeout_s
        self._publish_timeout_s = publish_timeout_s
        self._unlaunched_timeout_s = unlaunched_timeout_s
        self._landing_wait_timeout_s = landing_wait_timeout_s
        self._plan_authority = plan_authority
        self._queue_budget = queue_budget

        self._records: dict[RecordKey, TransferRecord] = {}
        self._requests: dict[int, RequestView] = {}
        self._plan_answers: dict[int, FetchPlan | None] = {}
        """Decided answers ``fetch_answer`` reads. A request with no entry is undecided (DEFER)."""
        self._answers_to_export: dict[int, FetchPlan | None] = {}
        """OWNER only: the answers the last ``advance`` decided, for ``export_plan_answers``."""
        self._pending_answers: dict[int, EncodedPlanAnswer] = {}
        """FOLLOWER only: adopted answers whose request is not a candidate here yet."""
        self._probe_answers: dict[int, dict[str, frozenset[bytes] | None]] = {}
        """Per request, each store source's answer to its probe once it is in; what the planner
        decides on. Kept until the request is forgotten, so a retry sees the same answer."""
        self._finished: set[int] = set()
        self._held: set[int] = set()
        """Finished requests the engine was told to hold; each owes a ``terminate_request``."""
        self._finished_by_gate: set[int] = set()
        """Finished requests the engine's release gate reported; the others were failed here and
        the gate has yet to ask about them."""
        self._terminated_before_gate: set[int] = set()
        """Held requests this coordinator terminated before the engine's release gate asked about
        them; the gate is answered "not yours" once, so the engine does not terminate twice."""
        self._any_rank_pending = False
        """Whether some rank of the collective reported pending work in the last round."""
        self._any_rank_drained = False
        """Whether some rank of the collective reported itself past its shutdown drain deadline
        in the last round."""

    # ---- loop entry points (``advance`` is the one collective: once per rank per round) ----

    def advance(
        self, candidates: Sequence[RequestView], now: float, *, drained: bool = False
    ) -> int:
        """Head of the loop: poll outcomes into votes, plan candidates, agree across ranks, apply.

        ``drained`` is whether this rank has waited long enough for its own pending work at
        shutdown; it goes in this rank's message, and a drained rank reports no pending work.

        Returns how many candidates are still undecided afterwards (``fetch_answer`` answers
        ``DEFER``): each is waiting on a store lookup, or on the ranks' agreement.
        """
        votes, expired = self._poll_and_vote(now)
        answers = self._plan(candidates, now)
        pending = not drained and self.has_pending_work()
        verdicts, expired, answers = self._agree(votes, expired, answers, pending, drained, now)
        self._apply(verdicts, expired, answers, now)
        return sum(1 for candidate in candidates if self.fetch_answer(candidate) is DEFER)

    def launch_reserved_fetches(
        self, reserved: Sequence[RequestView], now: float | None = None
    ) -> None:
        """After scheduling: start the fetch of every request the scheduler allocated for. A
        record already landed on the host (``LANDED``) places its landing into the pages instead
        of fetching; either way the request is parked until the delivery is agreed on."""
        now = time.monotonic() if now is None else now
        ready = []
        for request in reserved:
            record = self._records.get((request.py_request_id, "fetch"))
            if record is not None and self._wants_pages_now(record):
                ready.append((request, record))
        if not ready:
            return
        self._effects.prepare_fetch_resources([request for request, _ in ready])
        launched = [
            request for request, record in ready if self._launch_or_place(request, record, now)
        ]
        if launched:
            self._effects.park_for_fetch(launched)

    def publish_committed_blocks(
        self, requests: Sequence[RequestView], now: float | None = None
    ) -> None:
        """After a context step, before the response pass: offer the blocks each request committed.

        A request that ended this step is reported separately through ``notify_request_finished``
        (the engine's release gate); a publish still in flight then holds it.
        """
        now = time.monotonic() if now is None else now
        if self._publishers:
            for request in requests:
                self._publish_one(request, now)

    # ---- plan authority: owner hands out, follower takes ----

    @property
    def plan_authority(self) -> PlanAuthority:
        return self._plan_authority

    def export_plan_answers(self) -> PlanAnswers:
        """OWNER, after ``advance``: the answers it decided this round, on the wire, for the
        followers' ``adopt_plan_answers``. Empty in any other mode."""
        return [
            (request_id, encode_plan_answer(answer))
            for request_id, answer in sorted(self._answers_to_export.items())
        ]

    def adopt_plan_answers(
        self, candidates: Sequence[RequestView], answers: PlanAnswers, now: float | None = None
    ) -> int:
        """FOLLOWER: take the owner's answers. ``None`` decides a request local; ``(token_end,
        source)`` becomes this rank's own plan over its own layer groups. An answer for a request
        not yet among ``candidates`` (the undecided candidates here) is held until its request
        appears as a candidate, or is dropped when the request ends: the owner decides a request
        once and does not export it again. Returns how many of ``candidates`` got no answer and
        stay deferred."""
        now = time.monotonic() if now is None else now
        by_request_id = {candidate.py_request_id: candidate for candidate in candidates}
        self._pending_answers.update(answers)
        for request_id, encoded in list(self._pending_answers.items()):
            candidate = by_request_id.get(request_id)
            if candidate is None:
                continue
            del self._pending_answers[request_id]
            self._requests[request_id] = candidate
            if encoded is None:
                self._apply_answer(request_id, None, now)
            else:
                token_end, source = encoded
                self._apply_answer(
                    request_id, self._planner.plan_from_answer(candidate, token_end, source), now
                )
        return sum(1 for candidate in candidates if self.fetch_answer(candidate) is DEFER)

    # ---- scheduler hook (read-only, non-blocking) ----

    def fetch_answer(self, request: RequestView) -> PlanAnswer:
        """``FetchPlan`` to fetch, ``None`` to compute locally, ``DEFER`` to skip this round.

        A request the coordinator has not decided yet answers ``DEFER``; the engine passes such
        requests as ``candidates`` to the next ``advance``. So does a request whose rank gave up
        launching its plan: the ranks have yet to agree on what comes next, and meanwhile the
        scheduler must neither reserve pages for it nor plan it locally. A host-first fetch
        answers ``DEFER`` while its units are on their way to the backend (no pages are wanted
        yet) and its plan once they have landed (``LANDED``: reserve pages now). A request with a
        fetch record that holds pages, or was delivered, has nothing more to plan and answers
        ``None``.
        """
        request_id = request.py_request_id
        record = self._records.get((request_id, "fetch"))
        if record is not None:
            if record.state in (RecordState.PLANNED, RecordState.LANDED):
                return record.plan if self._wants_pages_now(record) else DEFER
            if record.state is RecordState.LANDING:
                return DEFER
            return None
        if request_id in self._plan_answers:
            return self._plan_answers[request_id]
        return DEFER

    # ---- control ----

    def notify_request_finished(self, request: RequestView, now: float | None = None) -> bool:
        """The engine's release gate: the request ended. Returns whether the engine may terminate
        it now: ``False`` while this coordinator holds it (it terminates the request itself once
        every record of it is gone), and ``False`` once for a request this coordinator failed and
        already terminated before the gate asked."""
        request_id = request.py_request_id
        if request_id in self._terminated_before_gate:
            self._terminated_before_gate.discard(request_id)
            return False
        self._finished_by_gate.add(request_id)
        self._finish_request(request, time.monotonic() if now is None else now)
        return request_id not in self._held

    def _finish_request(self, request: RequestView, now: float) -> None:
        """What a request's end does to its records; the request is held while any remains."""
        request_id = request.py_request_id
        if request_id in self._finished:
            return
        self._finished.add(request_id)
        self._requests[request_id] = request
        fetch = self._records.get((request_id, "fetch"))
        if fetch is not None:
            self._end_fetch_of_finished_request(fetch, now)
        publish = self._records.get((request_id, "publish"))
        if publish is not None:
            self._end_publish_of_finished_request(publish, now)
        self._hold_or_finish(request)

    def _hold_or_finish(self, request: RequestView) -> None:
        """A finished request is held while any record of it remains; with none left it is
        terminated now."""
        request_id = request.py_request_id
        if self._has_records_of(request_id):
            self._held.add(request_id)
            self._effects.hold_for_transfer([request])
        else:
            self._terminate_if_no_records(request_id)

    def _has_records_of(self, request_id: int) -> bool:
        return (request_id, "fetch") in self._records or (request_id, "publish") in self._records

    def _end_fetch_of_finished_request(self, record: TransferRecord, now: float) -> None:
        """A delivery into the pages (``IN_FLIGHT``) is abandoned and released when its outcome
        arrives, never quiesced here: that would block the engine thread on a transfer. One that
        ended (``DELIVERED``, fatally ``FAILED``) reaches its release point now. Anything else
        names no page: its landing, if any, goes at once (``Landing.close`` allows it before the
        outcome); the record itself goes too if it is rank-local already (expired), else it
        stays to vote until the ranks agree, so that every rank terminates the request in the
        same round."""
        if record.state is RecordState.IN_FLIGHT:
            return
        if record.state in (RecordState.DELIVERED, RecordState.FAILED):
            if self._quiesce(record):
                self._close_routes(record)
                self._release(record)
            return
        self._release_landing(record)
        if record.expired:
            self._release(record)
        else:
            self._bound_wait_for_verdict(record, now)

    def _end_publish_of_finished_request(self, record: TransferRecord, now: float) -> None:
        """A publish in flight ends on its own outcome. One nothing was ever submitted for is
        released now: no rank submitted either, since every rank offers the same pieces. One
        some publisher refused stays to vote FAILED until the ranks agree: a peer's may be in
        flight, and the store is missing this rank's part. A ``DELIVERED`` or ``FAILED`` record
        still here did not pass its release point (the backend refused to vouch for the pages)
        and stays."""
        if record.state is RecordState.PLANNED and not record.attempts and not record.rejected:
            self._release(record)
            return
        self._bound_wait_for_verdict(record, now)

    def _bound_wait_for_verdict(self, record: TransferRecord, now: float) -> None:
        """A record kept at its request's end waits for the ranks under a deadline, like any
        other, so a peer that stopped voting cannot hold the request forever."""
        if record.deadline is None:
            record.deadline = self._deadline_for(record, now)

    def has_backend_work(self) -> bool:
        """Some backend is working for this coordinator: a delivery into pages, or a landing."""
        return any(self._backend_busy_on(record) for record in self._records.values())

    def has_pending_work(self) -> bool:
        """Something here will move without a new request: a backend is working, a fetch waits
        for the scheduler's pages, or a record of a finished request waits for the ranks'
        agreement. The engine must not sleep on its request queue, and should yield briefly on
        an idle pass, while this is true."""
        return any(
            self._backend_busy_on(record)
            or self._idle_with_plan(record)
            or record.request_id in self._finished
            for record in self._records.values()
        )

    @property
    def any_rank_pending(self) -> bool:
        """Whether some rank of the collective reported pending work in the last round: the
        OR of every rank's message, so identical on every rank. ``False`` before the first
        round."""
        return self._any_rank_pending

    @property
    def any_rank_drained(self) -> bool:
        """Whether some rank of the collective reported itself past its shutdown drain deadline
        in the last round; the OR of every rank's message, so identical on every rank."""
        return self._any_rank_drained

    def inflight_request_ids(self) -> frozenset[int]:
        """Requests whose pages a backend may still touch: a record ``IN_FLIGHT``, or one whose
        backend refused to vouch for the pages at the release point."""
        return frozenset(
            record.request_id
            for record in self._records.values()
            if record.state is RecordState.IN_FLIGHT or record.quiesce_refused
        )

    # ---- what the engine's release gate and cancel path read (design §4.3) ----

    def parked_request_ids(self) -> frozenset[int]:
        """Running requests whose fetch is in flight: out of the scheduler's reach until it lands
        or is given back. A finished request with a fetch in flight is held, not parked."""
        return frozenset(
            record.request_id
            for record in self._records.values()
            if record.direction == "fetch"
            and record.state is RecordState.IN_FLIGHT
            and record.request_id not in self._held
        )

    def held_request_ids(self) -> frozenset[int]:
        """Finished requests held while a transfer still touches their pages or a record of
        theirs still waits for the ranks' agreement; each is terminated by this coordinator
        through ``terminate_request`` once every record of it is gone."""
        return frozenset(self._held)

    def owned_requests(self) -> list[RequestView]:
        """Every request this coordinator owns right now: parked or held."""
        owned = self.parked_request_ids() | self.held_request_ids()
        return [
            self._requests[request_id]
            for request_id in sorted(owned)
            if request_id in self._requests
        ]

    def status_dump(self) -> dict:
        """This coordinator's state as JSON-serializable values, for the status dump the hooks
        write at shutdown: the plan authority, one entry per record, the plan answers still
        remembered (``decided_plans``) and the finished requests not yet terminated
        (``finished_pending``) and what the ranks reported in the last round (``any_rank_pending``,
        ``any_rank_drained``). A record entry carries its state and clocks plus the retry
        bookkeeping (``retries_left``, ``consecutive_launch_failures``, ``retry_cap``) and how
        many of the plan's units the local cache had committed by launch (``committed_names``;
        the names themselves are hashes). Readers take entries by key, so an entry may gain
        keys."""
        return {
            "plan_authority": self._plan_authority.value,
            "any_rank_pending": self._any_rank_pending,
            "any_rank_drained": self._any_rank_drained,
            "records": [
                {
                    "request_id": record.request_id,
                    "direction": record.direction,
                    "state": record.state.value,
                    "try_index": record.try_index,
                    "attempts": len(record.attempts),
                    "outcomes": [
                        type(attempt.outcome).__name__
                        for attempt in record.attempts
                        if attempt.outcome
                    ],
                    "deadline": record.deadline,
                    "expired": record.expired,
                    "token_end": record.plan.token_end if record.plan else None,
                    "gave_up_launching": record.gave_up_launching,
                    "peer_launch_seen_at": record.peer_launch_seen_at,
                    "has_landing": record.landing is not None,
                    "resource_wait_since": record.resource_wait_since,
                    "retries_left": record.retries_left,
                    "consecutive_launch_failures": record.consecutive_launch_failures,
                    "retry_cap": record.retry_cap,
                    "committed_names": len(record.committed_names),
                }
                for record in self._records.values()
            ],
            "decided_plans": len(self._plan_answers),
            "finished_pending": sorted(self._finished),
        }

    # ---- advance, phase 1: poll and vote ----

    def _poll_and_vote(self, now: float) -> tuple[dict[RecordKey, Vote], list[RecordKey]]:
        """Poll every attempt in flight, then cast one vote per record that has a say this round.
        An expired record casts none: it is rank-local and ends here as soon as its own
        attempts are over. Every voting record past its deadline is reported expired, whatever
        its vote, so a peer that stopped voting cannot hold it forever."""
        self._queue.drain(self._queue_budget)
        votes: dict[RecordKey, Vote] = {}
        expired: list[RecordKey] = []
        ended_locally: list[tuple[TransferRecord, Vote]] = []
        for key, record in self._records.items():
            vote = self._vote(record, now)
            if vote is None:
                continue
            if record.expired:
                if vote.kind is not VoteKind.INFLIGHT:
                    ended_locally.append((record, vote))
                continue
            if self._deadline_passed(record, now):
                expired.append(key)
            votes[key] = vote
        self._end_expired_records(ended_locally)
        return votes, expired

    def _end_expired_records(self, ended: Sequence[tuple[TransferRecord, Vote]]) -> None:
        """After the vote: end every expired record whose attempts are over, on its own vote."""
        for record, vote in ended:
            self._end_expired_locally(record, vote)

    def _vote(self, record: TransferRecord, now: float) -> Vote | None:
        """This rank's vote on one record, or ``None`` for a record with nothing to say: a
        publish not submitted yet, a fetch delivered into the pages and waiting for its request's
        end, or a record whose pages the backend refused to vouch for (nothing is agreed on a
        poisoned page; the engine is fatal)."""
        if record.quiesce_refused:
            return None
        if record.state is RecordState.IN_FLIGHT:
            self._poll(record)
            return self._inflight_vote(record)
        if record.direction == "publish" and record.rejected:
            # Nothing of this rank's part reached the store: the publish has failed for every
            # rank, whether its request still runs or has ended.
            return Vote(VoteKind.FAILED)
        if record.request_id in self._finished:
            return self._finished_vote(record)
        if record.state is RecordState.LANDING:
            return self._landing_vote(record)
        if self._idle_with_plan(record):
            return self._unlaunched_vote(record, now)
        return None

    @staticmethod
    def _backend_busy_on(record: TransferRecord) -> bool:
        """A backend is working for this record: attempts in the pages, or a landing on its way.
        A finished request's landing is given back at once, so its LANDING record runs nothing."""
        if record.state is RecordState.LANDING:
            return record.landing is not None
        return record.state is RecordState.IN_FLIGHT

    @staticmethod
    def _poll(record: TransferRecord) -> None:
        for attempt in record.current_try_attempts():
            if attempt.outcome is None:
                attempt.outcome = attempt.attempt.poll()

    def _inflight_vote(self, record: TransferRecord) -> Vote:
        """INFLIGHT while any attempt of this rank is still running, so that a verdict never
        quiesces under a live attempt; once every attempt has its outcome, FAILED or TERMINAL."""
        if not record.has_all_outcomes():
            return Vote(VoteKind.INFLIGHT)
        if record.direction == "publish":
            # A pipelined publish has failed if any piece did or was refused; it lands once the
            # last piece has been offered, or once its request ended: no more pieces come then.
            if record.rejected or record.any_failed():
                return Vote(VoteKind.FAILED)
            if record.extent.is_last or record.request_id in self._finished:
                return Vote(VoteKind.TERMINAL)
            return Vote(VoteKind.INFLIGHT)
        if record.any_failed():
            return Vote(VoteKind.FAILED)
        # The retry aims no higher than what arrived: the probe answer is cached on the record,
        # so a retry above B would ask again for the very units that just came up short. A unit
        # the local cache committed while the fetch was on its way was never asked for, and
        # counts as arrived.
        reached_end = served_token_end(record.plan, record.served_names() | record.committed_names)
        return Vote(VoteKind.TERMINAL, reached_end)

    @staticmethod
    def _finished_vote(record: TransferRecord) -> Vote:
        """A record of a finished request with nothing running in the pages here: TERMINAL, so
        that it holds no rank up and fails none. A fetch votes its plan's target, so a peer still
        delivering lands on its own vote; without a plan (dropped after a failed try) no peer is
        delivering, and B does not matter."""
        if record.direction == "fetch" and record.plan is not None:
            return Vote(VoteKind.TERMINAL, record.plan.token_end)
        return Vote(VoteKind.TERMINAL)

    @staticmethod
    def _landing_vote(record: TransferRecord) -> Vote:
        """The ``LANDING`` counterpart of ``_inflight_vote``: the landing's outcome is the vote.
        Every rank lands its own layer groups, so the ranks take MIN(B) as for a delivery."""
        outcome = record.landing.poll()
        if outcome is None:
            return Vote(VoteKind.INFLIGHT)
        if is_failure(outcome):
            return Vote(VoteKind.FAILED)
        served = outcome.served if isinstance(outcome, Delivered) else frozenset()
        reached_end = served_token_end(record.plan, served)
        return Vote(VoteKind.TERMINAL, reached_end)

    def _unlaunched_vote(self, record: TransferRecord, now: float) -> Vote:
        """A planned record without an attempt here: UNLAUNCHED holds the others until this rank
        gets what it waits for (the scheduler's pages; for a host-first fetch, the landing memory
        first), FAILED once it gave up launching or waited too long by either clock: the peers'
        (``unlaunched_timeout_s``) or its own (``landing_wait_timeout_s``)."""
        if (
            record.gave_up_launching
            or self._unlaunched_too_long(record, now)
            or self._waited_too_long(record, now)
        ):
            return Vote(VoteKind.FAILED)
        return Vote(VoteKind.UNLAUNCHED)

    def _unlaunched_too_long(self, record: TransferRecord, now: float) -> bool:
        return (
            self._unlaunched_timeout_s is not None
            and record.peer_launch_seen_at is not None
            and now - record.peer_launch_seen_at >= self._unlaunched_timeout_s
        )

    def _waited_too_long(self, record: TransferRecord, now: float) -> bool:
        return (
            self._landing_wait_timeout_s is not None
            and record.resource_wait_since is not None
            and now - record.resource_wait_since >= self._landing_wait_timeout_s
        )

    @staticmethod
    def _deadline_passed(record: TransferRecord, now: float) -> bool:
        return record.deadline is not None and now >= record.deadline

    def _deadline_for(self, record: TransferRecord, now: float) -> float | None:
        timeout = self._fetch_timeout_s if record.direction == "fetch" else self._publish_timeout_s
        return None if timeout is None else now + timeout

    @staticmethod
    def _idle_with_plan(record: TransferRecord) -> bool:
        """A fetch record with a plan and nothing running for it here: PLANNED (the pages are not
        reserved yet, the launch never started, or the landing memory was refused) or LANDED
        (landed on the host, the pages not reserved yet). Attempts of earlier tries may remain
        on the record."""
        return (
            record.direction == "fetch"
            and record.state in (RecordState.PLANNED, RecordState.LANDED)
            and record.plan is not None
        )

    @staticmethod
    def _awaits_plan(record: TransferRecord | None) -> bool:
        """Whether a plan answer is due for the request: it has no fetch record, or a PLANNED one
        whose plan was dropped. Any other record is decided already."""
        return record is None or (record.state is RecordState.PLANNED and record.plan is None)

    def _wants_pages_now(self, record: TransferRecord) -> bool:
        """Whether the scheduler should reserve pages for the record: a device-direct plan not
        launched yet, or a host-first plan whose landing is complete. A host-first plan still
        PLANNED wants the backend's memory first, not pages."""
        if record.plan is None or record.gave_up_launching:
            return False
        if record.state is RecordState.LANDED:
            return True
        return record.state is RecordState.PLANNED and not self._lands_on_host(record)

    def _lands_on_host(self, record: TransferRecord) -> bool:
        return record.plan.source in self._host_first_sources

    def _end_expired_locally(self, record: TransferRecord, vote: Vote) -> None:
        """An expired record whose attempts are over: ended on this rank's vote alone. Its
        request, for a fetch, already failed at the expiry."""
        if record.direction == "publish":
            self._conclude_publish(record, failed=vote.kind is VoteKind.FAILED)
        else:
            self._release_finished_fetch(record)

    # ---- advance, phase 2: plan ----

    def _plan(self, candidates: Sequence[RequestView], now: float) -> dict[int, PlanAnswer]:
        """Ask the planner for an answer on every candidate that awaits one, probing the stores
        first; a FOLLOWER plans nothing."""
        answers: dict[int, PlanAnswer] = {}
        if self._plan_authority is PlanAuthority.FOLLOWER:
            # The owner plans; its answers arrive with the schedule (``adopt_plan_answers``).
            return answers
        for request in candidates:
            request_id = request.py_request_id
            record = self._records.get((request_id, "fetch"))
            # A record that already has a plan is not planned again, even when the scheduler is
            # answered DEFER for it because this rank gave up launching: what comes next is for
            # the ranks to agree on in the apply phase, not for this rank to decide alone.
            if not self._awaits_plan(record):
                continue
            self._requests[request_id] = request
            self._probe(request)
            answers[request_id] = self._planner.decide(
                request,
                self._probe_answers.get(request_id, {}),
                now=now,
                retry_cap=record.retry_cap if record is not None else None,
            )
        return answers

    def _probe(self, request: RequestView) -> None:
        """Ask every store source the request has no answer from yet what it holds, and cache
        each answer that is in on the request for the planner."""
        stores = [source for source in self._sources.values() if source.hint_key is None]
        if not stores:
            return
        cache = self._probe_answers.setdefault(request.py_request_id, {})
        unanswered = [source for source in stores if cache.get(source.name) is None]
        if not unanswered:
            return
        query = self._planner.probe_query(request)
        if query is None:
            return
        for source in unanswered:
            answer = self._probe_one_source(source, query)
            # A pending answer (``None``) is not recorded: an absent entry means the same thing
            # to the planner, and the backend is asked again next round.
            if answer is not None:
                cache[source.name] = answer

    @staticmethod
    def _probe_one_source(
        source: FetchSource, query: tuple[bytes, tuple[bytes, ...]]
    ) -> frozenset[bytes] | None:
        """One store's answer to the probe query; ``None`` while it is pending, and after a
        probe that raised: the answer then stays pending too."""
        name, units = query
        # Broad on purpose, against CODING_GUIDELINES: the contract says a failing probe
        # raises but leaves the type to the backend, and a store outage must not take the
        # engine loop down. The answer stays pending, so the planner defers until its probe
        # budget is spent and then plans without the store.
        try:
            return source.backend.probe(name, units)
        except Exception as exc:  # noqa: BLE001
            logger.warning("probe on %s failed, answer stays pending: %s", source.name, exc)
            return None

    # ---- advance, phase 3: one collective ----

    def _agree(
        self,
        votes: Mapping[RecordKey, Vote],
        expired: Sequence[RecordKey],
        answers: Mapping[int, PlanAnswer],
        pending: bool,
        drained: bool,
        now: float,
    ) -> tuple[dict[RecordKey, Verdict], list[RecordKey], dict[int, PlanAnswer]]:
        """Exchange this rank's message with the other ranks, then reduce the votes to verdicts
        and the plan answers to one answer per request, the same way on every rank."""
        plans_are_collective = self._plan_authority is PlanAuthority.ALL_RANKS
        message = RoundMessage(
            votes=[(key, vote.kind.value, vote.reached_end) for key, vote in sorted(votes.items())],
            expired=sorted(expired),
            plan_answers=(
                [
                    (request_id, encode_plan_answer(answer))
                    for request_id, answer in sorted(answers.items())
                ]
                if plans_are_collective
                else []
            ),
            pending=pending,
            drained=drained,
        )
        # A plain tuple goes out (no class tag in the pickle); a peer's message comes back as
        # whatever 5-sequence the transport made of it.
        gathered = [RoundMessage._make(raw) for raw in self._dist.allgather(tuple(message))]
        all_votes = votes_by_key(gathered)
        self._note_peer_launches(votes, all_votes, now)
        verdicts = reduce_votes(all_votes, len(gathered))
        self._any_rank_pending = any(peer.pending for peer in gathered)
        self._any_rank_drained = any(peer.drained for peer in gathered)
        expired_all = {tuple(key) for peer in gathered for key in peer.expired}
        # An owner's answers are its own; they reach the followers with the schedule.
        plan_answers = (
            reduce_plan_answers(gathered, answers) if plans_are_collective else dict(answers)
        )
        return verdicts, sorted(expired_all), plan_answers

    def _note_peer_launches(
        self,
        votes: Mapping[RecordKey, Vote],
        all_votes: Mapping[RecordKey, list[Vote]],
        now: float,
    ) -> None:
        """Start the unlaunched clock of every fetch this rank has not launched while some other
        rank has (its vote is anything but UNLAUNCHED). The collective carries no rank identity,
        so the kinds of the votes are all there is to read."""
        for key, vote in votes.items():
            if vote.kind is not VoteKind.UNLAUNCHED:
                continue
            record = self._records[key]
            if record.peer_launch_seen_at is None and any(
                peer.kind is not VoteKind.UNLAUNCHED for peer in all_votes.get(key, ())
            ):
                record.peer_launch_seen_at = now

    # ---- advance, phase 4: apply ----

    def _apply(
        self,
        verdicts: Mapping[RecordKey, Verdict],
        expired: Sequence[RecordKey],
        answers: Mapping[int, PlanAnswer],
        now: float,
    ) -> None:
        """Write what the ranks agreed on to the records: expiries first, then verdicts; then
        ask the backend again for the landing memory it refused to any host-first plan, and
        record the decided answers."""
        self._apply_expiries(expired, now)
        self._apply_verdicts(verdicts, now)
        self._retry_refused_landings(now)
        self._apply_plan_answers(answers, now)

    def _apply_expiries(self, expired: Sequence[RecordKey], now: float) -> None:
        """Expire every record some rank saw past its deadline, once."""
        for key in expired:
            record = self._records.get(key)
            if record is not None and not record.expired:
                self._expire(record, now)

    def _apply_verdicts(self, verdicts: Mapping[RecordKey, Verdict], now: float) -> None:
        """End, land or re-plan every record the ranks reached a verdict on."""
        for key in sorted(verdicts):
            record = self._records.get(key)
            if record is None:
                continue
            verdict = verdicts[key]
            if record.direction == "publish":
                self._conclude_publish(record, failed=verdict.failed)
            elif record.request_id in self._finished:
                # The request is gone; the agreement only says every rank releases it now.
                self._release_finished_fetch(record)
            elif self._backend_busy_on(record):
                self._apply_fetch_verdict(record, verdict, now)
            elif self._idle_with_plan(record):
                # An UNLAUNCHED vote blocks a landing, so the only verdict that reaches an
                # unlaunched record is a failure: drop the plan without touching pages.
                self._reset_for_replan(record)

    def _apply_plan_answers(self, answers: Mapping[int, PlanAnswer], now: float) -> None:
        """Record every decided answer; an OWNER also keeps them for ``export_plan_answers``."""
        self._answers_to_export = {}
        for request_id, answer in answers.items():
            if answer is DEFER:
                continue
            self._apply_answer(request_id, answer, now)
            if self._plan_authority is PlanAuthority.OWNER:
                self._answers_to_export[request_id] = answer

    def _apply_answer(self, request_id: int, answer: FetchPlan | None, now: float) -> None:
        """Write a decided answer: ``None`` releases a planned record, a plan goes on the record
        (created if needed) and ``fetch_answer`` reads it from now on. A plan from a host-first
        source is started right here, since it needs no pages to begin; any other waits for the
        scheduler's pages from now, on the wait clock. An answer for a record that is decided
        already is ignored."""
        key = (request_id, "fetch")
        record = self._records.get(key)
        if not self._awaits_plan(record):
            logger.debug(
                "request %d: plan answer ignored, record is %s", request_id, record.state.value
            )
            return
        self._plan_answers[request_id] = answer
        if answer is None:
            if record is not None:
                self._release(record)
            return
        if record is None:
            record = TransferRecord(request_id, "fetch", RecordState.PLANNED)
            self._records[key] = record
        record.plan = answer
        record.retry_cap = None
        if self._lands_on_host(record):
            self._start_host_landing(record, now)
        else:
            record.resource_wait_since = now

    # ---- host-first: a fetch lands in the backend's memory before pages are reserved ----

    def _start_host_landing(self, record: TransferRecord, now: float) -> None:
        """Phase one of a host-first fetch: ask the backend to land the plan's units in its own
        memory. No page is involved, so a refusal costs nothing but time: the record stays
        PLANNED, its wait is clocked from the first refusal, and the next ``advance`` asks again.
        It is not a failed launch (``consecutive_launch_failures`` is for page back-pressure)."""
        source = self._sources[record.plan.source]
        try:
            landing = source.backend.fetch_to_host(unit_names(record.plan))
        except SubmissionRejected as exc:
            logger.info(
                "request %d: landing refused by %s: %s", record.request_id, source.name, exc
            )
            if record.resource_wait_since is None:
                record.resource_wait_since = now
            return
        record.landing = landing
        record.resource_wait_since = None
        record.peer_launch_seen_at = None
        record.state = RecordState.LANDING
        record.deadline = self._deadline_for(record, now)

    def _retry_refused_landings(self, now: float) -> None:
        """Ask again for the landing memory of every host-first plan the backend refused."""
        for record in self._records.values():
            if (
                record.direction == "fetch"
                and record.state is RecordState.PLANNED
                and record.plan is not None
                and record.request_id not in self._finished
                and self._lands_on_host(record)
            ):
                self._start_host_landing(record, now)

    @staticmethod
    def _mark_landed(record: TransferRecord, now: float) -> None:
        """The ranks agreed the landing in the host memory is complete: the record waits for
        pages (``LANDED``), on the wait clock alone."""
        record.state = RecordState.LANDED
        record.resource_wait_since = now
        record.peer_launch_seen_at = None
        record.deadline = None  # the placement sets its own

    # ---- record endings: how a record leaves the table, and what its request is told ----

    def _expire(self, record: TransferRecord, now: float) -> None:
        """Some rank saw the record's deadline pass: from now on this rank ends it alone, once
        its own attempts are over (never under a live attempt: quiesce would block the engine
        thread). A publish is a warning, not a request failure. A fetch fails its request, unless
        the request ended already and the record is only waiting for its outcome."""
        record.expired = True
        if record.direction == "publish":
            # A long chunked prefill is still offering pieces: not worth a warning until the
            # last piece is out and the store has had its full say.
            log = logger.warning if record.extent.is_last else logger.debug
            log(
                "request %d: kv publish past its deadline; ended here once its attempts are over",
                record.request_id,
            )
        elif record.request_id not in self._finished:
            self._fail_expired_fetch(record, now)

    def _apply_fetch_verdict(self, record: TransferRecord, verdict: Verdict, now: float) -> None:
        """Land, deliver, or fail a fetch on the ranks' verdict."""
        if verdict.failed:
            self._fetch_failed_or_short(record, cap=None, reason="kv fetch failed")
        elif verdict.reached_end == record.plan.token_end:
            if record.state is RecordState.LANDING:
                self._mark_landed(record, now)
            else:
                self._mark_delivered(record)
        else:
            # The agreed MIN(reached_end) is as far as every rank got: the retry aims no higher.
            self._fetch_failed_or_short(
                record, cap=verdict.reached_end, reason="kv fetch served short"
            )

    def _mark_delivered(self, record: TransferRecord) -> None:
        """The ranks agreed the delivery into the pages is complete: the request is unparked and
        the landing, if any, is given back at once (the copy was complete before its outcome was
        reported); the record itself stays until the request ends."""
        record.state = RecordState.DELIVERED
        self._close_routes(record)
        request = self._requests[record.request_id]
        self._plan_answers[record.request_id] = None
        self._effects.unpark(
            request, record.plan.token_end, record.plan.no_local_fallback, self._aux(record)
        )
        self._release_landing(record)

    def _fetch_failed_or_short(
        self, record: TransferRecord, *, cap: int | None, reason: str
    ) -> None:
        """The ranks agreed the delivery failed or came up short. A landing on its way to the
        host names no page: nothing to quiesce or give back, the landing goes and the plan is
        retried or given up. A delivery into the pages passes the release point first."""
        was_landing = record.state is RecordState.LANDING
        record.state = RecordState.FAILED
        if was_landing:
            self._retry_or_give_up(record, cap=cap, reason=reason)
            return
        if not self._quiesce(record):
            return
        self._close_routes(record)
        request = self._requests[record.request_id]
        self._effects.give_back_fetch_pages([request])
        self._retry_or_give_up(record, cap=cap, reason=reason)

    def _reset_for_replan(self, record: TransferRecord) -> None:
        """The ranks agreed the fetch failed while this rank never launched it: same next step as
        after a failed attempt, minus the release point (no attempt, no pages to give back)."""
        reason = "kv fetch launch given up" if record.gave_up_launching else "kv fetch failed"
        self._retry_or_give_up(record, cap=None, reason=reason)

    def _retry_or_give_up(self, record: TransferRecord, *, cap: int | None, reason: str) -> None:
        """After a failure: a gen-init fetch fails its request; otherwise spend a retry and plan
        again next round, or, out of retries, compute the rest locally."""
        request = self._requests[record.request_id]
        if record.plan.no_local_fallback:
            self._release(record)
            self._plan_answers[record.request_id] = None
            self._effects.fail_requests([request], reason)
        elif record.retries_left > 0:
            record.retries_left -= 1
            record.retry_cap = cap
            self._release_landing(record)
            record.plan = None
            record.extent = None
            record.deadline = None
            record.gave_up_launching = False
            record.consecutive_launch_failures = 0
            record.peer_launch_seen_at = None
            record.state = RecordState.PLANNED
            self._plan_answers.pop(record.request_id, None)
        else:
            self._release(record)
            self._plan_answers[record.request_id] = None

    def _fail_expired_fetch(self, record: TransferRecord, now: float) -> None:
        """A fetch of a running request past its deadline: fail the request now.

        A hung store must not park a request forever, so the wait is bounded by the deadline
        and the request fails. A delivery into the pages cannot give them back yet: the backend
        may still be writing them, so the record stays in flight and the request is held until
        the outcome arrives (or ``close``); the late outcome then only releases. A record with
        nothing in the pages (planned, landing on the host, landed there) goes at once.
        """
        request = self._requests[record.request_id]
        self._effects.fail_requests([request], "kv fetch timed out")
        # The engine ends a failed request through its release gate, which notifies this
        # coordinator; the call is repeated here so the hold does not depend on the engine.
        self._finish_request(request, now)

    def _release_finished_fetch(self, record: TransferRecord) -> None:
        """A fetch whose request ended: its outcome is moot. Pass the release point if the
        current try touched the pages, release; no ``unpark`` and no ``give_back`` for a request
        that is gone. The held request is terminated once nothing else of it remains."""
        if self._owes_quiesce(record) and not self._quiesce(record):
            return
        self._close_routes(record)
        self._release(record)
        self._terminate_if_no_records(record.request_id)

    def _conclude_publish(self, record: TransferRecord, *, failed: bool) -> None:
        """A publish reached its end. A failure is a warning, never a request failure: a
        running request keeps running, and a finished one already has its response."""
        record.state = RecordState.FAILED if failed else RecordState.DELIVERED
        if self._owes_quiesce(record) and not self._quiesce(record):
            return
        self._release(record)
        request_id = record.request_id
        if failed:
            when = (
                "after the request ended" if request_id in self._finished else "while still running"
            )
            logger.warning("request %d: kv publish failed %s", request_id, when)
        self._terminate_if_no_records(request_id)

    def _terminate_if_no_records(self, request_id: int) -> None:
        """A finished request's last call to the engine, once no record of it remains.

        A held request is terminated here, whatever its last transfer's outcome. A request that
        was never held owes the engine nothing: it terminates as usual through the release gate.
        """
        if request_id not in self._finished or self._has_records_of(request_id):
            return
        request = self._requests.get(request_id)
        if request is None:
            logger.warning("request %d finished with no request object on record", request_id)
        elif request_id in self._held:
            if request_id not in self._finished_by_gate:
                self._terminated_before_gate.add(request_id)
            self._effects.terminate_request(request)
        self._forget_request(request_id)

    # ---- launch / publish helpers ----

    def _launch_or_place(self, request: RequestView, record: TransferRecord, now: float) -> bool:
        if record.state is RecordState.LANDED:
            return self._place_one(request, record, now)
        return self._launch_one(request, record, now)

    def _launch_one(self, request: RequestView, record: TransferRecord, now: float) -> bool:
        """Start the delivery of a device-direct plan into the pages the scheduler reserved;
        False when the launch did not start and the pages went back."""
        plan = record.plan
        request_id = request.py_request_id
        source = self._sources[plan.source]
        extent, committed = self._reader.fetch_extent_and_committed(request, plan)
        route = None
        if source.hint_key is not None and plan.hint is not None:
            route = self._open_route_or_drop(request, record, source)
            if route is None:
                return False
        try:
            attempt = source.backend.fetch(extent, route=route)
        except SubmissionRejected as exc:
            # Nothing escaped, so nothing to quiesce: give the pages back and try the same plan
            # again next round (this is the one back-pressure signal).
            logger.info("request %d: fetch rejected by %s: %s", request_id, source.name, exc)
            if route is not None:
                route.close()
            self._drop_launch(request, record)
            return False
        self._start_try(record, attempt, extent, committed, source.name, route, now)
        return True

    def _open_route_or_drop(
        self, request: RequestView, record: TransferRecord, source: FetchSource
    ) -> Route | None:
        """The route to the plan's hint, or ``None`` once the launch has been dropped: a refused
        route means the plan can never work here (give up), a failed one is worth another try.
        ``open_route`` itself always returns a ``Route``; a ``None`` from it would break the
        contract."""
        request_id = request.py_request_id
        # Broad on purpose, against CODING_GUIDELINES: ``open_route`` raises the backend's own
        # transport error for a hint that was fine but could not be prepared; that is worth
        # trying again, and it must not escape and strand the other requests in the queue.
        try:
            return source.backend.open_route(record.plan.hint)
        except (ValueError, NotImplementedError) as exc:
            # Bad hint, or a backend that cannot route: this plan can never work here.
            logger.warning("request %d: route refused by %s: %s", request_id, source.name, exc)
            self._drop_launch_and_give_up(request, record)
        except Exception as exc:  # noqa: BLE001
            logger.warning("request %d: route to %s failed: %s", request_id, source.name, exc)
            self._drop_launch(request, record)
        return None

    def _place_one(self, request: RequestView, record: TransferRecord, now: float) -> bool:
        """Phase two of a host-first fetch, once the scheduler reserved the pages: copy the
        landing into them. From here on the record is an ordinary delivery into pages; a refusal
        is page back-pressure and is counted as one."""
        request_id = request.py_request_id
        source = self._sources[record.plan.source]
        extent, committed = self._reader.fetch_extent_and_committed(request, record.plan)
        try:
            attempt = record.landing.place(extent)
        except SubmissionRejected as exc:
            logger.info("request %d: placement rejected by %s: %s", request_id, source.name, exc)
            self._drop_launch(request, record)
            return False
        self._start_try(record, attempt, extent, committed, source.name, None, now)
        return True

    def _start_try(
        self,
        record: TransferRecord,
        attempt: Attempt,
        extent: CacheExtent,
        committed: frozenset[bytes],
        backend_name: str,
        route: Route | None,
        now: float,
    ) -> None:
        """A delivery into the pages has started: the record is IN_FLIGHT on a new try, with the
        deadline counted from now."""
        record.consecutive_launch_failures = 0
        record.peer_launch_seen_at = None
        record.resource_wait_since = None
        try_index = record.try_index + 1 if record.attempts else 0
        record.extent = extent
        record.committed_names = committed
        record.attempts.append(
            AttemptRecord(attempt, try_index=try_index, backend_name=backend_name, route=route)
        )
        record.state = RecordState.IN_FLIGHT
        record.deadline = self._deadline_for(record, now)

    def _drop_launch(self, request: RequestView, record: TransferRecord) -> None:
        """A launch that never started: give the pages back, keep the plan and the retry budget.

        The plan stays so the scheduler reserves for it again next round; whether the rank
        launches then or votes the fetch failed is not decided here. A launch refused
        ``MAX_CONSECUTIVE_LAUNCH_FAILURES`` times in a row makes the record give up.
        """
        self._give_pages_back_after_failed_launch(request, record)
        if record.consecutive_launch_failures >= MAX_CONSECUTIVE_LAUNCH_FAILURES:
            self._give_up_launching(request, record)

    def _drop_launch_and_give_up(self, request: RequestView, record: TransferRecord) -> None:
        """A launch that can never work here (the route was refused): give the pages back and
        give up on the plan at once."""
        self._give_pages_back_after_failed_launch(request, record)
        self._give_up_launching(request, record)

    def _give_pages_back_after_failed_launch(
        self, request: RequestView, record: TransferRecord
    ) -> None:
        """Give the reserved pages back and count a launch that never started."""
        self._effects.give_back_fetch_pages([request])
        record.consecutive_launch_failures += 1

    @staticmethod
    def _give_up_launching(request: RequestView, record: TransferRecord) -> None:
        """From now on the record votes FAILED and answers the scheduler DEFER until the ranks
        agree, so that a rank never re-plans a fetch on its own."""
        record.gave_up_launching = True
        logger.warning(
            "request %d: giving up on launching the fetch from %s after %d failed launches",
            request.py_request_id,
            record.plan.source,
            record.consecutive_launch_failures,
        )

    def _publish_one(self, request: RequestView, now: float) -> None:
        """Offer what the request committed to every publisher, on a publish record created on
        first sight; the record goes IN_FLIGHT with the first submission a publisher accepts."""
        request_id = request.py_request_id
        self._requests[request_id] = request
        key = (request_id, "publish")
        record = self._records.get(key)
        if record is None:
            record = TransferRecord(request_id, "publish", RecordState.PLANNED)
            self._records[key] = record
        elif record.state not in (RecordState.PLANNED, RecordState.IN_FLIGHT):
            return
        extent, chunk = self._reader.publish_extent_and_chunk(request)
        record.extent = extent
        self._submit_publish_pieces(record, extent, chunk)
        if record.state is RecordState.PLANNED and record.attempts:
            # The deadline counts from the first submission a publisher accepted.
            record.state = RecordState.IN_FLIGHT
            record.deadline = self._deadline_for(record, now)

    def _submit_publish_pieces(
        self, record: TransferRecord, extent: CacheExtent, chunk: Chunk | None
    ) -> None:
        """Design §7.5: a publisher that places pieces works in series and hears every piece;
        one that does not is offered the content once, on the last piece, and never has to read
        ``is_last``."""
        for name, publisher in self._publishers.items():
            if isinstance(publisher, PlacesPieces):
                self._submit_publish(record, name, publisher.publish, extent)
                if chunk is not None:
                    self._submit_publish(record, name, publisher.place_piece, chunk)
            elif extent.is_last:
                self._submit_publish(record, name, publisher.publish, extent)

    def _submit_publish(
        self, record: TransferRecord, name: str, submit: Callable[[T], Attempt], arg: T
    ) -> None:
        try:
            attempt = submit(arg)
        except SubmissionRejected as exc:
            logger.warning("request %d: publish rejected by %s: %s", record.request_id, name, exc)
            record.rejected = True
            return
        record.attempts.append(AttemptRecord(attempt, backend_name=name))

    # ---- record bookkeeping ----

    @staticmethod
    def _owes_quiesce(record: TransferRecord) -> bool:
        """Whether the current try's attempts touched the pages and were not vouched for yet. A
        fetch quiesces each failed try as it fails, so only a try in flight, delivered, or
        fatally failed owes one; a publish has a single try."""
        if not record.current_try_attempts():
            return False
        if record.direction == "publish":
            return True
        return record.state in (RecordState.IN_FLIGHT, RecordState.DELIVERED, RecordState.FAILED)

    def _quiesce(self, record: TransferRecord) -> bool:
        """The release point. False means the backend cannot vouch for the memory: the engine is
        made fatal and the record stays, so the pages are never handed out again (design §4.3
        would retry until the deadline; the fatal path is what the engine has). A refusal is
        final: the backend is not asked again, and the engine is not made fatal twice."""
        if record.quiesce_refused:
            return False
        by_backend: dict[str, list[Attempt]] = {}
        for attempt in record.current_try_attempts():
            by_backend.setdefault(attempt.backend_name, []).append(attempt.attempt)
        for name, attempts in by_backend.items():
            if not self._backends[name].quiesce(attempts):
                record.quiesce_refused = True
                self._effects.fail_fatal(
                    RuntimeError(
                        f"backend {name} cannot confirm memory of request {record.request_id} "
                        f"({record.direction}) is untouched"
                    )
                )
                return False
        return True

    @staticmethod
    def _close_routes(record: TransferRecord) -> None:
        """Close the current try's routes. A close that raises has not closed: the handle stays
        on its attempt and is closed again only if the record reaches another release point; a
        record released right after takes it along. The other routes are still closed."""
        for attempt in record.current_try_attempts():
            if attempt.route is None:
                continue
            # Broad on purpose, against CODING_GUIDELINES: the contract says a failing close
            # raises but leaves the type to the backend, and a cleanup failure must not take
            # the engine loop down.
            try:
                attempt.route.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "request %d: closing the route from %s failed, handle kept: %s",
                    record.request_id,
                    attempt.backend_name,
                    exc,
                )
            else:
                attempt.route = None

    @staticmethod
    def _aux(record: TransferRecord) -> Mapping[str, object] | None:
        aux: dict[str, object] = {}
        for attempt in record.current_try_attempts():
            if isinstance(attempt.attempt, CarriesAux):
                aux.update(attempt.attempt.aux())
        return aux or None

    def _release(self, record: TransferRecord) -> None:
        """The record leaves the table; release is an event, not a state."""
        self._release_landing(record)
        self._records.pop(record.key, None)

    @staticmethod
    def _release_landing(record: TransferRecord) -> None:
        """Close a host-first landing and forget what went with it. Idempotent; a no-op for a
        record without one. Called when the delivery is complete (after ``unpark``), when the
        record leaves the table, when the plan is dropped for a retry, and at the end of a
        request whose landing is still on its way."""
        if record.landing is not None:
            record.landing.close()
        record.landing = None
        record.resource_wait_since = None
        record.committed_names = frozenset()

    def _forget_request(self, request_id: int) -> None:
        """Drop every per-request entry of a request that is gone, here and in the planner and
        the reader. For a held request this runs at its termination, so the reader's per-request
        cache keeps that one entry until then."""
        self._requests.pop(request_id, None)
        self._plan_answers.pop(request_id, None)
        self._pending_answers.pop(request_id, None)
        self._probe_answers.pop(request_id, None)
        self._finished.discard(request_id)
        self._finished_by_gate.discard(request_id)
        self._held.discard(request_id)
        self._planner.forget(request_id)
        self._reader.forget_request(request_id)

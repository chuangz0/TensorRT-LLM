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
(backends), never sees an address (contract) and never branches on backend type (assembly table
plus optional protocols).

Everything runs on the engine thread. There are no threads here.
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import TYPE_CHECKING, Callable, Mapping, NamedTuple, Sequence, TypeVar

from ...base.cache_backend import (
    Attempt,
    CacheExtent,
    Delivered,
    Fetches,
    Publishes,
    Route,
    SubmissionRejected,
)
from ...base.views import RequestView, ResourceReader
from ...remote_cache import DEFER, Defer, FetchPlan, FetchSource, Planner, merge, unit_names
from .interfaces import (
    CarriesAux,
    Collective,
    EngineQueue,
    KVTransferEffects,
    LandsOnHost,
    PlacesPieces,
    PlanAuthority,
)
from .records import AttemptRecord, RecordKey, RecordState, TransferRecord, is_failure

if TYPE_CHECKING:
    from ...base.backend import Chunk

__all__ = [
    "DEFER",
    "MAX_CONSECUTIVE_LAUNCH_FAILURES",
    "KVTransferCoordinator",
    "PlanAnswers",
    "Vote",
    "VoteKind",
]
# ``DEFER`` is re-exported: it is the coordinator's ``plan_fetch`` answer the engine compares
# against, and the engine side takes everything it needs from this package.

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_LAUNCH_FAILURES = 3
"""Launches in a row that never started (refused submission or failed route) after which a fetch
record gives up on its plan and lets the ranks agree on a failure."""

T = TypeVar("T")

_DEFER_ANSWER = "DEFER"
_PlanAnswer = FetchPlan | None | Defer
_EncodedPlanAnswer = tuple[int, str] | str | None
"""A plan answer on the wire: ``(token_end, source)``, ``"DEFER"``, or ``None``."""
PlanAnswers = list[tuple[int, _EncodedPlanAnswer]]
"""Decided answers as the owner hands them to the followers: ``[(request_id, encoded answer)]``,
never ``"DEFER"``. Built-in types only, so it rides in the pickled schedule."""


class VoteKind(Enum):
    """What one rank says about one record this round. Carried on the wire as the string value.

    UNLAUNCHED: fetch planned but not launched here (no pages yet); holds up a landing, not a failure.
    INFLIGHT: an attempt is still running here; nothing may be decided this round.
    FAILED: decisive: this rank's attempt failed, or it gave up launching.
    TERMINAL: every attempt here ended without failure; carries ``token_end`` for a fetch.
    """

    UNLAUNCHED = "UNLAUNCHED"
    INFLIGHT = "INFLIGHT"
    FAILED = "FAILED"
    TERMINAL = "TERMINAL"


class Vote(NamedTuple):
    """One rank's word on one record. ``token_end`` is the block boundary this rank's attempts
    reached, the merged ``B`` of design §7.1 (``remote_cache.merge``); it matters for ``TERMINAL``
    fetches only."""

    kind: VoteKind
    token_end: int = 0


_Verdict = tuple[int, bool]
"""``(MIN(token_end), failed)``: what the ranks agreed on for one record."""


class _RoundMessage(NamedTuple):
    """One rank's word per round: ``votes`` as ``[(key, kind, token_end)]``, ``expired`` as
    ``[key]``, ``plans`` as ``[(rid, encoded answer)]``. A plain 3-tuple on the wire."""

    votes: list
    expired: list
    plans: list


def _encode_plan_answer(answer: _PlanAnswer) -> _EncodedPlanAnswer:
    if answer is DEFER:
        return _DEFER_ANSWER
    if answer is None:
        return None
    return (answer.token_end, answer.source)


def _ballots_by_key(gathered: Sequence[_RoundMessage]) -> dict[RecordKey, list[Vote]]:
    """Every rank's vote on every record, in gathered order; keys come back as tuples."""
    ballots: dict[RecordKey, list[Vote]] = {}
    for message in gathered:
        for key, kind, token_end in message.votes:
            ballots.setdefault(tuple(key), []).append(Vote(VoteKind(kind), token_end))
    return ballots


def _reduce_votes(ballots: Mapping[RecordKey, list[Vote]], n: int) -> dict[RecordKey, _Verdict]:
    """The one reduction of design §7.1 "齐", for fetches and publishes alike.

    A record is decided only once all ``n`` ranks voted on it, and then in this order: any
    INFLIGHT holds the round (a failure landed now would quiesce under a running attempt and
    block the engine thread); else any FAILED is decisive for every rank, delivered data
    included; else any UNLAUNCHED holds the round (a landing needs every rank's pages); else
    every rank is TERMINAL and the landing takes MIN(B), since ranks hold different layer groups.
    """
    verdicts: dict[RecordKey, _Verdict] = {}
    for key, votes in ballots.items():
        if len(votes) < n:
            continue
        kinds = {vote.kind for vote in votes}
        if VoteKind.INFLIGHT in kinds:
            continue
        if VoteKind.FAILED in kinds:
            verdicts[key] = (0, True)
        elif VoteKind.UNLAUNCHED not in kinds:
            verdicts[key] = (min(v.token_end for v in votes), False)
    return verdicts


def _reduce_plans(
    gathered: Sequence[_RoundMessage], answers: Mapping[int, _PlanAnswer]
) -> dict[int, _PlanAnswer]:
    """Any DEFER -> DEFER; otherwise any disagreement on ``(token_end, source)`` -> None."""
    n = len(gathered)
    encoded: dict[int, list[_EncodedPlanAnswer]] = {}
    for message in gathered:
        for rid, answer in message.plans:
            encoded.setdefault(rid, []).append(
                tuple(answer) if isinstance(answer, list) else answer
            )
    consensus: dict[int, _PlanAnswer] = {}
    for rid, local in answers.items():
        v = encoded.get(rid, [])
        if len(v) < n or _DEFER_ANSWER in v:
            consensus[rid] = DEFER
        elif all(x == v[0] for x in v) and _encode_plan_answer(local) == v[0]:
            consensus[rid] = local
        else:
            consensus[rid] = None
    return consensus


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
            stops waiting for the others; the record still settles on its own outcome. ``None``
            disables.
        unlaunched_timeout_s: Longest this rank may stay unlaunched on a fetch another rank has
            already launched before it votes the fetch failed. ``None`` disables. Never starts
            counting on a single rank. Required: the default lives in ``KVTransferConfig``.
        landing_wait_timeout_s: Longest this rank waits for the scheduler's pages (from the
            plan's decision, or from landing on the host) or, host-first, for the backend's
            landing memory, before it votes the fetch failed. Counts on a single rank too, so a
            fetch the scheduler can never find pages for is given up and the request computes
            locally instead of stalling. ``None`` disables. Required, as above.
        plan_authority: Who decides plan answers on this rank. ``VOTED``: planned here and
            reduced in the collective. ``OWNER``: planned here alone, not carried in the
            payload, handed out with ``export_plan_answers``. ``FOLLOWER``: never planned or
            probed here; taken with ``adopt_plan_answers``. Votes and expiries travel through
            the collective in every mode.
        queue_budget: How many posted callables one ``advance`` runs.

    The engine must call ``notify_request_finished`` for every request it ever passed in, so that
    fetch records reach their release point and per-request state is dropped. Landings still
    held when the engine shuts down (``hooks.close`` frees parked and held requests only) are
    released by the backend's own ``close``.

    A record of a finished request keeps taking part in the round until the ranks agree on it,
    so that every rank terminates the request in the same round; a record past its deadline is
    rank-local instead and settles here as soon as its own attempts are over. A publish past its
    deadline whose attempt is still live stays ``IN_FLIGHT`` (it is never quiesced under a live
    attempt), so ``inflight_request_ids`` stays non-empty and the scheduler's deadlock detector
    is off for that time: the intended trade for never blocking the engine thread.
    """

    def __init__(
        self,
        sources: Sequence[FetchSource],
        publishers: Sequence[Publishes],
        planner: Planner,
        reader: ResourceReader,
        effects: KVTransferEffects,
        queue: EngineQueue,
        dist: Collective,
        *,
        unlaunched_timeout_s: float | None,
        landing_wait_timeout_s: float | None,
        fetch_timeout_s: float | None = None,
        publish_timeout_s: float | None = None,
        plan_authority: PlanAuthority = PlanAuthority.VOTED,
        queue_budget: int = 64,
    ) -> None:
        self._sources: dict[str, FetchSource] = {s.name: s for s in sources}
        self._publishers: dict[str, Publishes] = {
            f"publish:{i}": p for i, p in enumerate(publishers)
        }
        self._backends: dict[str, Fetches | LandsOnHost | Publishes] = {
            **{name: s.backend for name, s in self._sources.items()},
            **self._publishers,
        }
        self._host_sources: frozenset[str] = frozenset(
            name for name, s in self._sources.items() if isinstance(s.backend, LandsOnHost)
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
        self._plans: dict[int, FetchPlan | None] = {}
        """Decided answers ``plan_fetch`` reads. A request with no entry is undecided (DEFER)."""
        self._answers_to_export: dict[int, FetchPlan | None] = {}
        """OWNER only: the answers the last ``advance`` decided, for ``export_plan_answers``."""
        self._pending_answers: dict[int, _EncodedPlanAnswer] = {}
        """FOLLOWER only: adopted answers whose request is not a candidate here yet."""
        self._probe_answers: dict[int, dict[str, frozenset[bytes] | None]] = {}
        self._finished: set[int] = set()
        self._held: set[int] = set()
        """Finished requests the engine was told to hold; each owes a ``terminate_request``."""
        self._finished_by_gate: set[int] = set()
        """Finished requests the engine's release gate reported; the others were failed here and
        the gate has yet to ask about them."""
        self._terminated_before_gate: set[int] = set()
        """Held requests this coordinator terminated before the engine's release gate asked about
        them; the gate is answered "not yours" once, so the engine does not terminate twice."""

    # ---- loop entry points (``advance`` is the one collective: once per rank per round) ----

    def advance(self, candidates: Sequence[RequestView], now: float) -> int:
        """Head of the loop: poll outcomes into votes, plan candidates, agree across ranks, apply.

        Returns how many candidates are still undecided afterwards (``plan_fetch`` answers
        ``DEFER``): each is waiting on a store lookup, or on the ranks' agreement.
        """
        votes, expired = self._poll_and_vote(now)
        answers = self._plan(candidates, now)
        verdicts, expired, answers = self._agree(votes, expired, answers, now)
        self._apply(verdicts, expired, answers, now)
        return sum(1 for candidate in candidates if self.plan_fetch(candidate) is DEFER)

    def launch_reserved_fetches(
        self, queue: Sequence[RequestView], now: float | None = None
    ) -> None:
        """After scheduling: start the fetch of every request the scheduler allocated for. A
        record already landed on the host (``STAGED``) places its landing into the pages instead
        of fetching; either way the request is parked until the delivery is agreed on."""
        now = time.monotonic() if now is None else now
        ready = []
        for req in queue:
            rec = self._records.get((req.py_request_id, "fetch"))
            if rec is not None and self._wants_pages_now(rec):
                ready.append((req, rec))
        if not ready:
            return
        self._effects.prepare_fetch_resources([req for req, _ in ready])
        launched = [req for req, rec in ready if self._launch_or_place(req, rec, now)]
        if launched:
            self._effects.park_for_fetch(launched)

    def publish_committed_blocks(
        self, reqs: Sequence[RequestView], now: float | None = None
    ) -> None:
        """After a context step, before the response pass: offer the blocks each request committed.

        A request that ended this step is reported separately through ``notify_request_finished``
        (the engine's release gate); a publish still in flight then holds it.
        """
        now = time.monotonic() if now is None else now
        if self._publishers:
            for req in reqs:
                self._publish_one(req, now)

    # ---- plan authority: owner hands out, follower takes ----

    @property
    def plan_authority(self) -> PlanAuthority:
        return self._plan_authority

    def export_plan_answers(self) -> PlanAnswers:
        """OWNER, after ``advance``: the answers it decided this round, on the wire, for the
        followers' ``adopt_plan_answers``. Empty in any other mode."""
        return [
            (rid, _encode_plan_answer(ans)) for rid, ans in sorted(self._answers_to_export.items())
        ]

    def adopt_plan_answers(
        self, views: Sequence[RequestView], answers: PlanAnswers, now: float | None = None
    ) -> int:
        """FOLLOWER: take the owner's answers. ``None`` decides a request local; ``(token_end,
        source)`` becomes this rank's own plan over its own layer groups. An answer for a request
        not yet among ``views`` (the undecided candidates here) is held until its request appears
        as a candidate, or is dropped when the request ends: the owner decides a request once and
        does not export it again. Returns how many of ``views`` got no answer and stay deferred."""
        now = time.monotonic() if now is None else now
        by_rid = {view.py_request_id: view for view in views}
        self._pending_answers.update(answers)
        for rid, encoded in list(self._pending_answers.items()):
            view = by_rid.get(rid)
            if view is None:
                continue
            del self._pending_answers[rid]
            self._requests[rid] = view
            if encoded is None:
                self._record_answer(rid, None, now)
            else:
                token_end, source = encoded
                self._record_answer(rid, self._planner.materialize(view, token_end, source), now)
        answered = {rid for rid, _ in answers}
        return sum(1 for rid in by_rid if rid not in answered)

    # ---- scheduler hook (read-only, non-blocking) ----

    def plan_fetch(self, req: RequestView) -> _PlanAnswer:
        """``FetchPlan`` to fetch, ``None`` to compute locally, ``DEFER`` to skip this round.

        A request the coordinator has not decided yet answers ``DEFER``; the engine passes such
        requests as ``candidates`` to the next ``advance``. So does a request whose rank gave up
        launching its plan: the ranks have yet to agree on what comes next, and meanwhile the
        scheduler must neither reserve pages for it nor plan it locally. A host-first fetch
        answers ``DEFER`` while its units are on their way to the backend (no pages are wanted
        yet) and its plan once they have landed (``STAGED``: reserve pages now). A request with a
        fetch record that holds pages, or has landed, has nothing more to plan and answers
        ``None``.
        """
        rid = req.py_request_id
        rec = self._records.get((rid, "fetch"))
        if rec is not None:
            if rec.state in (RecordState.PLANNED, RecordState.STAGED):
                return rec.plan if self._wants_pages_now(rec) else DEFER
            if rec.state is RecordState.STAGING:
                return DEFER
            return None
        if rid in self._plans:
            return self._plans[rid]
        return DEFER

    # ---- control ----

    def notify_request_finished(self, req: RequestView, now: float | None = None) -> bool:
        """The engine's release gate: the request ended. Returns whether the engine may terminate
        it now: ``False`` while this coordinator holds it (it terminates the request itself once
        every record of it is gone), and ``False`` once for a request this coordinator failed and
        already terminated before the gate asked."""
        rid = req.py_request_id
        if rid in self._terminated_before_gate:
            self._terminated_before_gate.discard(rid)
            return False
        self._finished_by_gate.add(rid)
        self._finish_request(req, time.monotonic() if now is None else now)
        return rid not in self._held

    def _finish_request(self, req: RequestView, now: float) -> None:
        """What a request's end does to its records; the request is held while any remains."""
        rid = req.py_request_id
        if rid in self._finished:
            return
        self._finished.add(rid)
        self._requests[rid] = req
        fetch = self._records.get((rid, "fetch"))
        if fetch is not None:
            self._end_fetch_of_finished_request(fetch, now)
        publish = self._records.get((rid, "publish"))
        if publish is not None:
            self._end_publish_of_finished_request(publish, now)
        self._hold_or_finish(req)

    def _hold_or_finish(self, req: RequestView) -> None:
        """A finished request is held while any record of it remains; with none left it gets
        its last word to the engine now."""
        rid = req.py_request_id
        if self._has_records_of(rid):
            self._held.add(rid)
            self._effects.hold_for_transfer([req])
        else:
            self._finish_if_released(rid)

    def _has_records_of(self, rid: int) -> bool:
        return (rid, "fetch") in self._records or (rid, "publish") in self._records

    def _end_fetch_of_finished_request(self, rec: TransferRecord, now: float) -> None:
        """A delivery into the pages (``IN_FLIGHT``) is abandoned and released when its outcome
        arrives, never quiesced here: that would block the engine thread on a transfer. One that
        ended (``LANDED``, fatally ``FAILED``) reaches its release point now. Anything else names
        no page: its landing, if any, goes at once (``Landing.release`` allows it before the
        outcome); the record itself goes too if it is rank-local already (expired), else it
        stays to vote until the ranks agree, so that every rank terminates the request in the
        same round."""
        if rec.state is RecordState.IN_FLIGHT:
            return
        if rec.state in (RecordState.LANDED, RecordState.FAILED):
            if self._quiesce(rec):
                self._close_routes(rec)
                self._release(rec)
            return
        self._release_landing(rec)
        if rec.expired:
            self._release(rec)
        else:
            self._bound_wait_for_verdict(rec, now)

    def _end_publish_of_finished_request(self, rec: TransferRecord, now: float) -> None:
        """A publish in flight settles on its outcome. One nothing was ever submitted for is
        released now: no rank submitted either, since every rank offers the same pieces. One
        some publisher refused stays to vote FAILED until the ranks agree: a peer's may be in
        flight, and the store is missing this rank's part. A ``LANDED`` or ``FAILED`` record
        still here did not pass its release point (the backend refused to vouch for the pages)
        and stays."""
        if rec.state is RecordState.PLANNED and not rec.attempts and not rec.rejected:
            self._release(rec)
            return
        self._bound_wait_for_verdict(rec, now)

    def _bound_wait_for_verdict(self, rec: TransferRecord, now: float) -> None:
        """A record kept at its request's end waits for the ranks under a deadline, like any
        other, so a peer that stopped voting cannot hold the request forever."""
        if rec.deadline is None:
            rec.deadline = self._deadline_for(rec, now)

    def has_backend_work(self) -> bool:
        """Some backend is working for this coordinator: a delivery into pages, or a landing."""
        return any(self._backend_busy_on(rec) for rec in self._records.values())

    def has_pending_work(self) -> bool:
        """Something here will move without a new request: a backend is working, a fetch waits
        for the scheduler's pages, or a record of a finished request waits for the ranks'
        agreement. The engine must not sleep on its request queue, and should yield briefly on
        an idle pass, while this is true."""
        return any(
            self._backend_busy_on(rec)
            or self._idle_with_plan(rec)
            or rec.request_id in self._finished
            for rec in self._records.values()
        )

    def inflight_request_ids(self) -> frozenset[int]:
        """Requests whose pages a backend may still touch: a record ``IN_FLIGHT``, or one whose
        backend refused to vouch for the pages at the release point."""
        return frozenset(
            rec.request_id
            for rec in self._records.values()
            if rec.state is RecordState.IN_FLIGHT or rec.quiesce_refused
        )

    # ---- what the engine's release gate and cancel path read (design §4.3) ----

    def parked_request_ids(self) -> frozenset[int]:
        """Running requests whose fetch is in flight: out of the scheduler's reach until it lands
        or is given back. A finished request with a fetch in flight is held, not parked."""
        return frozenset(
            rec.request_id
            for rec in self._records.values()
            if rec.direction == "fetch"
            and rec.state is RecordState.IN_FLIGHT
            and rec.request_id not in self._held
        )

    def held_request_ids(self) -> frozenset[int]:
        """Finished requests held while a transfer still touches their pages or a record of
        theirs still waits for the ranks' agreement; each is terminated by this coordinator
        through ``terminate_request`` once every record of it is gone."""
        return frozenset(self._held)

    def owned_requests(self) -> list[RequestView]:
        """Every request this coordinator owns right now: parked or held."""
        owned = self.parked_request_ids() | self.held_request_ids()
        return [self._requests[rid] for rid in sorted(owned) if rid in self._requests]

    def status_dump(self) -> dict:
        return {
            "plan_authority": self._plan_authority.value,
            "records": [
                {
                    "request_id": rec.request_id,
                    "direction": rec.direction,
                    "state": rec.state.value,
                    "try_index": rec.try_index,
                    "attempts": len(rec.attempts),
                    "outcomes": [type(a.outcome).__name__ for a in rec.attempts if a.outcome],
                    "deadline": rec.deadline,
                    "expired": rec.expired,
                    "token_end": rec.plan.token_end if rec.plan else None,
                    "launch_gave_up": rec.launch_gave_up,
                    "peer_launched_at": rec.peer_launched_at,
                    "has_landing": rec.landing is not None,
                    "waiting_since": rec.waiting_since,
                }
                for rec in self._records.values()
            ],
            "decided_plans": len(self._plans),
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
        for key, rec in self._records.items():
            vote = self._vote(rec, now)
            if vote is None:
                continue
            if rec.expired:
                if vote.kind is not VoteKind.INFLIGHT:
                    ended_locally.append((rec, vote))
                continue
            if self._deadline_passed(rec, now):
                expired.append(key)
            votes[key] = vote
        self._end_expired_records(ended_locally)
        return votes, expired

    def _end_expired_records(self, ended: Sequence[tuple[TransferRecord, Vote]]) -> None:
        """After the vote: end every expired record whose attempts are over, on its own word."""
        for rec, vote in ended:
            self._end_expired_locally(rec, vote)

    def _vote(self, rec: TransferRecord, now: float) -> Vote | None:
        """This rank's word on one record, or ``None`` for a record with nothing to say: a
        publish not submitted yet, a fetch landed in the pages and waiting for its request's
        end, or a record whose pages the backend refused to vouch for (nothing is agreed on a
        poisoned page; the engine is fatal)."""
        if rec.quiesce_refused:
            return None
        if rec.state is RecordState.IN_FLIGHT:
            self._poll(rec)
            return self._inflight_vote(rec)
        if rec.direction == "publish" and rec.rejected:
            # Nothing of this rank's part reached the store: the publish has failed for every
            # rank, whether its request still runs or has ended.
            return Vote(VoteKind.FAILED)
        if rec.request_id in self._finished:
            return self._finished_vote(rec)
        if rec.state is RecordState.STAGING:
            return self._landing_vote(rec)
        if self._idle_with_plan(rec):
            return self._unlaunched_vote(rec, now)
        return None

    @staticmethod
    def _backend_busy_on(rec: TransferRecord) -> bool:
        """A backend is working for this record: attempts in the pages, or a landing on its way.
        A finished request's landing is given back at once, so its STAGING record runs nothing."""
        if rec.state is RecordState.STAGING:
            return rec.landing is not None
        return rec.state is RecordState.IN_FLIGHT

    @staticmethod
    def _poll(rec: TransferRecord) -> None:
        for a in rec.current_try_attempts():
            if a.outcome is None:
                a.outcome = a.attempt.poll()

    def _inflight_vote(self, rec: TransferRecord) -> Vote:
        """INFLIGHT while any attempt of this rank is still running, so that a verdict never
        quiesces under a live attempt; once every attempt has its outcome, FAILED or TERMINAL."""
        if not rec.is_terminal():
            return Vote(VoteKind.INFLIGHT)
        if rec.direction == "publish":
            # A pipelined publish has failed if any piece did or was refused; it lands once the
            # last piece has been offered, or once its request ended: no more pieces come then.
            if rec.rejected or rec.any_failed():
                return Vote(VoteKind.FAILED)
            if rec.extent.is_last or rec.request_id in self._finished:
                return Vote(VoteKind.TERMINAL)
            return Vote(VoteKind.INFLIGHT)
        if rec.any_failed():
            return Vote(VoteKind.FAILED)
        # The retry aims no higher than what arrived: the probe answer is cached on the record,
        # so a retry above B would ask again for the very units that just came up short. A unit
        # the local cache committed while the fetch was on its way was never asked for, and
        # counts as arrived.
        token_end = merge(rec.plan, rec.merged_served() | rec.committed_names)
        return Vote(VoteKind.TERMINAL, token_end)

    @staticmethod
    def _finished_vote(rec: TransferRecord) -> Vote:
        """A record of a finished request with nothing running in the pages here: TERMINAL, so
        that it holds no rank up and fails none. A fetch votes its plan's target, so a peer still
        delivering lands on its own word; without a plan (dropped after a failed try) no peer is
        delivering, and B does not matter."""
        if rec.direction == "fetch" and rec.plan is not None:
            return Vote(VoteKind.TERMINAL, rec.plan.token_end)
        return Vote(VoteKind.TERMINAL)

    @staticmethod
    def _landing_vote(rec: TransferRecord) -> Vote:
        """The ``STAGING`` counterpart of ``_inflight_vote``: the landing's outcome is the vote.
        Every rank lands its own layer groups, so the ranks take MIN(B) as for a delivery."""
        outcome = rec.landing.poll()
        if outcome is None:
            return Vote(VoteKind.INFLIGHT)
        if is_failure(outcome):
            return Vote(VoteKind.FAILED)
        served = outcome.served if isinstance(outcome, Delivered) else frozenset()
        token_end = merge(rec.plan, served)
        return Vote(VoteKind.TERMINAL, token_end)

    def _unlaunched_vote(self, rec: TransferRecord, now: float) -> Vote:
        """A planned record without an attempt here: UNLAUNCHED holds the others until this rank
        gets what it waits for (the scheduler's pages; for a host-first fetch, the landing memory
        first), FAILED once it gave up launching or waited too long by either clock: the peers'
        (``unlaunched_timeout_s``) or its own (``landing_wait_timeout_s``)."""
        if (
            rec.launch_gave_up
            or self._unlaunched_too_long(rec, now)
            or self._waited_too_long(rec, now)
        ):
            return Vote(VoteKind.FAILED)
        return Vote(VoteKind.UNLAUNCHED)

    def _unlaunched_too_long(self, rec: TransferRecord, now: float) -> bool:
        return (
            self._unlaunched_timeout_s is not None
            and rec.peer_launched_at is not None
            and now - rec.peer_launched_at >= self._unlaunched_timeout_s
        )

    def _waited_too_long(self, rec: TransferRecord, now: float) -> bool:
        return (
            self._landing_wait_timeout_s is not None
            and rec.waiting_since is not None
            and now - rec.waiting_since >= self._landing_wait_timeout_s
        )

    @staticmethod
    def _deadline_passed(rec: TransferRecord, now: float) -> bool:
        return rec.deadline is not None and now >= rec.deadline

    def _deadline_for(self, rec: TransferRecord, now: float) -> float | None:
        timeout = self._fetch_timeout_s if rec.direction == "fetch" else self._publish_timeout_s
        return None if timeout is None else now + timeout

    @staticmethod
    def _idle_with_plan(rec: TransferRecord) -> bool:
        """A fetch record with a plan and nothing running for it here: PLANNED (the pages are not
        reserved yet, the launch never started, or the landing memory was refused) or STAGED
        (landed on the host, the pages not reserved yet). Attempts of earlier tries may remain
        on the record."""
        return (
            rec.direction == "fetch"
            and rec.state in (RecordState.PLANNED, RecordState.STAGED)
            and rec.plan is not None
        )

    def _wants_pages_now(self, rec: TransferRecord) -> bool:
        """Whether the scheduler should reserve pages for the record: a device-direct plan not
        launched yet, or a host-first plan whose landing is complete. A host-first plan still
        PLANNED wants the backend's memory first, not pages."""
        if rec.plan is None or rec.launch_gave_up:
            return False
        if rec.state is RecordState.STAGED:
            return True
        return rec.state is RecordState.PLANNED and not self._lands_on_host(rec)

    def _lands_on_host(self, rec: TransferRecord) -> bool:
        return rec.plan.source in self._host_sources

    def _end_expired_locally(self, rec: TransferRecord, vote: Vote) -> None:
        """An expired record whose attempts are over: ended on this rank's word alone. Its
        request, for a fetch, already failed at the expiry."""
        if rec.direction == "publish":
            self._finish_publish(rec, failed=vote.kind is VoteKind.FAILED)
        else:
            self._release_finished_fetch(rec)

    # ---- advance, phase 2: plan ----

    def _plan(self, candidates: Sequence[RequestView], now: float) -> dict[int, _PlanAnswer]:
        answers: dict[int, _PlanAnswer] = {}
        if self._plan_authority is PlanAuthority.FOLLOWER:
            # The owner plans; its answers arrive with the schedule (``adopt_plan_answers``).
            return answers
        for req in candidates:
            rid = req.py_request_id
            rec = self._records.get((rid, "fetch"))
            # A record that already has a plan is not planned again, even when the scheduler is
            # answered DEFER for it because this rank gave up launching: what comes next is for
            # the ranks to agree on in the apply phase, not for this rank to decide alone.
            if rec is not None and (rec.state is not RecordState.PLANNED or rec.plan is not None):
                continue
            self._requests[rid] = req
            self._probe(req)
            answers[rid] = self._planner.decide(
                req,
                self._probe_answers.get(rid, {}),
                now=now,
                retry_hint=rec.retry_hint if rec is not None else None,
            )
        return answers

    def _probe(self, req: RequestView) -> None:
        """Ask every store source the request has no answer from yet what it holds, and cache
        each answer that is in on the request for the planner."""
        stores = [s for s in self._sources.values() if s.hint_key is None]
        if not stores:
            return
        cache = self._probe_answers.setdefault(req.py_request_id, {})
        unanswered = [s for s in stores if cache.get(s.name) is None]
        if not unanswered:
            return
        query = self._planner.probe_query(req)
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
        answers: Mapping[int, _PlanAnswer],
        now: float,
    ) -> tuple[dict[RecordKey, _Verdict], list[RecordKey], dict[int, _PlanAnswer]]:
        voted = self._plan_authority is PlanAuthority.VOTED
        message = _RoundMessage(
            votes=[(key, vote.kind.value, vote.token_end) for key, vote in sorted(votes.items())],
            expired=sorted(expired),
            plans=(
                [(rid, _encode_plan_answer(ans)) for rid, ans in sorted(answers.items())]
                if voted
                else []
            ),
        )
        # A plain tuple goes out (no class tag in the pickle); a peer's word comes back as
        # whatever 3-sequence the transport made of it.
        gathered = [_RoundMessage._make(word) for word in self._dist.allgather(tuple(message))]
        ballots = _ballots_by_key(gathered)
        self._note_peer_launches(votes, ballots, now)
        verdicts = _reduce_votes(ballots, len(gathered))
        expired_all = {tuple(key) for word in gathered for key in word.expired}
        # An owner's answers are its own word; they reach the followers with the schedule.
        consensus = _reduce_plans(gathered, answers) if voted else dict(answers)
        return verdicts, sorted(expired_all), consensus

    def _note_peer_launches(
        self, votes: Mapping[RecordKey, Vote], ballots: Mapping[RecordKey, list[Vote]], now: float
    ) -> None:
        """Start the unlaunched clock of every fetch this rank has not launched while some other
        rank has (its vote is anything but UNLAUNCHED). The collective carries no rank identity,
        so the kinds of the votes are all there is to read."""
        for key, vote in votes.items():
            if vote.kind is not VoteKind.UNLAUNCHED:
                continue
            rec = self._records[key]
            if rec.peer_launched_at is None and any(
                peer.kind is not VoteKind.UNLAUNCHED for peer in ballots.get(key, ())
            ):
                rec.peer_launched_at = now

    # ---- advance, phase 4: apply ----

    def _apply(
        self,
        verdicts: Mapping[RecordKey, _Verdict],
        expired: Sequence[RecordKey],
        answers: Mapping[int, _PlanAnswer],
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
            rec = self._records.get(key)
            if rec is not None and not rec.expired:
                self._expire(rec, now)

    def _apply_verdicts(self, verdicts: Mapping[RecordKey, _Verdict], now: float) -> None:
        """End, land or re-plan every record the ranks reached a verdict on."""
        for key in sorted(verdicts):
            rec = self._records.get(key)
            if rec is None:
                continue
            token_end, failed = verdicts[key]
            if rec.direction == "publish":
                self._finish_publish(rec, failed=failed)
            elif rec.request_id in self._finished:
                # The request is gone; the agreement only says every rank releases it now.
                self._release_finished_fetch(rec)
            elif self._backend_busy_on(rec):
                self._apply_fetch_verdict(rec, token_end, failed, now)
            elif self._idle_with_plan(rec):
                # An UNLAUNCHED vote blocks a landing, so the only verdict that reaches an
                # unlaunched record is a failure: drop the plan without touching pages.
                self._reset_for_replan(rec)

    def _apply_plan_answers(self, answers: Mapping[int, _PlanAnswer], now: float) -> None:
        """Record every decided answer; an OWNER also keeps them for ``export_plan_answers``."""
        self._answers_to_export = {}
        for rid, ans in answers.items():
            if ans is DEFER:
                continue
            self._record_answer(rid, ans, now)
            if self._plan_authority is PlanAuthority.OWNER:
                self._answers_to_export[rid] = ans

    def _record_answer(self, rid: int, ans: FetchPlan | None, now: float) -> None:
        """Write a decided answer: ``None`` releases a planned record, a plan goes on the record
        (created if needed) and ``plan_fetch`` reads it from now on. A plan from a host-first
        source is started right here, since it needs no pages to begin; any other waits for the
        scheduler's pages from now, on the wait clock."""
        self._plans[rid] = ans
        key = (rid, "fetch")
        rec = self._records.get(key)
        if ans is None:
            if rec is not None and rec.state is RecordState.PLANNED:
                self._release(rec)
            return
        if rec is None:
            rec = TransferRecord(rid, "fetch", RecordState.PLANNED)
            self._records[key] = rec
        rec.plan = ans
        rec.retry_hint = None
        if self._lands_on_host(rec):
            self._start_host_landing(rec, now)
        else:
            rec.waiting_since = now

    def _start_host_landing(self, rec: TransferRecord, now: float) -> None:
        """Phase one of a host-first fetch: ask the backend to land the plan's units in its own
        memory. No page is involved, so a refusal costs nothing but time: the record stays
        PLANNED, its wait is clocked from the first refusal, and the next ``advance`` asks again.
        It is not a failed launch (``consecutive_launch_failures`` is for page back-pressure)."""
        source = self._sources[rec.plan.source]
        try:
            landing = source.backend.fetch_to_host(
                f"fetch:{rec.request_id}".encode(), unit_names(rec.plan)
            )
        except SubmissionRejected as exc:
            logger.info("request %d: landing refused by %s: %s", rec.request_id, source.name, exc)
            if rec.waiting_since is None:
                rec.waiting_since = now
            return
        rec.landing = landing
        rec.waiting_since = None
        rec.peer_launched_at = None
        rec.state = RecordState.STAGING
        rec.deadline = self._deadline_for(rec, now)

    def _retry_refused_landings(self, now: float) -> None:
        """Ask again for the landing memory of every host-first plan the backend refused."""
        for rec in self._records.values():
            if (
                rec.direction == "fetch"
                and rec.state is RecordState.PLANNED
                and rec.plan is not None
                and rec.request_id not in self._finished
                and self._lands_on_host(rec)
            ):
                self._start_host_landing(rec, now)

    def _expire(self, rec: TransferRecord, now: float) -> None:
        """Some rank saw the record's deadline pass: from now on this rank ends it alone, once
        its own attempts are over (never under a live attempt: quiesce would block the engine
        thread). A publish is a warning, not a request failure. A fetch fails its request, unless
        the request ended already and the record is only waiting for its outcome."""
        rec.expired = True
        if rec.direction == "publish":
            # A long chunked prefill is still offering pieces: not worth a warning until the
            # last piece is out and the store has had its full say.
            log = logger.warning if rec.extent.is_last else logger.debug
            log(
                "request %d: kv publish past its deadline; ended here once its attempts are over",
                rec.request_id,
            )
        elif rec.request_id not in self._finished:
            self._fail_expired_fetch(rec, now)

    def _apply_fetch_verdict(
        self, rec: TransferRecord, token_end: int, failed: bool, now: float
    ) -> None:
        if failed:
            self._fetch_failed_or_short(rec, hint=None, reason="kv fetch failed")
        elif token_end == rec.plan.token_end:
            if rec.state is RecordState.STAGING:
                self._mark_staged(rec, now)
            else:
                self._land_in_pages(rec)
        else:
            # The agreed MIN(token_end) is as far as every rank got: the retry aims no higher.
            self._fetch_failed_or_short(rec, hint=token_end, reason="kv fetch served short")

    @staticmethod
    def _mark_staged(rec: TransferRecord, now: float) -> None:
        """The ranks agreed the landing in the host memory is complete: the record waits for
        pages (``STAGED``), on the wait clock alone."""
        rec.state = RecordState.STAGED
        rec.waiting_since = now
        rec.peer_launched_at = None
        rec.deadline = None  # the placement sets its own

    def _land_in_pages(self, rec: TransferRecord) -> None:
        """The ranks agreed the delivery into the pages is complete: the request is unparked and
        the landing, if any, is given back at once (the copy was complete before its outcome was
        reported); the record itself stays until the request ends."""
        rec.state = RecordState.LANDED
        self._close_routes(rec)
        req = self._requests[rec.request_id]
        self._plans[rec.request_id] = None
        self._effects.unpark(req, rec.plan.token_end, rec.plan.no_local_fallback, self._aux(rec))
        self._release_landing(rec)

    def _fetch_failed_or_short(self, rec: TransferRecord, *, hint: int | None, reason: str) -> None:
        """The ranks agreed the delivery failed or came up short. A landing on its way to the
        host names no page: nothing to quiesce or give back, the landing goes and the plan is
        retried or given up. A delivery into the pages passes the release point first."""
        was_staging = rec.state is RecordState.STAGING
        rec.state = RecordState.FAILED
        if was_staging:
            self._retry_or_give_up(rec, hint=hint, reason=reason)
            return
        if not self._quiesce(rec):
            return
        self._close_routes(rec)
        req = self._requests[rec.request_id]
        self._effects.give_back_fetch_pages([req])
        self._retry_or_give_up(rec, hint=hint, reason=reason)

    def _reset_for_replan(self, rec: TransferRecord) -> None:
        """The ranks agreed the fetch failed while this rank never launched it: same next step as
        after a failed attempt, minus the release point (no attempt, no pages to give back)."""
        reason = "kv fetch launch given up" if rec.launch_gave_up else "kv fetch failed"
        self._retry_or_give_up(rec, hint=None, reason=reason)

    def _retry_or_give_up(self, rec: TransferRecord, *, hint: int | None, reason: str) -> None:
        """After a failure: a gen-init fetch fails its request; otherwise spend a retry and plan
        again next round, or, out of retries, compute the rest locally."""
        req = self._requests[rec.request_id]
        if rec.plan.no_local_fallback:
            self._release(rec)
            self._plans[rec.request_id] = None
            self._effects.fail_requests([req], reason)
        elif rec.retries_left > 0:
            rec.retries_left -= 1
            rec.retry_hint = hint
            self._release_landing(rec)
            rec.plan = None
            rec.extent = None
            rec.deadline = None
            rec.launch_gave_up = False
            rec.consecutive_launch_failures = 0
            rec.peer_launched_at = None
            rec.state = RecordState.PLANNED
            self._plans.pop(rec.request_id, None)
        else:
            self._release(rec)
            self._plans[rec.request_id] = None

    def _fail_expired_fetch(self, rec: TransferRecord, now: float) -> None:
        """A fetch of a running request past its deadline: fail the request now.

        A hung store must not park a request forever, so the wait is bounded by the deadline
        and the request fails. A delivery into the pages cannot give them back yet: the backend
        may still be writing them, so the record stays in flight and the request is held until
        the outcome arrives (or ``close``); the late outcome then only releases. A record with
        nothing in the pages (planned, landing on the host, landed there) goes at once.
        """
        req = self._requests[rec.request_id]
        self._effects.fail_requests([req], "kv fetch timed out")
        # The engine ends a failed request through its release gate, which notifies this
        # coordinator; the call is repeated here so the hold does not depend on the engine.
        self._finish_request(req, now)

    def _release_finished_fetch(self, rec: TransferRecord) -> None:
        """A fetch whose request ended: its outcome is moot. Pass the release point if the
        current try touched the pages, release; no ``unpark`` and no ``give_back`` for a request
        that is gone. The held request is terminated once nothing else of it remains."""
        if self._owes_quiesce(rec) and not self._quiesce(rec):
            return
        self._close_routes(rec)
        self._release(rec)
        self._finish_if_released(rec.request_id)

    def _finish_publish(self, rec: TransferRecord, *, failed: bool) -> None:
        """A publish reached its end. A failure is a warning, never a request failure: a
        running request keeps running, and a finished one already has its response."""
        rec.state = RecordState.FAILED if failed else RecordState.LANDED
        if self._owes_quiesce(rec) and not self._quiesce(rec):
            return
        self._release(rec)
        rid = rec.request_id
        if failed:
            when = "after the request ended" if rid in self._finished else "while still running"
            logger.warning("request %d: kv publish failed %s", rid, when)
        self._finish_if_released(rid)

    def _finish_if_released(self, rid: int) -> None:
        """A finished request's last word to the engine, once no record of it remains.

        A held request is terminated here, whatever its last transfer's outcome. A request that
        was never held owes the engine nothing: it terminates as usual through the release gate.
        """
        if rid not in self._finished or self._has_records_of(rid):
            return
        req = self._requests.get(rid)
        if req is None:
            logger.warning("request %d finished with no request object on record", rid)
        elif rid in self._held:
            if rid not in self._finished_by_gate:
                self._terminated_before_gate.add(rid)
            self._effects.terminate_request(req)
        self._forget_request(rid)

    # ---- launch / publish helpers ----

    def _launch_or_place(self, req: RequestView, rec: TransferRecord, now: float) -> bool:
        if rec.state is RecordState.STAGED:
            return self._place_one(req, rec, now)
        return self._launch_one(req, rec, now)

    def _launch_one(self, req: RequestView, rec: TransferRecord, now: float) -> bool:
        """Start the delivery of a device-direct plan into the pages the scheduler reserved;
        False when the launch did not start and the pages went back."""
        plan = rec.plan
        rid = req.py_request_id
        source = self._sources[plan.source]
        extent, committed = self._reader.fetch_extent(req, plan)
        route = None
        if source.hint_key is not None and plan.hint is not None:
            route = self._open_route_or_drop(req, rec, source)
            if route is None:
                return False
        try:
            attempt = source.backend.fetch(extent, route=route)
        except SubmissionRejected as exc:
            # Nothing escaped, so nothing to quiesce: give the pages back and try the same plan
            # again next round (this is the one back-pressure signal).
            logger.info("request %d: fetch rejected by %s: %s", rid, source.name, exc)
            if route is not None:
                route.close()
            self._drop_launch(req, rec, give_up=False)
            return False
        self._start_try(rec, attempt, extent, committed, source.name, route, now)
        return True

    def _open_route_or_drop(
        self, req: RequestView, rec: TransferRecord, source: FetchSource
    ) -> Route | None:
        """The route to the plan's hint, or ``None`` once the launch has been dropped: a refused
        route means the plan can never work here (give up), a failed one is worth another try.
        ``open_route`` itself always returns a ``Route``; a ``None`` from it would break the
        contract."""
        rid = req.py_request_id
        # Broad on purpose, against CODING_GUIDELINES: ``open_route`` raises the backend's own
        # transport error for a hint that was fine but could not be prepared; that is worth
        # trying again, and it must not escape and strand the other requests in the queue.
        try:
            return source.backend.open_route(rec.plan.hint)
        except (ValueError, NotImplementedError) as exc:
            # Bad hint, or a backend that cannot route: this plan can never work here.
            logger.warning("request %d: route refused by %s: %s", rid, source.name, exc)
            self._drop_launch(req, rec, give_up=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("request %d: route to %s failed: %s", rid, source.name, exc)
            self._drop_launch(req, rec, give_up=False)
        return None

    def _place_one(self, req: RequestView, rec: TransferRecord, now: float) -> bool:
        """Phase two of a host-first fetch, once the scheduler reserved the pages: copy the
        landing into them. From here on the record is an ordinary delivery into pages; a refusal
        is page back-pressure and is counted as one."""
        rid = req.py_request_id
        source = self._sources[rec.plan.source]
        extent, committed = self._reader.fetch_extent(req, rec.plan)
        try:
            attempt = rec.landing.place(extent)
        except SubmissionRejected as exc:
            logger.info("request %d: placement rejected by %s: %s", rid, source.name, exc)
            self._drop_launch(req, rec, give_up=False)
            return False
        self._start_try(rec, attempt, extent, committed, source.name, None, now)
        return True

    def _start_try(
        self,
        rec: TransferRecord,
        attempt: Attempt,
        extent: CacheExtent,
        committed: frozenset[bytes],
        source: str,
        route: Route | None,
        now: float,
    ) -> None:
        """A delivery into the pages has started: the record is IN_FLIGHT on a new try, with the
        deadline counted from now."""
        rec.consecutive_launch_failures = 0
        rec.peer_launched_at = None
        rec.waiting_since = None
        try_index = rec.try_index + 1 if rec.attempts else 0
        rec.extent = extent
        rec.committed_names = committed
        rec.attempts.append(AttemptRecord(attempt, try_index=try_index, source=source, route=route))
        rec.state = RecordState.IN_FLIGHT
        rec.deadline = self._deadline_for(rec, now)

    def _drop_launch(self, req: RequestView, rec: TransferRecord, *, give_up: bool) -> None:
        """A launch that never started: give the pages back, keep the plan and the retry budget.

        The plan stays so the scheduler reserves for it again next round; whether the rank
        launches then or votes the fetch failed is not decided here. A launch that can never work
        (``give_up``), or one refused ``MAX_CONSECUTIVE_LAUNCH_FAILURES`` times in a row, makes
        the record give up: from then on it votes FAILED and answers the scheduler DEFER until
        the ranks agree, so that a rank never re-plans a fetch on its own.
        """
        self._effects.give_back_fetch_pages([req])
        rec.consecutive_launch_failures += 1
        if give_up or rec.consecutive_launch_failures >= MAX_CONSECUTIVE_LAUNCH_FAILURES:
            rec.launch_gave_up = True
            logger.warning(
                "request %d: giving up on launching the fetch from %s after %d failed launches",
                req.py_request_id,
                rec.plan.source,
                rec.consecutive_launch_failures,
            )

    def _publish_one(self, req: RequestView, now: float) -> None:
        """Offer what the request committed to every publisher, on a publish record created on
        first sight; the record goes IN_FLIGHT with the first submission a publisher accepts."""
        rid = req.py_request_id
        self._requests[rid] = req
        key = (rid, "publish")
        rec = self._records.get(key)
        if rec is None:
            rec = TransferRecord(rid, "publish", RecordState.PLANNED)
            self._records[key] = rec
        elif rec.state not in (RecordState.PLANNED, RecordState.IN_FLIGHT):
            return
        extent, chunk = self._reader.publish_extent_and_chunk(req)
        rec.extent = extent
        self._submit_publish_pieces(rec, extent, chunk)
        if rec.state is RecordState.PLANNED and rec.attempts:
            # The deadline counts from the first submission a publisher accepted.
            rec.state = RecordState.IN_FLIGHT
            rec.deadline = self._deadline_for(rec, now)

    def _submit_publish_pieces(
        self, rec: TransferRecord, extent: CacheExtent, chunk: Chunk | None
    ) -> None:
        """Design §7.5: a publisher that places pieces works in series and hears every piece;
        one that does not is offered the content once, on the last piece, and never has to read
        ``is_last``."""
        for name, publisher in self._publishers.items():
            if isinstance(publisher, PlacesPieces):
                self._submit_publish(rec, name, publisher.publish, extent)
                if chunk is not None:
                    self._submit_publish(rec, name, publisher.place, chunk)
            elif extent.is_last:
                self._submit_publish(rec, name, publisher.publish, extent)

    def _submit_publish(
        self, rec: TransferRecord, name: str, submit: Callable[[T], Attempt], arg: T
    ) -> None:
        try:
            attempt = submit(arg)
        except SubmissionRejected as exc:
            logger.warning("request %d: publish rejected by %s: %s", rec.request_id, name, exc)
            rec.rejected = True
            return
        rec.attempts.append(AttemptRecord(attempt, source=name))

    # ---- record bookkeeping ----

    @staticmethod
    def _owes_quiesce(rec: TransferRecord) -> bool:
        """Whether the current try's attempts touched the pages and were not vouched for yet. A
        fetch quiesces each failed try as it fails, so only a try in flight, landed, or fatally
        failed owes one; a publish has a single try."""
        if not rec.current_try_attempts():
            return False
        if rec.direction == "publish":
            return True
        return rec.state in (RecordState.IN_FLIGHT, RecordState.LANDED, RecordState.FAILED)

    def _quiesce(self, rec: TransferRecord) -> bool:
        """The release point. False means the backend cannot vouch for the memory: the engine is
        made fatal and the record stays, so the pages are never handed out again (design §4.3
        would retry until the deadline; the fatal path is what the engine has). A refusal is
        final: the backend is not asked again, and the engine is not made fatal twice."""
        if rec.quiesce_refused:
            return False
        by_backend: dict[str, list[Attempt]] = {}
        for a in rec.current_try_attempts():
            by_backend.setdefault(a.source, []).append(a.attempt)
        for name, attempts in by_backend.items():
            if not self._backends[name].quiesce(attempts):
                rec.quiesce_refused = True
                self._effects.fail_fatal(
                    RuntimeError(
                        f"backend {name} cannot confirm memory of request {rec.request_id} "
                        f"({rec.direction}) is untouched"
                    )
                )
                return False
        return True

    @staticmethod
    def _close_routes(rec: TransferRecord) -> None:
        for a in rec.current_try_attempts():
            if a.route is not None:
                a.route.close()
                a.route = None

    @staticmethod
    def _aux(rec: TransferRecord) -> Mapping[str, object] | None:
        aux: dict[str, object] = {}
        for a in rec.current_try_attempts():
            if isinstance(a.attempt, CarriesAux):
                aux.update(a.attempt.aux())
        return aux or None

    def _release(self, rec: TransferRecord) -> None:
        """The record leaves the table; release is an event, not a state."""
        self._release_landing(rec)
        self._records.pop(rec.key, None)

    @staticmethod
    def _release_landing(rec: TransferRecord) -> None:
        """Give a host-first landing back and forget what went with it. Idempotent; a no-op for a
        record without one. Called when the delivery is complete (after ``unpark``), when the
        record leaves the table, when the plan is dropped for a retry, and at the end of a
        request whose landing is still on its way."""
        if rec.landing is not None:
            rec.landing.release()
        rec.landing = None
        rec.waiting_since = None
        rec.committed_names = frozenset()

    def _forget_request(self, rid: int) -> None:
        """Drop every per-request entry of a request that is gone, here and in the planner and
        the reader. For a held request this runs at its termination, so the reader's per-request
        cache keeps that one entry until then."""
        self._requests.pop(rid, None)
        self._plans.pop(rid, None)
        self._pending_answers.pop(rid, None)
        self._probe_answers.pop(rid, None)
        self._finished.discard(rid)
        self._finished_by_gate.discard(rid)
        self._held.discard(rid)
        self._planner.forget(rid)
        self._reader.forget_request(rid)

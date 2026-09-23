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
from typing import Callable, Mapping, NamedTuple, Sequence, TypeVar

from ...base.cache_backend import Attempt, Fetches, Publishes, SubmissionRejected
from ...base.views import RequestView, ResourceReader
from ...remote_cache import DEFER, Defer, FetchPlan, FetchSource, Planner, merge
from .interfaces import (
    CarriesAux,
    DistLike,
    EngineQueue,
    KVTransferEffects,
    PlacesPieces,
    PlanAuthority,
)
from .records import AttemptRecord, RecordKey, RecordState, TransferRecord

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

_DEFER_WIRE = "DEFER"
_PlanAnswer = FetchPlan | None | Defer
_PlanWire = tuple[int, str] | str | None
"""A plan answer on the wire: ``(token_end, source)``, ``"DEFER"``, or ``None``."""
PlanAnswers = list[tuple[int, _PlanWire]]
"""Decided answers as the owner hands them to the followers: ``[(request_id, plan wire)]``, never
``"DEFER"``. Built-in types only, so it rides in the pickled schedule."""


class VoteKind(Enum):
    """What one rank says about one record this round. Carried on the wire as the string value.

    UNLAUNCHED: fetch planned but not launched here (no pages yet); holds up a landing, not a failure.
    INFLIGHT: an attempt is still running here; nothing may be decided this round.
    FAILED: decisive: this rank's attempt failed, or it gave up launching.
    TERMINAL: every attempt here ended without failure; carries ``(B, retry_hint)`` for a fetch,
        with ``retry_hint == B`` by construction (the wire keeps both fields).
    """

    UNLAUNCHED = "UNLAUNCHED"
    INFLIGHT = "INFLIGHT"
    FAILED = "FAILED"
    TERMINAL = "TERMINAL"


class Vote(NamedTuple):
    """One rank's word on one record. ``b`` and ``hint`` matter for ``TERMINAL`` fetches only."""

    kind: VoteKind
    b: int = 0
    hint: int = 0


_Verdict = tuple[int, int, bool]
"""``(MIN(B), MIN(retry_hint), failed)``: what the ranks agreed on for one record. Every rank votes
``retry_hint == B``, so the two minima agree; both are kept for the wire shape."""
_Payload = tuple[list, list, list]
"""``([(key, kind, B, hint)], [expired key], [(rid, plan wire)])``: one rank's word per round."""


def _wire(answer: _PlanAnswer) -> _PlanWire:
    if answer is DEFER:
        return _DEFER_WIRE
    if answer is None:
        return None
    return (answer.token_end, answer.source)


def _ballots_by_key(gathered: Sequence[_Payload]) -> dict[RecordKey, list[Vote]]:
    """Every rank's vote on every record, in gathered order; keys come back as tuples."""
    ballots: dict[RecordKey, list[Vote]] = {}
    for votes, _, _ in gathered:
        for key, kind, b, hint in votes:
            ballots.setdefault(tuple(key), []).append(Vote(VoteKind(kind), b, hint))
    return ballots


def _reduce_votes(ballots: Mapping[RecordKey, list[Vote]], n: int) -> dict[RecordKey, _Verdict]:
    """The one reduction of design §7.1 "齐", for fetches and publishes alike.

    A record is decided only once all ``n`` ranks voted on it, and then in this order: any
    INFLIGHT holds the round (a failure landed now would quiesce under a running attempt and
    block the engine thread); else any FAILED is decisive for every rank, delivered data
    included; else any UNLAUNCHED holds the round (a landing needs every rank's pages); else
    every rank is TERMINAL and the landing takes MIN(B), MIN(hint), since ranks hold different
    layer groups.
    """
    verdicts: dict[RecordKey, _Verdict] = {}
    for key, votes in ballots.items():
        if len(votes) < n:
            continue
        kinds = {vote.kind for vote in votes}
        if VoteKind.INFLIGHT in kinds:
            continue
        if VoteKind.FAILED in kinds:
            verdicts[key] = (0, 0, True)
        elif VoteKind.UNLAUNCHED not in kinds:
            verdicts[key] = (min(v.b for v in votes), min(v.hint for v in votes), False)
    return verdicts


def _reduce_plans(
    gathered: Sequence[_Payload], answers: Mapping[int, _PlanAnswer]
) -> dict[int, _PlanAnswer]:
    """Any DEFER -> DEFER; otherwise any disagreement on ``(token_end, source)`` -> None."""
    n = len(gathered)
    wires: dict[int, list[_PlanWire]] = {}
    for _, _, plans in gathered:
        for rid, wire in plans:
            wires.setdefault(rid, []).append(tuple(wire) if isinstance(wire, list) else wire)
    consensus: dict[int, _PlanAnswer] = {}
    for rid, local in answers.items():
        v = wires.get(rid, [])
        if len(v) < n or _DEFER_WIRE in v:
            consensus[rid] = DEFER
        elif all(x == v[0] for x in v) and _wire(local) == v[0]:
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
        fetch_timeout_s: Deadline for a fetch, from launch. ``None`` disables.
        publish_timeout_s: Deadline for a publish, from its first submission. ``None`` disables.
        unlaunched_timeout_s: Longest this rank may stay unlaunched on a fetch another rank has
            already launched before it votes the fetch failed. ``None`` disables. Never starts
            counting on a single rank.
        plan_authority: Who decides plan answers on this rank. ``VOTED``: planned here and
            reduced in the collective. ``OWNER``: planned here alone, not carried in the
            payload, handed out with ``export_plan_answers``. ``FOLLOWER``: never planned or
            probed here; taken with ``adopt_plan_answers``. Votes and expiries travel through
            the collective in every mode.
        queue_budget: How many posted callables one ``advance`` runs.

    The engine must call ``notify_request_finished`` for every request it ever passed in, so that
    fetch records reach their release point and per-request state is dropped.
    """

    def __init__(
        self,
        sources: Sequence[FetchSource],
        publishers: Sequence[Publishes],
        planner: Planner,
        reader: ResourceReader,
        effects: KVTransferEffects,
        queue: EngineQueue,
        dist: DistLike,
        *,
        fetch_timeout_s: float | None = None,
        publish_timeout_s: float | None = None,
        unlaunched_timeout_s: float | None = 30.0,
        plan_authority: PlanAuthority = PlanAuthority.VOTED,
        queue_budget: int = 64,
    ) -> None:
        self._sources: dict[str, FetchSource] = {s.name: s for s in sources}
        self._publishers: dict[str, Publishes] = {
            f"publish:{i}": p for i, p in enumerate(publishers)
        }
        self._backends: dict[str, Fetches | Publishes] = {
            **{name: s.backend for name, s in self._sources.items()},
            **self._publishers,
        }
        self._planner = planner
        self._reader = reader
        self._effects = effects
        self._queue = queue
        self._dist = dist
        self._fetch_timeout_s = fetch_timeout_s
        self._publish_timeout_s = publish_timeout_s
        self._unlaunched_timeout_s = unlaunched_timeout_s
        self._plan_authority = plan_authority
        self._queue_budget = queue_budget

        self._records: dict[RecordKey, TransferRecord] = {}
        self._requests: dict[int, RequestView] = {}
        self._plans: dict[int, FetchPlan | None] = {}
        """Decided answers ``plan_fetch`` reads. A request with no entry is undecided (DEFER)."""
        self._answers_to_export: dict[int, FetchPlan | None] = {}
        """OWNER only: the answers the last ``advance`` decided, for ``export_plan_answers``."""
        self._pending_answers: dict[int, _PlanWire] = {}
        """FOLLOWER only: adopted answers whose request is not a candidate here yet."""
        self._probe_answers: dict[int, dict[str, frozenset[bytes] | None]] = {}
        self._finished: set[int] = set()
        self._held: set[int] = set()
        """Finished requests the engine was told to hold; each owes a ``terminate_request``."""

    # ---- loop entry points (each rank calls each the same number of times per round) ----

    def advance(self, candidates: Sequence[RequestView], now: float) -> int:
        """Head of the loop: reap outcomes into votes, plan candidates, agree across ranks, apply.

        Returns how many candidates are still undecided afterwards (``plan_fetch`` answers
        ``DEFER``): each is waiting on a store lookup, or on the ranks' agreement.
        """
        votes, expired = self._reap(now)
        answers = self._plan(candidates, now)
        verdicts, expired, answers = self._sync(votes, expired, answers, now)
        self._apply(verdicts, expired, answers)
        return sum(1 for candidate in candidates if self.plan_fetch(candidate) is DEFER)

    def launch_fetches(self, queue: Sequence[RequestView], now: float | None = None) -> None:
        """After scheduling: start the fetch of every request the scheduler allocated for."""
        now = time.monotonic() if now is None else now
        ready = []
        for req in queue:
            rec = self._records.get((req.py_request_id, "fetch"))
            if rec is not None and self._is_launchable(rec):
                ready.append((req, rec))
        if not ready:
            return
        self._effects.prepare_fetch_resources([req for req, _ in ready])
        launched = [req for req, rec in ready if self._launch_one(req, rec, now)]
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
        return [(rid, _wire(ans)) for rid, ans in sorted(self._answers_to_export.items())]

    def adopt_plan_answers(self, views: Sequence[RequestView], answers: PlanAnswers) -> int:
        """FOLLOWER: take the owner's answers. ``None`` decides a request local; ``(token_end,
        source)`` becomes this rank's own plan over its own layer groups. An answer for a request
        not yet among ``views`` (the undecided candidates here) is held until its request appears
        as a candidate, or is dropped when the request ends: the owner decides a request once and
        does not export it again. Returns how many of ``views`` got no answer and stay deferred."""
        by_rid = {view.py_request_id: view for view in views}
        self._pending_answers.update(answers)
        for rid, wire in list(self._pending_answers.items()):
            view = by_rid.get(rid)
            if view is None:
                continue
            del self._pending_answers[rid]
            self._requests[rid] = view
            if wire is None:
                self._decide(rid, None)
            else:
                token_end, source = wire
                self._decide(rid, self._planner.materialize(view, token_end, source))
        answered = {rid for rid, _ in answers}
        return sum(1 for rid in by_rid if rid not in answered)

    # ---- scheduler hook (read-only, non-blocking) ----

    def plan_fetch(self, req: RequestView) -> _PlanAnswer:
        """``FetchPlan`` to fetch, ``None`` to compute locally, ``DEFER`` to skip this round.

        A request the coordinator has not decided yet answers ``DEFER``; the engine passes such
        requests as ``candidates`` to the next ``advance``. So does a request whose rank gave up
        launching its plan: the ranks have yet to agree on what comes next, and meanwhile the
        scheduler must neither reserve pages for it nor plan it locally. A request with a fetch
        record past ``PLANNED`` has nothing more to plan and answers ``None``.
        """
        rid = req.py_request_id
        rec = self._records.get((rid, "fetch"))
        if rec is not None:
            if rec.state is not RecordState.PLANNED:
                return None
            return rec.plan if self._is_launchable(rec) else DEFER
        if rid in self._plans:
            return self._plans[rid]
        return DEFER

    # ---- control ----

    def notify_request_finished(self, req: RequestView) -> None:
        """The request ended. A fetch not in flight reaches its release point now; one still in
        flight is abandoned and released when its outcome arrives (never quiesced here: that
        would block the engine thread on a transfer). Any record still in flight holds the
        request, so its pages stay put until the backend is done with them."""
        rid = req.py_request_id
        if rid in self._finished:
            return
        self._finished.add(rid)
        self._requests[rid] = req
        fetch = self._records.get((rid, "fetch"))
        if fetch is not None:
            if fetch.state is RecordState.IN_FLIGHT:
                fetch.abandoned = True
            else:
                # PLANNED with attempts = an earlier try that was quiesced when it failed.
                quiesce_due = fetch.attempts and fetch.state is not RecordState.PLANNED
                if quiesce_due and not self._quiesce(fetch):
                    return
                self._close_routes(fetch)
                self._release(fetch)
        publish = self._records.get((rid, "publish"))
        if publish is not None and publish.state is not RecordState.IN_FLIGHT:
            # Nothing is in flight for it: released now. A publish every publisher refused was
            # warned about at submission and owes the engine nothing more.
            self._release(publish)
        if (rid, "fetch") in self._records or (rid, "publish") in self._records:
            self._held.add(rid)
            self._effects.hold_for_transfer([req])
            return
        self._finish_if_released(rid)

    def has_inflight(self) -> bool:
        return any(rec.state is RecordState.IN_FLIGHT for rec in self._records.values())

    def inflight_request_ids(self) -> frozenset[int]:
        """Requests with a record still ``IN_FLIGHT``: a backend may still touch their pages."""
        return frozenset(
            rec.request_id for rec in self._records.values() if rec.state is RecordState.IN_FLIGHT
        )

    # ---- what the engine's release gate and cancel path read (design §4.3, plan §9 rule 5) ----

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
        """Finished requests held while a transfer still touches their pages; each is terminated
        by this coordinator through ``terminate_request`` once every record of it is gone."""
        return frozenset(self._held)

    def tracked_requests(self) -> list[RequestView]:
        """Every request this coordinator owns right now: parked or held."""
        tracked = self.parked_request_ids() | self.held_request_ids()
        return [self._requests[rid] for rid in sorted(tracked) if rid in self._requests]

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
                    "abandoned": rec.abandoned,
                    "token_end": rec.plan.token_end if rec.plan else None,
                    "launch_gave_up": rec.launch_gave_up,
                    "peer_launched_at": rec.peer_launched_at,
                }
                for rec in self._records.values()
            ],
            "decided_plans": len(self._plans),
            "finished_pending": sorted(self._finished),
        }

    # ---- advance, phase 1: reap ----

    def _reap(self, now: float) -> tuple[dict[RecordKey, Vote], list[RecordKey]]:
        """Poll every attempt in flight, then cast one vote per record that takes part in the
        round. A record of a finished request casts none: its outcome has no effect on the other
        ranks, which may already have released their record or never created it, so it settles
        here as soon as this rank's attempts are over."""
        self._queue.drain(self._queue_budget)
        votes: dict[RecordKey, Vote] = {}
        expired: list[RecordKey] = []
        settled_locally: list[tuple[TransferRecord, Vote]] = []
        for key, rec in self._records.items():
            if rec.state is RecordState.IN_FLIGHT:
                self._poll(rec)
                vote = self._inflight_vote(rec)
                if vote.kind is VoteKind.INFLIGHT and self._deadline_passed(rec, now):
                    expired.append(key)
            elif self._is_unlaunched_fetch(rec):
                vote = self._unlaunched_vote(rec, now)
            else:
                continue
            if self._is_voting(rec):
                votes[key] = vote
            elif rec.state is RecordState.IN_FLIGHT and vote.kind is not VoteKind.INFLIGHT:
                settled_locally.append((rec, vote))
        for rec, vote in settled_locally:
            self._settle_finished_record(rec, vote)
        return votes, expired

    def _is_voting(self, rec: TransferRecord) -> bool:
        """The plan's predicate: a record takes part in the round unless its request is finished,
        and only while in flight or planned with a plan."""
        return rec.request_id not in self._finished and (
            rec.state is RecordState.IN_FLIGHT or self._is_unlaunched_fetch(rec)
        )

    @staticmethod
    def _poll(rec: TransferRecord) -> None:
        for a in rec.current_try_attempts():
            if a.outcome is None:
                a.outcome = a.attempt.poll()

    @staticmethod
    def _inflight_vote(rec: TransferRecord) -> Vote:
        """INFLIGHT while any attempt of this rank is still running, so that a verdict never
        quiesces under a live attempt; once every attempt has its outcome, FAILED or TERMINAL."""
        if not rec.is_terminal():
            return Vote(VoteKind.INFLIGHT)
        if rec.direction == "publish":
            # A pipelined publish has failed if any piece did or was refused; it lands only once
            # the last piece has been offered.
            if rec.rejected or rec.any_failed():
                return Vote(VoteKind.FAILED)
            if rec.extent.is_last:
                return Vote(VoteKind.TERMINAL)
            return Vote(VoteKind.INFLIGHT)
        if rec.any_failed():
            return Vote(VoteKind.FAILED)
        # The retry aims no higher than what arrived: the probe answer is cached on the record,
        # so a retry above B would ask again for the very units that just came up short.
        b = merge(rec.plan, rec.merged_served())
        return Vote(VoteKind.TERMINAL, b, b)

    def _unlaunched_vote(self, rec: TransferRecord, now: float) -> Vote:
        if rec.launch_gave_up or self._unlaunched_too_long(rec, now):
            return Vote(VoteKind.FAILED)
        return Vote(VoteKind.UNLAUNCHED)

    def _unlaunched_too_long(self, rec: TransferRecord, now: float) -> bool:
        return (
            self._unlaunched_timeout_s is not None
            and rec.peer_launched_at is not None
            and now - rec.peer_launched_at >= self._unlaunched_timeout_s
        )

    @staticmethod
    def _deadline_passed(rec: TransferRecord, now: float) -> bool:
        # An abandoned fetch has already had its expiry (or its request ended) and is only
        # waiting for its outcome; abandoning a publish changes nothing about its deadline,
        # since the deadline is what fails the request the publish is holding (design §7.1).
        already_expired = rec.direction == "fetch" and rec.abandoned
        return rec.deadline is not None and now >= rec.deadline and not already_expired

    @staticmethod
    def _is_unlaunched_fetch(rec: TransferRecord) -> bool:
        """A fetch record PLANNED with a plan: the pages the plan needs are not reserved here yet,
        or the launch never started. Attempts of earlier tries may remain on the record."""
        return (
            rec.direction == "fetch" and rec.state is RecordState.PLANNED and rec.plan is not None
        )

    @staticmethod
    def _is_launchable(rec: TransferRecord) -> bool:
        return rec.state is RecordState.PLANNED and rec.plan is not None and not rec.launch_gave_up

    def _settle_finished_record(self, rec: TransferRecord, vote: Vote) -> None:
        if rec.direction == "fetch":
            self._release_finished_fetch(rec)
        else:
            failed = vote.kind is VoteKind.FAILED
            self._finish_publish(rec, failed=failed, reason="kv publish failed")

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
        stores = [s for s in self._sources.values() if s.hint_key is None]
        if not stores:
            return
        cache = self._probe_answers.setdefault(req.py_request_id, {})
        query = None
        for source in stores:
            if cache.get(source.name) is not None:
                continue
            if query is None:
                query = self._planner.probe_query(req)
                if query is None:
                    return
            name, units = query
            # Broad on purpose, against CODING_GUIDELINES: the contract says a failing probe
            # raises but leaves the type to the backend, and a store outage must not take the
            # engine loop down. The answer stays pending, so the planner defers until its probe
            # budget is spent and then plans without the store.
            try:
                answer = source.backend.probe(name, units)
            except Exception as exc:  # noqa: BLE001
                logger.warning("probe on %s failed, answer stays pending: %s", source.name, exc)
                continue
            # A pending answer (``None``) is not recorded: an absent entry means the same thing
            # to the planner, and the backend is asked again next round.
            if answer is not None:
                cache[source.name] = answer

    # ---- advance, phase 3: one collective ----

    def _sync(
        self,
        votes: Mapping[RecordKey, Vote],
        expired: Sequence[RecordKey],
        answers: Mapping[int, _PlanAnswer],
        now: float,
    ) -> tuple[dict[RecordKey, _Verdict], list[RecordKey], dict[int, _PlanAnswer]]:
        voted = self._plan_authority is PlanAuthority.VOTED
        payload: _Payload = (
            [(key, vote.kind.value, vote.b, vote.hint) for key, vote in sorted(votes.items())],
            sorted(expired),
            [(rid, _wire(ans)) for rid, ans in sorted(answers.items())] if voted else [],
        )
        gathered = self._dist.allgather(payload)
        ballots = _ballots_by_key(gathered)
        self._note_peer_launches(votes, ballots, now)
        verdicts = _reduce_votes(ballots, len(gathered))
        expired_all = {tuple(key) for _, exp, _ in gathered for key in exp}
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
    ) -> None:
        for key in expired:
            rec = self._records.get(key)
            if rec is None:
                continue
            if rec.state is RecordState.IN_FLIGHT:
                self._expire_inflight(rec)
            elif self._is_unlaunched_fetch(rec):
                # Another rank's attempt expired while this rank never launched: nothing is in
                # flight here, so the request fails now and its record goes with it.
                self._fail_unlaunched_fetch(rec, reason="kv fetch timed out")

        for key in sorted(verdicts):
            rec = self._records.get(key)
            if rec is None:
                continue
            b, hint, failed = verdicts[key]
            if rec.state is RecordState.IN_FLIGHT:
                self._settle_inflight(rec, b, hint, failed)
            elif self._is_unlaunched_fetch(rec):
                # An UNLAUNCHED vote blocks a landing, so the only verdict that reaches an
                # unlaunched record is a failure: drop the plan without touching pages.
                self._reset_for_replan(rec)

        self._answers_to_export = {}
        for rid, ans in answers.items():
            if ans is DEFER:
                continue
            self._decide(rid, ans)
            if self._plan_authority is PlanAuthority.OWNER:
                self._answers_to_export[rid] = ans

    def _decide(self, rid: int, ans: FetchPlan | None) -> None:
        """Write a decided answer: ``None`` releases a planned record, a plan goes on the record
        (created if needed) and ``plan_fetch`` reads it from now on."""
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

    def _expire_inflight(self, rec: TransferRecord) -> None:
        if rec.direction == "fetch" and rec.request_id in self._finished:
            # Already abandoned by the request's end; its outcome settles it. A publish of a
            # finished request is different: its expiry is what fails the held request.
            return
        rec.abandoned = True
        if rec.direction == "publish":
            self._finish_publish(rec, failed=True, reason="kv publish timed out")
        elif rec.plan.no_local_fallback:
            self._fetch_failed(rec, hint=None, reason="kv fetch timed out")
        else:
            self._fail_expired_fetch(rec)

    def _settle_inflight(self, rec: TransferRecord, b: int, hint: int, failed: bool) -> None:
        if rec.direction == "publish":
            self._finish_publish(rec, failed=failed, reason="kv publish failed")
        elif failed:
            self._fetch_failed(rec, hint=None, reason="kv fetch failed")
        elif b == rec.plan.token_end:
            self._fetch_landed(rec)
        else:
            self._fetch_failed(rec, hint=hint, reason="kv fetch served short")

    def _fetch_landed(self, rec: TransferRecord) -> None:
        if rec.request_id in self._finished:
            self._release_finished_fetch(rec)
            return
        rec.state = RecordState.LANDED
        self._close_routes(rec)
        req = self._requests[rec.request_id]
        self._plans[rec.request_id] = None
        self._effects.unpark(req, rec.plan.token_end, rec.plan.no_local_fallback, self._aux(rec))

    def _fetch_failed(self, rec: TransferRecord, *, hint: int | None, reason: str) -> None:
        if rec.request_id in self._finished:
            self._release_finished_fetch(rec)
            return
        rec.state = RecordState.FAILED
        if not self._quiesce(rec):
            return
        self._close_routes(rec)
        req = self._requests[rec.request_id]
        self._effects.give_back_fetch_pages([req])
        self._retry_or_settle(rec, hint=hint, reason=reason)

    def _reset_for_replan(self, rec: TransferRecord) -> None:
        """The ranks agreed the fetch failed while this rank never launched it: same next step as
        after a failed attempt, minus the release point (no attempt, no pages to give back)."""
        reason = "kv fetch launch given up" if rec.launch_gave_up else "kv fetch failed"
        self._retry_or_settle(rec, hint=None, reason=reason)

    def _retry_or_settle(self, rec: TransferRecord, *, hint: int | None, reason: str) -> None:
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

    def _fail_unlaunched_fetch(self, rec: TransferRecord, *, reason: str) -> None:
        req = self._requests[rec.request_id]
        self._release(rec)
        self._plans[rec.request_id] = None
        self._effects.fail_requests([req], reason)

    def _fail_expired_fetch(self, rec: TransferRecord) -> None:
        """A normal fetch past its deadline: fail the request now, keep its pages held.

        A hung store must not park a request forever, so the wait is bounded by the deadline
        and the request fails. Its pages cannot be given back yet: the backend may still be
        writing them, so the record stays in flight and the request is held until the outcome
        arrives (or ``close``). The late outcome then only releases; it no longer unparks.
        """
        req = self._requests[rec.request_id]
        self._effects.fail_requests([req], "kv fetch timed out")
        # The engine ends a failed request through its release gate, which notifies this
        # coordinator; the call is repeated here so the hold does not depend on the engine.
        self.notify_request_finished(req)

    def _release_finished_fetch(self, rec: TransferRecord) -> None:
        """A fetch whose request ended while it was in flight: its outcome is moot. Quiesce,
        release; no ``unpark`` and no ``give_back`` for a request that is gone. The held request
        is terminated once nothing else of it is in flight."""
        if not self._quiesce(rec):
            return
        self._close_routes(rec)
        self._release(rec)
        self._finish_if_released(rec.request_id)

    def _finish_publish(self, rec: TransferRecord, *, failed: bool, reason: str) -> None:
        """A publish reached its end. A failure is a warning, never a request failure: a
        running request keeps running, and a finished one already has its response."""
        rec.state = RecordState.FAILED if failed else RecordState.LANDED
        if not self._quiesce(rec):
            return
        self._release(rec)
        rid = rec.request_id
        if failed:
            when = "after the request ended" if rid in self._finished else "while still running"
            logger.warning("request %d: %s %s", rid, reason, when)
        self._finish_if_released(rid)

    def _finish_if_released(self, rid: int) -> None:
        """A finished request's last word to the engine, once no record of it remains.

        A held request is terminated here, whatever its last transfer's outcome. A request that
        was never held owes the engine nothing: it terminates as usual through the release gate.
        """
        if rid not in self._finished:
            return
        if (rid, "fetch") in self._records or (rid, "publish") in self._records:
            return
        req = self._requests.get(rid)
        if req is None:
            logger.warning("request %d finished with no request object on record", rid)
        elif rid in self._held:
            self._effects.terminate_request(req)
        self._forget_request(rid)

    # ---- launch / publish helpers ----

    def _launch_one(self, req: RequestView, rec: TransferRecord, now: float) -> bool:
        plan = rec.plan
        rid = req.py_request_id
        source = self._sources[plan.source]
        extent = self._reader.fetch_extent(req, plan)
        route = None
        if source.hint_key is not None and plan.hint is not None:
            # Broad on purpose, against CODING_GUIDELINES: ``open_route`` raises the backend's own
            # transport error for a hint that was fine but could not be prepared; that is worth
            # trying again, and it must not escape and strand the other requests in the queue.
            try:
                route = source.backend.open_route(plan.hint)
            except (ValueError, NotImplementedError) as exc:
                # Bad hint, or a backend that cannot route: this plan can never work here.
                logger.warning("request %d: route refused by %s: %s", rid, source.name, exc)
                self._drop_launch(req, rec, give_up=True)
                return False
            except Exception as exc:  # noqa: BLE001
                logger.warning("request %d: route to %s failed: %s", rid, source.name, exc)
                self._drop_launch(req, rec, give_up=False)
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
        rec.consecutive_launch_failures = 0
        rec.peer_launched_at = None
        try_index = rec.try_index + 1 if rec.attempts else 0
        rec.extent = extent
        rec.attempts.append(
            AttemptRecord(attempt, try_index=try_index, source=source.name, route=route)
        )
        rec.state = RecordState.IN_FLIGHT
        rec.deadline = None if self._fetch_timeout_s is None else now + self._fetch_timeout_s
        return True

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
        rid = req.py_request_id
        self._requests[rid] = req
        key = (rid, "publish")
        rec = self._records.get(key)
        if rec is None:
            deadline = None if self._publish_timeout_s is None else now + self._publish_timeout_s
            rec = TransferRecord(rid, "publish", RecordState.PLANNED, deadline=deadline)
            self._records[key] = rec
        elif rec.state not in (RecordState.PLANNED, RecordState.IN_FLIGHT):
            return
        extent, chunk = self._reader.publish_description(req)
        rec.extent = extent
        for name, publisher in self._publishers.items():
            # Design §7.5: a publisher that places pieces works in series and hears every piece;
            # one that does not is offered the content once, on the last piece, and never has
            # to read ``is_last``.
            if isinstance(publisher, PlacesPieces):
                self._submit_publish(rec, name, publisher.publish, extent)
                if chunk is not None:
                    self._submit_publish(rec, name, publisher.place, chunk)
            elif extent.is_last:
                self._submit_publish(rec, name, publisher.publish, extent)
        if rec.attempts:
            rec.state = RecordState.IN_FLIGHT

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

    def _quiesce(self, rec: TransferRecord) -> bool:
        """The release point. False means the backend cannot vouch for the memory. Design §4.3
        would keep the pages out of use until the deadline; this tightens that to the engine's
        existing poison path (fatal) and leaves retry-until-deadline as a follow-up."""
        by_backend: dict[str, list[Attempt]] = {}
        for a in rec.current_try_attempts():
            by_backend.setdefault(a.source, []).append(a.attempt)
        for name, attempts in by_backend.items():
            if not self._backends[name].quiesce(attempts):
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
        self._records.pop(rec.key, None)

    def _forget_request(self, rid: int) -> None:
        """Drop every per-request entry of a request that is gone, here and in the planner and
        the reader. For a held request this runs at its termination, so the reader's per-request
        cache keeps that one entry until then."""
        self._requests.pop(rid, None)
        self._plans.pop(rid, None)
        self._pending_answers.pop(rid, None)
        self._probe_answers.pop(rid, None)
        self._finished.discard(rid)
        self._held.discard(rid)
        self._planner.forget(rid)
        self._reader.forget_request(rid)

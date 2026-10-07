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
"""What the ranks say to each other each round, and how it is reduced (design §7.1 "齐").

Pure functions over plain values: one ``RoundMessage`` per rank goes through the collective, and
every rank reduces the gathered messages the same way, so the verdicts and plan answers come out
identical everywhere. Nothing here reads a record or touches the engine; ``coordinator.py`` builds
the message and applies the reductions.
"""

from __future__ import annotations

from enum import Enum
from typing import Mapping, NamedTuple, Sequence

from ...remote_cache import DEFER, Defer, FetchPlan
from .records import RecordKey

__all__ = [
    "DEFER_ANSWER",
    "EncodedPlanAnswer",
    "PlanAnswer",
    "PlanAnswers",
    "RoundMessage",
    "Verdict",
    "Vote",
    "VoteKind",
    "encode_plan_answer",
    "reduce_plan_answers",
    "reduce_votes",
    "votes_by_key",
]

DEFER_ANSWER = "DEFER"
"""``DEFER`` on the wire."""
PlanAnswer = FetchPlan | None | Defer
"""What the planner answers for one request: a plan, ``None`` (compute locally) or ``DEFER``."""
EncodedPlanAnswer = tuple[int, str] | str | None
"""A plan answer on the wire: ``(token_end, source)``, ``"DEFER"``, or ``None``."""
PlanAnswers = list[tuple[int, EncodedPlanAnswer]]
"""Decided answers as the owner hands them to the followers: ``[(request_id, encoded answer)]``,
never ``"DEFER"``. Built-in types only, so it rides in the pickled schedule."""


class VoteKind(Enum):
    """What one rank says about one record this round. Carried on the wire as the string value.

    UNLAUNCHED: fetch planned but not launched here (no pages yet); holds up a landing, not a failure.
    INFLIGHT: an attempt is still running here; nothing may be decided this round.
    FAILED: decisive: this rank's attempt failed, or it gave up launching.
    TERMINAL: every attempt here ended without failure; carries ``reached_end`` for a fetch.
    """

    UNLAUNCHED = "UNLAUNCHED"
    INFLIGHT = "INFLIGHT"
    FAILED = "FAILED"
    TERMINAL = "TERMINAL"


class Vote(NamedTuple):
    """One rank's vote on one record. ``reached_end`` is the block boundary this rank's attempts
    reached, the merged ``B`` of design §7.1 (``remote_cache.served_token_end``); it matters for
    ``TERMINAL`` fetches only."""

    kind: VoteKind
    reached_end: int = 0


class Verdict(NamedTuple):
    """What the ranks agreed on for one record: ``MIN(reached_end)`` over the ranks, and whether
    any rank failed."""

    reached_end: int
    failed: bool


class RoundMessage(NamedTuple):
    """One rank's message per round: ``votes`` as ``[(key, kind, reached_end)]``, ``expired`` as
    ``[key]``, ``plan_answers`` as ``[(request_id, encoded answer)]``, ``pending`` as whether
    this rank still has pending work, ``drained`` as whether it is past its shutdown drain
    deadline. A plain 5-tuple on the wire."""

    votes: list
    expired: list
    plan_answers: list
    pending: bool
    drained: bool


def encode_plan_answer(answer: PlanAnswer) -> EncodedPlanAnswer:
    if answer is DEFER:
        return DEFER_ANSWER
    if answer is None:
        return None
    return (answer.token_end, answer.source)


def votes_by_key(gathered: Sequence[RoundMessage]) -> dict[RecordKey, list[Vote]]:
    """Every rank's vote on every record, in gathered order; keys come back as tuples."""
    votes: dict[RecordKey, list[Vote]] = {}
    for message in gathered:
        for key, kind, reached_end in message.votes:
            votes.setdefault(tuple(key), []).append(Vote(VoteKind(kind), reached_end))
    return votes


def reduce_votes(votes: Mapping[RecordKey, list[Vote]], num_ranks: int) -> dict[RecordKey, Verdict]:
    """The one reduction of design §7.1 "齐", for fetches and publishes alike.

    A record is decided only once all ``num_ranks`` ranks voted on it, and then in this order:
    any INFLIGHT holds the round (a failure landed now would quiesce under a running attempt and
    block the engine thread); else any FAILED is decisive for every rank, delivered data
    included; else any UNLAUNCHED holds the round (a landing needs every rank's pages); else
    every rank is TERMINAL and the landing takes MIN(B), since ranks hold different layer groups.
    """
    verdicts: dict[RecordKey, Verdict] = {}
    for key, record_votes in votes.items():
        if len(record_votes) < num_ranks:
            continue
        kinds = {vote.kind for vote in record_votes}
        if VoteKind.INFLIGHT in kinds:
            continue
        if VoteKind.FAILED in kinds:
            verdicts[key] = Verdict(reached_end=0, failed=True)
        elif VoteKind.UNLAUNCHED not in kinds:
            verdicts[key] = Verdict(
                reached_end=min(vote.reached_end for vote in record_votes), failed=False
            )
    return verdicts


def reduce_plan_answers(
    gathered: Sequence[RoundMessage], local_answers: Mapping[int, PlanAnswer]
) -> dict[int, PlanAnswer]:
    """Any DEFER -> DEFER; otherwise any disagreement on ``(token_end, source)`` -> None."""
    num_ranks = len(gathered)
    encoded: dict[int, list[EncodedPlanAnswer]] = {}
    for message in gathered:
        for request_id, answer in message.plan_answers:
            encoded.setdefault(request_id, []).append(
                tuple(answer) if isinstance(answer, list) else answer
            )
    plan_answers: dict[int, PlanAnswer] = {}
    for request_id, local in local_answers.items():
        peer_answers = encoded.get(request_id, [])
        if len(peer_answers) < num_ranks or DEFER_ANSWER in peer_answers:
            plan_answers[request_id] = DEFER
        elif (
            all(answer == peer_answers[0] for answer in peer_answers)
            and encode_plan_answer(local) == peer_answers[0]
        ):
            plan_answers[request_id] = local
        else:
            plan_answers[request_id] = None
    return plan_answers

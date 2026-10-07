# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``KVTransferCoordinator`` at its seams: the engine queue drained at the head of ``advance``,
the one collective per round and what its payload carries, the gather seam with hand-written
peers, the status dump, and planning through the coordinator (probe budget, local reuse,
windowed groups).

Single rank. The fetch, publish and host-first lifecycles have their own files beside this
one (``test_coordinator_fetch.py``, ``test_coordinator_publish.py``,
``test_coordinator_host_first.py``).
"""

import json

import pytest

__extra_import_path__ = ["~/tensorrt_llm/_torch"]
from disaggregation.remote_cache import DEFER, FetchPlan  # noqa: E402
from fakes import (  # noqa: E402
    END,
    CoordinatorRig,
    FakePublishes,
    FakeRequest,
    ScriptedPeersCollective,
    full_attention,
    windowed,
    worker_request,
)

pytestmark = pytest.mark.cpu_only


# ---- engine queue, collective, dump ----


def test_engine_queue_drained_at_advance_start_under_budget():
    rig = CoordinatorRig(queue_budget=2)
    ran = []
    for i in range(3):
        rig.queue.post(lambda i=i: ran.append(i))
    rig.coord.advance([], 0.0)
    assert ran == [0, 1] and rig.queue.drains == [2]
    rig.coord.advance([], 1.0)
    assert ran == [0, 1, 2] and rig.queue.drains == [2, 2]


def test_queue_runs_before_polling():
    rig = CoordinatorRig()
    req = worker_request()
    attempt = rig.plan_and_launch(req)
    rig.queue.post(attempt.deliver_all)  # a backend completing on the engine thread
    rig.coord.advance([], 1.0)
    assert rig.effects.count("unpark") == 1  # reaped in the same advance


def test_one_allgather_per_advance_and_none_elsewhere():
    rig = CoordinatorRig()
    req = worker_request()
    rig.coord.advance([req], 0.0)
    rig.coord.launch_reserved_fetches([req], 0.0)
    rig.coord.advance([], 1.0)
    rig.coord.holds_finished_request(req)
    rig.coord.publish_committed_blocks([], now=2.0)
    assert len(rig.dist.calls) == 2


def test_allgather_payload_carries_plan_answers_and_arrivals():
    rig = CoordinatorRig()
    req = worker_request()
    rig.coord.advance([req], 0.0)
    arrivals, expired, plans, pending, _ = rig.payloads()[0]
    assert arrivals == [] and expired == [] and plans == [(1, (END, "worker"))]
    assert pending is False  # the plan is written after the gather: no record yet
    rig.coord.launch_reserved_fetches([req], 0.0)
    rig.worker.attempts[0].deliver_all()
    rig.coord.advance([], 1.0)
    arrivals, expired, plans, _, _ = rig.payloads()[1]
    assert arrivals == [((1, "fetch"), "TERMINAL", END)] and plans == []


def test_allgather_payload_carries_pending_and_drained_and_the_ors_are_last_rounds():
    """The 4th element is whether this rank has pending work after the poll, the 5th whether it
    is drained (``advance(..., drained=True)``): a drained rank reports no pending work while its
    attempt is still in flight. ``any_rank_pending`` and ``any_rank_drained`` are the gathered
    ORs, False before the first round."""
    pub = FakePublishes()
    rig = CoordinatorRig(publishers=[pub])
    assert rig.coord.any_rank_pending is False and rig.coord.any_rank_drained is False
    rig.coord.advance([], 0.0)
    assert rig.payloads()[-1] == ([], [], [], False, False)
    assert rig.coord.any_rank_pending is False and rig.coord.any_rank_drained is False
    req = FakeRequest(3, prompt_len=29)
    rig.coord.publish_committed_blocks([req], now=0.0)
    rig.coord.advance([], 1.0)
    assert rig.payloads()[-1][3:] == (True, False) and rig.coord.any_rank_pending is True
    rig.coord.advance([], 2.0, drained=True)
    assert rig.payloads()[-1][3:] == (False, True)
    assert rig.coord.any_rank_pending is False and rig.coord.any_rank_drained is True
    assert rig.record(3, "publish")["state"] == "IN_FLIGHT"
    dump = rig.coord.status_dump()
    assert dump["any_rank_pending"] is False and dump["any_rank_drained"] is True


def test_any_rank_pending_and_drained_are_true_on_a_peers_word_alone():
    rig = CoordinatorRig(
        dist=ScriptedPeersCollective(lambda local: ([], [], local[2], True, False))
    )
    rig.coord.advance([], 0.0)
    assert rig.payloads()[-1][3:] == (False, False)
    assert rig.coord.any_rank_pending is True and rig.coord.any_rank_drained is False
    rig = CoordinatorRig(
        dist=ScriptedPeersCollective(lambda local: ([], [], local[2], False, True))
    )
    rig.coord.advance([], 0.0)
    assert rig.coord.any_rank_pending is False and rig.coord.any_rank_drained is True


def test_allgather_payload_wires_none_and_defer():
    rig = CoordinatorRig(sources=("store",))
    a, b = FakeRequest(1, prompt_len=29), FakeRequest(2, prompt_len=29)
    rig.store.probe_answers.extend([frozenset(), None])  # a: holds nothing; b: unanswered
    rig.coord.advance([a, b], 0.0)
    assert rig.last_plan_answers == [(1, None), (2, "DEFER")]


# ---- the gather seam with hand-written peers ----


def test_peer_reporting_short_b_fails_the_local_landed_fetch():
    def peer(local):
        arrivals, expired, plans, pending, drained = local
        return (
            [(key, kind, token_end - 4) for key, kind, token_end in arrivals],
            expired,
            plans,
            pending,
            drained,
        )

    rig = CoordinatorRig(dist=ScriptedPeersCollective(peer))
    req = worker_request()
    rig.plan_and_launch(req).deliver_all()
    rig.coord.advance([], 1.0)
    assert rig.effects.count("unpark") == 0 and rig.effects.count("revert_fetch_pages") == 1
    assert rig.record(1)["state"] == "PLANNED"
    rig.coord.advance([req], 2.0)
    assert rig.coord.fetch_answer(req).token_end == END - 4  # MIN(B) from the peer


def test_peer_voting_a_different_source_makes_the_plan_none():
    def peer(local):
        arrivals, expired, plans, pending, drained = local
        return (arrivals, expired, [(rid, (v[0], "store")) for rid, v in plans], pending, drained)

    rig = CoordinatorRig(dist=ScriptedPeersCollective(peer))
    req = worker_request()
    rig.coord.advance([req], 0.0)
    assert rig.coord.fetch_answer(req) is None and rig.records() == []


def test_peer_not_reporting_an_arrival_keeps_it_in_flight():
    rig = CoordinatorRig(
        dist=ScriptedPeersCollective(lambda local: ([], [], local[2], False, False))
    )
    req = worker_request()
    rig.plan_and_launch(req).deliver_all()
    rig.coord.advance([], 1.0)
    assert rig.record(1)["state"] == "IN_FLIGHT" and rig.effects.count("unpark") == 0


def test_agreement_wait_is_bounded_by_fetch_timeout():
    """Two ranks; the peer's request is gone and it casts no vote on the record any more. The
    delivered rank waits for the agreement until its deadline, then fails the request and
    settles the record on its own word instead of waiting forever."""
    rig = CoordinatorRig(
        dist=ScriptedPeersCollective(lambda local: ([], [], local[2], False, False)),
        fetch_timeout_s=10.0,
    )
    req = worker_request()
    rig.plan_and_launch(req, now=0.0).deliver_all()
    rig.coord.advance([], 9.0)
    assert rig.record(1)["state"] == "IN_FLIGHT" and rig.effects.count("fail_requests") == 0
    rig.coord.advance([], 10.0)
    assert rig.effects.args_of("fail_requests") == [((req,), "kv fetch timed out")]
    assert rig.record(1)["expired"] and rig.effects.names()[-1:] == ["hold_for_transfer"]
    rig.coord.advance([], 11.0)  # the outcome is in: nothing more to wait for
    assert rig.last_votes == []
    assert rig.worker.count("quiesce") == 1 and rig.effects.count("unpark") == 0
    assert rig.effects.names()[-1:] == ["terminate_request"] and rig.records() == []


def test_status_dump_shape():
    rig = CoordinatorRig(fetch_timeout_s=10.0)
    req = worker_request()
    rig.plan_and_launch(req, now=1.0)
    dump = rig.coord.status_dump()
    assert dump["decided_plans"] == 1 and dump["finished_pending"] == []
    assert dump["records"] == [
        {
            "request_id": 1,
            "direction": "fetch",
            "state": "IN_FLIGHT",
            "try_index": 0,
            "attempts": 1,
            "outcomes": [],
            "deadline": 11.0,
            "expired": False,
            "token_end": END,
            "gave_up_launching": False,
            "peer_launch_seen_at": None,
            "has_landing": False,
            "resource_wait_since": None,
            "retries_left": 1,
            "consecutive_launch_failures": 0,
            "retry_cap": None,
            "committed_names": 0,
        }
    ]


def test_status_dump_reports_retry_bookkeeping_per_record():
    """A record entry carries what reading a stuck fetch needs: the retries left, the hint the
    next try is bounded by, the run of launches that never started, and how many of the plan's
    units the local cache had committed by launch (a count: the names are hashes). The dump is
    written as JSON at shutdown, so every value must survive ``json.dumps``."""
    rig = CoordinatorRig()
    req = worker_request()
    rig.reader.committed_blocks[1] = 2
    attempt = rig.plan_and_launch(req)
    rec = rig.record(1)
    assert rec["retries_left"] == 1 and rec["consecutive_launch_failures"] == 0
    assert rec["retry_cap"] is None and rec["committed_names"] == 2
    json.dumps(rig.coord.status_dump())

    # A short delivery: the retry is spent, the agreed boundary becomes the hint, and the
    # committed names of the dropped try are forgotten.
    attempt.deliver_all_but(*rig.reader.unit_names(req, [6]))
    rig.coord.advance([], 1.0)
    rec = rig.record(1)
    assert rec["state"] == "PLANNED" and rec["token_end"] is None
    assert rec["retries_left"] == 0 and rec["retry_cap"] == END - 4
    assert rec["consecutive_launch_failures"] == 0 and rec["committed_names"] == 0

    # Planned again within the hint, which the plan consumes; a rejected launch is counted
    # while the retry budget stays where it was.
    rig.advance_as_hooks_would(req, 2.0)
    rec = rig.record(1)
    assert rec["token_end"] == END - 4 and rec["retry_cap"] is None
    rig.worker.reject_next_calls = 1
    rig.coord.launch_reserved_fetches([req], 2.0)
    rec = rig.record(1)
    assert rec["consecutive_launch_failures"] == 1 and rec["retries_left"] == 0


# ---- planning through the coordinator ----


def test_plan_none_is_remembered_until_request_end():
    rig = CoordinatorRig()
    req = FakeRequest(1, prompt_len=29)  # no hint, store never answers -> DEFER twice, then None
    rig.coord.advance([req], 0.0)
    assert rig.coord.fetch_answer(req) is DEFER
    rig.coord.advance([req], 1.0)
    assert rig.coord.fetch_answer(req) is DEFER
    rig.coord.advance([req], 2.0)
    assert rig.coord.fetch_answer(req) is None
    assert rig.records() == [] and rig.coord.status_dump()["decided_plans"] == 1
    rig.coord.holds_finished_request(req)
    assert rig.coord.status_dump()["decided_plans"] == 0
    assert rig.coord.fetch_answer(req) is DEFER


def test_store_probe_is_asked_once_and_answer_is_cached():
    rig = CoordinatorRig(sources=("store",))
    req = FakeRequest(1, prompt_len=29)
    rig.store.probe_default = rig.reader.unit_names(req, range(5))
    rig.coord.advance([req], 0.0)
    assert rig.store.count("probe") == 1
    name, units = rig.store.calls[0][1]
    assert name == rig.reader.block_keys(req)[6] and len(units) == 7
    plan = rig.coord.fetch_answer(req)
    assert plan.source == "store" and plan.token_end == 20
    rig.coord.advance([req], 1.0)  # already PLANNED with a plan: not re-decided, not re-probed
    assert rig.store.count("probe") == 1


def test_store_probe_exception_keeps_the_answer_pending_until_budget_is_spent():
    rig = CoordinatorRig(sources=("store",))
    req = FakeRequest(1, prompt_len=29)
    rig.store.probe_answers.extend([RuntimeError("store unreachable")] * 3)
    rig.coord.advance([req], 0.0)
    assert rig.coord.fetch_answer(req) is DEFER
    rig.coord.advance([req], 1.0)
    assert rig.coord.fetch_answer(req) is DEFER
    rig.coord.advance([req], 2.0)  # probe_timeout_s spent: plan without the store
    assert rig.coord.fetch_answer(req) is None
    assert rig.store.count("probe") == 3  # re-asked every round while pending


def test_store_probe_recovering_within_budget_plans_from_the_store():
    rig = CoordinatorRig(sources=("store",))
    req = FakeRequest(1, prompt_len=29)
    rig.store.probe_answers.append(RuntimeError("blip"))
    rig.store.probe_default = rig.reader.unit_names(req, range(7))
    rig.coord.advance([req], 0.0)
    assert rig.coord.fetch_answer(req) is DEFER
    rig.coord.advance([req], 1.0)
    assert rig.coord.fetch_answer(req).source == "store"


def test_store_fetch_has_no_route():
    rig = CoordinatorRig(sources=("store",))
    req = FakeRequest(1, prompt_len=29)
    rig.store.probe_default = rig.reader.unit_names(req, range(7))
    rig.coord.advance([req], 0.0)
    rig.coord.launch_reserved_fetches([req], 0.0)
    assert rig.store.count("open_route") == 0
    extent, route = rig.store.calls[-1][1]
    assert route is None
    rig.store.attempts[0].deliver_all()
    rig.coord.advance([], 1.0)
    assert rig.effects.args_of("unpark") == [(req, END, False, None)]


def test_gen_first_context_defers_until_ready_and_builds_no_record():
    rig = CoordinatorRig()
    req = FakeRequest(
        1, prompt_len=29, is_generation_first_context=True, route_hints={"ctx": {"peer": "g"}}
    )
    rig.reader.ready[1] = False
    rig.coord.advance([req], 0.0)
    assert rig.coord.fetch_answer(req) is DEFER and rig.records() == []
    rig.reader.ready[1] = True
    rig.coord.advance([req], 1.0)
    assert isinstance(rig.coord.fetch_answer(req), FetchPlan)


def test_windowed_model_extent_carries_pruned_units():
    rig = CoordinatorRig(groups=[full_attention(0), windowed(1)])
    req = worker_request()
    rig.reader.reuse_tokens[1] = 8
    rig.plan_and_launch(req)
    extent = rig.worker.calls[-1][1][0]
    by_group = {}
    for u in extent.units:
        by_group.setdefault(u.local_group, set()).add(u.local)
    assert by_group == {0: {2, 3, 4, 5, 6}, 1: {4, 5, 6}}


@pytest.mark.parametrize("candidates_twice", [False, True])
def test_two_requests_progress_independently(candidates_twice):
    rig = CoordinatorRig()
    a, b = worker_request(1), worker_request(2, prompt_len=17)
    rig.coord.advance([a, b], 0.0)
    if candidates_twice:
        rig.coord.advance([a, b], 0.5)  # PLANNED with a plan: left alone
    rig.coord.launch_reserved_fetches([a, b], 1.0)
    assert rig.effects.args_of("park_for_fetch") == [((a, b),)]
    rig.worker.attempts[1].deliver_all()
    rig.coord.advance([], 2.0)
    assert rig.effects.args_of("unpark") == [(b, 16, False, None)]
    assert rig.record(1)["state"] == "IN_FLIGHT" and rig.record(2)["state"] == "DELIVERED"

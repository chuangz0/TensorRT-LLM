# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two real coordinators, one per rank, stepped in lockstep on one thread through the ``gather``
seam (``LockstepWorld``).

Covers the reductions of design §7.1 "齐": arrivals take MIN(B), MIN(hint), MAX(failed); plan
answers become DEFER if any rank defers, None on any disagreement about ``(token_end, source)``,
the plan when identical.
"""

import pytest

__extra_import_path__ = ["~/tensorrt_llm/_torch"]
from disaggregation.base.cache_backend import Failed  # noqa: E402
from disaggregation.orchestration.kv_transfer_interfaces import DEFER  # noqa: E402
from disaggregation.orchestration.planner import FetchPlan  # noqa: E402
from fakes import FakeRequest, LockstepWorld, Rig, worker_request  # noqa: E402

END = 28


@pytest.fixture
def world():
    return LockstepWorld(2)


def make_rigs(world, **kw):
    return [Rig(gather=world.gathers[r], **kw) for r in range(world.n)]


def advance_all(world, rigs, candidates, now):
    world.run(lambda r: rigs[r].coord.advance(candidates, now))


def launch_all(world, rigs, req, now=0.0):
    advance_all(world, rigs, [req], now)
    for rig in rigs:
        plan = rig.coord.plan_fetch(req)
        assert isinstance(plan, FetchPlan)
        rig.plans[req.py_request_id] = plan
        rig.coord.launch_fetches([req], now)
        assert len(rig.worker.attempts) == rig.worker.count("fetch")
    return [rig.worker.attempts[-1] for rig in rigs]


# ---- the harness itself ----


def test_lockstep_gathers_both_payloads_in_rank_order(world):
    rigs = make_rigs(world)
    req = worker_request()
    seen = {}

    def step(rank):
        rigs[rank].coord.advance([req], 0.0)
        seen[rank] = rigs[rank].payloads()[-1]

    world.run(step)
    assert set(seen) == {0, 1}
    assert seen[0][2] == seen[1][2] == [(1, (END, "worker"))]
    assert [len(g.calls) for g in world.gathers] == [1, 1]


def test_lockstep_detects_a_rank_that_never_gathers(world):
    with pytest.raises(AssertionError, match="never gathered"):
        world.run(lambda r: None)


# ---- arrivals ----


def test_both_ranks_land_when_both_serve_everything(world):
    rigs = make_rigs(world)
    req = worker_request()
    attempts = launch_all(world, rigs, req)
    for a in attempts:
        a.deliver_all()
    advance_all(world, rigs, [], 1.0)
    for rig in rigs:
        assert rig.effects.only("unpark") == [(req, END, False, None)]
        assert len(rig.payloads()) == 2


def test_min_b_across_ranks_fails_both_and_replans_to_the_min_hint(world):
    rigs = make_rigs(world)
    req = worker_request()
    attempts = launch_all(world, rigs, req)
    attempts[0].deliver_all()
    attempts[1].deliver_all_but(*rigs[1].reader.unit_names(req, [6]))
    advance_all(world, rigs, [], 1.0)

    for rig in rigs:
        assert rig.effects.count("unpark") == 0
        assert rig.effects.count("give_back_fetch_pages") == 1
        assert rig.worker.count("quiesce") == 1
        assert rig.record(1)["state"] == "PLANNED"
    # Rank 0 sent B = 28, rank 1 sent B = 24: the wire shows it, the reduction hides it.
    assert rigs[0].payloads()[1][0] == [((1, "fetch"), END, END, False)]
    assert rigs[1].payloads()[1][0] == [((1, "fetch"), END - 4, END - 4, False)]

    advance_all(world, rigs, [req], 2.0)
    plans = [rig.coord.plan_fetch(req) for rig in rigs]
    assert all(isinstance(p, FetchPlan) for p in plans)
    assert [p.token_end for p in plans] == [END - 4, END - 4]


def test_max_failed_across_ranks_fails_both(world):
    rigs = make_rigs(world)
    req = worker_request()
    attempts = launch_all(world, rigs, req)
    attempts[0].deliver_all()
    attempts[1].finish(Failed("nic"))
    advance_all(world, rigs, [], 1.0)
    for rig in rigs:
        assert rig.effects.count("unpark") == 0
        assert rig.effects.count("give_back_fetch_pages") == 1
        assert rig.record(1)["state"] == "PLANNED"


def test_arrival_waits_until_every_rank_is_terminal(world):
    rigs = make_rigs(world)
    req = worker_request()
    attempts = launch_all(world, rigs, req)
    attempts[0].deliver_all()
    advance_all(world, rigs, [], 1.0)
    for rig in rigs:
        assert rig.record(1)["state"] == "IN_FLIGHT"
        assert rig.effects.count("unpark") == 0 and rig.effects.count("give_back_fetch_pages") == 0
    attempts[1].deliver_all()
    advance_all(world, rigs, [], 2.0)
    for rig in rigs:
        assert rig.effects.count("unpark") == 1


def test_gen_init_expiry_on_one_rank_fails_both(world):
    rigs = make_rigs(world, fetch_timeout_s=10.0)
    req = FakeRequest(7, prompt_len=30, is_gen_init=True, route_hints={"ctx": {"peer": "c"}})
    advance_all(world, rigs, [req], 0.0)
    rigs[0].coord.launch_fetches([req], 0.0)
    rigs[1].coord.launch_fetches([req], 5.0)  # launched later on rank 1: its own deadline is 15
    advance_all(world, rigs, [], 10.0)
    for rig in rigs:
        assert rig.effects.only("fail_requests") == [((req,), "kv fetch timed out")]
        assert rig.records() == []


def test_finished_request_crossing_its_deadline_on_one_rank_fails_nothing(world):
    rigs = make_rigs(world, fetch_timeout_s=10.0)
    req = worker_request()
    advance_all(world, rigs, [req], 0.0)
    rigs[0].coord.launch_fetches([req], 0.0)  # deadline 10
    rigs[1].coord.launch_fetches([req], 5.0)  # deadline 15
    attempts = [rig.worker.attempts[-1] for rig in rigs]
    for rig in rigs:
        rig.coord.notify_request_finished(req)
    advance_all(world, rigs, [], 10.0)  # rank 0 is past its deadline, rank 1 is not
    for rig in rigs:
        assert rig.effects.count("give_back_fetch_pages") == 0
        assert rig.effects.count("fail_requests") == 0
        assert rig.record(1)["state"] == "IN_FLIGHT"
        assert rig.payloads()[-1][1] == []  # nothing expired on the wire either
    for a in attempts:
        a.deliver_all()
    advance_all(world, rigs, [], 11.0)
    for rig in rigs:
        assert rig.effects.count("unpark") == 0 and rig.effects.count("terminate_request") == 1
        assert rig.records() == []


def test_retry_after_min_b_lands_on_both_ranks(world):
    rigs = make_rigs(world)
    req = worker_request()
    attempts = launch_all(world, rigs, req)
    attempts[0].deliver_all_but(*rigs[0].reader.unit_names(req, [6]))
    attempts[1].deliver_all_but(*rigs[1].reader.unit_names(req, [5, 6]))
    advance_all(world, rigs, [], 1.0)
    attempts = launch_all(world, rigs, req, now=2.0)
    assert [rig.plans[1].token_end for rig in rigs] == [END - 8, END - 8]
    for a in attempts:
        a.deliver_all()
    advance_all(world, rigs, [], 3.0)
    for rig in rigs:
        assert rig.effects.only("unpark") == [(req, END - 8, False, None)]
        assert rig.record(1)["state"] == "LANDED" and rig.record(1)["try_index"] == 1


# ---- plan answers ----


def test_one_rank_deferring_defers_both(world):
    rigs = make_rigs(world, sources=("store",))
    req = FakeRequest(1, prompt_len=29)
    rigs[0].store.probe_default = rigs[0].reader.unit_names(req, range(7))
    rigs[1].store.probe_answers.append(None)  # rank 1's store has not answered yet
    advance_all(world, rigs, [req], 0.0)
    for rig in rigs:
        assert rig.coord.plan_fetch(req) is DEFER
        assert rig.records() == [] and rig.coord.status_dump()["decided_plans"] == 0
    assert rigs[0].payloads()[0][2] == [(1, (END, "store"))]
    assert rigs[1].payloads()[0][2] == [(1, "DEFER")]

    rigs[1].store.probe_default = rigs[1].reader.unit_names(req, range(7))
    advance_all(world, rigs, [req], 1.0)
    plans = [rig.coord.plan_fetch(req) for rig in rigs]
    assert [p.token_end for p in plans] == [END, END]


def test_disagreeing_token_end_becomes_none_on_both(world):
    rigs = make_rigs(world, sources=("store",))
    req = FakeRequest(1, prompt_len=29)
    rigs[0].store.probe_default = rigs[0].reader.unit_names(req, range(7))  # 28
    rigs[1].store.probe_default = rigs[1].reader.unit_names(req, range(4))  # 16
    advance_all(world, rigs, [req], 0.0)
    for rig in rigs:
        assert rig.coord.plan_fetch(req) is None
        assert rig.records() == []


def test_same_token_end_different_source_becomes_none_on_both(world):
    rigs = make_rigs(world, sources=("store", "worker"))
    req = worker_request()
    rigs[0].store.probe_default = rigs[0].reader.unit_names(req, range(7))  # store, 28
    rigs[1].store.probe_default = frozenset()  # falls through to the worker, 28
    advance_all(world, rigs, [req], 0.0)
    assert rigs[0].payloads()[0][2] == [(1, (END, "store"))]
    assert rigs[1].payloads()[0][2] == [(1, (END, "worker"))]
    for rig in rigs:
        assert rig.coord.plan_fetch(req) is None and rig.records() == []


def test_none_versus_plan_is_none(world):
    rigs = make_rigs(world, sources=("store",))
    req = FakeRequest(1, prompt_len=29)
    rigs[0].store.probe_default = rigs[0].reader.unit_names(req, range(7))
    rigs[1].store.probe_default = frozenset()  # answered: holds nothing
    advance_all(world, rigs, [req], 0.0)
    assert [rig.coord.plan_fetch(req) for rig in rigs] == [None, None]


def test_identical_plans_are_kept_and_local_reuse_may_differ(world):
    rigs = make_rigs(world)
    req = worker_request()
    rigs[1].reader.reuse_tokens[1] = 8
    advance_all(world, rigs, [req], 0.0)
    plans = [rig.coord.plan_fetch(req) for rig in rigs]
    assert [p.token_end for p in plans] == [END, END]
    assert plans[0].units_by_group == {0: tuple(range(7))}
    assert plans[1].units_by_group == {0: (2, 3, 4, 5, 6)}
    for rig in rigs:
        rig.coord.launch_fetches([req], 1.0)
    for rig in rigs:
        rig.worker.attempts[-1].deliver_all()
    advance_all(world, rigs, [], 2.0)
    for rig in rigs:
        assert rig.effects.only("unpark") == [(req, END, False, None)]

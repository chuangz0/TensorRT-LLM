# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owner and follower on a ``LockstepWorld(2)``.

Rank 0 is the OWNER: it plans, its payload carries no plans, and ``export_plan_answers`` hands
its answers out. Rank 1 is the FOLLOWER: it never plans or probes; ``adopt_plan_answers`` builds
its own plan from the owner's ``(token_end, source)``. Votes still travel in both modes, so the
one-round lag between the owner's launch and the follower's is held by the ``UNLAUNCHED`` vote
and bounded by ``unlaunched_timeout_s`` exactly as between two voting ranks.
"""

import pytest

__extra_import_path__ = ["~/tensorrt_llm/_torch"]
from disaggregation.orchestration.kv_transfer.interfaces import PlanAuthority  # noqa: E402
from disaggregation.remote_cache import DEFER, FetchPlan  # noqa: E402
from fakes import FakeRequest, LockstepWorld, Rig, plan_unit_names, worker_request  # noqa: E402

pytestmark = pytest.mark.cpu_only

OWNER, FOLLOWER = 0, 1
END = 28  # 7 nameable blocks of 4
UNLAUNCHED_TIMEOUT_S = 10.0


def authority_of(rank: int) -> PlanAuthority:
    return PlanAuthority.OWNER if rank == OWNER else PlanAuthority.FOLLOWER


@pytest.fixture
def world():
    return LockstepWorld(2, authority_of=authority_of)


@pytest.fixture
def rigs(world):
    """Store-only ranks whose store holds the whole prompt."""
    rigs = [
        Rig(sources=("store",), unlaunched_timeout_s=UNLAUNCHED_TIMEOUT_S, **world.rig_kwargs(r))
        for r in range(world.n)
    ]
    for rig in rigs:
        rig.store.probe_default = rig.reader.unit_names(store_request(), range(7))
    return rigs


def store_request() -> FakeRequest:
    return FakeRequest(1, prompt_len=29)


def advance_all(world, rigs, req, now):
    """Every rank advances as its hooks would: the request is a candidate where undecided."""
    world.run(
        lambda r: rigs[r].coord.advance(
            [req] if rigs[r].coord.plan_fetch(req) is DEFER else [], now
        )
    )


def propagate(rigs, req):
    """Stage 0 as the PP loop plays it: the owner exports, the follower adopts."""
    answers = rigs[OWNER].coord.export_plan_answers()
    rigs[FOLLOWER].coord.adopt_plan_answers([req], answers)
    return answers


def launch_on(rig, req, now):
    plan = rig.coord.plan_fetch(req)
    assert isinstance(plan, FetchPlan)
    rig.coord.launch_reserved_fetches([req], now)
    return rig.store.attempts[-1]


def plan_section(rig):
    return rig.payloads()[-1][2]


# ---- decide, export, adopt ----


def test_owner_decides_and_the_follower_adopts_without_planning(world, rigs):
    req = store_request()
    advance_all(world, rigs, req, 0.0)

    assert isinstance(rigs[OWNER].coord.plan_fetch(req), FetchPlan)
    assert rigs[FOLLOWER].coord.plan_fetch(req) is DEFER  # nothing decided here yet
    assert plan_section(rigs[OWNER]) == [] and plan_section(rigs[FOLLOWER]) == []
    assert rigs[FOLLOWER].store.count("probe") == 0

    answers = propagate(rigs, req)
    assert answers == [(1, (END, "store"))]
    owner_plan, follower_plan = (rig.coord.plan_fetch(req) for rig in rigs)
    assert (follower_plan.token_end, follower_plan.source) == (
        owner_plan.token_end,
        owner_plan.source,
    )
    # Same groups, same reuse depth here: the same units.
    assert plan_unit_names(follower_plan) == plan_unit_names(owner_plan)
    assert rigs[FOLLOWER].store.count("probe") == 0
    assert rigs[OWNER].coord.export_plan_answers() == answers  # until the next advance


def test_owner_exports_a_local_decision_and_the_follower_releases_its_candidate(world, rigs):
    req = store_request()
    for rig in rigs:
        rig.store.probe_default = frozenset()  # the store holds nothing: compute locally
    advance_all(world, rigs, req, 0.0)
    assert rigs[OWNER].coord.plan_fetch(req) is None

    assert propagate(rigs, req) == [(1, None)]
    assert rigs[FOLLOWER].coord.plan_fetch(req) is None
    assert rigs[FOLLOWER].records() == []


def test_follower_holds_an_answer_until_its_request_appears_then_both_land(world, rigs):
    """The owner exports a request's answer once; a follower whose engine has not yet made
    the request a candidate keeps the answer and applies it the round the request appears."""
    req = store_request()
    world.run(lambda r: rigs[r].coord.advance([req] if r == OWNER else [], 0.0))
    answers = rigs[OWNER].coord.export_plan_answers()
    rigs[FOLLOWER].coord.adopt_plan_answers([], answers)  # not a candidate here yet
    assert rigs[FOLLOWER].records() == [] and rigs[FOLLOWER].coord.plan_fetch(req) is DEFER
    launch_on(rigs[OWNER], req, 0.0).deliver_all()

    advance_all(world, rigs, req, 1.0)  # the follower's engine now holds the request
    assert rigs[OWNER].coord.export_plan_answers() == []  # decided once, not exported again
    rigs[FOLLOWER].coord.adopt_plan_answers([req], [])  # this round's answers: none
    assert isinstance(rigs[FOLLOWER].coord.plan_fetch(req), FetchPlan)
    launch_on(rigs[FOLLOWER], req, 1.0).deliver_all()

    advance_all(world, rigs, req, 2.0)
    for rig in rigs:
        assert rig.effects.only("unpark") == [(req, END, False, None)]


def test_follower_drops_a_held_answer_when_the_request_ends(rigs):
    """Two answers arrive for requests not yet candidates here; request 7 ends before it ever
    is one, so its answer is dropped, while request 8's is still applied the round it appears."""
    stray, late = FakeRequest(7, prompt_len=29), FakeRequest(8, prompt_len=29)
    rigs[FOLLOWER].coord.adopt_plan_answers([], [(7, (END, "store")), (8, None)])
    assert rigs[FOLLOWER].records() == [] and rigs[FOLLOWER].coord.plan_fetch(stray) is DEFER
    assert rigs[FOLLOWER].coord.status_dump()["decided_plans"] == 0  # a None is held too
    rigs[FOLLOWER].coord.notify_request_finished(stray)
    rigs[FOLLOWER].coord.adopt_plan_answers([stray, late], [])
    assert rigs[FOLLOWER].coord.plan_fetch(stray) is DEFER  # its answer is gone
    assert rigs[FOLLOWER].coord.plan_fetch(late) is None  # its answer was kept
    assert rigs[FOLLOWER].coord.status_dump()["decided_plans"] == 1


@pytest.mark.xfail(
    strict=True,
    reason="adopt_plan_answers counts only this call's answers, not a held answer it applies",
)
def test_follower_counts_a_held_answer_it_applies_as_decided():
    """``adopt_plan_answers`` returns how many of the views got no answer and stay deferred; a
    view whose answer was held from an earlier round gets that answer now and is decided."""
    world = LockstepWorld(2, authority_of=authority_of)
    rig = Rig(sources=("store",), **world.rig_kwargs(FOLLOWER))
    req = store_request()
    rig.coord.adopt_plan_answers([], [(1, (END, "store"))])  # held: not a candidate yet
    assert rig.coord.adopt_plan_answers([req], []) == 0
    assert isinstance(rig.coord.plan_fetch(req), FetchPlan)


@pytest.mark.xfail(
    strict=True,
    reason="_record_answer re-plans a STAGING record and starts a second landing without "
    "releasing the first",
)
def test_follower_refuses_an_answer_for_a_staging_record(world):
    """The owner decides a request once; an answer that nevertheless reaches the follower again
    while the record it decided is landing on the host must not start a second landing (the
    first would be forgotten without a release) or restart the record's clock."""
    rigs = [
        Rig(sources=("host",), unlaunched_timeout_s=UNLAUNCHED_TIMEOUT_S, **world.rig_kwargs(r))
        for r in range(world.n)
    ]
    req = store_request()
    advance_all(world, rigs, req, 0.0)
    answers = rigs[OWNER].coord.export_plan_answers()
    follower = rigs[FOLLOWER]
    follower.coord.adopt_plan_answers([req], answers, 0.0)
    assert follower.record(1)["state"] == "STAGING" and follower.host.count("fetch_to_host") == 1

    follower.coord.adopt_plan_answers([req], answers, 0.5)  # the same answer once more

    assert follower.host.count("fetch_to_host") == 1
    assert len(follower.host.landings) == 1 and follower.host.releases() == 0
    assert follower.record(1)["state"] == "STAGING" and follower.record(1)["has_landing"]


def test_materialize_refuses_a_routed_source(world):
    rig = Rig(**world.rig_kwargs(FOLLOWER))  # worker (routed) and store
    with pytest.raises(ValueError, match="routed source 'worker'"):
        rig.planner.materialize(worker_request(), END, "worker")
    with pytest.raises(ValueError, match="no fetch source named 'nowhere'"):
        rig.planner.materialize(worker_request(), END, "nowhere")


def test_voted_ranks_still_carry_plans_in_the_payload():
    world = LockstepWorld(2)
    rigs = [Rig(sources=("store",), **world.rig_kwargs(r)) for r in range(2)]
    req = store_request()
    for rig in rigs:
        rig.store.probe_default = rig.reader.unit_names(req, range(7))
    advance_all(world, rigs, req, 0.0)
    assert plan_section(rigs[0]) == [(1, (END, "store"))]
    assert rigs[0].coord.export_plan_answers() == []


# ---- the one-round lag, held by UNLAUNCHED and bounded by the unlaunched timeout ----


def test_follower_launching_one_round_late_lands_both_once(world, rigs):
    req = store_request()
    advance_all(world, rigs, req, 0.0)
    propagate(rigs, req)
    owner_attempt = launch_on(rigs[OWNER], req, 0.0)  # round R: the owner launches at Stage 0
    owner_attempt.deliver_all()

    advance_all(world, rigs, req, 1.0)  # owner TERMINAL, follower UNLAUNCHED: no landing
    assert [rig.effects.count("unpark") for rig in rigs] == [0, 0]
    assert rigs[OWNER].record(1)["state"] == "IN_FLIGHT"
    assert rigs[FOLLOWER].record(1)["peer_launched_at"] == 1.0

    follower_attempt = launch_on(rigs[FOLLOWER], req, 1.0)  # round R+1: pages found here now
    follower_attempt.deliver_all()
    assert rigs[FOLLOWER].record(1)["peer_launched_at"] is None

    advance_all(world, rigs, req, 2.0)  # both TERMINAL: both land in the same round
    for rig in rigs:
        assert rig.effects.only("unpark") == [(req, END, False, None)]
        assert rig.effects.count("give_back_fetch_pages") == 0
        assert [r["state"] for r in rig.records()] == ["LANDED"]


def test_follower_that_never_launches_resets_both_ranks_and_the_owner_exports_again(world, rigs):
    req = store_request()
    advance_all(world, rigs, req, 0.0)
    propagate(rigs, req)
    launch_on(rigs[OWNER], req, 0.0).deliver_all()

    advance_all(world, rigs, req, 1.0)  # the follower's clock starts
    advance_all(world, rigs, req, 1.0 + UNLAUNCHED_TIMEOUT_S)  # the follower votes FAILED
    assert rigs[OWNER].store.count("quiesce") == 1
    assert rigs[OWNER].effects.count("give_back_fetch_pages") == 1
    assert rigs[FOLLOWER].store.count("quiesce") == 0
    assert rigs[FOLLOWER].effects.count("give_back_fetch_pages") == 0
    for rig in rigs:
        assert rig.effects.count("unpark") == 0
        assert rig.fetch_record(1).retries_left == 0
        assert rig.coord.plan_fetch(req) is DEFER  # planned again next round

    advance_all(world, rigs, req, 2.0 + UNLAUNCHED_TIMEOUT_S)  # the owner decides once more
    assert isinstance(rigs[OWNER].coord.plan_fetch(req), FetchPlan)
    assert rigs[FOLLOWER].coord.plan_fetch(req) is DEFER
    assert propagate(rigs, req) == [(1, (END, "store"))]
    assert isinstance(rigs[FOLLOWER].coord.plan_fetch(req), FetchPlan)
    assert rigs[FOLLOWER].store.count("probe") == 0


# ---- host-first: the follower starts its own landing when it adopts the answer ----


def test_follower_starts_its_landing_when_it_adopts_a_host_first_answer(world):
    rigs = [
        Rig(sources=("host",), unlaunched_timeout_s=UNLAUNCHED_TIMEOUT_S, **world.rig_kwargs(r))
        for r in range(world.n)
    ]
    req = store_request()
    advance_all(world, rigs, req, 0.0)
    assert rigs[OWNER].record(1)["state"] == "STAGING"  # decided here: landing started here
    assert rigs[FOLLOWER].records() == [] and rigs[FOLLOWER].host.count("fetch_to_host") == 0

    rigs[FOLLOWER].coord.adopt_plan_answers([req], rigs[OWNER].coord.export_plan_answers(), 0.0)
    assert rigs[FOLLOWER].record(1)["state"] == "STAGING"
    assert (
        rigs[FOLLOWER].host.count("fetch_to_host") == 1 and rigs[FOLLOWER].host.count("probe") == 0
    )
    assert rigs[FOLLOWER].coord.plan_fetch(req) is DEFER

    for rig in rigs:
        rig.host.landings[-1].deliver_all()
    advance_all(world, rigs, req, 1.0)
    for rig in rigs:
        assert rig.record(1)["state"] == "STAGED" and isinstance(
            rig.coord.plan_fetch(req), FetchPlan
        )

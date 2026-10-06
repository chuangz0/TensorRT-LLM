# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``KVTransferCoordinator`` driven from the loop entry points, one edge of design §4.1 per test.

Single rank. Assertions are on the record table (via ``status_dump``), the effects the engine was
asked to perform, and the calls each backend saw -- including the order of ``quiesce`` relative
to ``give_back_fetch_pages`` (the release point, design §4.3).
"""

import pytest

__extra_import_path__ = ["~/tensorrt_llm/_torch"]
from disaggregation.base.cache_backend import Cancelled, Delivered, Failed  # noqa: E402
from disaggregation.remote_cache import DEFER, FetchPlan  # noqa: E402
from fakes import (  # noqa: E402
    FakeChunk,
    FakePlacingPublishes,
    FakePublishes,
    FakeRequest,
    PeerGather,
    Rig,
    extent_names,
    full_attention,
    ordinals_by_group,
    plan_unit_names,
    windowed,
    worker_request,
)

END = 28  # prompt_len 29, tpb 4


def quiesce_indices(trace):
    return [i for i, (name, _) in enumerate(trace) if name == "quiesce"]


def effect_indices(trace, name):
    return [i for i, (n, _) in enumerate(trace) if n == name]


def gen_init_request() -> FakeRequest:
    return FakeRequest(7, prompt_len=30, is_gen_init=True, route_hints={"ctx": {"peer": "c"}})


def loop_advance(rig: Rig, req: FakeRequest, now: float) -> None:
    """One round's ``advance`` as the engine hooks issue it: the request is a candidate only
    while its answer is ``DEFER``."""
    rig.coord.advance([req] if rig.coord.plan_fetch(req) is DEFER else [], now)


def drive_rounds(rig: Rig, req: FakeRequest, rounds: int) -> int:
    """The engine loop for one request: ``advance``, then launch when planned, round after round,
    until the answer is ``None`` (compute locally). Returns how many rounds it took."""
    for round_index in range(rounds):
        loop_advance(rig, req, float(round_index))
        plan = rig.coord.plan_fetch(req)
        if plan is None:
            return round_index
        if isinstance(plan, FetchPlan):
            rig.coord.launch_fetches([req], float(round_index))
    raise AssertionError(f"request {req.py_request_id} still undecided after {rounds} rounds")


# ---- PLANNED -> IN_FLIGHT -> LANDED -> released ----


def test_undecided_request_answers_defer_and_has_no_record():
    rig = Rig()
    req = worker_request()
    assert rig.coord.plan_fetch(req) is DEFER
    assert rig.records() == []
    assert rig.coord.has_inflight() is False


def test_advance_plans_and_plan_fetch_reads_the_plan():
    rig = Rig()
    req = worker_request()
    rig.coord.advance([req], 0.0)
    plan = rig.coord.plan_fetch(req)
    assert isinstance(plan, FetchPlan) and plan.token_end == END and plan.source == "worker"
    rec = rig.record(1)
    assert rec["state"] == "PLANNED" and rec["attempts"] == 0 and rec["token_end"] == END
    assert rig.effects.calls == []  # planning has no side effects
    assert rig.coord.has_inflight() is False


def test_launch_opens_route_fetches_and_parks():
    rig = Rig()
    req = worker_request()
    attempt = rig.plan_and_launch(req)
    assert rig.fetch_record(1).waiting_since is None  # the wait for pages ended at the launch
    assert rig.effects.names() == ["prepare_fetch_resources", "park_for_fetch"]
    assert rig.effects.only("park_for_fetch") == [((req,),)]
    assert [m for m, _ in rig.worker.calls] == ["open_route", "fetch"]
    assert rig.worker.calls[0][1] == (req.route_hints["ctx"],)
    extent, route = rig.worker.calls[1][1]
    assert route is rig.worker.routes[0] and route.closed == 0
    assert extent_names(extent) == plan_unit_names(rig.plans[1])
    assert attempt.payload is extent
    rec = rig.record(1)
    assert rec["state"] == "IN_FLIGHT" and rec["attempts"] == 1 and rec["try_index"] == 0
    assert rig.coord.has_inflight() is True
    # Past PLANNED there is nothing more to plan: the scheduler hook answers None.
    assert rig.coord.plan_fetch(req) is None
    # The store is never asked for a route.
    assert rig.store.count("open_route") == 0


def test_launch_is_a_noop_for_requests_without_a_plan():
    rig = Rig()
    req = worker_request()
    rig.coord.launch_fetches([req], 0.0)
    assert rig.effects.calls == [] and rig.worker.calls == []


def test_in_flight_attempt_is_polled_each_advance_without_effects():
    rig = Rig()
    req = worker_request()
    attempt = rig.plan_and_launch(req)
    rig.coord.advance([], 1.0)
    rig.coord.advance([], 2.0)
    assert attempt.polls == 2
    assert rig.effects.names() == ["prepare_fetch_resources", "park_for_fetch"]
    assert rig.record(1)["state"] == "IN_FLIGHT"


def test_delivered_lands_unparks_and_closes_route_without_quiesce():
    rig = Rig()
    req = worker_request()
    attempt = rig.plan_and_launch(req)
    attempt.deliver_all()
    rig.coord.advance([], 1.0)
    assert rig.effects.only("unpark") == [(req, END, False, None)]
    assert rig.worker.routes[0].closed == 1
    assert rig.worker.count("quiesce") == 0  # not a release point yet
    rec = rig.record(1)
    assert rec["state"] == "LANDED" and rec["outcomes"] == ["Delivered"]
    assert rig.coord.has_inflight() is False
    assert rig.coord.plan_fetch(req) is None  # decided: compute locally from here


def test_request_end_releases_landed_fetch_with_exactly_one_quiesce():
    rig = Rig()
    req = worker_request()
    attempt = rig.plan_and_launch(req)
    attempt.deliver_all()
    rig.coord.advance([], 1.0)
    rig.coord.notify_request_finished(req)
    quiesces = [args for m, args in rig.worker.calls if m == "quiesce"]
    assert quiesces == [((attempt,), True)]
    assert rig.records() == []
    assert rig.worker.routes[0].closed == 1  # idempotent close, not closed twice
    assert rig.coord.status_dump() == {
        "plan_authority": "VOTED",
        "records": [],
        "decided_plans": 0,
        "finished_pending": [],
    }
    # Nothing else happened at the release point of a landed fetch.
    assert rig.effects.names() == ["prepare_fetch_resources", "park_for_fetch", "unpark"]


def test_notify_request_finished_is_idempotent():
    rig = Rig()
    req = worker_request()
    attempt = rig.plan_and_launch(req)
    attempt.deliver_all()
    rig.coord.advance([], 1.0)
    rig.coord.notify_request_finished(req)
    rig.coord.notify_request_finished(req)
    assert rig.worker.count("quiesce") == 1


@pytest.mark.parametrize("late", ["delivered", "failed"])
def test_request_end_while_in_flight_abandons_then_releases_on_outcome(late):
    rig = Rig()
    req = worker_request()
    attempt = rig.plan_and_launch(req)
    rig.coord.notify_request_finished(req)
    # Never quiesce on the engine thread for a transfer still running: abandon it and hold the
    # request so its pages stay put until the backend is done with them.
    assert rig.worker.count("quiesce") == 0
    rec = rig.record(1)
    assert rec["state"] == "IN_FLIGHT"
    assert rig.worker.routes[0].closed == 0
    assert rig.effects.names()[-1:] == ["hold_for_transfer"]
    assert rig.effects.only("hold_for_transfer") == [((req,),)]
    assert rig.coord.status_dump()["finished_pending"] == [1]

    if late == "delivered":
        attempt.deliver_all()
    else:
        attempt.finish(Failed("late"))
    rig.coord.advance([], 5.0)
    assert rig.worker.count("quiesce") == 1
    assert rig.worker.routes[0].closed == 1
    assert rig.records() == []
    assert rig.effects.names() == [
        "prepare_fetch_resources",
        "park_for_fetch",
        "hold_for_transfer",
        "terminate_request",
    ]
    assert rig.effects.only("terminate_request") == [(req,)]
    assert quiesce_indices(rig.trace)[0] < effect_indices(rig.trace, "terminate_request")[0]
    assert rig.coord.status_dump() == {
        "plan_authority": "VOTED",
        "records": [],
        "decided_plans": 0,
        "finished_pending": [],
    }


def test_request_end_while_in_flight_survives_its_deadline():
    rig = Rig(fetch_timeout_s=10.0)
    req = worker_request()
    attempt = rig.plan_and_launch(req, now=0.0)
    rig.coord.notify_request_finished(req)
    rig.coord.advance([], 50.0)  # past the deadline: already abandoned, still held
    assert rig.record(1)["state"] == "IN_FLIGHT" and rig.effects.count("terminate_request") == 0
    attempt.deliver_all()
    rig.coord.advance([], 51.0)
    assert rig.effects.count("terminate_request") == 1 and rig.records() == []


# ---- FAILED -> PLANNED (retry) -> FAILED -> released ----


def test_failed_quiesces_before_give_back_and_keeps_one_retry():
    rig = Rig()
    req = worker_request()
    attempt = rig.plan_and_launch(req)
    attempt.finish(Failed("link down"))
    rig.coord.advance([], 1.0)

    q, gb = quiesce_indices(rig.trace), effect_indices(rig.trace, "give_back_fetch_pages")
    assert len(q) == 1 and len(gb) == 1 and q[0] < gb[0]
    assert rig.effects.only("give_back_fetch_pages") == [((req,),)]
    assert rig.effects.count("unpark") == 0 and rig.effects.count("fail_requests") == 0
    assert rig.worker.routes[0].closed == 1
    rec = rig.record(1)
    assert rec["state"] == "PLANNED" and rec["token_end"] is None
    assert rig.coord.plan_fetch(req) is DEFER  # planned but not yet re-decided
    assert rig.coord.has_inflight() is False


def test_second_failure_releases_and_request_is_replanned_as_none():
    rig = Rig()
    req = worker_request()
    rig.plan_and_launch(req).finish(Failed("first"))
    rig.coord.advance([], 1.0)

    second = rig.plan_and_launch(req, now=2.0)  # re-plan + re-launch: try 1
    assert rig.record(1)["try_index"] == 1 and rig.record(1)["attempts"] == 2
    assert rig.worker.count("open_route") == 2
    second.finish(Failed("second"))
    rig.coord.advance([], 3.0)

    q, gb = quiesce_indices(rig.trace), effect_indices(rig.trace, "give_back_fetch_pages")
    assert len(q) == 2 and len(gb) == 2 and q[0] < gb[0] < q[1] < gb[1]
    # The second quiesce covers the current try only; the first try was quiesced when it failed.
    assert rig.worker.calls[-1] == ("quiesce", ((second,), True))
    assert rig.records() == []
    assert rig.coord.plan_fetch(req) is None
    assert all(r.closed == 1 for r in rig.worker.routes)
    assert rig.effects.count("fail_requests") == 0  # local fallback, not a failure


def test_cancelled_by_peer_counts_as_failed():
    rig = Rig()
    req = worker_request()
    rig.plan_and_launch(req).finish(Cancelled(by_peer=True))
    rig.coord.advance([], 1.0)
    assert rig.effects.count("give_back_fetch_pages") == 1
    assert rig.record(1)["state"] == "PLANNED"
    assert rig.payloads()[-1][0] == [((1, "fetch"), "FAILED", 0)]


def test_local_cancel_is_delivered_nothing_not_a_failure():
    rig = Rig()
    req = worker_request()
    rig.plan_and_launch(req).finish(Cancelled(by_peer=False))
    rig.coord.advance([], 1.0)
    # On the wire it is a short serve (TERMINAL with B = 0), so the retry path applies.
    assert rig.payloads()[-1][0] == [((1, "fetch"), "TERMINAL", 0)]
    assert rig.effects.count("give_back_fetch_pages") == 1
    assert rig.effects.count("fail_requests") == 0
    assert rig.record(1)["state"] == "PLANNED"
    # The hint is 0, so the retry has nothing to aim for and the request computes locally.
    rig.coord.advance([req], 2.0)
    assert rig.coord.plan_fetch(req) is None and rig.records() == []


def test_gen_init_local_cancel_fails_the_request():
    rig = Rig()
    req = FakeRequest(7, prompt_len=30, is_gen_init=True, route_hints={"ctx": {"peer": "c"}})
    rig.plan_and_launch(req).finish(Cancelled(by_peer=False))
    rig.coord.advance([], 1.0)
    assert rig.effects.only("fail_requests") == [((req,), "kv fetch served short")]


def test_second_short_serve_quiesces_only_the_second_try():
    rig = Rig()
    req = worker_request()
    first = rig.plan_and_launch(req)
    first.deliver_all_but(*rig.reader.unit_names(req, [6]))
    rig.coord.advance([], 1.0)
    second = rig.plan_and_launch(req, now=2.0)
    assert rig.plans[1].token_end == END - 4
    second.deliver_all_but(*rig.reader.unit_names(req, [5]))
    rig.coord.advance([], 3.0)

    quiesces = [args for m, args in rig.worker.calls if m == "quiesce"]
    assert quiesces == [((first,), True), ((second,), True)]
    assert len(rig.worker.routes) == 2 and [r.closed for r in rig.worker.routes] == [1, 1]
    assert rig.records() == [] and rig.coord.plan_fetch(req) is None


@pytest.mark.parametrize("gen_init", [False, True])
def test_transport_error_and_rejection_alternation_is_bounded(gen_init):
    """A failed route and a refused submission are the same thing to the record: a launch that
    never started. Neither re-plans the request on this rank alone; together they count towards
    ``MAX_CONSECUTIVE_LAUNCH_FAILURES``, and the failure lands through the ranks' agreement."""
    rig = Rig()
    req = gen_init_request() if gen_init else worker_request()
    rid = req.py_request_id

    # 1. transport error, 2. rejection, 3. transport error: the plan survives each of them.
    for index, failure in enumerate(("route", "reject", "route")):
        if failure == "route":
            rig.worker.open_route_errors.append(RuntimeError(f"t{index}"))
        else:
            rig.worker.reject_next = 1
        loop_advance(rig, req, float(index))
        assert isinstance(rig.coord.plan_fetch(req), FetchPlan)
        rig.coord.launch_fetches([req], float(index))
        rec = rig.fetch_record(rid)
        assert rig.record(rid)["state"] == "PLANNED" and rec.plan is not None
        assert rec.consecutive_launch_failures == index + 1 and rec.retries_left == 1
    assert rig.effects.count("give_back_fetch_pages") == 3 and rig.worker.count("quiesce") == 0
    # At the cap the rank gives up: DEFER to the scheduler, nothing failed yet.
    assert rig.fetch_record(rid).launch_gave_up and rig.coord.plan_fetch(req) is DEFER
    assert rig.effects.count("fail_requests") == 0

    # 4. Its FAILED vote lands on the next advance.
    loop_advance(rig, req, 3.0)
    assert rig.payloads()[-1][0] == [((rid, "fetch"), "FAILED", 0)]
    if gen_init:
        assert rig.effects.only("fail_requests") == [((req,), "kv fetch launch given up")]
        assert rig.records() == [] and rig.coord.plan_fetch(req) is None
    else:
        rec = rig.fetch_record(rid)
        assert rec.retries_left == 0 and rec.plan is None
        assert not rec.launch_gave_up and rec.consecutive_launch_failures == 0
        assert rig.coord.plan_fetch(req) is DEFER  # planned afresh next round
        assert rig.effects.count("fail_requests") == 0


def test_quiesce_false_at_release_gate_holds_the_request():
    """The backend cannot vouch for the pages: fatal, and the request is held so the engine does
    not free them; the poisoned record is never asked about again and never released."""
    rig = Rig()
    req = worker_request()
    attempt = rig.plan_and_launch(req)
    attempt.deliver_all()
    rig.coord.advance([], 1.0)
    rig.worker.quiesce_answers.append(False)
    assert rig.coord.notify_request_finished(req, 2.0) is False
    assert rig.effects.count("fail_fatal") == 1
    assert rig.effects.only("hold_for_transfer") == [((req,),)]
    assert rig.effects.count("terminate_request") == 0
    assert rig.record(1)["state"] == "LANDED"
    assert rig.coord.held_request_ids() == {1} and rig.coord.inflight_request_ids() == {1}
    assert rig.coord.status_dump()["finished_pending"] == [1]
    rig.coord.advance([], 3.0)
    assert rig.worker.count("quiesce") == 1 and rig.effects.count("terminate_request") == 0
    assert rig.record(1)["state"] == "LANDED" and rig.payloads()[-1][0] == []


# ---- open_route failures ----


@pytest.mark.parametrize(
    "error", [NotImplementedError("single destination"), ValueError("bad hint")]
)
def test_route_refused_gives_up_and_settles_on_local_compute_after_two_agreements(error):
    rig = Rig()
    req = worker_request()
    rig.worker.open_route_errors.append(error)
    rig.coord.advance([req], 0.0)
    rig.coord.launch_fetches([req], 0.0)
    assert rig.effects.names() == ["prepare_fetch_resources", "give_back_fetch_pages"]
    assert rig.worker.count("fetch") == 0 and rig.worker.count("quiesce") == 0
    # This plan can never work here: the rank gives up at once and answers the scheduler DEFER
    # until the ranks agree, rather than deciding "compute locally" on its own.
    assert rig.fetch_record(1).launch_gave_up and rig.coord.plan_fetch(req) is DEFER

    # First agreement: the retry is spent and the request is planned again, on the same route
    # (the planner knows nothing of the refusal) ...
    loop_advance(rig, req, 1.0)
    assert rig.fetch_record(1).retries_left == 0 and rig.coord.plan_fetch(req) is DEFER
    rig.worker.open_route_errors.append(error)
    loop_advance(rig, req, 2.0)
    rig.coord.launch_fetches([req], 2.0)
    assert rig.fetch_record(1).launch_gave_up
    # ... which is refused again; the second agreement settles on local compute.
    loop_advance(rig, req, 3.0)
    assert rig.records() == [] and rig.coord.plan_fetch(req) is None
    assert rig.effects.count("give_back_fetch_pages") == 2
    assert rig.effects.count("fail_requests") == 0


def test_gen_init_route_refused_fails_the_request_at_the_next_agreement():
    rig = Rig()
    req = gen_init_request()
    rig.worker.open_route_errors.append(ValueError("unknown peer"))
    rig.coord.advance([req], 0.0)
    rig.coord.launch_fetches([req], 0.0)
    # NB-5: pages first; the verdict waits for the ranks' agreement on the next advance.
    assert rig.effects.names() == ["prepare_fetch_resources", "give_back_fetch_pages"]
    assert rig.coord.plan_fetch(req) is DEFER and rig.record(7)["state"] == "PLANNED"
    loop_advance(rig, req, 1.0)
    assert rig.effects.names()[-1:] == ["fail_requests"]
    ((reqs, reason),) = rig.effects.only("fail_requests")
    assert reqs == (req,) and reason == "kv fetch launch given up"
    assert rig.coord.plan_fetch(req) is None and rig.records() == []


def test_route_transport_error_keeps_the_plan_and_the_retry():
    rig = Rig()
    req = worker_request()
    rig.worker.open_route_errors.append(RuntimeError("peer metadata fetch failed"))
    rig.coord.advance([req], 0.0)
    rig.coord.launch_fetches([req], 0.0)
    assert rig.effects.names() == ["prepare_fetch_resources", "give_back_fetch_pages"]
    assert rig.worker.count("fetch") == 0 and rig.worker.count("quiesce") == 0
    # The record keeps its plan: the scheduler reserves for the same fetch again next round.
    rec = rig.record(1)
    assert rec["state"] == "PLANNED" and rec["attempts"] == 0 and rec["token_end"] == END
    assert rig.coord.plan_fetch(req) is rig.fetch_record(1).plan
    assert rig.fetch_record(1).retries_left == 1

    # The route works next round; a real failure afterwards still has its retry.
    rig.plan_and_launch(req, now=1.0).finish(Failed("later"))
    rig.coord.advance([], 2.0)
    assert rig.record(1)["state"] == "PLANNED" and rig.fetch_record(1).retries_left == 0


def test_repeated_route_transport_errors_give_up_then_settle_on_local_compute():
    rig = Rig()
    req = worker_request()
    rig.worker.open_route_errors.extend(RuntimeError(f"e{i}") for i in range(6))
    drive_rounds(rig, req, rounds=12)
    # Three errors give the plan up; the agreement spends the retry on a fresh plan; three more
    # give that up too; the next agreement settles on local compute.
    assert rig.worker.count("open_route") == 6 and rig.worker.count("fetch") == 0
    assert rig.effects.count("give_back_fetch_pages") == 6
    assert rig.effects.count("fail_requests") == 0
    assert rig.records() == [] and rig.coord.plan_fetch(req) is None


def test_repeated_route_transport_errors_fail_gen_init():
    rig = Rig()
    req = gen_init_request()
    rig.worker.open_route_errors.extend(RuntimeError(f"e{i}") for i in range(3))
    for round_index in range(3):
        loop_advance(rig, req, float(round_index))
        rig.coord.launch_fetches([req], float(round_index))
    assert rig.effects.count("fail_requests") == 0 and rig.coord.plan_fetch(req) is DEFER
    loop_advance(rig, req, 3.0)  # the FAILED vote lands: a gen-init fetch has no retry
    ((reqs, reason),) = rig.effects.only("fail_requests")
    assert reqs == (req,) and reason == "kv fetch launch given up"
    assert rig.effects.names()[-2:] == ["give_back_fetch_pages", "fail_requests"]
    assert rig.records() == [] and rig.coord.plan_fetch(req) is None


def test_route_failure_does_not_strand_other_requests_in_the_same_launch():
    rig = Rig()
    a, b, c = worker_request(1), worker_request(2), worker_request(3)
    rig.worker.open_route_errors.append(RuntimeError("only a's route fails"))
    rig.coord.advance([a, b, c], 0.0)
    rig.coord.launch_fetches([a, b, c], 0.0)
    assert rig.effects.only("prepare_fetch_resources") == [((a, b, c),)]
    assert rig.effects.only("give_back_fetch_pages") == [((a,),)]
    assert rig.effects.only("park_for_fetch") == [((b, c),)]
    assert rig.record(1)["state"] == "PLANNED" and rig.record(1)["attempts"] == 0
    assert rig.record(2)["state"] == "IN_FLIGHT" and rig.record(3)["state"] == "IN_FLIGHT"


# ---- aligned prompt ----


def test_aligned_gen_init_prompt_lands():
    # prompt_len 28 = 7 blocks exactly; the reader names 6 (the last prompt token is not
    # reusable), the plan asks for ordinals 0..6 and the nameless block 6 is skipped on both sides.
    rig = Rig()
    req = FakeRequest(7, prompt_len=28, is_gen_init=True, route_hints={"ctx": {"peer": "c"}})
    attempt = rig.plan_and_launch(req)
    assert rig.plans[7].token_end == 28 and ordinals_by_group(rig.plans[7]) == {0: tuple(range(7))}
    assert len(attempt.payload.units) == 6
    attempt.deliver_all()
    rig.coord.advance([], 1.0)
    assert rig.effects.only("unpark") == [(req, 28, True, None)]


def test_aligned_context_prompt_fetches_to_the_last_nameable_block():
    rig = Rig()
    req = worker_request(1, prompt_len=28)
    attempt = rig.plan_and_launch(req)
    assert rig.plans[1].token_end == 24 and len(attempt.payload.units) == 6
    attempt.deliver_all()
    rig.coord.advance([], 1.0)
    assert rig.effects.only("unpark") == [(req, 24, False, None)]


def test_served_short_retries_with_min_b():
    rig = Rig()
    req = worker_request()
    attempt = rig.plan_and_launch(req)
    plan = rig.plans[1]
    last_block = rig.reader.unit_names(req, [6])
    attempt.finish(Delivered(plan_unit_names(plan) - last_block))
    rig.coord.advance([], 1.0)
    assert rig.effects.count("give_back_fetch_pages") == 1 and rig.effects.count("unpark") == 0
    assert rig.record(1)["state"] == "PLANNED"

    rig.coord.advance([req], 2.0)
    replan = rig.coord.plan_fetch(req)
    assert isinstance(replan, FetchPlan) and replan.token_end == END - 4
    assert ordinals_by_group(replan) == {0: tuple(range(6))}

    rig.coord.launch_fetches([req], 2.0)
    rig.worker.attempts[-1].deliver_all()
    rig.coord.advance([], 3.0)
    assert rig.effects.only("unpark") == [(req, END - 4, False, None)]


def test_short_served_after_the_retry_gives_up_locally():
    rig = Rig()
    req = worker_request()
    rig.plan_and_launch(req).deliver_all_but(*rig.reader.unit_names(req, [6]))
    rig.coord.advance([], 1.0)
    rig.plan_and_launch(req, now=2.0).deliver_all_but(*rig.reader.unit_names(req, [5]))
    rig.coord.advance([], 3.0)
    assert rig.records() == [] and rig.coord.plan_fetch(req) is None
    assert rig.effects.count("give_back_fetch_pages") == 2


def test_quiesce_false_is_fatal_and_pages_are_not_given_back():
    rig = Rig()
    req = worker_request()
    rig.worker.quiesce_answers.append(False)
    rig.plan_and_launch(req).finish(Failed("x"))
    rig.coord.advance([], 1.0)
    assert rig.effects.count("fail_fatal") == 1
    assert rig.effects.count("give_back_fetch_pages") == 0
    assert rig.record(1)["state"] == "FAILED"
    assert rig.worker.routes[0].closed == 0
    # The refusal is final: the request's end holds the request without asking the backend
    # again or making the engine fatal twice.
    assert rig.coord.notify_request_finished(req, 2.0) is False
    assert rig.worker.count("quiesce") == 1 and rig.effects.count("fail_fatal") == 1
    assert rig.effects.names()[-1:] == ["hold_for_transfer"]
    rig.coord.advance([], 3.0)
    assert rig.worker.count("quiesce") == 1 and rig.effects.count("terminate_request") == 0


# ---- SubmissionRejected ----


def test_submission_rejected_gives_back_without_quiesce_or_retry_cost():
    rig = Rig()
    req = worker_request()
    rig.worker.reject_next = 1
    rig.coord.advance([req], 0.0)
    plan = rig.coord.plan_fetch(req)
    rig.coord.launch_fetches([req], 0.0)
    # NB-5: the resources were prepared for the launch, so they are given back, in that order.
    assert rig.effects.names() == ["prepare_fetch_resources", "give_back_fetch_pages"]
    assert rig.worker.count("quiesce") == 0
    assert rig.worker.routes[0].closed == 1
    # The record keeps its plan: the same fetch is reserved for and launched again next round,
    # the retry budget untouched; only the run of failed launches is counted.
    rec = rig.record(1)
    assert rec["state"] == "PLANNED" and rec["token_end"] == END and rec["attempts"] == 0
    assert rig.coord.plan_fetch(req) is plan
    assert rig.fetch_record(1).consecutive_launch_failures == 1
    assert rig.fetch_record(1).retries_left == 1

    # The retry budget is intact: a real failure afterwards still gets its one retry.
    rig.plan_and_launch(req, now=1.0).finish(Failed("later"))
    rig.coord.advance([], 2.0)
    assert rig.record(1)["state"] == "PLANNED"


# ---- expiry ----


def test_gen_init_expiry_fails_the_request_and_holds_its_pages_until_the_outcome():
    """A gen-init fetch expires like any other: the request fails now, the pages stay held
    while the attempt may still write them, and the late outcome only releases."""
    rig = Rig(fetch_timeout_s=10.0)
    req = FakeRequest(7, prompt_len=30, is_gen_init=True, route_hints={"ctx": {"peer": "c"}})
    attempt = rig.plan_and_launch(req, now=0.0)
    assert rig.plans[7].no_local_fallback is True
    rig.coord.advance([], 9.9)
    assert rig.effects.count("fail_requests") == 0
    rig.coord.advance([], 10.0)
    assert rig.worker.count("quiesce") == 0 and rig.effects.count("give_back_fetch_pages") == 0
    assert rig.effects.names()[-2:] == ["fail_requests", "hold_for_transfer"]
    assert rig.effects.only("fail_requests") == [((req,), "kv fetch timed out")]
    rec = rig.record(7)
    assert rec["state"] == "IN_FLIGHT" and rec["expired"]
    # The late delivery lands nothing: the release point, then the held request is terminated.
    attempt.deliver_all()
    rig.coord.advance([], 11.0)
    assert rig.worker.count("quiesce") == 1 and rig.effects.count("unpark") == 0
    assert rig.effects.names()[-1:] == ["terminate_request"]
    assert rig.records() == [] and rig.coord.plan_fetch(req) is DEFER


def test_gen_init_failure_does_not_retry():
    rig = Rig()
    req = FakeRequest(7, prompt_len=30, is_gen_init=True, route_hints={"ctx": {"peer": "c"}})
    rig.plan_and_launch(req).finish(Failed("x"))
    rig.coord.advance([], 1.0)
    assert rig.effects.only("fail_requests") == [((req,), "kv fetch failed")]
    assert rig.records() == []


def test_gen_init_delivery_unparks_at_prompt_len_with_no_local_fallback():
    rig = Rig()
    req = FakeRequest(7, prompt_len=30, is_gen_init=True, route_hints={"ctx": {"peer": "c"}})
    rig.plan_and_launch(req).deliver_all()
    rig.coord.advance([], 1.0)
    assert rig.effects.only("unpark") == [(req, 30, True, None)]


def test_normal_fetch_expiry_fails_the_request_and_holds_its_pages_until_the_outcome():
    """A fetch past its deadline fails the request; its pages stay held (the backend may still
    write them) until the outcome arrives, and a late Delivered lands nothing."""
    rig = Rig(fetch_timeout_s=10.0)
    req = worker_request()
    attempt = rig.plan_and_launch(req, now=0.0)
    rig.coord.advance([], 10.0)
    rec = rig.record(1)
    assert rec["state"] == "IN_FLIGHT" and rec["expired"]
    assert rig.effects.names() == [
        "prepare_fetch_resources",
        "park_for_fetch",
        "fail_requests",
        "hold_for_transfer",
    ]
    assert rig.effects.only("fail_requests") == [((req,), "kv fetch timed out")]
    attempt.deliver_all()
    rig.coord.advance([], 20.0)
    assert rig.effects.names()[-1:] == ["terminate_request"]
    assert rig.effects.count("unpark") == 0 and rig.effects.count("give_back_fetch_pages") == 0
    assert rig.worker.count("quiesce") == 1
    assert rig.records() == [] and rig.coord.status_dump()["finished_pending"] == []


def test_no_timeout_means_no_deadline():
    rig = Rig()
    req = worker_request()
    rig.plan_and_launch(req, now=0.0)
    rig.coord.advance([], 1e9)
    rec = rig.record(1)
    assert rec["deadline"] is None and rec["expired"] is False


# ---- publish ----


def test_publish_flow_holds_then_terminates_after_quiesce():
    pub = FakePublishes(name="store")
    rig = Rig(publishers=[pub])
    req = FakeRequest(3, prompt_len=29)
    ext = rig.reader.script_publish(req, [(range(0, 4), False, None), (range(4, 7), True, None)])

    rig.coord.publish_committed_blocks([req], now=0.0)
    # A publisher that does not place pieces is not offered an intermediate piece.
    assert pub.payloads("publish") == []
    rec = rig.record(3, "publish")
    assert rec["state"] == "PLANNED" and rec["attempts"] == 0
    assert rig.coord.has_inflight() is False
    rig.coord.advance([], 1.0)
    assert rig.record(3, "publish")["state"] == "PLANNED" and pub.count("quiesce") == 0

    rig.coord.publish_committed_blocks([req], now=2.0)
    rig.coord.notify_request_finished(req)
    assert pub.payloads("publish") == [ext[1]]
    assert rig.record(3, "publish")["state"] == "IN_FLIGHT" and rig.coord.has_inflight()
    assert rig.effects.names() == ["hold_for_transfer"]
    assert rig.effects.only("hold_for_transfer") == [((req,),)]
    assert rig.coord.status_dump()["finished_pending"] == [3]

    pub.attempts[0].deliver_all()
    rig.coord.advance([], 3.0)
    assert pub.count("quiesce") == 1
    quiesced = [args[0] for m, args in pub.calls if m == "quiesce"][0]
    assert set(quiesced) == set(pub.attempts)
    assert rig.effects.names() == ["hold_for_transfer", "terminate_request"]
    assert rig.effects.only("terminate_request") == [(req,)]
    assert rig.records() == [] and rig.coord.status_dump()["finished_pending"] == []
    q = quiesce_indices(rig.trace)
    assert q[0] < effect_indices(rig.trace, "terminate_request")[0]


def test_publish_landed_before_request_end_terminates_immediately_at_end():
    pub = FakePublishes()
    rig = Rig(publishers=[pub])
    req = FakeRequest(3, prompt_len=29)
    rig.coord.publish_committed_blocks([req], now=0.0)  # default: everything, is_last
    pub.attempts[0].deliver_all()
    rig.coord.advance([], 1.0)
    assert rig.records() == [] and pub.count("quiesce") == 1
    assert rig.effects.calls == []  # request still running: nothing to tell the engine
    rig.coord.notify_request_finished(req)
    assert rig.effects.calls == []  # release already happened; the engine terminates as usual
    assert rig.coord.status_dump()["finished_pending"] == []


def test_publish_failure_after_request_end_terminates_the_held_request():
    """The request already answered its client; a publish that then fails is the store's loss,
    not the request's: held while in flight, then terminated, never failed."""
    pub = FakePublishes()
    rig = Rig(publishers=[pub])
    req = FakeRequest(3, prompt_len=29)
    rig.coord.publish_committed_blocks([req], now=0.0)
    rig.coord.notify_request_finished(req)
    pub.attempts[0].finish(Failed("peer gone"))
    rig.coord.advance([], 1.0)
    assert rig.effects.names() == ["hold_for_transfer", "terminate_request"]
    assert rig.effects.count("fail_requests") == 0
    assert rig.records() == []


def test_publish_expiry_warns_and_waits_for_the_outcome():
    """A publish past its deadline is never quiesced under its live attempt: the record is
    marked expired, and the outcome settles it on this rank alone."""
    pub = FakePublishes()
    rig = Rig(publishers=[pub], publish_timeout_s=5.0)
    req = FakeRequest(3, prompt_len=29)
    rig.coord.publish_committed_blocks([req], now=0.0)
    rig.coord.notify_request_finished(req, 0.0)
    rig.coord.advance([], 5.0)
    assert rig.effects.names() == ["hold_for_transfer"]
    assert pub.count("quiesce") == 0
    rec = rig.record(3, "publish")
    assert rec["state"] == "IN_FLIGHT" and rec["expired"]
    rig.coord.advance([], 6.0)
    assert rig.payloads()[-1] == ([], [], [])  # expired: no longer the ranks' business
    pub.attempts[0].deliver_all()
    rig.coord.advance([], 7.0)
    assert rig.effects.names() == ["hold_for_transfer", "terminate_request"]
    assert rig.effects.count("fail_requests") == 0
    assert pub.count("quiesce") == 1 and rig.records() == []


def test_publish_deadline_counts_from_the_first_accepted_submission():
    pub = FakePublishes()
    rig = Rig(publishers=[pub], publish_timeout_s=5.0)
    req = FakeRequest(3, prompt_len=29)
    rig.reader.script_publish(req, [(range(0, 4), False, None), (range(4, 7), True, None)])
    rig.coord.publish_committed_blocks([req], now=0.0)  # a plain publisher hears only the last
    assert rig.record(3, "publish")["deadline"] is None
    rig.coord.publish_committed_blocks([req], now=3.0)
    assert rig.record(3, "publish")["deadline"] == 8.0
    rig.coord.advance([], 7.9)
    assert rig.record(3, "publish")["expired"] is False
    rig.coord.advance([], 8.0)
    assert rig.record(3, "publish")["expired"] is True


def test_publish_rejected_outright_votes_failed_and_terminates_with_the_agreement():
    """Nothing escaped here, but a peer's publish may be in flight and the store is missing
    this rank's part: the record stays to vote FAILED, and the request is terminated with the
    agreement, in the same round everywhere."""
    pub = FakePublishes()
    rig = Rig(publishers=[pub])
    req = FakeRequest(3, prompt_len=29)
    pub.reject_next = 1
    rig.coord.publish_committed_blocks([req], now=0.0)
    assert rig.coord.notify_request_finished(req, 0.0) is False
    assert pub.attempts == [] and pub.count("quiesce") == 0  # nothing escaped, nothing to quiesce
    assert rig.effects.names() == ["hold_for_transfer"]
    assert rig.record(3, "publish")["state"] == "PLANNED"
    rig.coord.advance([], 1.0)
    assert rig.payloads()[-1][0] == [((3, "publish"), "FAILED", 0)]
    assert rig.effects.names() == ["hold_for_transfer", "terminate_request"]
    assert pub.count("quiesce") == 0
    assert rig.records() == [] and rig.coord.status_dump()["finished_pending"] == []


def test_publish_rejected_while_the_request_runs_votes_failed_and_is_released():
    """A running request's refused publish does not wait for its end: it votes FAILED, so a
    peer's in-flight publish is failed too instead of waiting on this rank's silence."""
    pub = FakePublishes()
    rig = Rig(publishers=[pub])
    req = FakeRequest(3, prompt_len=29)
    pub.reject_next = 1
    rig.coord.publish_committed_blocks([req], now=0.0)
    rig.coord.advance([], 1.0)
    assert rig.payloads()[-1][0] == [((3, "publish"), "FAILED", 0)]
    assert rig.records() == [] and rig.effects.calls == []  # a warning is the only word
    assert rig.coord.notify_request_finished(req, 2.0) is True


def test_publish_never_submitted_owes_the_finished_request_nothing():
    """A plain publisher hears only the last piece; a request cancelled before it has nothing
    in flight on any rank, so the record goes at once and the engine terminates as usual."""
    pub = FakePublishes()
    rig = Rig(publishers=[pub])
    req = FakeRequest(3, prompt_len=29)
    rig.reader.script_publish(req, [(range(0, 4), False, None), (range(4, 7), True, None)])
    rig.coord.publish_committed_blocks([req], now=0.0)
    assert rig.coord.notify_request_finished(req, 1.0) is True
    assert rig.effects.calls == [] and pub.calls == []
    assert rig.records() == [] and rig.coord.status_dump()["finished_pending"] == []


def test_quiesce_false_publish_record_is_not_released_without_quiesce():
    """A publish whose backend refused to vouch for the pages keeps its record and its pages:
    the request's end does not release it, and nothing is agreed on it again."""
    pub = FakePublishes()
    rig = Rig(publishers=[pub])
    req = FakeRequest(3, prompt_len=29)
    rig.coord.publish_committed_blocks([req], now=0.0)
    pub.attempts[0].deliver_all()
    pub.quiesce_answers.append(False)
    rig.coord.advance([], 1.0)
    assert rig.effects.only("fail_fatal") and rig.record(3, "publish")["state"] == "LANDED"
    assert rig.coord.inflight_request_ids() == {3}
    assert rig.coord.notify_request_finished(req, 2.0) is False
    assert rig.effects.names() == ["fail_fatal", "hold_for_transfer"]
    rig.coord.advance([], 3.0)
    assert pub.count("quiesce") == 1 and rig.record(3, "publish")["state"] == "LANDED"
    assert rig.effects.count("terminate_request") == 0


def test_partial_publish_rejection_terminates_the_held_request_at_next_reap():
    # Publish accepted, place rejected: the pieces that were offered may land, the publish as a
    # whole has still failed.
    placing = FakePlacingPublishes()
    rig = Rig(publishers=[placing])
    req = FakeRequest(3, prompt_len=29)
    rig.reader.script_publish(req, [(range(0, 7), True, FakeChunk(3, 0))])
    placing.reject_methods.add("place")
    rig.coord.publish_committed_blocks([req], now=0.0)
    rig.coord.notify_request_finished(req)
    assert placing.count("publish") == 1 and placing.count("place") == 1
    assert len(placing.attempts) == 1
    assert rig.effects.names() == ["hold_for_transfer"]  # one piece is genuinely in flight
    placing.attempts[0].deliver_all()
    rig.coord.advance([], 1.0)
    assert placing.count("quiesce") == 1
    assert rig.effects.names() == ["hold_for_transfer", "terminate_request"]
    assert rig.effects.count("fail_requests") == 0
    assert rig.records() == []


def test_one_of_two_publishers_rejecting_still_terminates_once_the_other_lands():
    ok, bad = FakePublishes(name="ok"), FakePublishes(name="bad")
    rig = Rig(publishers=[ok, bad])
    req = FakeRequest(3, prompt_len=29)
    bad.reject_next = 1
    rig.coord.publish_committed_blocks([req], now=0.0)
    rig.coord.notify_request_finished(req)
    assert len(ok.attempts) == 1 and bad.attempts == []
    assert rig.effects.names() == ["hold_for_transfer"]
    ok.attempts[0].deliver_all()
    rig.coord.advance([], 1.0)  # a failed publish even without polling anything from ``bad``
    assert rig.effects.names() == ["hold_for_transfer", "terminate_request"]
    assert rig.effects.count("fail_requests") == 0
    assert ok.count("quiesce") == 1 and bad.count("quiesce") == 0
    assert rig.records() == []


def test_failed_intermediate_piece_is_reported_before_is_last():
    placing = FakePlacingPublishes()
    rig = Rig(publishers=[placing])
    req = FakeRequest(3, prompt_len=29)
    rig.reader.script_publish(
        req, [(range(0, 3), False, FakeChunk(3, 0)), (range(3, 7), True, FakeChunk(3, 1))]
    )
    rig.coord.publish_committed_blocks([req], now=0.0)
    placing.attempts[0].deliver_all()  # publish of piece 0 fine ...
    placing.attempts[1].finish(Failed("place lost"))  # ... its place failed
    rig.coord.advance([], 1.0)
    # Failed as soon as any piece fails, without waiting for the last piece.
    assert placing.count("quiesce") == 1 and rig.records() == []
    assert rig.effects.calls == []  # request still running: nothing to tell the engine yet


def test_publish_with_terminal_intermediate_pieces_settles_when_the_request_ends():
    """No piece was ever the last one: the record stays in flight (more pieces may come) past
    its deadline too; the request's end says no more will, and the pieces so far settle it."""
    placing = FakePlacingPublishes()
    rig = Rig(publishers=[placing], publish_timeout_s=5.0)
    req = FakeRequest(3, prompt_len=29)
    rig.reader.script_publish(
        req, [(range(0, 3), False, FakeChunk(3, 0)), (range(3, 7), True, FakeChunk(3, 1))]
    )
    rig.coord.publish_committed_blocks([req], now=0.0)
    for a in placing.attempts:
        a.finish(Delivered(frozenset()))  # publish and place of piece 0 both done
    rig.coord.advance([], 4.0)
    assert rig.record(3, "publish")["state"] == "IN_FLIGHT"  # every piece so far landed, not last
    rig.coord.advance([], 5.0)
    assert rig.record(3, "publish")["expired"] and placing.count("quiesce") == 0
    assert rig.effects.calls == []  # request still running: the warning is the only word
    rig.coord.notify_request_finished(req, 6.0)
    assert rig.effects.names() == ["hold_for_transfer"]
    rig.coord.advance([], 7.0)
    assert placing.count("quiesce") == 1 and rig.records() == []
    assert rig.effects.names() == ["hold_for_transfer", "terminate_request"]


def test_publish_rejected_while_fetch_in_flight_terminates_once_the_fetch_releases():
    pub = FakePublishes()
    rig = Rig(publishers=[pub])
    req = worker_request()
    attempt = rig.plan_and_launch(req)
    pub.reject_next = 1
    rig.coord.publish_committed_blocks([req], now=1.0)
    rig.coord.notify_request_finished(req, 1.0)
    # Both records wait for the agreement: the fetch still names the pages, the refused
    # publish votes FAILED meanwhile.
    assert rig.effects.names()[-1:] == ["hold_for_transfer"]
    assert rig.effects.count("fail_requests") == 0
    assert rig.record(1, "publish")["state"] == "PLANNED" and rig.record(1)["state"] == "IN_FLIGHT"
    attempt.deliver_all()
    rig.coord.advance([], 2.0)
    assert rig.worker.count("quiesce") == 1 and pub.count("quiesce") == 0
    assert rig.effects.names()[-1:] == ["terminate_request"]
    assert rig.effects.count("fail_requests") == 0
    assert rig.effects.count("unpark") == 0
    assert rig.coord.status_dump() == {
        "plan_authority": "VOTED",
        "records": [],
        "decided_plans": 0,
        "finished_pending": [],
    }


def test_publish_landed_first_waits_for_the_in_flight_fetch_before_terminating():
    pub = FakePublishes()
    rig = Rig(publishers=[pub])
    req = worker_request()
    attempt = rig.plan_and_launch(req)
    rig.coord.publish_committed_blocks([req], now=1.0)
    rig.coord.notify_request_finished(req)
    assert rig.effects.names()[-1:] == ["hold_for_transfer"]
    pub.attempts[0].deliver_all()
    rig.coord.advance([], 2.0)
    assert pub.count("quiesce") == 1 and rig.record(1, "publish") is None
    assert rig.effects.count("terminate_request") == 0  # the fetch is still in flight
    attempt.deliver_all()
    rig.coord.advance([], 3.0)
    assert rig.effects.names()[-1:] == ["terminate_request"]
    assert rig.effects.count("unpark") == 0
    assert rig.coord.status_dump() == {
        "plan_authority": "VOTED",
        "records": [],
        "decided_plans": 0,
        "finished_pending": [],
    }


def test_both_records_in_flight_fetch_releases_first_then_publish_lands():
    pub = FakePublishes()
    rig = Rig(publishers=[pub])
    req = worker_request()
    attempt = rig.plan_and_launch(req)
    rig.coord.publish_committed_blocks([req], now=1.0)
    rig.coord.notify_request_finished(req)
    assert rig.effects.names()[-1:] == ["hold_for_transfer"]
    assert (
        rig.record(1)["state"] == "IN_FLIGHT" and rig.record(1, "publish")["state"] == "IN_FLIGHT"
    )

    attempt.deliver_all()
    rig.coord.advance([], 2.0)
    assert rig.record(1) is None and rig.record(1, "publish")["state"] == "IN_FLIGHT"
    assert rig.effects.count("terminate_request") == 0
    assert rig.effects.count("unpark") == 0

    pub.attempts[0].deliver_all()
    rig.coord.advance([], 3.0)
    assert rig.effects.count("terminate_request") == 1
    assert rig.effects.names()[-1:] == ["terminate_request"]
    assert rig.worker.count("quiesce") == 1 and pub.count("quiesce") == 1
    assert rig.coord.status_dump() == {
        "plan_authority": "VOTED",
        "records": [],
        "decided_plans": 0,
        "finished_pending": [],
    }


def test_no_publishers_means_no_publish_records():
    rig = Rig()
    req = FakeRequest(3, prompt_len=29)
    rig.coord.publish_committed_blocks([req], now=0.0)
    rig.coord.notify_request_finished(req)
    assert rig.records() == [] and rig.effects.calls == []
    assert rig.reader.calls == []  # publish_description is not even asked


def test_places_pieces_gets_every_chunk_plain_publisher_only_the_last():
    placing, plain = FakePlacingPublishes(name="worker"), FakePublishes(name="store")
    rig = Rig(publishers=[placing, plain])
    req = FakeRequest(3, prompt_len=29)
    chunks = [FakeChunk(3, i) for i in range(3)]
    ext = rig.reader.script_publish(
        req,
        [
            (range(0, 2), False, chunks[0]),
            (range(2, 4), False, chunks[1]),
            (range(4, 7), True, chunks[2]),
        ],
    )
    for i in range(3):
        rig.coord.publish_committed_blocks([req], now=float(i))

    assert placing.payloads("place") == chunks
    assert placing.payloads("publish") == ext
    # Design §7.5: a publisher that does not place pieces hears once, on the last piece, and
    # must not need to read ``is_last``.
    assert plain.payloads("publish") == [ext[-1]]


def test_places_pieces_not_called_without_a_chunk():
    placing = FakePlacingPublishes()
    rig = Rig(publishers=[placing])
    req = FakeRequest(3, prompt_len=29)
    rig.coord.publish_committed_blocks([req], now=0.0)
    assert placing.count("publish") == 1 and placing.count("place") == 0


# ---- CarriesAux ----


def test_aux_from_attempt_reaches_unpark():
    rig = Rig()
    req = FakeRequest(7, prompt_len=30, is_gen_init=True, route_hints={"ctx": {"peer": "c"}})
    rig.worker.aux_for_next = {"first_token": 42, "ctx_usage": 0.5}
    rig.plan_and_launch(req).deliver_all()
    rig.coord.advance([], 1.0)
    assert rig.effects.only("unpark") == [(req, 30, True, {"first_token": 42, "ctx_usage": 0.5})]


# ---- engine queue, collective, dump ----


def test_engine_queue_drained_at_advance_start_under_budget():
    rig = Rig(queue_budget=2)
    ran = []
    for i in range(3):
        rig.queue.post(lambda i=i: ran.append(i))
    rig.coord.advance([], 0.0)
    assert ran == [0, 1] and rig.queue.drains == [2]
    rig.coord.advance([], 1.0)
    assert ran == [0, 1, 2] and rig.queue.drains == [2, 2]


def test_queue_runs_before_polling():
    rig = Rig()
    req = worker_request()
    attempt = rig.plan_and_launch(req)
    rig.queue.post(attempt.deliver_all)  # a backend completing on the engine thread
    rig.coord.advance([], 1.0)
    assert rig.effects.count("unpark") == 1  # reaped in the same advance


def test_one_allgather_per_advance_and_none_elsewhere():
    rig = Rig()
    req = worker_request()
    rig.coord.advance([req], 0.0)
    rig.coord.launch_fetches([req], 0.0)
    rig.coord.advance([], 1.0)
    rig.coord.notify_request_finished(req)
    rig.coord.publish_committed_blocks([], now=2.0)
    assert len(rig.dist.calls) == 2


def test_allgather_payload_carries_plan_answers_and_arrivals():
    rig = Rig()
    req = worker_request()
    rig.coord.advance([req], 0.0)
    arrivals, expired, plans = rig.payloads()[0]
    assert arrivals == [] and expired == [] and plans == [(1, (END, "worker"))]
    rig.coord.launch_fetches([req], 0.0)
    rig.worker.attempts[0].deliver_all()
    rig.coord.advance([], 1.0)
    arrivals, expired, plans = rig.payloads()[1]
    assert arrivals == [((1, "fetch"), "TERMINAL", END)] and plans == []


def test_allgather_payload_wires_none_and_defer():
    rig = Rig(sources=("store",))
    a, b = FakeRequest(1, prompt_len=29), FakeRequest(2, prompt_len=29)
    rig.store.probe_answers.extend([frozenset(), None])  # a: holds nothing; b: unanswered
    rig.coord.advance([a, b], 0.0)
    assert rig.payloads()[0][2] == [(1, None), (2, "DEFER")]


# ---- the gather seam with hand-written peers ----


def test_peer_reporting_short_b_fails_the_local_landed_fetch():
    def peer(local):
        arrivals, expired, plans = local
        return (
            [(key, kind, token_end - 4) for key, kind, token_end in arrivals],
            expired,
            plans,
        )

    rig = Rig(dist=PeerGather(peer))
    req = worker_request()
    rig.plan_and_launch(req).deliver_all()
    rig.coord.advance([], 1.0)
    assert rig.effects.count("unpark") == 0 and rig.effects.count("give_back_fetch_pages") == 1
    assert rig.record(1)["state"] == "PLANNED"
    rig.coord.advance([req], 2.0)
    assert rig.coord.plan_fetch(req).token_end == END - 4  # MIN(B) from the peer


def test_peer_voting_a_different_source_makes_the_plan_none():
    def peer(local):
        arrivals, expired, plans = local
        return (arrivals, expired, [(rid, (v[0], "store")) for rid, v in plans])

    rig = Rig(dist=PeerGather(peer))
    req = worker_request()
    rig.coord.advance([req], 0.0)
    assert rig.coord.plan_fetch(req) is None and rig.records() == []


def test_peer_not_reporting_an_arrival_keeps_it_in_flight():
    rig = Rig(dist=PeerGather(lambda local: ([], [], local[2])))
    req = worker_request()
    rig.plan_and_launch(req).deliver_all()
    rig.coord.advance([], 1.0)
    assert rig.record(1)["state"] == "IN_FLIGHT" and rig.effects.count("unpark") == 0


def test_consensus_wait_is_bounded_by_fetch_timeout():
    """Two ranks; the peer's request is gone and it casts no vote on the record any more. The
    delivered rank waits for the agreement until its deadline, then fails the request and
    settles the record on its own word instead of waiting forever."""
    rig = Rig(dist=PeerGather(lambda local: ([], [], local[2])), fetch_timeout_s=10.0)
    req = worker_request()
    rig.plan_and_launch(req, now=0.0).deliver_all()
    rig.coord.advance([], 9.0)
    assert rig.record(1)["state"] == "IN_FLIGHT" and rig.effects.count("fail_requests") == 0
    rig.coord.advance([], 10.0)
    assert rig.effects.only("fail_requests") == [((req,), "kv fetch timed out")]
    assert rig.record(1)["expired"] and rig.effects.names()[-1:] == ["hold_for_transfer"]
    rig.coord.advance([], 11.0)  # the outcome is in: nothing more to wait for
    assert rig.payloads()[-1][0] == []
    assert rig.worker.count("quiesce") == 1 and rig.effects.count("unpark") == 0
    assert rig.effects.names()[-1:] == ["terminate_request"] and rig.records() == []


def test_status_dump_shape():
    rig = Rig(fetch_timeout_s=10.0)
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
            "launch_gave_up": False,
            "peer_launched_at": None,
            "has_landing": False,
            "waiting_since": None,
        }
    ]


# ---- planning through the coordinator ----


def test_plan_none_is_remembered_until_request_end():
    rig = Rig()
    req = FakeRequest(1, prompt_len=29)  # no hint, store never answers -> DEFER twice, then None
    rig.coord.advance([req], 0.0)
    assert rig.coord.plan_fetch(req) is DEFER
    rig.coord.advance([req], 1.0)
    assert rig.coord.plan_fetch(req) is DEFER
    rig.coord.advance([req], 2.0)
    assert rig.coord.plan_fetch(req) is None
    assert rig.records() == [] and rig.coord.status_dump()["decided_plans"] == 1
    rig.coord.notify_request_finished(req)
    assert rig.coord.status_dump()["decided_plans"] == 0
    assert rig.coord.plan_fetch(req) is DEFER


def test_store_probe_is_asked_once_and_answer_is_cached():
    rig = Rig(sources=("store",))
    req = FakeRequest(1, prompt_len=29)
    rig.store.probe_default = rig.reader.unit_names(req, range(5))
    rig.coord.advance([req], 0.0)
    assert rig.store.count("probe") == 1
    name, units = rig.store.calls[0][1]
    assert name == rig.reader.block_keys(req)[6] and len(units) == 7
    plan = rig.coord.plan_fetch(req)
    assert plan.source == "store" and plan.token_end == 20
    rig.coord.advance([req], 1.0)  # already PLANNED with a plan: not re-decided, not re-probed
    assert rig.store.count("probe") == 1


def test_store_probe_exception_keeps_the_answer_pending_until_budget_is_spent():
    rig = Rig(sources=("store",))
    req = FakeRequest(1, prompt_len=29)
    rig.store.probe_answers.extend([RuntimeError("store unreachable")] * 3)
    rig.coord.advance([req], 0.0)
    assert rig.coord.plan_fetch(req) is DEFER
    rig.coord.advance([req], 1.0)
    assert rig.coord.plan_fetch(req) is DEFER
    rig.coord.advance([req], 2.0)  # probe_timeout_s spent: plan without the store
    assert rig.coord.plan_fetch(req) is None
    assert rig.store.count("probe") == 3  # re-asked every round while pending


def test_store_probe_recovering_within_budget_plans_from_the_store():
    rig = Rig(sources=("store",))
    req = FakeRequest(1, prompt_len=29)
    rig.store.probe_answers.append(RuntimeError("blip"))
    rig.store.probe_default = rig.reader.unit_names(req, range(7))
    rig.coord.advance([req], 0.0)
    assert rig.coord.plan_fetch(req) is DEFER
    rig.coord.advance([req], 1.0)
    assert rig.coord.plan_fetch(req).source == "store"


def test_store_fetch_has_no_route():
    rig = Rig(sources=("store",))
    req = FakeRequest(1, prompt_len=29)
    rig.store.probe_default = rig.reader.unit_names(req, range(7))
    rig.coord.advance([req], 0.0)
    rig.coord.launch_fetches([req], 0.0)
    assert rig.store.count("open_route") == 0
    extent, route = rig.store.calls[-1][1]
    assert route is None
    rig.store.attempts[0].deliver_all()
    rig.coord.advance([], 1.0)
    assert rig.effects.only("unpark") == [(req, END, False, None)]


def test_gen_first_context_defers_until_ready_and_builds_no_record():
    rig = Rig()
    req = FakeRequest(
        1, prompt_len=29, is_gen_first_context=True, route_hints={"ctx": {"peer": "g"}}
    )
    rig.reader.ready[1] = False
    rig.coord.advance([req], 0.0)
    assert rig.coord.plan_fetch(req) is DEFER and rig.records() == []
    rig.reader.ready[1] = True
    rig.coord.advance([req], 1.0)
    assert isinstance(rig.coord.plan_fetch(req), FetchPlan)


def test_windowed_model_extent_carries_pruned_units():
    rig = Rig(groups=[full_attention(0), windowed(1)])
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
    rig = Rig()
    a, b = worker_request(1), worker_request(2, prompt_len=17)
    rig.coord.advance([a, b], 0.0)
    if candidates_twice:
        rig.coord.advance([a, b], 0.5)  # PLANNED with a plan: left alone
    rig.coord.launch_fetches([a, b], 1.0)
    assert rig.effects.only("park_for_fetch") == [((a, b),)]
    rig.worker.attempts[1].deliver_all()
    rig.coord.advance([], 2.0)
    assert rig.effects.only("unpark") == [(b, 16, False, None)]
    assert rig.record(1)["state"] == "IN_FLIGHT" and rig.record(2)["state"] == "LANDED"


# ---- back-pressure that never lets up ----


def test_consecutive_rejections_give_up_then_the_request_computes_locally():
    rig = Rig()
    req = worker_request()
    rig.worker.reject_next = 10
    drive_rounds(rig, req, rounds=12)
    # Three rejections give the plan up; the agreement spends the retry on a fresh plan; three
    # more give that up too; the next agreement settles on local compute.
    assert rig.worker.count("fetch") == 6
    assert rig.effects.count("give_back_fetch_pages") == 6
    assert rig.effects.count("fail_requests") == 0
    assert rig.records() == [] and rig.coord.plan_fetch(req) is None


def test_three_consecutive_rejections_fail_a_gen_init_fetch():
    rig = Rig()
    req = FakeRequest(1, prompt_len=29, is_gen_init=True, route_hints={"ctx": {"peer": "peer1"}})
    rig.worker.reject_next = 10
    for round_index in range(3):
        loop_advance(rig, req, float(round_index))
        assert isinstance(rig.coord.plan_fetch(req), FetchPlan)
        rig.coord.launch_fetches([req], float(round_index))
    assert rig.worker.count("fetch") == 3
    assert rig.effects.count("give_back_fetch_pages") == 3
    assert rig.effects.count("fail_requests") == 0 and rig.coord.plan_fetch(req) is DEFER
    loop_advance(rig, req, 3.0)  # the FAILED vote lands: a gen-init fetch has no retry
    assert rig.effects.only("fail_requests") == [((req,), "kv fetch launch given up")]
    assert rig.coord.plan_fetch(req) is None and rig.records() == []


# =============================================================================================
# Host-first fetch: PLANNED -> STAGING -> STAGED -> IN_FLIGHT (placement) -> LANDED
# =============================================================================================


def host_rig(**kw) -> Rig:
    """A single ``LandsOnHost`` store that holds every prompt."""
    return Rig(sources=("host",), **kw)


def host_request(rid: int = 1, prompt_len: int = 29) -> FakeRequest:
    return FakeRequest(rid, prompt_len)


def land_and_stage(rig: Rig, req: FakeRequest, now: float = 0.0):
    """Decide the plan (which starts the landing), deliver the landing, agree: ``STAGED``."""
    landing = rig.plan_and_land(req, now)
    landing.deliver_all()
    rig.coord.advance([], now + 1.0)
    assert rig.record(req.py_request_id)["state"] == "STAGED"
    return landing


def test_host_first_plan_starts_its_landing_when_decided_and_parks_nothing():
    rig = host_rig()
    req = host_request()
    rig.plan_and_land(req)
    assert [m for m, _ in rig.host.calls] == ["probe", "fetch_to_host"]
    name, units = rig.host.calls[-1][1]
    assert name == b"fetch:1" and frozenset(units) == rig.reader.unit_names(req, range(7))
    rec = rig.record(1)
    assert rec["state"] == "STAGING" and rec["has_landing"] and rec["attempts"] == 0
    assert rig.effects.calls == []  # no pages: nothing prepared, nothing parked
    assert rig.coord.plan_fetch(req) is DEFER
    # A landing is work in flight for pacing, but it names no page and parks no request.
    assert rig.coord.has_inflight() is True
    assert rig.coord.inflight_request_ids() == frozenset()
    assert rig.coord.parked_request_ids() == frozenset()


def test_plan_fetch_answers_defer_defer_plan_none_none_along_the_host_first_path():
    rig = host_rig()
    req = host_request()
    landing = rig.plan_and_land(req)
    assert rig.coord.plan_fetch(req) is DEFER  # STAGING: the units are on their way
    landing.deliver_all()
    rig.coord.advance([], 1.0)
    plan = rig.coord.plan_fetch(req)  # STAGED: reserve pages for exactly this plan
    assert isinstance(plan, FetchPlan) and plan.token_end == END and plan.source == "host"
    assert plan is rig.fetch_record(1).plan
    attempt = rig.reserve_and_place(req, 1.0)
    assert rig.coord.plan_fetch(req) is None  # IN_FLIGHT: parked, out of the scheduler's reach
    attempt.deliver_all()
    rig.coord.advance([], 2.0)
    assert rig.record(1)["state"] == "LANDED" and rig.coord.plan_fetch(req) is None


def test_refused_landing_keeps_the_plan_votes_unlaunched_and_is_asked_again_next_round():
    rig = host_rig()
    req = host_request()
    rig.host.reject_next = 1
    rig.coord.advance([req], 0.0)
    rec = rig.fetch_record(1)
    assert rec.state.value == "PLANNED" and rec.plan is not None and rec.landing is None
    assert rec.consecutive_launch_failures == 0  # not page back-pressure
    assert rec.waiting_since == 0.0
    assert rig.coord.plan_fetch(req) is DEFER and rig.effects.calls == []

    loop_advance(rig, req, 1.0)
    assert rig.payloads()[-1][0] == [((1, "fetch"), "UNLAUNCHED", 0)]
    assert rig.host.count("fetch_to_host") == 2 and rig.host.count("probe") == 1
    rec = rig.record(1)
    assert rec["state"] == "STAGING" and rec["waiting_since"] is None and rec["has_landing"]


@pytest.mark.parametrize("ending", ["short", "failed"])
def test_landing_that_fails_or_comes_up_short_is_released_and_replanned_without_pages(ending):
    rig = host_rig()
    req = host_request()
    landing = rig.plan_and_land(req)
    if ending == "short":
        landing.deliver_all_but(*rig.reader.unit_names(req, [6]))
    else:
        landing.finish(Failed("store gone"))
    rig.coord.advance([], 1.0)
    kind = ("TERMINAL", END - 4) if ending == "short" else ("FAILED", 0)
    assert rig.payloads()[-1][0] == [((1, "fetch"), *kind)]
    assert landing.releases == 1
    assert rig.effects.count("give_back_fetch_pages") == 0 and rig.host.count("quiesce") == 0
    assert rig.effects.count("unpark") == 0 and rig.effects.count("fail_requests") == 0
    rec = rig.fetch_record(1)
    assert rec.state.value == "PLANNED" and rec.plan is None and rec.landing is None
    assert rec.retries_left == 0 and rig.coord.plan_fetch(req) is DEFER

    rig.coord.advance([req], 2.0)  # replanned, and the new landing starts in the same round
    assert rig.host.count("fetch_to_host") == 2
    replan = rig.fetch_record(1).plan
    assert replan.token_end == (END - 4 if ending == "short" else END)
    assert rig.record(1)["state"] == "STAGING"


def test_placement_lands_unparks_and_releases_the_landing_in_the_same_round():
    rig = host_rig()
    req = host_request()
    landing = land_and_stage(rig, req)
    rec = rig.record(1)
    assert rec["has_landing"] and rec["waiting_since"] == 1.0  # waiting for pages since landed

    attempt = rig.reserve_and_place(req, 2.0)
    assert rig.effects.names() == ["prepare_fetch_resources", "park_for_fetch"]
    assert extent_names(attempt.payload) == plan_unit_names(rig.plans[1])
    rec = rig.record(1)
    assert rec["state"] == "IN_FLIGHT" and rec["attempts"] == 1 and rec["try_index"] == 0
    assert rec["waiting_since"] is None and landing.releases == 0
    assert rig.coord.parked_request_ids() == {1} and rig.coord.inflight_request_ids() == {1}

    attempt.deliver_all()
    rig.coord.advance([], 3.0)
    assert rig.effects.only("unpark") == [(req, END, False, None)]
    assert landing.releases == 1  # the copy was complete before its outcome: gone at once
    rec = rig.record(1)
    assert rec["state"] == "LANDED" and rec["has_landing"] is False  # the record stays
    assert rig.host.count("quiesce") == 0
    rig.coord.notify_request_finished(req)
    assert rig.host.calls[-1] == ("quiesce", ((attempt,), True))  # the placement's release point
    assert rig.records() == [] and landing.releases == 1


def test_placement_served_short_quiesces_gives_back_and_releases_the_landing():
    rig = host_rig()
    req = host_request()
    landing = land_and_stage(rig, req)
    attempt = rig.reserve_and_place(req, 2.0)
    attempt.deliver_all_but(*rig.reader.unit_names(req, [6]))
    rig.coord.advance([], 3.0)
    q, gb = quiesce_indices(rig.trace), effect_indices(rig.trace, "give_back_fetch_pages")
    assert len(q) == 1 and len(gb) == 1 and q[0] < gb[0]
    assert rig.host.calls[-1] == ("release", (landing,))  # after quiesce and give-back
    assert landing.releases == 1 and rig.effects.count("unpark") == 0
    rec = rig.fetch_record(1)
    assert rec.state.value == "PLANNED" and rec.plan is None and rec.landing is None
    assert rec.retries_left == 0 and rec.retry_hint == END - 4

    rig.coord.advance([req], 4.0)  # the retry lands on the host afresh
    assert rig.host.count("fetch_to_host") == 2 and rig.record(1)["state"] == "STAGING"
    assert rig.fetch_record(1).plan.token_end == END - 4


@pytest.mark.parametrize("source", ["host", "worker"])
def test_units_the_local_cache_committed_meanwhile_count_as_served(source):
    """Pages the reservation found already committed (another request computed the same
    prefix while the fetch waited) are not fetched over, and do not make the delivery short."""
    rig = Rig(sources=(source,))
    req = worker_request() if source == "worker" else host_request()
    rig.reader.committed_blocks[1] = 2
    if source == "host":
        land_and_stage(rig, req)
        attempt = rig.reserve_and_place(req, 2.0)
    else:
        attempt = rig.plan_and_launch(req)
    assert extent_names(attempt.payload) == rig.reader.unit_names(req, range(2, 7))
    assert rig.fetch_record(1).committed_names == rig.reader.unit_names(req, range(2))
    attempt.deliver_all()
    rig.coord.advance([], 3.0)
    assert rig.payloads()[-1][0] == [((1, "fetch"), "TERMINAL", END)]
    assert rig.effects.only("unpark") == [(req, END, False, None)]
    assert rig.effects.count("give_back_fetch_pages") == 0


def test_placement_of_an_empty_extent_parks_for_one_round_then_unparks():
    rig = host_rig()
    req = host_request()
    landing = land_and_stage(rig, req)
    rig.reader.committed_blocks[1] = 7  # every block committed locally while landing
    rig.host.script_place(Delivered(frozenset()))  # nothing to copy: done at once
    attempt = rig.reserve_and_place(req, 2.0)
    assert attempt.payload.units == ()
    assert rig.effects.names() == ["prepare_fetch_resources", "park_for_fetch"]
    assert rig.record(1)["state"] == "IN_FLIGHT"
    rig.coord.advance([], 3.0)
    assert rig.payloads()[-1][0] == [((1, "fetch"), "TERMINAL", END)]
    assert rig.effects.only("unpark") == [(req, END, False, None)]
    assert landing.releases == 1


@pytest.mark.parametrize("source", ["host", "worker"])
def test_units_the_reservation_has_no_page_for_make_the_delivery_short(source):
    rig = Rig(sources=(source,))
    req = worker_request() if source == "worker" else host_request()
    rig.reader.reserved_blocks[1] = 5  # the scheduler reserved fewer pages than planned
    if source == "host":
        landing = land_and_stage(rig, req)
        attempt = rig.reserve_and_place(req, 2.0)
    else:
        attempt = rig.plan_and_launch(req)
    assert extent_names(attempt.payload) == rig.reader.unit_names(req, range(5))
    assert rig.fetch_record(1).committed_names == frozenset()
    attempt.deliver_all()
    rig.coord.advance([], 3.0)
    assert rig.payloads()[-1][0] == [((1, "fetch"), "TERMINAL", 20)]
    assert rig.effects.count("unpark") == 0 and rig.effects.count("give_back_fetch_pages") == 1
    assert rig.record(1)["state"] == "PLANNED"
    if source == "host":
        assert landing.releases == 1


def test_placement_refused_three_times_gives_up_and_the_agreement_replans():
    rig = host_rig()
    req = host_request()
    landing = land_and_stage(rig, req)
    rig.host.reject_place_next = 3
    for index, now in enumerate((2.0, 3.0, 4.0)):
        assert isinstance(rig.coord.plan_fetch(req), FetchPlan)
        rig.coord.launch_fetches([req], now)
        rec = rig.fetch_record(1)
        assert rec.state.value == "STAGED" and rec.consecutive_launch_failures == index + 1
    assert rig.effects.count("give_back_fetch_pages") == 3  # page back-pressure, counted as such
    assert rig.fetch_record(1).launch_gave_up and rig.coord.plan_fetch(req) is DEFER
    assert landing.releases == 0

    loop_advance(rig, req, 5.0)
    assert rig.payloads()[-1][0] == [((1, "fetch"), "FAILED", 0)]
    rec = rig.fetch_record(1)
    assert landing.releases == 1 and rec.landing is None
    assert rec.state.value == "PLANNED" and rec.plan is None and rec.retries_left == 0
    assert rig.host.count("quiesce") == 0 and rig.effects.count("fail_requests") == 0


def test_staging_expiry_fails_the_request_and_releases_the_landing_at_once():
    rig = host_rig(fetch_timeout_s=10.0)
    req = host_request()
    landing = rig.plan_and_land(req, now=0.0)
    assert rig.record(1)["deadline"] == 10.0
    rig.coord.advance([], 10.0)
    assert rig.effects.names() == ["fail_requests"]  # no pages: nothing to hold
    assert rig.effects.only("fail_requests") == [((req,), "kv fetch timed out")]
    # The landing names no page, so nothing waits for its outcome: record and landing go now.
    assert landing.releases == 1 and rig.records() == []
    assert rig.coord.held_request_ids() == frozenset()
    assert rig.coord.status_dump() == {
        "plan_authority": "VOTED",
        "records": [],
        "decided_plans": 0,
        "finished_pending": [],
    }
    landing.deliver_all()  # late, and moot
    rig.coord.advance([], 11.0)
    assert landing.releases == 1 and rig.effects.names() == ["fail_requests"]


@pytest.mark.parametrize("stage", ["STAGING", "STAGED"])
def test_request_ending_before_placement_releases_the_landing_and_votes_until_agreed(stage):
    """The landing names no page and goes at once; the record stays one more round to vote
    TERMINAL at the plan's target, so that a peer still delivering lands on its own word and
    every rank terminates the request in the same round."""
    rig = host_rig()
    req = host_request()
    landing = land_and_stage(rig, req) if stage == "STAGED" else rig.plan_and_land(req)
    assert rig.coord.notify_request_finished(req, 2.0) is False
    assert landing.releases == 1 and rig.host.count("quiesce") == 0
    assert rig.effects.names() == ["hold_for_transfer"]
    rec = rig.record(1)
    assert rec["state"] == stage and rec["has_landing"] is False
    assert rig.coord.has_inflight() is False  # nothing runs for it any more
    rig.coord.advance([], 3.0)
    assert rig.payloads()[-1][0] == [((1, "fetch"), "TERMINAL", END)]
    assert rig.effects.names() == ["hold_for_transfer", "terminate_request"]
    assert rig.records() == [] and rig.coord.status_dump()["finished_pending"] == []


def test_request_ending_while_planned_is_terminated_with_the_agreement():
    """A planned fetch never launched here: no pages, nothing to quiesce; the record votes once
    so a peer that did launch is not left without a verdict."""
    rig = Rig()
    req = worker_request()
    rig.coord.advance([req], 0.0)
    assert rig.coord.notify_request_finished(req, 1.0) is False
    assert rig.effects.names() == ["hold_for_transfer"]
    rig.coord.advance([], 2.0)
    assert rig.payloads()[-1][0] == [((1, "fetch"), "TERMINAL", END)]
    assert rig.effects.names() == ["hold_for_transfer", "terminate_request"]
    assert rig.records() == [] and rig.worker.count("quiesce") == 0


def test_request_failed_by_the_coordinator_answers_the_late_release_gate_once():
    """The engine may terminate a failed request later than the failure (deferred under
    attention DP); the coordinator, having terminated the held request itself meanwhile, tells
    the gate once that the request is not the engine's to terminate again."""
    rig = Rig(fetch_timeout_s=10.0)
    req = worker_request()
    attempt = rig.plan_and_launch(req, now=0.0)
    rig.coord.advance([], 10.0)  # fails the request; the effects here never call the gate back
    assert rig.effects.names()[-2:] == ["fail_requests", "hold_for_transfer"]
    attempt.deliver_all()
    rig.coord.advance([], 11.0)
    assert rig.effects.only("terminate_request") == [(req,)]
    assert rig.coord.notify_request_finished(req, 12.0) is False  # the engine's late gate
    assert rig.coord.notify_request_finished(req, 13.0) is True  # a fresh request of that id
    assert rig.effects.count("terminate_request") == 1


def test_reserve_wait_of_a_device_direct_plan_spends_a_retry_then_computes_locally():
    """The scheduler never finds pages for the plan (``launch_fetches`` is never called): the
    wait is clocked from the plan's decision, the fetch is given up at the wait timeout, the
    retry waits once more, and the request computes locally; the scheduler is never stalled."""
    rig = Rig(landing_wait_timeout_s=5.0)
    req = worker_request()
    rig.coord.advance([req], 0.0)
    assert isinstance(rig.coord.plan_fetch(req), FetchPlan)
    assert rig.fetch_record(1).waiting_since == 0.0
    loop_advance(rig, req, 4.9)
    assert rig.payloads()[-1][0] == [((1, "fetch"), "UNLAUNCHED", 0)]
    loop_advance(rig, req, 5.0)
    assert rig.payloads()[-1][0] == [((1, "fetch"), "FAILED", 0)]
    rec = rig.fetch_record(1)
    assert rec.retries_left == 0 and rec.plan is None and rec.waiting_since is None
    assert rig.coord.plan_fetch(req) is DEFER and rig.effects.count("fail_requests") == 0

    loop_advance(rig, req, 6.0)  # planned afresh; the wait starts over
    assert isinstance(rig.coord.plan_fetch(req), FetchPlan)
    assert rig.fetch_record(1).waiting_since == 6.0
    loop_advance(rig, req, 10.9)
    assert rig.payloads()[-1][0] == [((1, "fetch"), "UNLAUNCHED", 0)]
    loop_advance(rig, req, 11.0)
    assert rig.payloads()[-1][0] == [((1, "fetch"), "FAILED", 0)]
    assert rig.records() == [] and rig.coord.plan_fetch(req) is None
    assert rig.effects.calls == [] and rig.worker.count("quiesce") == 0


def test_landing_wait_timeout_spends_a_retry_at_each_wait_then_computes_locally():
    """The rank's own clock: refused landing memory (PLANNED) and pages that never come (STAGED)
    each time out once; the second timeout is out of retries and the request computes locally."""
    rig = host_rig(landing_wait_timeout_s=5.0)
    req = host_request()
    rig.host.reject_next = 100
    rig.coord.advance([req], 0.0)
    loop_advance(rig, req, 4.9)
    assert rig.payloads()[-1][0] == [((1, "fetch"), "UNLAUNCHED", 0)]
    loop_advance(rig, req, 5.0)
    assert rig.payloads()[-1][0] == [((1, "fetch"), "FAILED", 0)]
    rec = rig.fetch_record(1)
    assert rec.retries_left == 0 and rec.plan is None and rec.waiting_since is None
    assert rig.coord.plan_fetch(req) is DEFER and rig.effects.count("fail_requests") == 0

    rig.host.reject_next = 0
    loop_advance(rig, req, 6.0)  # planned afresh; the landing memory is there this time
    landing = rig.host.landings[-1]
    landing.deliver_all()
    rig.coord.advance([], 7.0)
    assert rig.record(1)["state"] == "STAGED" and rig.record(1)["waiting_since"] == 7.0
    loop_advance(rig, req, 11.9)  # the scheduler never finds pages
    assert rig.payloads()[-1][0] == [((1, "fetch"), "UNLAUNCHED", 0)]
    loop_advance(rig, req, 12.0)
    assert rig.payloads()[-1][0] == [((1, "fetch"), "FAILED", 0)]
    assert landing.releases == 1 and rig.records() == []
    assert rig.coord.plan_fetch(req) is None and rig.effects.count("fail_requests") == 0
    assert rig.effects.count("give_back_fetch_pages") == 0 and rig.host.count("quiesce") == 0

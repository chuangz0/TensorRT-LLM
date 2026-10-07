# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fetch record from the loop entry points, one edge of design §4.1 per test: PLANNED ->
IN_FLIGHT -> DELIVERED and its release; the one retry after a failure or a short serve; routes
that fail to open and submissions the backend refuses; the fetch and wait deadlines; the
request's end at every point of the record's life.

Single rank. Assertions are on the record table (via ``status_dump``), the effects the engine
was asked to perform, and the calls each backend saw -- including the order of ``quiesce``
relative to ``revert_fetch_pages`` (the release point, design §4.3).
"""

import pytest

__extra_import_path__ = ["~/tensorrt_llm/_torch"]
from disaggregation.base.cache_backend import Cancelled, Delivered, Failed  # noqa: E402
from disaggregation.remote_cache import DEFER, FetchPlan  # noqa: E402
from fakes import (  # noqa: E402
    EMPTY_DUMP,
    END,
    CoordinatorRig,
    FakeRequest,
    ScriptedPeersCollective,
    assert_quiesce_precedes,
    extent_names,
    gen_init_request,
    host_rig,
    ordinals_by_group,
    plan_unit_names,
    store_request,
    worker_request,
)

pytestmark = pytest.mark.cpu_only


# ---- PLANNED -> IN_FLIGHT -> DELIVERED -> released ----


def test_deferred_request_answers_defer_and_has_no_record():
    rig = CoordinatorRig()
    req = worker_request()
    assert rig.coord.fetch_answer(req) is DEFER
    assert rig.records() == []
    assert rig.coord.has_backend_work() is False


def test_advance_plans_and_plan_fetch_reads_the_plan():
    rig = CoordinatorRig()
    req = worker_request()
    rig.coord.advance([req], 0.0)
    plan = rig.coord.fetch_answer(req)
    assert isinstance(plan, FetchPlan) and plan.token_end == END and plan.source == "worker"
    rec = rig.record(1)
    assert rec["state"] == "PLANNED" and rec["attempts"] == 0 and rec["token_end"] == END
    assert rig.effects.calls == []  # planning has no side effects
    assert rig.coord.has_backend_work() is False


def test_launch_opens_route_fetches_and_parks():
    rig = CoordinatorRig()
    req = worker_request()
    attempt = rig.plan_and_launch(req)
    assert rig.record(1)["resource_wait_since"] is None  # the wait for pages ended at the launch
    assert rig.effects.names() == ["prepare_fetch_resources", "park_for_fetch"]
    assert rig.effects.args_of("park_for_fetch") == [((req,),)]
    assert [m for m, _ in rig.worker.calls] == ["open_route", "fetch"]
    assert rig.worker.calls[0][1] == (req.route_hints["ctx"],)
    extent, route = rig.worker.calls[1][1]
    assert route is rig.worker.routes[0] and route.closed == 0
    assert extent_names(extent) == plan_unit_names(rig.plans[1])
    assert attempt.payload is extent
    rec = rig.record(1)
    assert rec["state"] == "IN_FLIGHT" and rec["attempts"] == 1 and rec["try_index"] == 0
    assert rig.coord.has_backend_work() is True
    # Past PLANNED there is nothing more to plan: the scheduler hook answers None.
    assert rig.coord.fetch_answer(req) is None
    # The store is never asked for a route.
    assert rig.store.count("open_route") == 0


def test_launch_is_a_noop_for_requests_without_a_plan():
    rig = CoordinatorRig()
    req = worker_request()
    rig.coord.launch_reserved_fetches([req], 0.0)
    assert rig.effects.calls == [] and rig.worker.calls == []


def test_in_flight_attempt_is_polled_each_advance_without_effects():
    rig = CoordinatorRig()
    req = worker_request()
    attempt = rig.plan_and_launch(req)
    rig.coord.advance([], 1.0)
    rig.coord.advance([], 2.0)
    assert attempt.polls == 2
    assert rig.effects.names() == ["prepare_fetch_resources", "park_for_fetch"]
    assert rig.record(1)["state"] == "IN_FLIGHT"


def test_delivered_lands_unparks_and_closes_route_without_quiesce():
    rig = CoordinatorRig()
    req = worker_request()
    attempt = rig.plan_and_launch(req)
    attempt.deliver_all()
    rig.coord.advance([], 1.0)
    assert rig.effects.args_of("unpark") == [(req, END, False, None)]
    assert rig.worker.routes[0].closed == 1
    assert rig.worker.count("quiesce") == 0  # not a release point yet
    rec = rig.record(1)
    assert rec["state"] == "DELIVERED" and rec["outcomes"] == ["Delivered"]
    assert rig.coord.has_backend_work() is False
    assert rig.coord.fetch_answer(req) is None  # decided: compute locally from here


def test_request_end_releases_landed_fetch_with_exactly_one_quiesce():
    rig = CoordinatorRig()
    req = worker_request()
    attempt = rig.plan_and_launch(req)
    attempt.deliver_all()
    rig.coord.advance([], 1.0)
    rig.coord.holds_finished_request(req)
    quiesces = [args for m, args in rig.worker.calls if m == "quiesce"]
    assert quiesces == [((attempt,), True)]
    assert rig.records() == []
    assert rig.worker.routes[0].closed == 1  # idempotent close, not closed twice
    assert rig.coord.status_dump() == EMPTY_DUMP
    # Nothing else happened at the release point of a landed fetch.
    assert rig.effects.names() == ["prepare_fetch_resources", "park_for_fetch", "unpark"]


def test_holds_finished_request_is_idempotent():
    rig = CoordinatorRig()
    req = worker_request()
    attempt = rig.plan_and_launch(req)
    attempt.deliver_all()
    rig.coord.advance([], 1.0)
    rig.coord.holds_finished_request(req)
    rig.coord.holds_finished_request(req)
    assert rig.worker.count("quiesce") == 1


@pytest.mark.parametrize("late", ["delivered", "failed"])
def test_request_end_while_in_flight_abandons_then_releases_on_outcome(late):
    rig = CoordinatorRig()
    req = worker_request()
    attempt = rig.plan_and_launch(req)
    rig.coord.holds_finished_request(req)
    # Never quiesce on the engine thread for a transfer still running: abandon it and hold the
    # request so its pages stay put until the backend is done with them.
    assert rig.worker.count("quiesce") == 0
    rec = rig.record(1)
    assert rec["state"] == "IN_FLIGHT"
    assert rig.worker.routes[0].closed == 0
    assert rig.effects.names()[-1:] == ["hold_for_transfer"]
    assert rig.effects.args_of("hold_for_transfer") == [((req,),)]
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
    assert rig.effects.args_of("terminate_request") == [(req,)]
    assert_quiesce_precedes(rig.trace, "terminate_request")
    assert rig.coord.status_dump() == EMPTY_DUMP


def test_request_end_while_in_flight_survives_its_deadline():
    rig = CoordinatorRig(fetch_timeout_s=10.0)
    req = worker_request()
    attempt = rig.plan_and_launch(req, now=0.0)
    rig.coord.holds_finished_request(req)
    rig.coord.advance([], 50.0)  # past the deadline: already abandoned, still held
    assert rig.record(1)["state"] == "IN_FLIGHT" and rig.effects.count("terminate_request") == 0
    attempt.deliver_all()
    rig.coord.advance([], 51.0)
    assert rig.effects.count("terminate_request") == 1 and rig.records() == []


# ---- FAILED -> PLANNED (retry) -> FAILED -> released ----


def test_failed_quiesces_before_give_back_and_keeps_one_retry():
    rig = CoordinatorRig()
    req = worker_request()
    attempt = rig.plan_and_launch(req)
    attempt.finish(Failed("link down"))
    rig.coord.advance([], 1.0)

    assert_quiesce_precedes(rig.trace, "revert_fetch_pages")
    assert rig.effects.args_of("revert_fetch_pages") == [((req,),)]
    assert rig.effects.count("unpark") == 0 and rig.effects.count("fail_requests") == 0
    assert rig.worker.routes[0].closed == 1
    rec = rig.record(1)
    assert rec["state"] == "PLANNED" and rec["token_end"] is None
    assert rig.coord.fetch_answer(req) is DEFER  # planned but not yet re-decided
    assert rig.coord.has_backend_work() is False


def test_second_failure_releases_and_request_is_replanned_as_none():
    rig = CoordinatorRig()
    req = worker_request()
    rig.plan_and_launch(req).finish(Failed("first"))
    rig.coord.advance([], 1.0)

    second = rig.plan_and_launch(req, now=2.0)  # re-plan + re-launch: try 1
    assert rig.record(1)["try_index"] == 1 and rig.record(1)["attempts"] == 2
    assert rig.worker.count("open_route") == 2
    second.finish(Failed("second"))
    rig.coord.advance([], 3.0)

    assert_quiesce_precedes(rig.trace, "revert_fetch_pages", quiesces=2, effects=2)
    # The second quiesce covers the current try only; the first try was quiesced when it failed.
    assert rig.worker.calls[-1] == ("quiesce", ((second,), True))
    assert rig.records() == []
    assert rig.coord.fetch_answer(req) is None
    assert all(r.closed == 1 for r in rig.worker.routes)
    assert rig.effects.count("fail_requests") == 0  # local fallback, not a failure


def test_cancelled_by_peer_counts_as_failed():
    rig = CoordinatorRig()
    req = worker_request()
    rig.plan_and_launch(req).finish(Cancelled(by_peer=True))
    rig.coord.advance([], 1.0)
    assert rig.effects.count("revert_fetch_pages") == 1
    assert rig.record(1)["state"] == "PLANNED"
    assert rig.last_votes == [((1, "fetch"), "FAILED", 0)]


def test_local_cancel_is_delivered_nothing_not_a_failure():
    rig = CoordinatorRig()
    req = worker_request()
    rig.plan_and_launch(req).finish(Cancelled(by_peer=False))
    rig.coord.advance([], 1.0)
    # On the wire it is a short serve (TERMINAL with B = 0), so the retry path applies.
    assert rig.last_votes == [((1, "fetch"), "TERMINAL", 0)]
    assert rig.effects.count("revert_fetch_pages") == 1
    assert rig.effects.count("fail_requests") == 0
    assert rig.record(1)["state"] == "PLANNED"
    # The hint is 0, so the retry has nothing to aim for and the request computes locally.
    rig.coord.advance([req], 2.0)
    assert rig.coord.fetch_answer(req) is None and rig.records() == []


def test_gen_init_local_cancel_fails_the_request():
    rig = CoordinatorRig()
    req = gen_init_request()
    rig.plan_and_launch(req).finish(Cancelled(by_peer=False))
    rig.coord.advance([], 1.0)
    assert rig.effects.args_of("fail_requests") == [((req,), "kv fetch served short")]


def test_second_short_serve_quiesces_only_the_second_try():
    rig = CoordinatorRig()
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
    assert rig.records() == [] and rig.coord.fetch_answer(req) is None


@pytest.mark.parametrize("gen_init", [False, True])
def test_transport_error_and_rejection_alternation_is_bounded(gen_init):
    """A failed route and a refused submission are the same thing to the record: a launch that
    never started. Neither re-plans the request on this rank alone; together they count towards
    ``MAX_CONSECUTIVE_LAUNCH_FAILURES``, and the failure lands through the ranks' agreement."""
    rig = CoordinatorRig()
    req = gen_init_request() if gen_init else worker_request()
    rid = req.py_request_id

    # 1. transport error, 2. rejection, 3. transport error: the plan survives each of them.
    for index, failure in enumerate(("route", "reject", "route")):
        if failure == "route":
            rig.worker.open_route_errors.append(RuntimeError(f"t{index}"))
        else:
            rig.worker.reject_next_calls = 1
        rig.advance_as_hooks_would(req, float(index))
        assert isinstance(rig.coord.fetch_answer(req), FetchPlan)
        rig.coord.launch_reserved_fetches([req], float(index))
        rec = rig.record(rid)
        assert rec["state"] == "PLANNED" and rec["token_end"] is not None
        assert rec["consecutive_launch_failures"] == index + 1 and rec["retries_left"] == 1
    assert rig.effects.count("revert_fetch_pages") == 3 and rig.worker.count("quiesce") == 0
    # At the cap the rank gives up: DEFER to the scheduler, nothing failed yet.
    assert rig.record(rid)["gave_up_launching"] and rig.coord.fetch_answer(req) is DEFER
    assert rig.effects.count("fail_requests") == 0

    # 4. Its FAILED vote lands on the next advance.
    rig.advance_as_hooks_would(req, 3.0)
    assert rig.last_votes == [((rid, "fetch"), "FAILED", 0)]
    if gen_init:
        assert rig.effects.args_of("fail_requests") == [((req,), "kv fetch launch given up")]
        assert rig.records() == [] and rig.coord.fetch_answer(req) is None
    else:
        rec = rig.record(rid)
        assert rec["token_end"] is None and not rec["gave_up_launching"]
        assert rec["retries_left"] == 0 and rec["consecutive_launch_failures"] == 0
        assert rig.coord.fetch_answer(req) is DEFER  # planned afresh next round
        assert rig.effects.count("fail_requests") == 0


def test_quiesce_false_on_a_finished_requests_late_outcome_keeps_the_hold():
    """The request ended while its fetch was in flight; the outcome arrives and the backend
    cannot vouch for the pages at the release point: fatal, the record stays, the request stays
    held and its pages stay in flight, and the backend is not asked again."""
    rig = CoordinatorRig()
    req = worker_request()
    attempt = rig.plan_and_launch(req)
    assert rig.coord.holds_finished_request(req, 1.0) is True
    rig.worker.quiesce_answers.append(False)
    attempt.deliver_all()
    rig.coord.advance([], 2.0)
    assert rig.effects.count("fail_fatal") == 1 and rig.effects.count("terminate_request") == 0
    assert rig.coord.held_request_ids() == {1} and rig.coord.inflight_request_ids() == {1}
    assert rig.record(1)["state"] == "IN_FLIGHT"
    rig.coord.advance([], 3.0)
    assert rig.worker.count("quiesce") == 1 and rig.effects.count("fail_fatal") == 1
    assert rig.effects.count("terminate_request") == 0 and rig.last_votes == []
    assert rig.coord.held_request_ids() == {1}


def test_quiesce_false_on_a_short_placement_is_fatal_and_keeps_the_pages():
    """A host-first placement served short passes the release point before the pages go back;
    a backend that refuses there makes the engine fatal, and nothing moves afterwards: no
    give-back, no retry, and the landing stays with the record (it is released only when the
    record leaves the table or its plan is dropped, neither of which happens)."""
    rig = host_rig()
    req = store_request()
    landing = rig.land_and_agree(req)
    attempt = rig.launch_reserved(req, 2.0)
    rig.host.quiesce_answers.append(False)
    attempt.deliver_all_but(*rig.reader.unit_names(req, [6]))
    rig.coord.advance([], 3.0)
    assert rig.effects.count("fail_fatal") == 1
    assert rig.effects.count("revert_fetch_pages") == 0 and rig.effects.count("unpark") == 0
    assert rig.record(1)["state"] == "FAILED" and rig.record(1)["has_landing"]
    assert landing.closes == 0 and rig.host.count("fetch_to_host") == 1  # no retry
    assert rig.coord.inflight_request_ids() == {1} and rig.coord.fetch_answer(req) is None
    assert rig.coord.holds_finished_request(req, 4.0) is True
    assert rig.effects.count("fail_fatal") == 1 and rig.host.count("quiesce") == 1
    assert rig.effects.names()[-1:] == ["hold_for_transfer"]


def test_quiesce_false_at_release_gate_holds_the_request():
    """The backend cannot vouch for the pages: fatal, and the request is held so the engine does
    not free them; the poisoned record is never asked about again and never released."""
    rig = CoordinatorRig()
    req = worker_request()
    attempt = rig.plan_and_launch(req)
    attempt.deliver_all()
    rig.coord.advance([], 1.0)
    rig.worker.quiesce_answers.append(False)
    assert rig.coord.holds_finished_request(req, 2.0) is True
    assert rig.effects.count("fail_fatal") == 1
    assert rig.effects.args_of("hold_for_transfer") == [((req,),)]
    assert rig.effects.count("terminate_request") == 0
    assert rig.record(1)["state"] == "DELIVERED"
    assert rig.coord.held_request_ids() == {1} and rig.coord.inflight_request_ids() == {1}
    assert rig.coord.status_dump()["finished_pending"] == [1]
    rig.coord.advance([], 3.0)
    assert rig.worker.count("quiesce") == 1 and rig.effects.count("terminate_request") == 0
    assert rig.record(1)["state"] == "DELIVERED" and rig.last_votes == []


# ---- open_route failures ----


@pytest.mark.parametrize(
    "error", [NotImplementedError("single destination"), ValueError("bad hint")]
)
def test_route_refused_gives_up_and_settles_on_local_compute_after_two_agreements(error):
    rig = CoordinatorRig()
    req = worker_request()
    rig.worker.open_route_errors.append(error)
    rig.coord.advance([req], 0.0)
    rig.coord.launch_reserved_fetches([req], 0.0)
    assert rig.effects.names() == ["prepare_fetch_resources", "revert_fetch_pages"]
    assert rig.worker.count("fetch") == 0 and rig.worker.count("quiesce") == 0
    # This plan can never work here: the rank gives up at once and answers the scheduler DEFER
    # until the ranks agree, rather than deciding "compute locally" on its own.
    assert rig.record(1)["gave_up_launching"] and rig.coord.fetch_answer(req) is DEFER

    # First agreement: the retry is spent and the request is planned again, on the same route
    # (the planner knows nothing of the refusal) ...
    rig.advance_as_hooks_would(req, 1.0)
    assert rig.record(1)["retries_left"] == 0 and rig.coord.fetch_answer(req) is DEFER
    rig.worker.open_route_errors.append(error)
    rig.advance_as_hooks_would(req, 2.0)
    rig.coord.launch_reserved_fetches([req], 2.0)
    assert rig.record(1)["gave_up_launching"]
    # ... which is refused again; the second agreement settles on local compute.
    rig.advance_as_hooks_would(req, 3.0)
    assert rig.records() == [] and rig.coord.fetch_answer(req) is None
    assert rig.effects.count("revert_fetch_pages") == 2
    assert rig.effects.count("fail_requests") == 0


def test_gen_init_route_refused_fails_the_request_at_the_next_agreement():
    rig = CoordinatorRig()
    req = gen_init_request()
    rig.worker.open_route_errors.append(ValueError("unknown peer"))
    rig.coord.advance([req], 0.0)
    rig.coord.launch_reserved_fetches([req], 0.0)
    # The pages go back first; the verdict waits for the ranks' agreement on the next advance.
    assert rig.effects.names() == ["prepare_fetch_resources", "revert_fetch_pages"]
    assert rig.coord.fetch_answer(req) is DEFER and rig.record(7)["state"] == "PLANNED"
    rig.advance_as_hooks_would(req, 1.0)
    assert rig.effects.names()[-1:] == ["fail_requests"]
    ((reqs, reason),) = rig.effects.args_of("fail_requests")
    assert reqs == (req,) and reason == "kv fetch launch given up"
    assert rig.coord.fetch_answer(req) is None and rig.records() == []


def test_route_transport_error_keeps_the_plan_and_the_retry():
    rig = CoordinatorRig()
    req = worker_request()
    rig.worker.open_route_errors.append(RuntimeError("peer metadata fetch failed"))
    rig.coord.advance([req], 0.0)
    plan = rig.coord.fetch_answer(req)
    rig.coord.launch_reserved_fetches([req], 0.0)
    assert rig.effects.names() == ["prepare_fetch_resources", "revert_fetch_pages"]
    assert rig.worker.count("fetch") == 0 and rig.worker.count("quiesce") == 0
    # The record keeps its plan: the scheduler reserves for the same fetch again next round.
    rec = rig.record(1)
    assert rec["state"] == "PLANNED" and rec["attempts"] == 0 and rec["token_end"] == END
    assert rig.coord.fetch_answer(req) is plan and rec["retries_left"] == 1

    # The route works next round; a real failure afterwards still has its retry.
    rig.plan_and_launch(req, now=1.0).finish(Failed("later"))
    rig.coord.advance([], 2.0)
    rec = rig.record(1)
    assert rec["state"] == "PLANNED" and rec["retries_left"] == 0


def test_repeated_route_transport_errors_give_up_then_settle_on_local_compute():
    rig = CoordinatorRig()
    req = worker_request()
    rig.worker.open_route_errors.extend(RuntimeError(f"e{i}") for i in range(6))
    rig.drive_rounds(req, rounds=12)
    # Three errors give the plan up; the agreement spends the retry on a fresh plan; three more
    # give that up too; the next agreement settles on local compute.
    assert rig.worker.count("open_route") == 6 and rig.worker.count("fetch") == 0
    assert rig.effects.count("revert_fetch_pages") == 6
    assert rig.effects.count("fail_requests") == 0
    assert rig.records() == [] and rig.coord.fetch_answer(req) is None


def test_repeated_route_transport_errors_fail_gen_init():
    rig = CoordinatorRig()
    req = gen_init_request()
    rig.worker.open_route_errors.extend(RuntimeError(f"e{i}") for i in range(3))
    for round_index in range(3):
        rig.advance_as_hooks_would(req, float(round_index))
        rig.coord.launch_reserved_fetches([req], float(round_index))
    assert rig.effects.count("fail_requests") == 0 and rig.coord.fetch_answer(req) is DEFER
    rig.advance_as_hooks_would(req, 3.0)  # the FAILED vote lands: a gen-init fetch has no retry
    ((reqs, reason),) = rig.effects.args_of("fail_requests")
    assert reqs == (req,) and reason == "kv fetch launch given up"
    assert rig.effects.names()[-2:] == ["revert_fetch_pages", "fail_requests"]
    assert rig.records() == [] and rig.coord.fetch_answer(req) is None


def test_route_close_raising_is_contained():
    """A route whose ``close`` raises has not closed (the contract leaves the handle open for
    another try). The raise must not escape ``advance``: the delivery it belongs to is complete
    and the request is unparked all the same; the handle is kept, not abandoned, and the
    request's end closes it on the next try and releases the record."""
    rig = CoordinatorRig()
    req = worker_request()
    attempt = rig.plan_and_launch(req)
    route = rig.worker.routes[0]
    route.close_errors.append(RuntimeError("connector gone"))
    attempt.deliver_all()
    rig.coord.advance([], 1.0)
    assert rig.effects.args_of("unpark") == [(req, END, False, None)]
    assert rig.record(1)["state"] == "DELIVERED"
    assert route.closed == 0  # the failed close has not closed; the handle is still held
    assert rig.coord.holds_finished_request(req, 2.0) is False
    assert route.closed == 1  # the next try, at the release point, closed it
    assert rig.records() == [] and rig.worker.count("quiesce") == 1


def test_route_failure_does_not_strand_other_requests_in_the_same_launch():
    rig = CoordinatorRig()
    a, b, c = worker_request(1), worker_request(2), worker_request(3)
    rig.worker.open_route_errors.append(RuntimeError("only a's route fails"))
    rig.coord.advance([a, b, c], 0.0)
    rig.coord.launch_reserved_fetches([a, b, c], 0.0)
    assert rig.effects.args_of("prepare_fetch_resources") == [((a, b, c),)]
    assert rig.effects.args_of("revert_fetch_pages") == [((a,),)]
    assert rig.effects.args_of("park_for_fetch") == [((b, c),)]
    assert rig.record(1)["state"] == "PLANNED" and rig.record(1)["attempts"] == 0
    assert rig.record(2)["state"] == "IN_FLIGHT" and rig.record(3)["state"] == "IN_FLIGHT"


# ---- aligned prompt ----


def test_aligned_gen_init_prompt_lands():
    # prompt_len 28 = 7 blocks exactly; the reader names 6 (the last prompt token is not
    # reusable), the plan asks for ordinals 0..6 and the nameless block 6 is skipped on both sides.
    rig = CoordinatorRig()
    req = FakeRequest(
        7, prompt_len=28, is_disagg_generation_init=True, route_hints={"ctx": {"peer": "c"}}
    )
    attempt = rig.plan_and_launch(req)
    assert rig.plans[7].token_end == 28 and ordinals_by_group(rig.plans[7]) == {0: tuple(range(7))}
    assert len(attempt.payload.units) == 6
    attempt.deliver_all()
    rig.coord.advance([], 1.0)
    assert rig.effects.args_of("unpark") == [(req, 28, True, None)]


def test_aligned_context_prompt_fetches_to_the_last_nameable_block():
    rig = CoordinatorRig()
    req = worker_request(1, prompt_len=28)
    attempt = rig.plan_and_launch(req)
    assert rig.plans[1].token_end == 24 and len(attempt.payload.units) == 6
    attempt.deliver_all()
    rig.coord.advance([], 1.0)
    assert rig.effects.args_of("unpark") == [(req, 24, False, None)]


def test_served_short_retries_with_min_b():
    rig = CoordinatorRig()
    req = worker_request()
    attempt = rig.plan_and_launch(req)
    plan = rig.plans[1]
    last_block = rig.reader.unit_names(req, [6])
    attempt.finish(Delivered(plan_unit_names(plan) - last_block))
    rig.coord.advance([], 1.0)
    assert rig.effects.count("revert_fetch_pages") == 1 and rig.effects.count("unpark") == 0
    assert rig.record(1)["state"] == "PLANNED"

    rig.coord.advance([req], 2.0)
    replan = rig.coord.fetch_answer(req)
    assert isinstance(replan, FetchPlan) and replan.token_end == END - 4
    assert ordinals_by_group(replan) == {0: tuple(range(6))}

    rig.coord.launch_reserved_fetches([req], 2.0)
    rig.worker.attempts[-1].deliver_all()
    rig.coord.advance([], 3.0)
    assert rig.effects.args_of("unpark") == [(req, END - 4, False, None)]


def test_short_served_after_the_retry_gives_up_locally():
    rig = CoordinatorRig()
    req = worker_request()
    rig.plan_and_launch(req).deliver_all_but(*rig.reader.unit_names(req, [6]))
    rig.coord.advance([], 1.0)
    rig.plan_and_launch(req, now=2.0).deliver_all_but(*rig.reader.unit_names(req, [5]))
    rig.coord.advance([], 3.0)
    assert rig.records() == [] and rig.coord.fetch_answer(req) is None
    assert rig.effects.count("revert_fetch_pages") == 2


def test_quiesce_false_is_fatal_and_pages_are_not_given_back():
    rig = CoordinatorRig()
    req = worker_request()
    rig.worker.quiesce_answers.append(False)
    rig.plan_and_launch(req).finish(Failed("x"))
    rig.coord.advance([], 1.0)
    assert rig.effects.count("fail_fatal") == 1
    assert rig.effects.count("revert_fetch_pages") == 0
    assert rig.record(1)["state"] == "FAILED"
    assert rig.worker.routes[0].closed == 0
    # The refusal is final: the request's end holds the request without asking the backend
    # again or making the engine fatal twice.
    assert rig.coord.holds_finished_request(req, 2.0) is True
    assert rig.worker.count("quiesce") == 1 and rig.effects.count("fail_fatal") == 1
    assert rig.effects.names()[-1:] == ["hold_for_transfer"]
    rig.coord.advance([], 3.0)
    assert rig.worker.count("quiesce") == 1 and rig.effects.count("terminate_request") == 0


# ---- SubmissionRejected ----


def test_submission_rejected_gives_back_without_quiesce_or_retry_cost():
    rig = CoordinatorRig()
    req = worker_request()
    rig.worker.reject_next_calls = 1
    rig.coord.advance([req], 0.0)
    plan = rig.coord.fetch_answer(req)
    rig.coord.launch_reserved_fetches([req], 0.0)
    # The resources were prepared for the launch, so they are given back, in that order.
    assert rig.effects.names() == ["prepare_fetch_resources", "revert_fetch_pages"]
    assert rig.worker.count("quiesce") == 0
    assert rig.worker.routes[0].closed == 1
    # The record keeps its plan: the same fetch is reserved for and launched again next round,
    # the retry budget untouched; only the run of failed launches is counted.
    rec = rig.record(1)
    assert rec["state"] == "PLANNED" and rec["token_end"] == END and rec["attempts"] == 0
    assert rig.coord.fetch_answer(req) is plan
    assert rec["consecutive_launch_failures"] == 1 and rec["retries_left"] == 1

    # The retry budget is intact: a real failure afterwards still gets its one retry.
    rig.plan_and_launch(req, now=1.0).finish(Failed("later"))
    rig.coord.advance([], 2.0)
    assert rig.record(1)["state"] == "PLANNED"


# ---- expiry ----


def test_gen_init_expiry_fails_the_request_and_holds_its_pages_until_the_outcome():
    """A gen-init fetch expires like any other: the request fails now, the pages stay held
    while the attempt may still write them, and the late outcome only releases."""
    rig = CoordinatorRig(fetch_timeout_s=10.0)
    req = gen_init_request()
    attempt = rig.plan_and_launch(req, now=0.0)
    assert rig.plans[7].no_local_fallback is True
    rig.coord.advance([], 9.9)
    assert rig.effects.count("fail_requests") == 0
    rig.coord.advance([], 10.0)
    assert rig.worker.count("quiesce") == 0 and rig.effects.count("revert_fetch_pages") == 0
    assert rig.effects.names()[-2:] == ["fail_requests", "hold_for_transfer"]
    assert rig.effects.args_of("fail_requests") == [((req,), "kv fetch timed out")]
    rec = rig.record(7)
    assert rec["state"] == "IN_FLIGHT" and rec["expired"]
    # The late delivery lands nothing: the release point, then the held request is terminated.
    attempt.deliver_all()
    rig.coord.advance([], 11.0)
    assert rig.worker.count("quiesce") == 1 and rig.effects.count("unpark") == 0
    assert rig.effects.names()[-1:] == ["terminate_request"]
    assert rig.records() == [] and rig.coord.fetch_answer(req) is DEFER


def test_gen_init_failure_does_not_retry():
    rig = CoordinatorRig()
    req = gen_init_request()
    rig.plan_and_launch(req).finish(Failed("x"))
    rig.coord.advance([], 1.0)
    assert rig.effects.args_of("fail_requests") == [((req,), "kv fetch failed")]
    assert rig.records() == []


def test_gen_init_delivery_unparks_at_prompt_len_with_no_local_fallback():
    rig = CoordinatorRig()
    req = gen_init_request()
    rig.plan_and_launch(req).deliver_all()
    rig.coord.advance([], 1.0)
    assert rig.effects.args_of("unpark") == [(req, 30, True, None)]


def test_normal_fetch_expiry_fails_the_request_and_holds_its_pages_until_the_outcome():
    """A fetch past its deadline fails the request; its pages stay held (the backend may still
    write them) until the outcome arrives, and a late Delivered lands nothing."""
    rig = CoordinatorRig(fetch_timeout_s=10.0)
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
    assert rig.effects.args_of("fail_requests") == [((req,), "kv fetch timed out")]
    attempt.deliver_all()
    rig.coord.advance([], 20.0)
    assert rig.effects.names()[-1:] == ["terminate_request"]
    assert rig.effects.count("unpark") == 0 and rig.effects.count("revert_fetch_pages") == 0
    assert rig.worker.count("quiesce") == 1
    assert rig.records() == [] and rig.coord.status_dump()["finished_pending"] == []


def test_no_timeout_means_no_deadline():
    rig = CoordinatorRig()
    req = worker_request()
    rig.plan_and_launch(req, now=0.0)
    rig.coord.advance([], 1e9)
    rec = rig.record(1)
    assert rec["deadline"] is None and rec["expired"] is False


def test_expired_gen_init_fetch_is_held_not_parked_until_its_outcome():
    """What the engine's release gate and cancel path read while an expired gen-init fetch
    waits for its outcome: the request is held (the engine may not free it), not parked (it is
    not the scheduler's), and its pages are in flight (nothing may evict them); the outcome
    clears all three."""
    rig = CoordinatorRig(fetch_timeout_s=10.0)
    req = gen_init_request()
    attempt = rig.plan_and_launch(req, now=0.0)
    assert rig.coord.parked_request_ids() == {7} and rig.coord.held_request_ids() == frozenset()
    rig.coord.advance([], 10.0)
    assert rig.coord.held_request_ids() == {7}
    assert rig.coord.parked_request_ids() == frozenset()
    assert rig.coord.inflight_request_ids() == {7}
    assert rig.coord.owned_requests() == [req]
    assert rig.coord.has_backend_work() and rig.coord.has_pending_work()
    attempt.deliver_all()
    rig.coord.advance([], 11.0)
    assert rig.coord.held_request_ids() == frozenset()
    assert rig.coord.inflight_request_ids() == frozenset()
    assert rig.coord.owned_requests() == []
    assert not rig.coord.has_backend_work() and not rig.coord.has_pending_work()


def test_finished_request_whose_peer_stays_in_flight_is_terminated_at_its_deadline():
    """A planned fetch never launched here; the request ends and the record stays to vote so
    that the peer, still delivering, is not left without a verdict. The peer never finishes:
    the record waits under the fetch deadline like any other, is expired on this rank alone,
    and the held request is terminated here without a failure, since it already ended."""

    def peer_in_flight(local):
        votes, expired, plans, pending, drained = local
        return ([(key, "INFLIGHT", 0) for key, _, _ in votes], expired, plans, pending, drained)

    rig = CoordinatorRig(dist=ScriptedPeersCollective(peer_in_flight), fetch_timeout_s=10.0)
    req = worker_request()
    rig.coord.advance([req], 0.0)  # planned, no pages this round
    assert rig.coord.holds_finished_request(req, 1.0) is True  # deadline 11.0 from here
    assert rig.effects.names() == ["hold_for_transfer"]
    rig.coord.advance([], 10.9)
    assert rig.last_votes == [((1, "fetch"), "TERMINAL", END)]
    assert rig.coord.held_request_ids() == {1} and rig.effects.count("terminate_request") == 0
    rig.coord.advance([], 11.0)  # the deadline: this rank reports the expiry and stops waiting
    assert rig.record(1)["expired"] and rig.effects.count("terminate_request") == 0
    rig.coord.advance([], 12.0)  # an expired record casts no vote and ends on its own word
    assert rig.last_votes == []
    assert rig.effects.names() == ["hold_for_transfer", "terminate_request"]
    assert rig.effects.count("fail_requests") == 0 and rig.worker.count("quiesce") == 0
    assert rig.records() == [] and rig.coord.status_dump()["finished_pending"] == []


# ---- CarriesAux ----


def test_aux_from_attempt_reaches_unpark():
    rig = CoordinatorRig()
    req = gen_init_request()
    rig.worker.aux_for_next = {"first_token": 42, "ctx_usage": 0.5}
    rig.plan_and_launch(req).deliver_all()
    rig.coord.advance([], 1.0)
    assert rig.effects.args_of("unpark") == [(req, 30, True, {"first_token": 42, "ctx_usage": 0.5})]


# ---- back-pressure that never lets up ----


def test_consecutive_rejections_give_up_then_the_request_computes_locally():
    rig = CoordinatorRig()
    req = worker_request()
    rig.worker.reject_next_calls = 10
    rig.drive_rounds(req, rounds=12)
    # Three rejections give the plan up; the agreement spends the retry on a fresh plan; three
    # more give that up too; the next agreement settles on local compute.
    assert rig.worker.count("fetch") == 6
    assert rig.effects.count("revert_fetch_pages") == 6
    assert rig.effects.count("fail_requests") == 0
    assert rig.records() == [] and rig.coord.fetch_answer(req) is None


def test_three_consecutive_rejections_fail_a_gen_init_fetch():
    rig = CoordinatorRig()
    req = FakeRequest(
        1, prompt_len=29, is_disagg_generation_init=True, route_hints={"ctx": {"peer": "peer1"}}
    )
    rig.worker.reject_next_calls = 10
    for round_index in range(3):
        rig.advance_as_hooks_would(req, float(round_index))
        assert isinstance(rig.coord.fetch_answer(req), FetchPlan)
        rig.coord.launch_reserved_fetches([req], float(round_index))
    assert rig.worker.count("fetch") == 3
    assert rig.effects.count("revert_fetch_pages") == 3
    assert rig.effects.count("fail_requests") == 0 and rig.coord.fetch_answer(req) is DEFER
    rig.advance_as_hooks_would(req, 3.0)  # the FAILED vote lands: a gen-init fetch has no retry
    assert rig.effects.args_of("fail_requests") == [((req,), "kv fetch launch given up")]
    assert rig.coord.fetch_answer(req) is None and rig.records() == []


# ---- the request's end before, during and after the fetch ----


def test_request_ending_while_planned_is_terminated_with_the_agreement():
    """A planned fetch never launched here: no pages, nothing to quiesce; the record votes once
    so a peer that did launch is not left without a verdict."""
    rig = CoordinatorRig()
    req = worker_request()
    rig.coord.advance([req], 0.0)
    assert rig.coord.holds_finished_request(req, 1.0) is True
    assert rig.effects.names() == ["hold_for_transfer"]
    rig.coord.advance([], 2.0)
    assert rig.last_votes == [((1, "fetch"), "TERMINAL", END)]
    assert rig.effects.names() == ["hold_for_transfer", "terminate_request"]
    assert rig.records() == [] and rig.worker.count("quiesce") == 0


def test_request_failed_by_the_coordinator_answers_the_late_release_gate_once():
    """The engine may terminate a failed request later than the failure (deferred under
    attention DP); the coordinator, having terminated the held request itself meanwhile, tells
    the gate once that the request is not the engine's to terminate again."""
    rig = CoordinatorRig(fetch_timeout_s=10.0)
    req = worker_request()
    attempt = rig.plan_and_launch(req, now=0.0)
    rig.coord.advance([], 10.0)  # fails the request; the effects here never call the gate back
    assert rig.effects.names()[-2:] == ["fail_requests", "hold_for_transfer"]
    attempt.deliver_all()
    rig.coord.advance([], 11.0)
    assert rig.effects.args_of("terminate_request") == [(req,)]
    assert rig.coord.holds_finished_request(req, 12.0) is True  # the engine's late gate
    assert rig.coord.holds_finished_request(req, 13.0) is False  # a fresh request of that id
    assert rig.effects.count("terminate_request") == 1


def test_reserve_wait_of_a_device_direct_plan_spends_a_retry_then_computes_locally():
    """The scheduler never finds pages for the plan (``launch_reserved_fetches`` is never called): the
    wait is clocked from the plan's decision, the fetch is given up at the wait timeout, the
    retry waits once more, and the request computes locally; the scheduler is never stalled."""
    rig = CoordinatorRig(fetch_wait_timeout_s=5.0)
    req = worker_request()
    rig.coord.advance([req], 0.0)
    assert isinstance(rig.coord.fetch_answer(req), FetchPlan)
    assert rig.record(1)["resource_wait_since"] == 0.0
    rig.advance_as_hooks_would(req, 4.9)
    assert rig.last_votes == [((1, "fetch"), "UNLAUNCHED", 0)]
    rig.advance_as_hooks_would(req, 5.0)
    assert rig.last_votes == [((1, "fetch"), "FAILED", 0)]
    rec = rig.record(1)
    assert (
        rec["token_end"] is None and rec["resource_wait_since"] is None and rec["retries_left"] == 0
    )
    assert rig.coord.fetch_answer(req) is DEFER and rig.effects.count("fail_requests") == 0

    rig.advance_as_hooks_would(req, 6.0)  # planned afresh; the wait starts over
    assert isinstance(rig.coord.fetch_answer(req), FetchPlan)
    assert rig.record(1)["resource_wait_since"] == 6.0
    rig.advance_as_hooks_would(req, 10.9)
    assert rig.last_votes == [((1, "fetch"), "UNLAUNCHED", 0)]
    rig.advance_as_hooks_would(req, 11.0)
    assert rig.last_votes == [((1, "fetch"), "FAILED", 0)]
    assert rig.records() == [] and rig.coord.fetch_answer(req) is None
    assert rig.effects.calls == [] and rig.worker.count("quiesce") == 0

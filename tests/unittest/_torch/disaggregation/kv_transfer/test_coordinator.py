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
from disaggregation.orchestration.kv_transfer_interfaces import DEFER  # noqa: E402
from disaggregation.orchestration.remote_cache import FetchPlan  # noqa: E402
from fakes import (  # noqa: E402
    FakeChunk,
    FakePlacingPublishes,
    FakePublishes,
    FakeRequest,
    PeerGather,
    Rig,
    extent_names,
    full_attention,
    windowed,
    worker_request,
)

END = 28  # prompt_len 29, tpb 4


def quiesce_indices(trace):
    return [i for i, (name, _) in enumerate(trace) if name == "quiesce"]


def effect_indices(trace, name):
    return [i for i, (n, _) in enumerate(trace) if n == name]


# ---- PLANNED -> IN_FLIGHT -> LANDED -> RELEASED ----


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
    assert rig.effects.names() == ["prepare_fetch_resources", "park_for_fetch"]
    assert rig.effects.only("park_for_fetch") == [((req,),)]
    assert [m for m, _ in rig.worker.calls] == ["open_route", "fetch"]
    assert rig.worker.calls[0][1] == (req.route_hints["ctx"],)
    extent, route = rig.worker.calls[1][1]
    assert route is rig.worker.routes[0] and route.closed == 0
    assert extent_names(extent) == rig.plans[1].unit_names
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
    assert rig.coord.status_dump() == {"records": [], "decided_plans": 0, "finished_pending": []}
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
    assert rec["state"] == "IN_FLIGHT" and rec["abandoned"] is True
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
    assert rig.coord.status_dump() == {"records": [], "decided_plans": 0, "finished_pending": []}


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


# ---- FAILED -> PLANNED (retry) -> FAILED -> RELEASED ----


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
    assert rig.payloads()[-1][0] == [((1, "fetch"), 0, 0, True)]


def test_local_cancel_is_delivered_nothing_not_a_failure():
    rig = Rig()
    req = worker_request()
    rig.plan_and_launch(req).finish(Cancelled(by_peer=False))
    rig.coord.advance([], 1.0)
    # On the wire it is a short serve (B = 0, not failed), so the retry path applies.
    assert rig.payloads()[-1][0] == [((1, "fetch"), 0, 0, False)]
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
    rig = Rig()
    if gen_init:
        req = FakeRequest(7, prompt_len=30, is_gen_init=True, route_hints={"ctx": {"peer": "c"}})
    else:
        req = worker_request()
    rid = req.py_request_id

    # 1. transport error: costs the one retry, record kept.
    rig.worker.open_route_errors.append(RuntimeError("t1"))
    rig.coord.advance([req], 0.0)
    rig.coord.launch_fetches([req], 0.0)
    assert rig.record(rid)["state"] == "PLANNED" and rig.coord.plan_fetch(req) is DEFER
    # 2. rejection: free, record kept.
    rig.worker.reject_next = 1
    rig.coord.advance([req], 1.0)
    rig.coord.launch_fetches([req], 1.0)
    assert rig.record(rid)["state"] == "PLANNED" and rig.coord.plan_fetch(req) is DEFER
    assert rig.effects.count("fail_requests") == 0
    # 3. transport error again: budget exhausted.
    rig.worker.open_route_errors.append(RuntimeError("t2"))
    rig.coord.advance([req], 2.0)
    rig.coord.launch_fetches([req], 2.0)
    assert rig.effects.count("give_back_fetch_pages") == 3
    assert rig.records() == [] and rig.coord.plan_fetch(req) is None
    assert rig.worker.count("quiesce") == 0
    if gen_init:
        assert rig.effects.only("fail_requests") == [((req,), "kv route failed: t2")]
    else:
        assert rig.effects.count("fail_requests") == 0


def test_quiesce_false_at_request_end_is_fatal_and_keeps_the_record():
    rig = Rig()
    req = worker_request()
    attempt = rig.plan_and_launch(req)
    attempt.deliver_all()
    rig.coord.advance([], 1.0)
    rig.worker.quiesce_answers.append(False)
    rig.coord.notify_request_finished(req)
    assert rig.effects.count("fail_fatal") == 1
    assert rig.effects.count("hold_for_transfer") == 0
    assert rig.effects.count("terminate_request") == 0
    assert rig.record(1)["state"] == "LANDED"
    assert rig.coord.status_dump()["finished_pending"] == [1]


# ---- open_route failures ----


@pytest.mark.parametrize(
    "error", [NotImplementedError("single destination"), ValueError("bad hint")]
)
def test_route_refused_gives_back_and_settles_on_local_compute(error):
    rig = Rig()
    req = worker_request()
    rig.worker.open_route_errors.append(error)
    rig.coord.advance([req], 0.0)
    rig.coord.launch_fetches([req], 0.0)
    assert rig.effects.names() == ["prepare_fetch_resources", "give_back_fetch_pages"]
    assert rig.worker.count("fetch") == 0 and rig.worker.count("quiesce") == 0
    assert rig.records() == []
    # This plan can never work: decided as "compute locally", not re-planned.
    assert rig.coord.plan_fetch(req) is None
    assert rig.coord.status_dump()["decided_plans"] == 1


def test_gen_init_route_refused_fails_the_request():
    rig = Rig()
    req = FakeRequest(7, prompt_len=30, is_gen_init=True, route_hints={"ctx": {"peer": "c"}})
    rig.worker.open_route_errors.append(ValueError("unknown peer"))
    rig.coord.advance([req], 0.0)
    rig.coord.launch_fetches([req], 0.0)
    # NB-5: pages first, then the verdict.
    assert rig.effects.names() == [
        "prepare_fetch_resources",
        "give_back_fetch_pages",
        "fail_requests",
    ]
    ((reqs, reason),) = rig.effects.only("fail_requests")
    assert reqs == (req,) and reason == "kv route refused: unknown peer"
    assert rig.coord.plan_fetch(req) is None and rig.records() == []


def test_route_transport_error_costs_the_retry_and_replans_once():
    rig = Rig()
    req = worker_request()
    rig.worker.open_route_errors.append(RuntimeError("peer metadata fetch failed"))
    rig.coord.advance([req], 0.0)
    rig.coord.launch_fetches([req], 0.0)
    assert rig.effects.names() == ["prepare_fetch_resources", "give_back_fetch_pages"]
    assert rig.worker.count("fetch") == 0 and rig.worker.count("quiesce") == 0
    # The record stays, PLANNED without a plan, so the consumed retry is remembered.
    rec = rig.record(1)
    assert rec["state"] == "PLANNED" and rec["attempts"] == 0 and rec["token_end"] is None
    assert rig.coord.plan_fetch(req) is DEFER

    # Planned again next round; the one retry is spent, so a real failure now gives up.
    rig.plan_and_launch(req, now=1.0).finish(Failed("later"))
    rig.coord.advance([], 2.0)
    assert rig.records() == [] and rig.coord.plan_fetch(req) is None


def test_second_route_transport_error_settles_on_local_compute():
    rig = Rig()
    req = worker_request()
    rig.worker.open_route_errors.extend([RuntimeError("first"), RuntimeError("second")])
    rig.coord.advance([req], 0.0)
    rig.coord.launch_fetches([req], 0.0)
    rig.coord.advance([req], 1.0)
    rig.coord.launch_fetches([req], 1.0)
    assert rig.effects.count("give_back_fetch_pages") == 2
    assert rig.effects.count("fail_requests") == 0
    assert rig.records() == [] and rig.coord.plan_fetch(req) is None


def test_second_route_transport_error_fails_gen_init():
    rig = Rig()
    req = FakeRequest(7, prompt_len=30, is_gen_init=True, route_hints={"ctx": {"peer": "c"}})
    rig.worker.open_route_errors.extend([RuntimeError("first"), RuntimeError("second")])
    rig.coord.advance([req], 0.0)
    rig.coord.launch_fetches([req], 0.0)
    assert rig.effects.count("fail_requests") == 0 and rig.coord.plan_fetch(req) is DEFER
    rig.coord.advance([req], 1.0)
    rig.coord.launch_fetches([req], 1.0)
    ((reqs, reason),) = rig.effects.only("fail_requests")
    assert reqs == (req,) and reason == "kv route failed: second"
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
    assert rig.plans[7].token_end == 28 and rig.plans[7].units_by_group == {0: tuple(range(7))}
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


def test_served_short_retries_with_min_hint():
    rig = Rig()
    req = worker_request()
    attempt = rig.plan_and_launch(req)
    plan = rig.plans[1]
    last_block = rig.reader.unit_names(req, [6])
    attempt.finish(Delivered(plan.unit_names - last_block))
    rig.coord.advance([], 1.0)
    assert rig.effects.count("give_back_fetch_pages") == 1 and rig.effects.count("unpark") == 0
    assert rig.record(1)["state"] == "PLANNED"

    rig.coord.advance([req], 2.0)
    replan = rig.coord.plan_fetch(req)
    assert isinstance(replan, FetchPlan) and replan.token_end == END - 4
    assert replan.units_by_group == {0: tuple(range(6))}

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


# ---- SubmissionRejected ----


def test_submission_rejected_gives_back_without_quiesce_or_retry_cost():
    rig = Rig()
    req = worker_request()
    rig.worker.reject_next = 1
    rig.coord.advance([req], 0.0)
    rig.coord.launch_fetches([req], 0.0)
    # NB-5: the resources were prepared for the launch, so they are given back, in that order.
    assert rig.effects.names() == ["prepare_fetch_resources", "give_back_fetch_pages"]
    assert rig.worker.count("quiesce") == 0
    assert rig.worker.routes[0].closed == 1
    # The record persists, PLANNED without a plan, so the retry budget is one budget for the
    # request rather than one per rejection.
    rec = rig.record(1)
    assert rec["state"] == "PLANNED" and rec["token_end"] is None and rec["attempts"] == 0
    assert rig.coord.plan_fetch(req) is DEFER  # back to undecided, planned again next round

    # The retry budget is intact: a real failure afterwards still gets its one retry.
    rig.plan_and_launch(req, now=1.0).finish(Failed("later"))
    rig.coord.advance([], 2.0)
    assert rig.record(1)["state"] == "PLANNED"


# ---- expiry ----


def test_gen_init_expiry_fails_the_request():
    rig = Rig(fetch_timeout_s=10.0)
    req = FakeRequest(7, prompt_len=30, is_gen_init=True, route_hints={"ctx": {"peer": "c"}})
    attempt = rig.plan_and_launch(req, now=0.0)
    assert rig.plans[7].no_local_fallback is True
    rig.coord.advance([], 9.9)
    assert rig.effects.count("fail_requests") == 0
    rig.coord.advance([], 10.0)
    q, gb = quiesce_indices(rig.trace), effect_indices(rig.trace, "give_back_fetch_pages")
    assert len(q) == 1 and q[0] < gb[0]
    assert rig.effects.only("fail_requests") == [((req,), "kv fetch timed out")]
    assert rig.records() == [] and rig.coord.plan_fetch(req) is None
    # A late delivery is dropped.
    attempt.deliver_all()
    rig.coord.advance([], 11.0)
    assert rig.effects.count("unpark") == 0


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
    assert rec["state"] == "IN_FLIGHT" and rec["abandoned"] is True
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
    assert rec["deadline"] is None and rec["abandoned"] is False


def test_abandon_marks_records_but_stops_nothing():
    rig = Rig(fetch_timeout_s=10.0)
    req = worker_request()
    attempt = rig.plan_and_launch(req, now=0.0)
    rig.coord.abandon(req)
    assert rig.record(1)["abandoned"] is True
    rig.coord.advance([], 50.0)  # past the deadline: already abandoned, nothing new
    attempt.deliver_all()
    rig.coord.advance([], 51.0)
    assert rig.effects.count("unpark") == 1


# ---- publish ----


def test_publish_flow_holds_then_terminates_after_quiesce():
    pub = FakePublishes(name="store")
    rig = Rig(publishers=[pub])
    req = FakeRequest(3, prompt_len=29)
    ext = rig.reader.script_publish(req, [(range(0, 4), False, None), (range(4, 7), True, None)])

    rig.coord.publish_committed_blocks([req], finished=(), now=0.0)
    # A publisher that does not place pieces is not offered an intermediate piece.
    assert pub.payloads("publish") == []
    rec = rig.record(3, "publish")
    assert rec["state"] == "PLANNED" and rec["attempts"] == 0
    assert rig.coord.has_inflight() is False
    rig.coord.advance([], 1.0)
    assert rig.record(3, "publish")["state"] == "PLANNED" and pub.count("quiesce") == 0

    rig.coord.publish_committed_blocks([req], finished=[3], now=2.0)
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
    assert rig.effects.names() == [
        "hold_for_transfer",
        "stage_transfer_response",
        "terminate_request",
    ]
    assert rig.effects.only("terminate_request") == [(req,)]
    assert rig.records() == [] and rig.coord.status_dump()["finished_pending"] == []
    q = quiesce_indices(rig.trace)
    assert q[0] < effect_indices(rig.trace, "terminate_request")[0]


def test_publish_landed_before_request_end_terminates_immediately_at_end():
    pub = FakePublishes()
    rig = Rig(publishers=[pub])
    req = FakeRequest(3, prompt_len=29)
    rig.coord.publish_committed_blocks([req], finished=(), now=0.0)  # default: everything, is_last
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
    rig.coord.publish_committed_blocks([req], finished=[3], now=0.0)
    pub.attempts[0].finish(Failed("peer gone"))
    rig.coord.advance([], 1.0)
    assert rig.effects.names() == ["hold_for_transfer", "terminate_request"]
    assert rig.effects.count("fail_requests") == 0
    assert rig.effects.count("stage_transfer_response") == 0
    assert rig.records() == []


def test_publish_expiry_terminates_the_held_request():
    pub = FakePublishes()
    rig = Rig(publishers=[pub], publish_timeout_s=5.0)
    req = FakeRequest(3, prompt_len=29)
    rig.coord.publish_committed_blocks([req], finished=[3], now=0.0)
    rig.coord.advance([], 5.0)
    assert rig.effects.names() == ["hold_for_transfer", "terminate_request"]
    assert rig.effects.count("fail_requests") == 0
    assert pub.count("quiesce") == 1 and rig.records() == []


def test_publish_rejected_outright_terminates_the_finished_request():
    pub = FakePublishes()
    rig = Rig(publishers=[pub])
    req = FakeRequest(3, prompt_len=29)
    pub.reject_next = 1
    rig.coord.publish_committed_blocks([req], finished=[3], now=0.0)
    assert pub.attempts == [] and pub.count("quiesce") == 0  # nothing escaped, nothing to quiesce
    # Nothing was ever offered: not held, terminated straight away, and the record is gone.
    assert rig.effects.names() == ["terminate_request"]
    assert rig.effects.count("fail_requests") == 0
    assert rig.records() == [] and rig.coord.status_dump()["finished_pending"] == []


def test_partial_publish_rejection_terminates_the_held_request_at_next_reap():
    # Publish accepted, place rejected: the pieces that were offered may land, the publish as a
    # whole has still failed.
    placing = FakePlacingPublishes()
    rig = Rig(publishers=[placing])
    req = FakeRequest(3, prompt_len=29)
    rig.reader.script_publish(req, [(range(0, 7), True, FakeChunk(3, 0))])
    placing.reject_methods.add("place")
    rig.coord.publish_committed_blocks([req], finished=[3], now=0.0)
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
    rig.coord.publish_committed_blocks([req], finished=[3], now=0.0)
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
    rig.coord.publish_committed_blocks([req], finished=(), now=0.0)
    placing.attempts[0].deliver_all()  # publish of piece 0 fine ...
    placing.attempts[1].finish(Failed("place lost"))  # ... its place failed
    rig.coord.advance([], 1.0)
    # Failed as soon as any piece fails, without waiting for the last piece.
    assert placing.count("quiesce") == 1 and rig.records() == []
    assert rig.effects.calls == []  # request still running: nothing to tell the engine yet


def test_publish_deadline_expires_with_terminal_intermediate_pieces():
    placing = FakePlacingPublishes()
    rig = Rig(publishers=[placing], publish_timeout_s=5.0)
    req = FakeRequest(3, prompt_len=29)
    rig.reader.script_publish(
        req, [(range(0, 3), False, FakeChunk(3, 0)), (range(3, 7), True, FakeChunk(3, 1))]
    )
    rig.coord.publish_committed_blocks([req], finished=(), now=0.0)
    for a in placing.attempts:
        a.finish(Delivered(frozenset()))  # publish and place of piece 0 both done
    rig.coord.advance([], 4.0)
    assert rig.record(3, "publish")["state"] == "IN_FLIGHT"  # every piece so far landed, not last
    rig.coord.advance([], 5.0)
    # Timed out: quiesced and released although no piece was ever the last one.
    assert placing.count("quiesce") == 1 and rig.records() == []
    assert rig.effects.calls == []  # request still running: the warning is the only word


def test_publish_rejected_while_fetch_in_flight_terminates_once_the_fetch_releases():
    pub = FakePublishes()
    rig = Rig(publishers=[pub])
    req = worker_request()
    attempt = rig.plan_and_launch(req)
    pub.reject_next = 1
    rig.coord.publish_committed_blocks([req], finished=[1], now=1.0)
    # The publish verdict waits: the fetch still names the pages.
    assert rig.effects.names()[-1:] == ["hold_for_transfer"]
    assert rig.effects.count("fail_requests") == 0
    assert rig.record(1, "publish") is None and rig.record(1)["abandoned"] is True
    attempt.deliver_all()
    rig.coord.advance([], 2.0)
    assert rig.worker.count("quiesce") == 1
    assert rig.effects.names()[-1:] == ["terminate_request"]
    assert rig.effects.count("fail_requests") == 0
    assert rig.effects.count("stage_transfer_response") == 0
    assert rig.effects.count("unpark") == 0
    assert rig.coord.status_dump() == {"records": [], "decided_plans": 0, "finished_pending": []}


def test_publish_landed_first_waits_for_the_in_flight_fetch_before_terminating():
    pub = FakePublishes()
    rig = Rig(publishers=[pub])
    req = worker_request()
    attempt = rig.plan_and_launch(req)
    rig.coord.publish_committed_blocks([req], finished=[1], now=1.0)
    assert rig.effects.names()[-1:] == ["hold_for_transfer"]
    pub.attempts[0].deliver_all()
    rig.coord.advance([], 2.0)
    assert pub.count("quiesce") == 1 and rig.record(1, "publish") is None
    assert rig.effects.count("terminate_request") == 0  # the fetch is still in flight
    assert rig.effects.count("stage_transfer_response") == 0
    attempt.deliver_all()
    rig.coord.advance([], 3.0)
    assert rig.effects.names()[-2:] == ["stage_transfer_response", "terminate_request"]
    assert rig.effects.count("unpark") == 0
    assert rig.coord.status_dump() == {"records": [], "decided_plans": 0, "finished_pending": []}


def test_abandoned_publish_of_held_request_still_times_out_and_terminates():
    pub = FakePublishes()
    rig = Rig(publishers=[pub], publish_timeout_s=5.0)
    req = FakeRequest(3, prompt_len=29)
    rig.coord.publish_committed_blocks([req], finished=[3], now=0.0)
    rig.coord.abandon(req)
    assert rig.record(3, "publish")["abandoned"] is True
    rig.coord.advance([], 5.0)
    # Abandoning stops nothing and forgives nothing: the deadline is what releases a held
    # request, and a finished request is terminated rather than failed.
    assert rig.effects.names() == ["hold_for_transfer", "terminate_request"]
    assert rig.effects.count("fail_requests") == 0
    assert pub.count("quiesce") == 1 and rig.records() == []


def test_both_records_in_flight_fetch_releases_first_then_publish_lands():
    pub = FakePublishes()
    rig = Rig(publishers=[pub])
    req = worker_request()
    attempt = rig.plan_and_launch(req)
    rig.coord.publish_committed_blocks([req], finished=[1], now=1.0)
    assert rig.effects.names()[-1:] == ["hold_for_transfer"]
    assert (
        rig.record(1)["state"] == "IN_FLIGHT" and rig.record(1, "publish")["state"] == "IN_FLIGHT"
    )

    attempt.deliver_all()
    rig.coord.advance([], 2.0)
    assert rig.record(1) is None and rig.record(1, "publish")["state"] == "IN_FLIGHT"
    assert rig.effects.count("stage_transfer_response") == 0
    assert rig.effects.count("terminate_request") == 0
    assert rig.effects.count("unpark") == 0

    pub.attempts[0].deliver_all()
    rig.coord.advance([], 3.0)
    assert rig.effects.count("stage_transfer_response") == 1
    assert rig.effects.count("terminate_request") == 1
    assert rig.effects.names()[-2:] == ["stage_transfer_response", "terminate_request"]
    assert rig.worker.count("quiesce") == 1 and pub.count("quiesce") == 1
    assert rig.coord.status_dump() == {"records": [], "decided_plans": 0, "finished_pending": []}


def test_no_publishers_means_no_publish_records():
    rig = Rig()
    req = FakeRequest(3, prompt_len=29)
    rig.coord.publish_committed_blocks([req], finished=[3], now=0.0)
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
        rig.coord.publish_committed_blocks([req], finished=[3] if i == 2 else (), now=float(i))

    assert placing.payloads("place") == chunks
    assert placing.payloads("publish") == ext
    # Design §7.5: a publisher that does not place pieces hears once, on the last piece, and
    # must not need to read ``is_last``.
    assert plain.payloads("publish") == [ext[-1]]


def test_places_pieces_not_called_without_a_chunk():
    placing = FakePlacingPublishes()
    rig = Rig(publishers=[placing])
    req = FakeRequest(3, prompt_len=29)
    rig.coord.publish_committed_blocks([req], finished=(), now=0.0)
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


def test_one_allgather_per_advance_over_world():
    rig = Rig()
    req = worker_request()
    rig.coord.advance([req], 0.0)
    rig.coord.launch_fetches([req], 0.0)
    rig.coord.advance([], 1.0)
    rig.coord.notify_request_finished(req)
    rig.coord.publish_committed_blocks([], finished=(), now=2.0)
    assert len(rig.dist.calls) == 2
    assert all(scope == "world" for scope, _ in rig.dist.calls)


def test_attention_dp_gathers_over_pp_group():
    rig = Rig(attention_dp=True)
    rig.coord.advance([], 0.0)
    assert rig.dist.calls[0][0] == "pp"


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
    assert arrivals == [((1, "fetch"), END, END, False)] and plans == []


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
            [(key, b - 4, hint - 4, failed) for key, b, hint, failed in arrivals],
            expired,
            plans,
        )

    rig = Rig(gather=PeerGather(peer))
    req = worker_request()
    rig.plan_and_launch(req).deliver_all()
    rig.coord.advance([], 1.0)
    assert rig.effects.count("unpark") == 0 and rig.effects.count("give_back_fetch_pages") == 1
    assert rig.record(1)["state"] == "PLANNED"
    rig.coord.advance([req], 2.0)
    assert rig.coord.plan_fetch(req).token_end == END - 4  # MIN(hint) from the peer


def test_peer_voting_a_different_source_makes_the_plan_none():
    def peer(local):
        arrivals, expired, plans = local
        return (arrivals, expired, [(rid, (v[0], "store")) for rid, v in plans])

    rig = Rig(gather=PeerGather(peer))
    req = worker_request()
    rig.coord.advance([req], 0.0)
    assert rig.coord.plan_fetch(req) is None and rig.records() == []


def test_peer_not_reporting_an_arrival_keeps_it_in_flight():
    rig = Rig(gather=PeerGather(lambda local: ([], [], local[2])))
    req = worker_request()
    rig.plan_and_launch(req).deliver_all()
    rig.coord.advance([], 1.0)
    assert rig.record(1)["state"] == "IN_FLIGHT" and rig.effects.count("unpark") == 0


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
            "abandoned": False,
            "token_end": END,
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
    rig.coord.advance([req], 2.0)  # probe_budget_rounds = 2 spent: plan without the store
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


def test_three_consecutive_rejections_send_a_normal_fetch_to_local_compute():
    rig = Rig()
    req = worker_request()
    rig.worker.reject_next = 10
    for round_index in range(6):
        if rig.coord.plan_fetch(req) is DEFER:  # undecided: a candidate for this round
            rig.coord.advance([req], float(round_index))
        plan = rig.coord.plan_fetch(req)
        if plan is None:
            break
        assert isinstance(plan, FetchPlan)
        rig.coord.launch_fetches([req], float(round_index))
    assert rig.coord.plan_fetch(req) is None
    assert rig.worker.count("fetch") == 3
    assert rig.effects.count("give_back_fetch_pages") == 3
    assert rig.effects.count("fail_requests") == 0
    assert rig.records() == []


def test_three_consecutive_rejections_fail_a_gen_init_fetch():
    rig = Rig()
    req = FakeRequest(1, prompt_len=29, is_gen_init=True, route_hints={"ctx": {"peer": "peer1"}})
    rig.worker.reject_next = 10
    for round_index in range(3):
        rig.coord.advance([req], float(round_index))
        assert isinstance(rig.coord.plan_fetch(req), FetchPlan)
        rig.coord.launch_fetches([req], float(round_index))
    assert rig.worker.count("fetch") == 3
    assert rig.effects.count("give_back_fetch_pages") == 3
    (failed,) = rig.effects.only("fail_requests")
    assert failed[0] == (req,) and failed[1].startswith("kv fetch rejected")
    assert rig.coord.plan_fetch(req) is None
    assert rig.records() == []

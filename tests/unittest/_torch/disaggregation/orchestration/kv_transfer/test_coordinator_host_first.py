# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A fetch from a ``LandsOnHost`` store: PLANNED -> LANDING -> LANDED -> IN_FLIGHT (the placement
into pages) -> DELIVERED. The landing starts when the plan is decided, with no pages; the
scheduler reserves only once the content is on the host; the landing is released with the
placement's outcome, the record's failure, or the request's end.

Single rank, over the ``host`` source of ``fakes.CoordinatorRig`` (``host_rig``).
"""

import pytest

__extra_import_path__ = ["~/tensorrt_llm/_torch"]
from disaggregation.base.cache_backend import Failed  # noqa: E402
from disaggregation.remote_cache import DEFER, FetchPlan  # noqa: E402
from fakes import (  # noqa: E402
    EMPTY_DUMP,
    END,
    CoordinatorRig,
    ScriptedPeersCollective,
    assert_quiesce_precedes,
    extent_names,
    host_rig,
    plan_unit_names,
    store_request,
    worker_request,
)

pytestmark = pytest.mark.cpu_only


def test_host_first_plan_starts_its_landing_when_decided_and_parks_nothing():
    rig = host_rig()
    req = store_request()
    rig.plan_and_land(req)
    assert [m for m, _ in rig.host.calls] == ["probe", "fetch_to_host"]
    (units,) = rig.host.calls[-1][1]
    assert frozenset(units) == rig.reader.unit_names(req, range(7))
    rec = rig.record(1)
    assert rec["state"] == "LANDING" and rec["has_landing"] and rec["attempts"] == 0
    assert rig.effects.calls == []  # no pages: nothing prepared, nothing parked
    assert rig.coord.fetch_answer(req) is DEFER
    # A landing is work in flight for pacing, but it names no page and parks no request.
    assert rig.coord.has_backend_work() is True
    assert rig.coord.inflight_request_ids() == frozenset()
    assert rig.coord.parked_request_ids() == frozenset()


def test_plan_fetch_answers_defer_defer_plan_none_none_along_the_host_first_path():
    rig = host_rig()
    req = store_request()
    landing = rig.plan_and_land(req)
    assert rig.coord.fetch_answer(req) is DEFER  # LANDING: the units are on their way
    landing.deliver_all()
    rig.coord.advance([], 1.0)
    plan = rig.coord.fetch_answer(req)  # LANDED: reserve pages for exactly this plan
    assert isinstance(plan, FetchPlan) and plan.token_end == END and plan.source == "host"
    rec = rig.record(1)
    assert rec["state"] == "LANDED" and rec["token_end"] == plan.token_end
    attempt = rig.launch_reserved(req, 1.0)
    assert rig.coord.fetch_answer(req) is None  # IN_FLIGHT: parked, out of the scheduler's reach
    attempt.deliver_all()
    rig.coord.advance([], 2.0)
    assert rig.record(1)["state"] == "DELIVERED" and rig.coord.fetch_answer(req) is None


def test_refused_landing_keeps_the_plan_votes_unlaunched_and_is_asked_again_next_round():
    rig = host_rig()
    req = store_request()
    rig.host.reject_next_calls = 1
    rig.coord.advance([req], 0.0)
    rec = rig.record(1)
    assert rec["state"] == "PLANNED" and rec["token_end"] is not None and not rec["has_landing"]
    assert rec["consecutive_launch_failures"] == 0  # not page back-pressure
    assert rec["resource_wait_since"] == 0.0
    assert rig.coord.fetch_answer(req) is DEFER and rig.effects.calls == []

    rig.advance_as_hooks_would(req, 1.0)
    assert rig.last_votes == [((1, "fetch"), "UNLAUNCHED", 0)]
    assert rig.host.count("fetch_to_host") == 2 and rig.host.count("probe") == 1
    rec = rig.record(1)
    assert rec["state"] == "LANDING" and rec["resource_wait_since"] is None and rec["has_landing"]


def test_peer_launched_at_is_cleared_when_the_landing_memory_arrives():
    """A rank refused landing memory votes UNLAUNCHED; once a peer has landed, its unlaunched
    clock runs. The clock is for the peers' sake, and it stops the moment this rank's landing
    is accepted: a LANDING record is on the fetch deadline, not on the unlaunched clock."""

    def peer_landed(local):
        votes, expired, plans, pending, drained = local
        return ([(key, "TERMINAL", END) for key, _, _ in votes], expired, plans, pending, drained)

    rig = host_rig(dist=ScriptedPeersCollective(peer_landed))
    req = store_request()
    rig.host.reject_next_calls = 2
    rig.coord.advance([req], 0.0)  # planned; the first landing request is refused
    assert rig.record(1)["state"] == "PLANNED" and rig.record(1)["peer_launch_seen_at"] is None
    rig.advance_as_hooks_would(req, 1.0)  # the peer has landed: the clock starts; refused once more
    rec = rig.record(1)
    assert (
        rec["state"] == "PLANNED" and rec["peer_launch_seen_at"] == 1.0 and not rec["has_landing"]
    )
    rig.advance_as_hooks_would(req, 2.0)  # the landing memory arrives
    rec = rig.record(1)
    assert rec["state"] == "LANDING" and rec["has_landing"]
    assert rec["peer_launch_seen_at"] is None and rec["resource_wait_since"] is None
    assert rig.host.count("fetch_to_host") == 3


@pytest.mark.parametrize("ending", ["short", "failed"])
def test_landing_that_fails_or_comes_up_short_is_released_and_replanned_without_pages(ending):
    rig = host_rig()
    req = store_request()
    landing = rig.plan_and_land(req)
    if ending == "short":
        landing.deliver_all_but(*rig.reader.unit_names(req, [6]))
    else:
        landing.finish(Failed("store gone"))
    rig.coord.advance([], 1.0)
    kind = ("TERMINAL", END - 4) if ending == "short" else ("FAILED", 0)
    assert rig.last_votes == [((1, "fetch"), *kind)]
    assert landing.closes == 1
    assert rig.effects.count("revert_fetch_pages") == 0 and rig.host.count("quiesce") == 0
    assert rig.effects.count("unpark") == 0 and rig.effects.count("fail_requests") == 0
    rec = rig.record(1)
    assert rec["state"] == "PLANNED" and rec["token_end"] is None and not rec["has_landing"]
    assert rec["retries_left"] == 0 and rig.coord.fetch_answer(req) is DEFER

    rig.coord.advance([req], 2.0)  # replanned, and the new landing starts in the same round
    assert rig.host.count("fetch_to_host") == 2
    assert rig.record(1)["token_end"] == (END - 4 if ending == "short" else END)
    assert rig.record(1)["state"] == "LANDING"


def test_placement_lands_unparks_and_releases_the_landing_in_the_same_round():
    rig = host_rig()
    req = store_request()
    landing = rig.land_and_agree(req)
    rec = rig.record(1)
    assert (
        rec["has_landing"] and rec["resource_wait_since"] == 1.0
    )  # waiting for pages since landed

    attempt = rig.launch_reserved(req, 2.0)
    assert rig.effects.names() == ["prepare_fetch_resources", "park_for_fetch"]
    assert extent_names(attempt.payload) == plan_unit_names(rig.plans[1])
    rec = rig.record(1)
    assert rec["state"] == "IN_FLIGHT" and rec["attempts"] == 1 and rec["try_index"] == 0
    assert rec["resource_wait_since"] is None and landing.closes == 0
    assert rig.coord.parked_request_ids() == {1} and rig.coord.inflight_request_ids() == {1}

    attempt.deliver_all()
    rig.coord.advance([], 3.0)
    assert rig.effects.args_of("unpark") == [(req, END, False, None)]
    assert landing.closes == 1  # the copy was complete before its outcome: gone at once
    rec = rig.record(1)
    assert rec["state"] == "DELIVERED" and rec["has_landing"] is False  # the record stays
    assert rig.host.count("quiesce") == 0
    rig.coord.holds_finished_request(req)
    assert rig.host.calls[-1] == ("quiesce", ((attempt,), True))  # the placement's release point
    assert rig.records() == [] and landing.closes == 1


def test_placement_served_short_quiesces_gives_back_and_releases_the_landing():
    rig = host_rig()
    req = store_request()
    landing = rig.land_and_agree(req)
    attempt = rig.launch_reserved(req, 2.0)
    attempt.deliver_all_but(*rig.reader.unit_names(req, [6]))
    rig.coord.advance([], 3.0)
    assert_quiesce_precedes(rig.trace, "revert_fetch_pages")
    assert rig.host.calls[-1] == ("close", (landing,))  # after quiesce and give-back
    assert landing.closes == 1 and rig.effects.count("unpark") == 0
    rec = rig.record(1)
    assert rec["state"] == "PLANNED" and rec["token_end"] is None and not rec["has_landing"]
    assert rec["retries_left"] == 0 and rec["retry_cap"] == END - 4

    rig.coord.advance([req], 4.0)  # the retry lands on the host afresh
    assert rig.host.count("fetch_to_host") == 2 and rig.record(1)["state"] == "LANDING"
    assert rig.record(1)["token_end"] == END - 4


@pytest.mark.parametrize("source", ["host", "worker"])
def test_units_the_local_cache_committed_meanwhile_count_as_served(source):
    """Pages the reservation found already committed (another request computed the same
    prefix while the fetch waited) are not fetched over, and do not make the delivery short."""
    rig = CoordinatorRig(sources=(source,))
    req = worker_request() if source == "worker" else store_request()
    rig.reader.committed_blocks[1] = 2
    if source == "host":
        rig.land_and_agree(req)
        attempt = rig.launch_reserved(req, 2.0)
    else:
        attempt = rig.plan_and_launch(req)
    assert extent_names(attempt.payload) == rig.reader.unit_names(req, range(2, 7))
    assert rig.record(1)["committed_names"] == len(rig.reader.unit_names(req, range(2)))
    attempt.deliver_all()
    rig.coord.advance([], 3.0)
    assert rig.last_votes == [((1, "fetch"), "TERMINAL", END)]
    assert rig.effects.args_of("unpark") == [(req, END, False, None)]
    assert rig.effects.count("revert_fetch_pages") == 0


def test_placement_of_an_empty_extent_parks_for_one_round_then_unparks():
    rig = host_rig()
    req = store_request()
    landing = rig.land_and_agree(req)
    rig.reader.committed_blocks[1] = 7  # every block committed locally while landing
    attempt = rig.launch_reserved(req, 2.0)
    assert attempt.payload.units == ()  # nothing to copy: the placement is done at once
    assert rig.effects.names() == ["prepare_fetch_resources", "park_for_fetch"]
    assert rig.record(1)["state"] == "IN_FLIGHT"
    rig.coord.advance([], 3.0)
    assert rig.last_votes == [((1, "fetch"), "TERMINAL", END)]
    assert rig.effects.args_of("unpark") == [(req, END, False, None)]
    assert landing.closes == 1


@pytest.mark.parametrize("source", ["host", "worker"])
def test_units_the_reservation_has_no_page_for_make_the_delivery_short(source):
    rig = CoordinatorRig(sources=(source,))
    req = worker_request() if source == "worker" else store_request()
    rig.reader.reserved_blocks[1] = 5  # the scheduler reserved fewer pages than planned
    if source == "host":
        landing = rig.land_and_agree(req)
        attempt = rig.launch_reserved(req, 2.0)
    else:
        attempt = rig.plan_and_launch(req)
    assert extent_names(attempt.payload) == rig.reader.unit_names(req, range(5))
    assert rig.record(1)["committed_names"] == 0
    attempt.deliver_all()
    rig.coord.advance([], 3.0)
    assert rig.last_votes == [((1, "fetch"), "TERMINAL", 20)]
    assert rig.effects.count("unpark") == 0 and rig.effects.count("revert_fetch_pages") == 1
    assert rig.record(1)["state"] == "PLANNED"
    if source == "host":
        assert landing.closes == 1


def test_placement_refused_three_times_gives_up_and_the_agreement_replans():
    rig = host_rig()
    req = store_request()
    landing = rig.land_and_agree(req)
    rig.host.reject_next_place_calls = 3
    for index, now in enumerate((2.0, 3.0, 4.0)):
        assert isinstance(rig.coord.fetch_answer(req), FetchPlan)
        rig.coord.launch_reserved_fetches([req], now)
        rec = rig.record(1)
        assert rec["state"] == "LANDED" and rec["consecutive_launch_failures"] == index + 1
    assert rig.effects.count("revert_fetch_pages") == 3  # page back-pressure, counted as such
    assert rig.record(1)["gave_up_launching"] and rig.coord.fetch_answer(req) is DEFER
    assert landing.closes == 0

    rig.advance_as_hooks_would(req, 5.0)
    assert rig.last_votes == [((1, "fetch"), "FAILED", 0)]
    rec = rig.record(1)
    assert landing.closes == 1 and not rec["has_landing"]
    assert rec["state"] == "PLANNED" and rec["token_end"] is None and rec["retries_left"] == 0
    assert rig.host.count("quiesce") == 0 and rig.effects.count("fail_requests") == 0


def test_landing_expiry_fails_the_request_and_releases_the_landing_at_once():
    rig = host_rig(fetch_timeout_s=10.0)
    req = store_request()
    landing = rig.plan_and_land(req, now=0.0)
    assert rig.record(1)["deadline"] == 10.0
    rig.coord.advance([], 10.0)
    assert rig.effects.names() == ["fail_requests"]  # no pages: nothing to hold
    assert rig.effects.args_of("fail_requests") == [((req,), "kv fetch timed out")]
    # The landing names no page, so nothing waits for its outcome: record and landing go now.
    assert landing.closes == 1 and rig.records() == []
    assert rig.coord.held_request_ids() == frozenset()
    assert rig.coord.status_dump() == EMPTY_DUMP
    landing.deliver_all()  # late, and moot
    rig.coord.advance([], 11.0)
    assert landing.closes == 1 and rig.effects.names() == ["fail_requests"]


@pytest.mark.parametrize("stage", ["LANDING", "LANDED"])
def test_request_ending_before_placement_releases_the_landing_and_votes_until_agreed(stage):
    """The landing names no page and goes at once; the record stays one more round to vote
    TERMINAL at the plan's target, so that a peer still delivering lands on its own word and
    every rank terminates the request in the same round."""
    rig = host_rig()
    req = store_request()
    landing = rig.land_and_agree(req) if stage == "LANDED" else rig.plan_and_land(req)
    assert rig.coord.holds_finished_request(req, 2.0) is True
    assert landing.closes == 1 and rig.host.count("quiesce") == 0
    assert rig.effects.names() == ["hold_for_transfer"]
    rec = rig.record(1)
    assert rec["state"] == stage and rec["has_landing"] is False
    assert rig.coord.has_backend_work() is False  # nothing runs for it any more
    rig.coord.advance([], 3.0)
    assert rig.last_votes == [((1, "fetch"), "TERMINAL", END)]
    assert rig.effects.names() == ["hold_for_transfer", "terminate_request"]
    assert rig.records() == [] and rig.coord.status_dump()["finished_pending"] == []


def test_fetch_wait_timeout_spends_a_retry_at_each_wait_then_computes_locally():
    """The rank's own clock: refused landing memory (PLANNED) and pages that never come (LANDED)
    each time out once; the second timeout is out of retries and the request computes locally."""
    rig = host_rig(fetch_wait_timeout_s=5.0)
    req = store_request()
    rig.host.reject_next_calls = 100
    rig.coord.advance([req], 0.0)
    rig.advance_as_hooks_would(req, 4.9)
    assert rig.last_votes == [((1, "fetch"), "UNLAUNCHED", 0)]
    rig.advance_as_hooks_would(req, 5.0)
    assert rig.last_votes == [((1, "fetch"), "FAILED", 0)]
    rec = rig.record(1)
    assert (
        rec["token_end"] is None and rec["resource_wait_since"] is None and rec["retries_left"] == 0
    )
    assert rig.coord.fetch_answer(req) is DEFER and rig.effects.count("fail_requests") == 0

    rig.host.reject_next_calls = 0
    rig.advance_as_hooks_would(req, 6.0)  # planned afresh; the landing memory is there this time
    landing = rig.host.landings[-1]
    landing.deliver_all()
    rig.coord.advance([], 7.0)
    assert rig.record(1)["state"] == "LANDED" and rig.record(1)["resource_wait_since"] == 7.0
    rig.advance_as_hooks_would(req, 11.9)  # the scheduler never finds pages
    assert rig.last_votes == [((1, "fetch"), "UNLAUNCHED", 0)]
    rig.advance_as_hooks_would(req, 12.0)
    assert rig.last_votes == [((1, "fetch"), "FAILED", 0)]
    assert landing.closes == 1 and rig.records() == []
    assert rig.coord.fetch_answer(req) is None and rig.effects.count("fail_requests") == 0
    assert rig.effects.count("revert_fetch_pages") == 0 and rig.host.count("quiesce") == 0

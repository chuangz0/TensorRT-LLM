# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``n`` real coordinators, one per rank, stepped in lockstep on one thread through the ``gather``
seam (``LockstepWorld``), for ``n`` in {2, 3, 4}. Rank 1 is the rank that diverges; every other
rank behaves like rank 0.

Covers the one reduction of design §7.1 "齐" over the four vote kinds: an INFLIGHT vote holds
the round, then any FAILED is decisive for every rank, then an UNLAUNCHED vote holds a landing,
else the landing takes MIN(B). Plan answers become DEFER if any rank defers, None on any
disagreement about ``(token_end, source)``, the plan when identical. A record of a finished
request keeps voting until the ranks agree, so every rank terminates the request in the same
round; a record past its deadline is rank-local and settles on its own outcome.
"""

import pytest

__extra_import_path__ = ["~/tensorrt_llm/_torch"]
from disaggregation.base.cache_backend import Failed  # noqa: E402
from disaggregation.remote_cache import DEFER, FetchPlan  # noqa: E402
from fakes import (  # noqa: E402
    FakeChunk,
    FakePlacingPublishes,
    FakePublishes,
    FakeRequest,
    LockstepWorld,
    Rig,
    ordinals_by_group,
    worker_request,
)

pytestmark = pytest.mark.cpu_only

END = 28
KEY = (1, "fetch")
DIVERGENT = 1
"""The rank whose fakes are scripted differently from the others in every scenario."""
UNLAUNCHED_TIMEOUT_S = 10.0


@pytest.fixture(params=[2, 3, 4], ids=lambda n: f"n{n}")
def world(request):
    return LockstepWorld(request.param)


def make_rigs(world, **kw):
    kw.setdefault("unlaunched_timeout_s", UNLAUNCHED_TIMEOUT_S)
    return [Rig(dist=world.gathers[r], **kw) for r in range(world.n)]


def peers(rigs):
    """Every rank but the divergent one."""
    return [rig for rank, rig in enumerate(rigs) if rank != DIVERGENT]


def advance_all(world, rigs, candidates, now):
    world.run(lambda r: rigs[r].coord.advance(candidates, now))


def loop_advance_all(world, rigs, req, now):
    """Each rank advances as its engine hooks would: ``req`` is a candidate on the ranks whose
    answer for it is ``DEFER``."""

    def step(rank):
        rig = rigs[rank]
        rig.coord.advance([req] if rig.coord.fetch_answer(req) is DEFER else [], now)

    world.run(step)


def launch_on(rig, req, now):
    plan = rig.coord.fetch_answer(req)
    assert isinstance(plan, FetchPlan)
    rig.plans[req.py_request_id] = plan
    before = len(rig.worker.attempts)
    rig.coord.launch_reserved_fetches([req], now)
    assert len(rig.worker.attempts) == before + 1, "launch did not create a worker attempt"
    return rig.worker.attempts[-1]


def launch_all(world, rigs, req, now=0.0):
    advance_all(world, rigs, [req], now)
    return [launch_on(rig, req, now) for rig in rigs]


def votes_of(rig):
    """The vote section of the last payload this rank sent."""
    return rig.payloads()[-1][0]


# ---- the harness itself ----


def test_lockstep_gathers_every_payload_in_rank_order(world):
    rigs = make_rigs(world)
    req = worker_request()
    seen = {}

    def step(rank):
        rigs[rank].coord.advance([req], 0.0)
        seen[rank] = rigs[rank].payloads()[-1]

    world.run(step)
    assert set(seen) == set(range(world.n))
    assert all(seen[rank][2] == [(1, (END, "worker"))] for rank in seen)
    assert [len(g.calls) for g in world.gathers] == [1] * world.n


def test_lockstep_detects_a_rank_that_never_gathers(world):
    with pytest.raises(AssertionError, match="never gathered"):
        world.run(lambda r: None)


# ---- arrivals ----


def test_every_rank_lands_when_every_rank_serves_everything(world):
    rigs = make_rigs(world)
    req = worker_request()
    attempts = launch_all(world, rigs, req)
    for a in attempts:
        a.deliver_all()
    advance_all(world, rigs, [], 1.0)
    for rig in rigs:
        assert rig.effects.only("unpark") == [(req, END, False, None)]
        assert len(rig.payloads()) == 2


def test_min_b_across_ranks_fails_every_rank_and_replans_to_the_min_b(world):
    rigs = make_rigs(world)
    req = worker_request()
    attempts = launch_all(world, rigs, req)
    for rank, a in enumerate(attempts):
        if rank == DIVERGENT:
            a.deliver_all_but(*rigs[rank].reader.unit_names(req, [6]))
        else:
            a.deliver_all()
    advance_all(world, rigs, [], 1.0)

    for rig in rigs:
        assert rig.effects.count("unpark") == 0
        assert rig.effects.count("give_back_fetch_pages") == 1
        assert rig.worker.count("quiesce") == 1
        assert rig.record(1)["state"] == "PLANNED"
    # The wire shows B = 28 on the full ranks and B = 24 on rank 1; the reduction hides it.
    for rig in peers(rigs):
        assert votes_of(rig) == [(KEY, "TERMINAL", END)]
    assert votes_of(rigs[DIVERGENT]) == [(KEY, "TERMINAL", END - 4)]

    advance_all(world, rigs, [req], 2.0)
    plans = [rig.coord.fetch_answer(req) for rig in rigs]
    assert all(isinstance(p, FetchPlan) for p in plans)
    assert [p.token_end for p in plans] == [END - 4] * world.n


def test_failed_on_one_rank_fails_every_rank(world):
    rigs = make_rigs(world)
    req = worker_request()
    attempts = launch_all(world, rigs, req)
    for rank, a in enumerate(attempts):
        if rank == DIVERGENT:
            a.finish(Failed("nic"))
        else:
            a.deliver_all()
    advance_all(world, rigs, [], 1.0)
    for rig in rigs:
        assert rig.effects.count("unpark") == 0
        assert rig.effects.count("give_back_fetch_pages") == 1
        assert rig.record(1)["state"] == "PLANNED"


def test_arrival_waits_until_every_rank_is_terminal(world):
    rigs = make_rigs(world)
    req = worker_request()
    attempts = launch_all(world, rigs, req)
    for rank, a in enumerate(attempts):
        if rank != DIVERGENT:
            a.deliver_all()
    advance_all(world, rigs, [], 1.0)
    assert votes_of(rigs[DIVERGENT]) == [(KEY, "INFLIGHT", 0)]
    for rig in rigs:
        assert rig.record(1)["state"] == "IN_FLIGHT"
        assert rig.effects.count("unpark") == 0 and rig.effects.count("give_back_fetch_pages") == 0
    attempts[DIVERGENT].deliver_all()
    advance_all(world, rigs, [], 2.0)
    for rig in rigs:
        assert rig.effects.count("unpark") == 1


def test_gen_init_expiry_on_one_rank_fails_every_rank(world):
    rigs = make_rigs(world, fetch_timeout_s=10.0)
    req = FakeRequest(
        7, prompt_len=30, is_disagg_generation_init=True, route_hints={"ctx": {"peer": "c"}}
    )
    advance_all(world, rigs, [req], 0.0)
    for rank, rig in enumerate(rigs):
        # Launched later on rank 1: its own deadline is 15.
        rig.coord.launch_reserved_fetches([req], 5.0 if rank == DIVERGENT else 0.0)
    advance_all(world, rigs, [], 10.0)
    for rig in rigs:
        # Every rank fails the request in this round; the pages stay held while the attempt may
        # still write them, and no attempt is quiesced while live.
        assert rig.effects.only("fail_requests") == [((req,), "kv fetch timed out")]
        assert rig.effects.count("hold_for_transfer") == 1 and rig.worker.count("quiesce") == 0
        assert rig.record(7)["state"] == "IN_FLIGHT" and rig.record(7)["expired"]
    for rig in rigs:
        rig.worker.attempts[-1].deliver_all()
    advance_all(world, rigs, [], 11.0)
    for rig in rigs:
        assert rig.records() == [] and rig.effects.count("terminate_request") == 1
        assert rig.effects.count("unpark") == 0


def test_finished_request_crossing_its_deadline_on_one_rank_fails_nothing(world):
    rigs = make_rigs(world, fetch_timeout_s=10.0)
    req = worker_request()
    advance_all(world, rigs, [req], 0.0)
    for rank, rig in enumerate(rigs):
        rig.coord.launch_reserved_fetches(
            [req], 5.0 if rank == DIVERGENT else 0.0
        )  # deadlines 15 / 10
    attempts = [rig.worker.attempts[-1] for rig in rigs]
    for rig in rigs:
        rig.coord.notify_request_finished(req, 6.0)
    advance_all(world, rigs, [], 10.0)  # the full ranks are past their deadline, rank 1 is not
    for rank, rig in enumerate(rigs):
        assert rig.effects.count("give_back_fetch_pages") == 0
        assert rig.effects.count("fail_requests") == 0
        # A finished fetch still votes; the full ranks report the expiry, and the broadcast
        # makes every rank's record rank-local from here on.
        assert votes_of(rig) == [(KEY, "INFLIGHT", 0)]
        assert rig.payloads()[-1][1] == ([] if rank == DIVERGENT else [KEY])
        assert rig.record(1)["state"] == "IN_FLIGHT" and rig.record(1)["expired"]
    for a in attempts:
        a.deliver_all()
    advance_all(world, rigs, [], 11.0)
    for rig in rigs:
        assert rig.effects.count("unpark") == 0 and rig.effects.count("terminate_request") == 1
        assert rig.records() == []


def test_retry_after_min_b_lands_on_every_rank(world):
    rigs = make_rigs(world)
    req = worker_request()
    attempts = launch_all(world, rigs, req)
    for rank, a in enumerate(attempts):
        missing = [5, 6] if rank == DIVERGENT else [6]
        a.deliver_all_but(*rigs[rank].reader.unit_names(req, missing))
    advance_all(world, rigs, [], 1.0)
    attempts = launch_all(world, rigs, req, now=2.0)
    assert [rig.plans[1].token_end for rig in rigs] == [END - 8] * world.n
    for a in attempts:
        a.deliver_all()
    advance_all(world, rigs, [], 3.0)
    for rig in rigs:
        assert rig.effects.only("unpark") == [(req, END - 8, False, None)]
        assert rig.record(1)["state"] == "DELIVERED" and rig.record(1)["try_index"] == 1


# ---- launch outcomes: UNLAUNCHED, FAILED before launch, the unlaunched clock ----


def test_unlaunched_clock_does_not_start_while_no_rank_has_launched(world):
    """No peer has launched, so the unlaunched clock never starts; the wait for pages is
    bounded by every rank's own wait clock instead, and runs out on all of them alike."""
    rigs = make_rigs(world, landing_wait_timeout_s=30.0)
    req = worker_request()
    advance_all(world, rigs, [req], 0.0)
    advance_all(world, rigs, [], 1.0)  # nobody could reserve pages this round
    advance_all(world, rigs, [], 29.9)
    for rig in rigs:
        assert votes_of(rig) == [(KEY, "UNLAUNCHED", 0)]
        assert rig.record(1)["peer_launch_seen_at"] is None
        assert isinstance(rig.coord.fetch_answer(req), FetchPlan)  # still waiting for pages
        assert rig.effects.count("fail_requests") == 0
    advance_all(world, rigs, [], 30.0)
    for rig in rigs:
        assert votes_of(rig) == [(KEY, "FAILED", 0)]
        rec = rig.record(1)
        assert rec["token_end"] is None and rec["retries_left"] == 0
        assert rig.effects.calls == []  # nothing launched: nothing to give back or fail


def test_unlaunched_rank_holds_the_landing_until_it_launches_too(world):
    rigs = make_rigs(world)
    req = worker_request()
    advance_all(world, rigs, [req], 0.0)
    early = [launch_on(rig, req, 0.0) for rig in peers(rigs)]  # rank 1 got no pages this round
    for a in early:
        a.deliver_all()
    advance_all(world, rigs, [], 1.0)
    for rig in peers(rigs):
        assert votes_of(rig) == [(KEY, "TERMINAL", END)]
        assert rig.effects.count("unpark") == 0 and rig.record(1)["state"] == "IN_FLIGHT"
        assert rig.record(1)["peer_launch_seen_at"] is None
    assert votes_of(rigs[DIVERGENT]) == [(KEY, "UNLAUNCHED", 0)]
    assert rigs[DIVERGENT].record(1)["peer_launch_seen_at"] == 1.0  # from the peers' first vote

    late = launch_on(rigs[DIVERGENT], req, 2.0)
    assert rigs[DIVERGENT].record(1)["peer_launch_seen_at"] is None
    late.deliver_all()
    advance_all(world, rigs, [], 3.0)
    for rig in rigs:
        assert rig.effects.only("unpark") == [(req, END, False, None)]


def test_rank_unlaunched_past_the_timeout_votes_failed_and_every_rank_replans(world):
    rigs = make_rigs(world)
    req = worker_request()
    advance_all(world, rigs, [req], 0.0)
    early = [launch_on(rig, req, 0.0) for rig in peers(rigs)]
    for a in early:
        a.deliver_all()
    advance_all(world, rigs, [], 1.0)  # rank 1's clock starts here
    advance_all(world, rigs, [], 1.0 + UNLAUNCHED_TIMEOUT_S - 0.1)
    for rig in rigs:
        assert rig.effects.count("unpark") == 0 and rig.effects.count("give_back_fetch_pages") == 0

    advance_all(world, rigs, [], 1.0 + UNLAUNCHED_TIMEOUT_S)
    assert votes_of(rigs[DIVERGENT]) == [(KEY, "FAILED", 0)]
    for rig in peers(rigs):
        # Delivered data is dropped with the failure: the release point, then the pages back.
        assert rig.worker.count("quiesce") == 1 and rig.effects.count("give_back_fetch_pages") == 1
        assert rig.effects.count("unpark") == 0
    assert rigs[DIVERGENT].worker.count("quiesce") == 0
    assert rigs[DIVERGENT].effects.count("give_back_fetch_pages") == 0
    for rig in rigs:
        rec = rig.record(1)
        assert rec["token_end"] is None and rec["peer_launch_seen_at"] is None
        assert rec["retries_left"] == 0
        assert rig.coord.fetch_answer(req) is DEFER

    loop_advance_all(world, rigs, req, 12.0)
    plans = [rig.coord.fetch_answer(req) for rig in rigs]
    assert [p.token_end for p in plans] == [END] * world.n


def test_rank_that_gives_up_launching_holds_no_one_and_fails_the_fetch_for_every_rank(world):
    rigs = make_rigs(world)
    req = worker_request()
    advance_all(world, rigs, [req], 0.0)
    running = [launch_on(rig, req, 0.0) for rig in peers(rigs)]
    laggard = rigs[DIVERGENT]
    laggard.worker.reject_next = 3
    for now in (1.0, 2.0, 3.0):
        assert isinstance(laggard.coord.fetch_answer(req), FetchPlan)
        laggard.coord.launch_reserved_fetches([req], now)
        advance_all(world, rigs, [], now)
        for rig in peers(rigs):  # the running attempts are not disturbed
            assert rig.record(1)["state"] == "IN_FLIGHT" and rig.worker.count("quiesce") == 0
    assert laggard.record(1)["gave_up_launching"] and laggard.coord.fetch_answer(req) is DEFER
    assert laggard.effects.count("give_back_fetch_pages") == 3
    assert votes_of(laggard) == [(KEY, "FAILED", 0)]
    laggard.coord.launch_reserved_fetches([req], 4.0)  # the scheduler queue may still name it
    assert laggard.worker.count("fetch") == 3

    for a in running:
        a.deliver_all()
    loop_advance_all(world, rigs, req, 5.0)  # the peers are TERMINAL: the failure lands
    for rig in peers(rigs):
        assert rig.worker.count("quiesce") == 1 and rig.effects.count("give_back_fetch_pages") == 1
    for rig in rigs:
        rec = rig.record(1)
        assert rec["token_end"] is None and not rec["gave_up_launching"]
        assert rec["retries_left"] == 0
        assert rig.coord.fetch_answer(req) is DEFER and rig.effects.count("unpark") == 0

    # The retry runs into the same refusals: the next agreement settles on local compute.
    loop_advance_all(world, rigs, req, 6.0)
    running = [launch_on(rig, req, 6.0) for rig in peers(rigs)]
    laggard.worker.reject_next = 3
    for now in (7.0, 8.0, 9.0):
        laggard.coord.launch_reserved_fetches([req], now)
        advance_all(world, rigs, [], now)
    for a in running:
        a.deliver_all()
    loop_advance_all(world, rigs, req, 10.0)
    for rig in rigs:
        assert rig.coord.fetch_answer(req) is None and rig.records() == []
        assert rig.effects.count("fail_requests") == 0


def test_route_refused_on_one_rank_fails_the_fetch_for_every_rank(world):
    rigs = make_rigs(world)
    req = worker_request()
    advance_all(world, rigs, [req], 0.0)
    running = [launch_on(rig, req, 0.0) for rig in peers(rigs)]
    laggard = rigs[DIVERGENT]
    laggard.worker.open_route_errors.append(ValueError("bad hint"))
    laggard.coord.launch_reserved_fetches([req], 0.0)
    assert laggard.record(1)["gave_up_launching"] and laggard.coord.fetch_answer(req) is DEFER
    loop_advance_all(world, rigs, req, 1.0)
    for rig in peers(rigs):  # in flight: nothing lands yet
        assert rig.record(1)["state"] == "IN_FLIGHT" and rig.worker.count("quiesce") == 0
    for a in running:
        a.deliver_all()
    loop_advance_all(world, rigs, req, 2.0)
    for rig in peers(rigs):
        assert rig.worker.count("quiesce") == 1 and rig.effects.count("give_back_fetch_pages") == 1
    for rig in rigs:
        assert rig.record(1)["retries_left"] == 0 and rig.coord.fetch_answer(req) is DEFER


def test_a_given_up_rank_and_an_unlaunched_rank_agree_without_anyone_launching(world):
    rigs = make_rigs(world)
    req = worker_request()
    for rig in peers(rigs):
        rig.worker.open_route_errors.append(ValueError("bad hint"))
    advance_all(world, rigs, [req], 0.0)
    for rig in peers(rigs):
        rig.coord.launch_reserved_fetches([req], 0.0)
        assert rig.record(1)["gave_up_launching"]
    # Rank 1 never launched: an UNLAUNCHED vote does not hold a failure back.
    loop_advance_all(world, rigs, req, 1.0)
    for rig in peers(rigs):
        assert votes_of(rig) == [(KEY, "FAILED", 0)]
        assert rig.effects.count("give_back_fetch_pages") == 1  # from the refused launch only
    assert votes_of(rigs[DIVERGENT]) == [(KEY, "UNLAUNCHED", 0)]
    assert rigs[DIVERGENT].effects.count("give_back_fetch_pages") == 0
    for rig in rigs:
        assert rig.worker.count("quiesce") == 0
        rec = rig.record(1)
        assert rec["token_end"] is None and not rec["gave_up_launching"]
        assert rec["retries_left"] == 0
        assert rig.coord.fetch_answer(req) is DEFER


def test_expiry_on_a_launched_rank_fails_and_releases_the_unlaunched_rank_at_once(world):
    rigs = make_rigs(world, fetch_timeout_s=10.0)
    req = worker_request()
    advance_all(world, rigs, [req], 0.0)
    running = [launch_on(rig, req, 0.0) for rig in peers(rigs)]  # deadline 10
    advance_all(world, rigs, [], 5.0)  # rank 1's unlaunched clock starts; the deadline is first
    advance_all(world, rigs, [], 10.0)
    for rig in rigs:
        assert rig.effects.only("fail_requests") == [((req,), "kv fetch timed out")]
    # Nothing is in flight on rank 1: its record goes at once, no hold.
    laggard = rigs[DIVERGENT]
    assert laggard.records() == [] and laggard.effects.count("hold_for_transfer") == 0
    for rig in peers(rigs):
        assert rig.record(1)["state"] == "IN_FLIGHT" and rig.record(1)["expired"]
        assert rig.effects.count("hold_for_transfer") == 1 and rig.coord.has_backend_work()

    for a in running:
        a.deliver_all()
    advance_all(world, rigs, [], 11.0)  # the late outcomes settle locally, without rank 1
    for rig in peers(rigs):
        assert rig.records() == [] and not rig.coord.has_backend_work()
        assert rig.effects.count("terminate_request") == 1 and rig.effects.count("unpark") == 0
        assert votes_of(rig) == []


def test_expiry_on_one_rank_after_another_delivered_fails_every_rank_then_each_releases_alone(
    world,
):
    rigs = make_rigs(world, fetch_timeout_s=10.0)
    req = worker_request()
    attempts = launch_all(world, rigs, req)  # deadline 10 everywhere
    for rank, a in enumerate(attempts):
        if rank != DIVERGENT:
            a.deliver_all()
    advance_all(world, rigs, [], 10.0)
    for rig in rigs:
        assert rig.effects.only("fail_requests") == [((req,), "kv fetch timed out")]
        assert rig.effects.count("hold_for_transfer") == 1 and rig.effects.count("unpark") == 0
        assert rig.record(1)["state"] == "IN_FLIGHT"
    advance_all(world, rigs, [], 11.0)
    for rig in peers(rigs):  # delivered: released on their own, rank 1 still in flight
        assert rig.records() == [] and rig.effects.count("terminate_request") == 1
    laggard = rigs[DIVERGENT]
    assert laggard.record(1)["state"] == "IN_FLIGHT"
    assert laggard.effects.count("terminate_request") == 0
    attempts[DIVERGENT].deliver_all()
    advance_all(world, rigs, [], 12.0)
    assert laggard.records() == [] and laggard.effects.count("terminate_request") == 1


# ---- publish through the same reduction ----


def make_publishing_rigs(world):
    """Rank 1 publishes with a piece-placing publisher, the others with a plain one."""
    rigs = []
    for rank in range(world.n):
        publisher = FakePlacingPublishes() if rank == DIVERGENT else FakePublishes()
        rigs.append(Rig(dist=world.gathers[rank], publishers=[publisher]))
    return rigs


def test_publish_rejected_piece_on_one_rank_waits_for_every_running_publish(world):
    rigs = make_publishing_rigs(world)
    req = FakeRequest(3, prompt_len=29)
    laggard = rigs[DIVERGENT]
    laggard.reader.script_publish(req, [(range(0, 7), True, FakeChunk(3, 0))])
    laggard.publishers[0].reject_methods.add("place_piece")
    for rig in rigs:
        rig.coord.publish_committed_blocks([req], now=0.0)
    # Rank 1's ``place`` was refused, but its ``publish`` piece is still running: INFLIGHT, so
    # no verdict quiesces under it.
    advance_all(world, rigs, [], 1.0)
    for rig in rigs:
        assert votes_of(rig) == [((3, "publish"), "INFLIGHT", 0)]
        assert rig.record(3, "publish")["state"] == "IN_FLIGHT"

    laggard.publishers[0].attempts[0].deliver_all()
    advance_all(world, rigs, [], 2.0)
    assert votes_of(laggard) == [((3, "publish"), "FAILED", 0)]
    for rig in peers(rigs):  # their publishes still run: the failure waits for them
        assert votes_of(rig) == [((3, "publish"), "INFLIGHT", 0)]
        assert rig.record(3, "publish")["state"] == "IN_FLIGHT"
    assert laggard.record(3, "publish")["state"] == "IN_FLIGHT"

    for rig in peers(rigs):
        rig.publishers[0].attempts[0].deliver_all()
    advance_all(world, rigs, [], 3.0)
    for rig in rigs:
        assert rig.records() == [] and rig.publishers[0].count("quiesce") == 1
        assert rig.effects.calls == []  # the request is still running: nothing to tell it


def test_publish_rejected_on_one_rank_fails_the_publish_on_every_rank(world):
    """Rank 1's publisher refused the blocks: the store is missing its layer group, so the
    publish has failed for every rank. Its FAILED vote is cast at once and lands as soon as the
    peers' attempts are over, instead of waiting for rank 1's request to end."""
    rigs = make_publishing_rigs(world)
    req = FakeRequest(3, prompt_len=29)
    rigs[DIVERGENT].publishers[0].reject_methods.update({"publish", "place_piece"})
    for rig in rigs:
        rig.coord.publish_committed_blocks([req], now=0.0)
    assert rigs[DIVERGENT].publishers[0].attempts == []
    advance_all(world, rigs, [], 1.0)
    assert votes_of(rigs[DIVERGENT]) == [((3, "publish"), "FAILED", 0)]
    for rig in peers(rigs):  # still running: no verdict quiesces under them
        assert votes_of(rig) == [((3, "publish"), "INFLIGHT", 0)]
        assert rig.record(3, "publish")["state"] == "IN_FLIGHT"
    for rig in peers(rigs):
        rig.publishers[0].attempts[0].deliver_all()
    advance_all(world, rigs, [], 2.0)
    for rig in rigs:
        assert rig.records() == [] and rig.effects.calls == []  # running: a warning only
    assert all(rig.publishers[0].count("quiesce") == 1 for rig in peers(rigs))
    assert rigs[DIVERGENT].publishers[0].count("quiesce") == 0


def test_publish_of_a_finished_request_terminates_on_every_rank_in_the_same_round(world):
    """The publishes end in different rounds; the held request is terminated nowhere until
    every rank's has, then everywhere at once, so the ranks free the pages in step."""
    rigs = make_publishing_rigs(world)
    req = FakeRequest(3, prompt_len=29)
    for rig in rigs:
        rig.coord.publish_committed_blocks([req], now=0.0)
        rig.coord.notify_request_finished(req, 0.0)
        assert rig.effects.names() == ["hold_for_transfer"]
    rigs[0].publishers[0].attempts[0].deliver_all()
    advance_all(world, rigs, [], 1.0)
    assert votes_of(rigs[0]) == [((3, "publish"), "TERMINAL", 0)]
    for rig in rigs[1:]:
        assert votes_of(rig) == [((3, "publish"), "INFLIGHT", 0)]
    for rig in rigs:
        assert rig.record(3, "publish")["state"] == "IN_FLIGHT"
        assert rig.effects.count("terminate_request") == 0
    for rig in rigs[1:]:
        for attempt in rig.publishers[0].attempts:
            attempt.deliver_all()
    advance_all(world, rigs, [], 2.0)
    for rig in rigs:
        assert rig.effects.names() == ["hold_for_transfer", "terminate_request"]
        assert rig.records() == [] and rig.publishers[0].count("quiesce") == 1


def test_publish_verdict_reaches_a_rank_whose_request_has_not_ended_yet():
    """Two ranks whose requests end in different rounds: the agreement releases both publish
    records at once; the rank already finished is terminated here, the other has nothing to
    hold once its request ends and terminates as usual."""
    world = LockstepWorld(2)
    rigs = make_publishing_rigs(world)
    req = FakeRequest(3, prompt_len=29)
    for rig in rigs:
        rig.coord.publish_committed_blocks([req], now=0.0)
        for attempt in rig.publishers[0].attempts:
            attempt.deliver_all()
    rigs[0].coord.notify_request_finished(req, 0.5)
    assert rigs[0].effects.names() == ["hold_for_transfer"]
    advance_all(world, rigs, [], 1.0)
    assert rigs[0].effects.names() == ["hold_for_transfer", "terminate_request"]
    assert rigs[1].effects.names() == []
    for rig in rigs:
        assert rig.records() == [] and rig.publishers[0].count("quiesce") == 1
    assert rigs[1].coord.notify_request_finished(req, 2.0) is True
    assert rigs[1].effects.names() == []


# ---- plan answers ----


def test_one_rank_deferring_defers_every_rank(world):
    rigs = make_rigs(world, sources=("store",))
    req = FakeRequest(1, prompt_len=29)
    for rig in peers(rigs):
        rig.store.probe_default = rig.reader.unit_names(req, range(7))
    rigs[DIVERGENT].store.probe_answers.append(None)  # rank 1's store has not answered yet
    advance_all(world, rigs, [req], 0.0)
    for rig in rigs:
        assert rig.coord.fetch_answer(req) is DEFER
        assert rig.records() == [] and rig.coord.status_dump()["decided_plans"] == 0
    for rig in peers(rigs):
        assert rig.payloads()[0][2] == [(1, (END, "store"))]
    assert rigs[DIVERGENT].payloads()[0][2] == [(1, "DEFER")]

    rigs[DIVERGENT].store.probe_default = rigs[DIVERGENT].reader.unit_names(req, range(7))
    advance_all(world, rigs, [req], 1.0)
    plans = [rig.coord.fetch_answer(req) for rig in rigs]
    assert [p.token_end for p in plans] == [END] * world.n


def test_disagreeing_token_end_becomes_none_on_every_rank(world):
    rigs = make_rigs(world, sources=("store",))
    req = FakeRequest(1, prompt_len=29)
    for rank, rig in enumerate(rigs):
        blocks = 4 if rank == DIVERGENT else 7  # 16 on rank 1, 28 elsewhere
        rig.store.probe_default = rig.reader.unit_names(req, range(blocks))
    advance_all(world, rigs, [req], 0.0)
    for rig in rigs:
        assert rig.coord.fetch_answer(req) is None
        assert rig.records() == []


def test_same_token_end_different_source_becomes_none_on_every_rank(world):
    rigs = make_rigs(world, sources=("store", "worker"))
    req = worker_request()
    for rig in peers(rigs):
        rig.store.probe_default = rig.reader.unit_names(req, range(7))  # store, 28
    rigs[DIVERGENT].store.probe_default = frozenset()  # falls through to the worker, 28
    advance_all(world, rigs, [req], 0.0)
    for rig in peers(rigs):
        assert rig.payloads()[0][2] == [(1, (END, "store"))]
    assert rigs[DIVERGENT].payloads()[0][2] == [(1, (END, "worker"))]
    for rig in rigs:
        assert rig.coord.fetch_answer(req) is None and rig.records() == []


def test_none_versus_plan_is_none(world):
    rigs = make_rigs(world, sources=("store",))
    req = FakeRequest(1, prompt_len=29)
    for rig in peers(rigs):
        rig.store.probe_default = rig.reader.unit_names(req, range(7))
    rigs[DIVERGENT].store.probe_default = frozenset()  # answered: holds nothing
    advance_all(world, rigs, [req], 0.0)
    assert [rig.coord.fetch_answer(req) for rig in rigs] == [None] * world.n


def test_identical_plans_are_kept_and_local_reuse_may_differ(world):
    rigs = make_rigs(world)
    req = worker_request()
    rigs[DIVERGENT].reader.reuse_tokens[1] = 8
    advance_all(world, rigs, [req], 0.0)
    plans = [rig.coord.fetch_answer(req) for rig in rigs]
    assert [p.token_end for p in plans] == [END] * world.n
    for rank, plan in enumerate(plans):
        if rank == DIVERGENT:
            assert ordinals_by_group(plan) == {0: (2, 3, 4, 5, 6)}
        else:
            assert ordinals_by_group(plan) == {0: tuple(range(7))}
    for rig in rigs:
        rig.coord.launch_reserved_fetches([req], 1.0)
    for rig in rigs:
        rig.worker.attempts[-1].deliver_all()
    advance_all(world, rigs, [], 2.0)
    for rig in rigs:
        assert rig.effects.only("unpark") == [(req, END, False, None)]


# ---- host-first: landing and placement through the same reduction ----


def make_host_rigs(world, **kw):
    """Every rank fetches from a ``LandsOnHost`` store that holds the whole prompt."""
    return make_rigs(world, sources=("host",), **kw)


def host_request() -> FakeRequest:
    return FakeRequest(1, prompt_len=29)


def stage_all(world, rigs, req, now=0.0):
    """Every rank decides the plan (starting its landing), lands, and agrees: all ``LANDED``."""
    advance_all(world, rigs, [req], now)
    for rig in rigs:
        rig.host.landings[-1].deliver_all()
    advance_all(world, rigs, [], now + 1.0)
    for rig in rigs:
        assert rig.record(1)["state"] == "LANDED"


def place_on(rig, req, now):
    plan = rig.coord.fetch_answer(req)
    assert isinstance(plan, FetchPlan)
    rig.plans[req.py_request_id] = plan
    before = len(rig.host.attempts)
    rig.coord.launch_reserved_fetches([req], now)
    assert len(rig.host.attempts) == before + 1, "launch did not place the landing"
    return rig.host.attempts[-1]


def test_no_rank_is_staged_until_every_rank_has_landed(world):
    rigs = make_host_rigs(world)
    req = host_request()
    advance_all(world, rigs, [req], 0.0)
    for rig in peers(rigs):
        rig.host.landings[-1].deliver_all()
    advance_all(world, rigs, [], 1.0)
    for rig in peers(rigs):
        assert votes_of(rig) == [(KEY, "TERMINAL", END)]
    assert votes_of(rigs[DIVERGENT]) == [(KEY, "INFLIGHT", 0)]
    for rig in rigs:
        assert rig.record(1)["state"] == "LANDING" and rig.coord.fetch_answer(req) is DEFER
    rigs[DIVERGENT].host.landings[-1].deliver_all()
    advance_all(world, rigs, [], 2.0)
    for rig in rigs:
        assert rig.record(1)["state"] == "LANDED" and isinstance(
            rig.coord.fetch_answer(req), FetchPlan
        )
        assert rig.record(1)["resource_wait_since"] == 2.0


def test_a_placed_rank_waits_for_a_rank_without_pages_and_a_wait_timeout_fails_both_alike(world):
    rigs = make_host_rigs(world, landing_wait_timeout_s=10.0)
    req = host_request()
    stage_all(world, rigs, req)  # LANDED at 1.0: the wait for pages is clocked from there
    placed = [place_on(rig, req, 2.0) for rig in peers(rigs)]  # rank 1 got no pages
    for a in placed:
        a.deliver_all()
    advance_all(world, rigs, [], 3.0)
    for rig in peers(rigs):
        assert votes_of(rig) == [(KEY, "TERMINAL", END)]
        assert rig.effects.count("unpark") == 0 and rig.record(1)["state"] == "IN_FLIGHT"
    assert votes_of(rigs[DIVERGENT]) == [(KEY, "UNLAUNCHED", 0)]
    assert rigs[DIVERGENT].record(1)["state"] == "LANDED"

    advance_all(world, rigs, [], 11.0)  # 10 s since rank 1 was staged
    assert votes_of(rigs[DIVERGENT]) == [(KEY, "FAILED", 0)]
    for rig in peers(rigs):
        assert rig.host.count("quiesce") == 1 and rig.effects.count("give_back_fetch_pages") == 1
        assert rig.effects.count("unpark") == 0
    laggard = rigs[DIVERGENT]
    assert (
        laggard.host.count("quiesce") == 0 and laggard.effects.count("give_back_fetch_pages") == 0
    )
    for rig in rigs:
        assert rig.host.closes() == 1
        rec = rig.record(1)
        assert rec["token_end"] is None and not rec["has_landing"]
        assert rec["retries_left"] == 0
        assert rig.coord.fetch_answer(req) is DEFER and rig.effects.count("fail_requests") == 0


def test_staged_ranks_wait_on_the_landing_clock_not_the_unlaunched_clock():
    """Rank 1 was refused landing memory while rank 0 landed, so its unlaunched clock started;
    once its landing is granted and both are LANDED, that clock is off: two ranks waiting for
    pages alike are bounded by ``landing_wait_timeout_s``, not by ``unlaunched_timeout_s``."""
    world = LockstepWorld(2)
    rigs = make_host_rigs(world, landing_wait_timeout_s=30.0)
    req = host_request()
    rigs[DIVERGENT].host.reject_next = 1
    advance_all(world, rigs, [req], 0.0)
    rigs[0].host.landings[-1].deliver_all()
    loop_advance_all(world, rigs, req, 1.0)  # rank 1's clock starts here, and it is granted
    assert rigs[DIVERGENT].record(1)["state"] == "LANDING"
    assert rigs[DIVERGENT].record(1)["peer_launch_seen_at"] is None
    rigs[DIVERGENT].host.landings[-1].deliver_all()
    advance_all(world, rigs, [], 2.0)
    for rig in rigs:
        assert rig.record(1)["state"] == "LANDED" and rig.record(1)["peer_launch_seen_at"] is None

    advance_all(world, rigs, [], 2.0 + UNLAUNCHED_TIMEOUT_S)  # no pages anywhere
    for rig in rigs:
        assert votes_of(rig) == [(KEY, "UNLAUNCHED", 0)]
        assert (
            rig.record(1)["state"] == "LANDED" and rig.effects.count("give_back_fetch_pages") == 0
        )
    advance_all(world, rigs, [], 2.0 + 30.0)  # 30 s since both were staged
    for rig in rigs:
        assert votes_of(rig) == [(KEY, "FAILED", 0)]
        assert rig.record(1)["token_end"] is None and rig.host.closes() == 1


def test_a_rank_refused_landing_memory_holds_the_landing_of_the_others(world):
    rigs = make_host_rigs(world)
    req = host_request()
    rigs[DIVERGENT].host.reject_next = 1
    advance_all(world, rigs, [req], 0.0)
    assert rigs[DIVERGENT].record(1)["state"] == "PLANNED"
    for rig in peers(rigs):
        rig.host.landings[-1].deliver_all()
    loop_advance_all(world, rigs, req, 1.0)
    for rig in peers(rigs):
        assert votes_of(rig) == [(KEY, "TERMINAL", END)]
        assert rig.record(1)["state"] == "LANDING"  # held by the UNLAUNCHED vote
    assert votes_of(rigs[DIVERGENT]) == [(KEY, "UNLAUNCHED", 0)]
    assert rigs[DIVERGENT].record(1)["state"] == "LANDING"  # asked again, accepted this round
    rigs[DIVERGENT].host.landings[-1].deliver_all()
    advance_all(world, rigs, [], 2.0)
    for rig in rigs:
        assert rig.record(1)["state"] == "LANDED"


def test_expiry_while_placing_on_one_rank_fails_every_rank(world):
    rigs = make_host_rigs(world, fetch_timeout_s=10.0)
    req = host_request()
    stage_all(world, rigs, req)
    place_on(rigs[DIVERGENT], req, 2.0)  # deadline 12; the others never get pages
    advance_all(world, rigs, [], 12.0)
    for rig in rigs:
        assert rig.effects.only("fail_requests") == [((req,), "kv fetch timed out")]
        assert rig.host.closes() == (1 if rig is not rigs[DIVERGENT] else 0)
    for rig in peers(rigs):  # nothing in the pages: released at once, not held
        assert rig.records() == [] and rig.effects.count("hold_for_transfer") == 0
    laggard = rigs[DIVERGENT]
    assert laggard.record(1)["state"] == "IN_FLIGHT" and laggard.record(1)["expired"]
    assert laggard.effects.count("hold_for_transfer") == 1
    laggard.host.attempts[-1].deliver_all()
    advance_all(world, rigs, [], 13.0)
    assert laggard.records() == [] and laggard.host.closes() == 1
    assert laggard.effects.count("terminate_request") == 1 and laggard.effects.count("unpark") == 0


def test_expiry_while_staging_on_one_rank_fails_every_rank(world):
    """The landings started together, so they expire together: no rank can be LANDED while
    another is still LANDING (the landing needs every rank's word), and every rank fails the
    request and releases its landing in the same round, without a hold."""
    rigs = make_host_rigs(world, fetch_timeout_s=10.0)
    req = host_request()
    advance_all(world, rigs, [req], 0.0)
    for rig in peers(rigs):
        rig.host.landings[-1].deliver_all()
    advance_all(world, rigs, [], 10.0)
    for rig in rigs:
        assert rig.effects.names() == ["fail_requests"]
        assert rig.records() == [] and rig.host.closes() == 1
        assert rig.coord.held_request_ids() == frozenset()
    rigs[DIVERGENT].host.landings[-1].deliver_all()  # late, and moot
    advance_all(world, rigs, [], 11.0)
    for rig in rigs:
        assert rig.effects.names() == ["fail_requests"] and rig.host.closes() == 1

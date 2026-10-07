# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The publish record from the loop entry points: a running request's publish owes the engine
nothing; a finished request is held until its publish lands, fails or is refused and is then
terminated exactly once; piece-placing publishers hear every piece and plain ones only the
last; a publish and a fetch of one request release together.

Single rank. Assertions are on the record table (via ``status_dump``), the effects the engine
was asked to perform, and the calls each publisher saw.
"""

import pytest

__extra_import_path__ = ["~/tensorrt_llm/_torch"]
from disaggregation.base.cache_backend import Delivered, Failed  # noqa: E402
from fakes import (  # noqa: E402
    EMPTY_DUMP,
    CoordinatorRig,
    FakeChunk,
    FakePlacingPublishes,
    FakePublishes,
    FakeRequest,
    ScriptedPeersCollective,
    assert_quiesce_precedes,
    worker_request,
)

pytestmark = pytest.mark.cpu_only


def test_publish_flow_holds_then_terminates_after_quiesce():
    pub = FakePublishes(name="store")
    rig = CoordinatorRig(publishers=[pub])
    req = FakeRequest(3, prompt_len=29)
    ext = rig.reader.script_publish(req, [(range(0, 4), False, None), (range(4, 7), True, None)])

    rig.coord.publish_committed_blocks([req], now=0.0)
    # A publisher that does not place pieces is not offered an intermediate piece.
    assert pub.payloads("publish") == []
    rec = rig.record(3, "publish")
    assert rec["state"] == "PLANNED" and rec["attempts"] == 0
    assert rig.coord.has_backend_work() is False
    rig.coord.advance([], 1.0)
    assert rig.record(3, "publish")["state"] == "PLANNED" and pub.count("quiesce") == 0

    rig.coord.publish_committed_blocks([req], now=2.0)
    rig.coord.holds_finished_request(req)
    assert pub.payloads("publish") == [ext[1]]
    assert rig.record(3, "publish")["state"] == "IN_FLIGHT" and rig.coord.has_backend_work()
    assert rig.effects.names() == ["hold_for_transfer"]
    assert rig.effects.args_of("hold_for_transfer") == [((req,),)]
    assert rig.coord.status_dump()["finished_pending"] == [3]

    pub.attempts[0].deliver_all()
    rig.coord.advance([], 3.0)
    assert pub.count("quiesce") == 1
    quiesced = [args[0] for m, args in pub.calls if m == "quiesce"][0]
    assert set(quiesced) == set(pub.attempts)
    assert rig.effects.names() == ["hold_for_transfer", "terminate_request"]
    assert rig.effects.args_of("terminate_request") == [(req,)]
    assert rig.records() == [] and rig.coord.status_dump()["finished_pending"] == []
    assert_quiesce_precedes(rig.trace, "terminate_request")


def test_publish_landed_before_request_end_terminates_immediately_at_end():
    pub = FakePublishes()
    rig = CoordinatorRig(publishers=[pub])
    req = FakeRequest(3, prompt_len=29)
    rig.coord.publish_committed_blocks([req], now=0.0)  # default: everything, is_last
    pub.attempts[0].deliver_all()
    rig.coord.advance([], 1.0)
    assert rig.records() == [] and pub.count("quiesce") == 1
    assert rig.effects.calls == []  # request still running: nothing to tell the engine
    rig.coord.holds_finished_request(req)
    assert rig.effects.calls == []  # release already happened; the engine terminates as usual
    assert rig.coord.status_dump()["finished_pending"] == []


def test_publish_failure_after_request_end_terminates_the_held_request():
    """The request already answered its client; a publish that then fails is the store's loss,
    not the request's: held while in flight, then terminated, never failed."""
    pub = FakePublishes()
    rig = CoordinatorRig(publishers=[pub])
    req = FakeRequest(3, prompt_len=29)
    rig.coord.publish_committed_blocks([req], now=0.0)
    rig.coord.holds_finished_request(req)
    pub.attempts[0].finish(Failed("peer gone"))
    rig.coord.advance([], 1.0)
    assert rig.effects.names() == ["hold_for_transfer", "terminate_request"]
    assert rig.effects.count("fail_requests") == 0
    assert rig.records() == []


def test_publish_expiry_warns_and_waits_for_the_outcome():
    """A publish past its deadline is never quiesced under its live attempt: the record is
    marked expired, and the outcome settles it on this rank alone."""
    pub = FakePublishes()
    rig = CoordinatorRig(publishers=[pub], publish_timeout_s=5.0)
    req = FakeRequest(3, prompt_len=29)
    rig.coord.publish_committed_blocks([req], now=0.0)
    rig.coord.holds_finished_request(req, 0.0)
    rig.coord.advance([], 5.0)
    assert rig.effects.names() == ["hold_for_transfer"]
    assert pub.count("quiesce") == 0
    rec = rig.record(3, "publish")
    assert rec["state"] == "IN_FLIGHT" and rec["expired"]
    rig.coord.advance([], 6.0)
    assert rig.payloads()[-1] == ([], [], [], True, False)  # expired: no vote, but work here
    pub.attempts[0].deliver_all()
    rig.coord.advance([], 7.0)
    assert rig.effects.names() == ["hold_for_transfer", "terminate_request"]
    assert rig.effects.count("fail_requests") == 0
    assert pub.count("quiesce") == 1 and rig.records() == []


def test_publish_deadline_counts_from_the_first_accepted_submission():
    pub = FakePublishes()
    rig = CoordinatorRig(publishers=[pub], publish_timeout_s=5.0)
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


def test_pipelined_publish_deadline_counts_from_the_first_piece_not_the_last():
    """A publisher that places pieces hears every piece: its first accepted piece starts the
    clock, and a later piece does not restart it."""
    placing = FakePlacingPublishes()
    rig = CoordinatorRig(publishers=[placing], publish_timeout_s=5.0)
    req = FakeRequest(3, prompt_len=29)
    rig.reader.script_publish(
        req, [(range(0, 4), False, FakeChunk(3, 0)), (range(4, 7), True, FakeChunk(3, 1))]
    )
    rig.coord.publish_committed_blocks([req], now=0.0)
    assert rig.record(3, "publish")["deadline"] == 5.0
    rig.coord.publish_committed_blocks([req], now=3.0)
    assert rig.record(3, "publish")["deadline"] == 5.0 and placing.count("publish") == 2
    rig.coord.advance([], 5.0)
    assert rig.record(3, "publish")["expired"] is True


def test_publish_expiry_does_not_quiesce_a_live_attempt():
    """A running request's publish past its deadline: the attempt may still be reading the
    pages, so it is never quiesced while live. The record stays in flight and keeps the pages
    protected (``inflight_request_ids``), the documented trade for never blocking the engine
    thread; the outcome then settles it, with the one quiesce at the release point."""
    pub = FakePublishes()
    rig = CoordinatorRig(publishers=[pub], publish_timeout_s=5.0)
    req = FakeRequest(3, prompt_len=29)
    rig.coord.publish_committed_blocks([req], now=0.0)
    rig.coord.advance([], 5.0)
    rec = rig.record(3, "publish")
    assert rec["state"] == "IN_FLIGHT" and rec["expired"]
    assert pub.count("quiesce") == 0
    assert rig.coord.inflight_request_ids() == {3} and rig.coord.has_backend_work()
    rig.coord.advance([], 6.0)  # still live, still not quiesced, still protected
    assert pub.count("quiesce") == 0 and rig.coord.inflight_request_ids() == {3}
    pub.attempts[0].deliver_all()
    rig.coord.advance([], 7.0)
    assert pub.count("quiesce") == 1 and rig.records() == []
    assert rig.coord.inflight_request_ids() == frozenset()
    assert rig.effects.calls == []  # a running request's publish owes the engine nothing


def test_pipelined_publish_of_finished_request_without_last_piece_is_bounded():
    """A piece-placing publish whose request ended before its last piece was offered: the
    pieces so far settle it, but the agreement needs every rank, and a peer that never votes
    on the record would hold the request forever. The wait is bounded by the publish deadline:
    past it the record is rank-local and the held request is terminated on this rank's word."""
    placing = FakePlacingPublishes()
    rig = CoordinatorRig(
        publishers=[placing],
        publish_timeout_s=5.0,
        dist=ScriptedPeersCollective(
            lambda local: ([], [], local[2], False, False)
        ),  # a peer with no vote on it
    )
    req = FakeRequest(3, prompt_len=29)
    rig.reader.script_publish(
        req, [(range(0, 3), False, FakeChunk(3, 0)), (range(3, 7), True, FakeChunk(3, 1))]
    )
    rig.coord.publish_committed_blocks([req], now=0.0)  # piece 0 only: deadline 5.0
    for attempt in placing.attempts:
        attempt.finish(Delivered(frozenset()))
    assert rig.coord.holds_finished_request(req, 1.0) is True
    assert rig.effects.names() == ["hold_for_transfer"]
    rig.coord.advance([], 4.0)
    assert rig.last_votes == [((3, "publish"), "TERMINAL", 0)]  # no more pieces will come
    assert rig.coord.held_request_ids() == {3} and placing.count("quiesce") == 0
    rig.coord.advance([], 5.0)  # the deadline: expired, rank-local from here
    assert rig.record(3, "publish")["expired"] and rig.effects.count("terminate_request") == 0
    rig.coord.advance([], 6.0)
    assert placing.count("quiesce") == 1
    assert rig.effects.names() == ["hold_for_transfer", "terminate_request"]
    assert rig.effects.count("fail_requests") == 0
    assert rig.records() == [] and rig.coord.status_dump()["finished_pending"] == []


def test_publish_rejected_outright_votes_failed_and_terminates_with_the_agreement():
    """Nothing escaped here, but a peer's publish may be in flight and the store is missing
    this rank's part: the record stays to vote FAILED, and the request is terminated with the
    agreement, in the same round everywhere."""
    pub = FakePublishes()
    rig = CoordinatorRig(publishers=[pub])
    req = FakeRequest(3, prompt_len=29)
    pub.reject_next_calls = 1
    rig.coord.publish_committed_blocks([req], now=0.0)
    assert rig.coord.holds_finished_request(req, 0.0) is True
    assert pub.attempts == [] and pub.count("quiesce") == 0  # nothing escaped, nothing to quiesce
    assert rig.effects.names() == ["hold_for_transfer"]
    assert rig.record(3, "publish")["state"] == "PLANNED"
    rig.coord.advance([], 1.0)
    assert rig.last_votes == [((3, "publish"), "FAILED", 0)]
    assert rig.effects.names() == ["hold_for_transfer", "terminate_request"]
    assert pub.count("quiesce") == 0
    assert rig.records() == [] and rig.coord.status_dump()["finished_pending"] == []


def test_publish_rejected_while_the_request_runs_votes_failed_and_is_released():
    """A running request's refused publish does not wait for its end: it votes FAILED, so a
    peer's in-flight publish is failed too instead of waiting on this rank's silence."""
    pub = FakePublishes()
    rig = CoordinatorRig(publishers=[pub])
    req = FakeRequest(3, prompt_len=29)
    pub.reject_next_calls = 1
    rig.coord.publish_committed_blocks([req], now=0.0)
    rig.coord.advance([], 1.0)
    assert rig.last_votes == [((3, "publish"), "FAILED", 0)]
    assert rig.records() == [] and rig.effects.calls == []  # a warning is the only word
    assert rig.coord.holds_finished_request(req, 2.0) is False


def test_publish_never_submitted_owes_the_finished_request_nothing():
    """A plain publisher hears only the last piece; a request cancelled before it has nothing
    in flight on any rank, so the record goes at once and the engine terminates as usual."""
    pub = FakePublishes()
    rig = CoordinatorRig(publishers=[pub])
    req = FakeRequest(3, prompt_len=29)
    rig.reader.script_publish(req, [(range(0, 4), False, None), (range(4, 7), True, None)])
    rig.coord.publish_committed_blocks([req], now=0.0)
    assert rig.coord.holds_finished_request(req, 1.0) is False
    assert rig.effects.calls == [] and pub.calls == []
    assert rig.records() == [] and rig.coord.status_dump()["finished_pending"] == []


def test_quiesce_false_publish_record_is_not_released_without_quiesce():
    """A publish whose backend refused to vouch for the pages keeps its record and its pages:
    the request's end does not release it, and nothing is agreed on it again."""
    pub = FakePublishes()
    rig = CoordinatorRig(publishers=[pub])
    req = FakeRequest(3, prompt_len=29)
    rig.coord.publish_committed_blocks([req], now=0.0)
    pub.attempts[0].deliver_all()
    pub.quiesce_answers.append(False)
    rig.coord.advance([], 1.0)
    assert rig.effects.args_of("fail_fatal") and rig.record(3, "publish")["state"] == "DELIVERED"
    assert rig.coord.inflight_request_ids() == {3}
    assert rig.coord.holds_finished_request(req, 2.0) is True
    assert rig.effects.names() == ["fail_fatal", "hold_for_transfer"]
    rig.coord.advance([], 3.0)
    assert pub.count("quiesce") == 1 and rig.record(3, "publish")["state"] == "DELIVERED"
    assert rig.effects.count("terminate_request") == 0


def test_partial_publish_rejection_terminates_the_held_request_at_next_reap():
    # Publish accepted, place rejected: the pieces that were offered may land, the publish as a
    # whole has still failed.
    placing = FakePlacingPublishes()
    rig = CoordinatorRig(publishers=[placing])
    req = FakeRequest(3, prompt_len=29)
    rig.reader.script_publish(req, [(range(0, 7), True, FakeChunk(3, 0))])
    placing.reject_methods.add("place_piece")
    rig.coord.publish_committed_blocks([req], now=0.0)
    rig.coord.holds_finished_request(req)
    assert placing.count("publish") == 1 and placing.count("place_piece") == 1
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
    rig = CoordinatorRig(publishers=[ok, bad])
    req = FakeRequest(3, prompt_len=29)
    bad.reject_next_calls = 1
    rig.coord.publish_committed_blocks([req], now=0.0)
    rig.coord.holds_finished_request(req)
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
    rig = CoordinatorRig(publishers=[placing])
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
    rig = CoordinatorRig(publishers=[placing], publish_timeout_s=5.0)
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
    rig.coord.holds_finished_request(req, 6.0)
    assert rig.effects.names() == ["hold_for_transfer"]
    rig.coord.advance([], 7.0)
    assert placing.count("quiesce") == 1 and rig.records() == []
    assert rig.effects.names() == ["hold_for_transfer", "terminate_request"]


def test_publish_rejected_while_fetch_in_flight_terminates_once_the_fetch_releases():
    pub = FakePublishes()
    rig = CoordinatorRig(publishers=[pub])
    req = worker_request()
    attempt = rig.plan_and_launch(req)
    pub.reject_next_calls = 1
    rig.coord.publish_committed_blocks([req], now=1.0)
    rig.coord.holds_finished_request(req, 1.0)
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
    assert rig.coord.status_dump() == EMPTY_DUMP


def test_publish_landed_first_waits_for_the_in_flight_fetch_before_terminating():
    pub = FakePublishes()
    rig = CoordinatorRig(publishers=[pub])
    req = worker_request()
    attempt = rig.plan_and_launch(req)
    rig.coord.publish_committed_blocks([req], now=1.0)
    rig.coord.holds_finished_request(req)
    assert rig.effects.names()[-1:] == ["hold_for_transfer"]
    pub.attempts[0].deliver_all()
    rig.coord.advance([], 2.0)
    assert pub.count("quiesce") == 1 and rig.record(1, "publish") is None
    assert rig.effects.count("terminate_request") == 0  # the fetch is still in flight
    attempt.deliver_all()
    rig.coord.advance([], 3.0)
    assert rig.effects.names()[-1:] == ["terminate_request"]
    assert rig.effects.count("unpark") == 0
    assert rig.coord.status_dump() == EMPTY_DUMP


def test_both_records_in_flight_fetch_releases_first_then_publish_lands():
    pub = FakePublishes()
    rig = CoordinatorRig(publishers=[pub])
    req = worker_request()
    attempt = rig.plan_and_launch(req)
    rig.coord.publish_committed_blocks([req], now=1.0)
    rig.coord.holds_finished_request(req)
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
    assert rig.coord.status_dump() == EMPTY_DUMP


def test_no_publishers_means_no_publish_records():
    rig = CoordinatorRig()
    req = FakeRequest(3, prompt_len=29)
    rig.coord.publish_committed_blocks([req], now=0.0)
    rig.coord.holds_finished_request(req)
    assert rig.records() == [] and rig.effects.calls == []
    assert rig.reader.calls == []  # publish_extent_and_chunk is not even asked


def test_places_pieces_gets_every_chunk_plain_publisher_only_the_last():
    placing, plain = FakePlacingPublishes(name="worker"), FakePublishes(name="store")
    rig = CoordinatorRig(publishers=[placing, plain])
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

    assert placing.payloads("place_piece") == chunks
    assert placing.payloads("publish") == ext
    # Design §7.5: a publisher that does not place pieces hears once, on the last piece, and
    # must not need to read ``is_last``.
    assert plain.payloads("publish") == [ext[-1]]


def test_places_pieces_not_called_without_a_chunk():
    placing = FakePlacingPublishes()
    rig = CoordinatorRig(publishers=[placing])
    req = FakeRequest(3, prompt_len=29)
    rig.coord.publish_committed_blocks([req], now=0.0)
    assert placing.count("publish") == 1 and placing.count("place_piece") == 0

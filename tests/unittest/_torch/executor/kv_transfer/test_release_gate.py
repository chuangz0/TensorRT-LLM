# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The release gate (``holds_finished_request`` behind ``_terminate_request``) and every
termination, cancel and expiry path through it: cases A-E of the design (E synthesized: the
disagg send never comes back), the cancel path, a held request whose publish fails or expires,
an expired fetch, the engine's fatal error path with a parked request, a termination the engine
defers under attention DP, cancellation while parked, and a real ``_handle_responses`` pass.
"""

import json
from unittest.mock import Mock

import pytest
from engine_fakes import CONTEXT_INIT, EngineRig, FakeKVCache, finish_prefill, make_request

from tensorrt_llm._torch.disaggregation.base.cache_backend import Failed
from tensorrt_llm._torch.pyexecutor.kv_transfer.effects import (
    KV_FETCH_IN_PROGRESS,
    KV_HELD_FOR_TRANSFER,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState

pytestmark = pytest.mark.cpu_only


# =============================================================================================
# The release gate: cases A-E
# =============================================================================================


class TestReleaseGate:
    def test_case_a_publish_landed_engine_terminates_now(self, rig):
        req = make_request(1, 100)
        attempt = rig.publish(req)
        attempt.deliver_all()
        rig.advance()  # the publish record is reaped and released
        assert rig.records() == []
        assert rig.terminations() == 0

        rig.executor._terminate_request(req)

        rig.executor._do_terminate_request.assert_called_once_with(req)
        assert req.state != KV_HELD_FOR_TRANSFER
        assert not rig.hooks.owns(req)
        assert rig.kv.count("free_resources") == 0  # _do_terminate_request owns that
        assert rig.coord.status_dump()["finished_pending"] == []

    def test_case_b_publish_in_flight_layer_holds_then_terminates(self, rig):
        req = make_request(1, 100)
        attempt = rig.publish(req)

        rig.executor._terminate_request(req)  # _handle_responses' release point

        assert rig.terminations() == 0
        assert req.state == KV_HELD_FOR_TRANSFER
        assert rig.hooks.owns(req) and rig.hooks.has_pending_work()
        rig.advance()  # still in flight
        assert rig.terminations() == 0
        attempt.deliver_all()
        rig.advance()
        rig.executor._do_terminate_request.assert_called_once_with(req)
        assert not rig.hooks.owns(req) and not rig.coord.has_pending_work()
        assert rig.records() == [] and rig.coord.status_dump()["finished_pending"] == []
        assert rig.hooks.has_pending_work()  # that round's gathered word still saw the record
        rig.advance()
        assert not rig.hooks.has_pending_work()

    def test_case_b_publish_failure_still_terminates_the_finished_request_once(self, rig, caplog):
        """The request already answered its client; a store that then fails to take its blocks
        is a warning, not a request failure. The layer terminates it exactly once."""
        req = make_request(1, 100)
        attempt = rig.publish(req)
        rig.executor._terminate_request(req)
        attempt.finish(Failed("store down"))
        with caplog.at_level("WARNING"):
            rig.advance()
        rig.executor._handle_errors.assert_not_called()
        rig.executor._do_terminate_request.assert_called_once_with(req)
        assert any("kv publish failed" in r.getMessage() for r in caplog.records)
        assert not rig.hooks.owns(req)
        assert rig.records() == [] and rig.coord.status_dump()["finished_pending"] == []

    def test_case_c_ctx_only_send_done_first_publish_in_flight(self, rig):
        """The disagg ``release_transfer`` reaches ``_terminate_request`` first, with the seq and
        index slots already released by the send; the gate holds until the publish lands."""
        req = make_request(1, 100)
        attempt = rig.publish(req)
        req.state = KV_HELD_FOR_TRANSFER  # disagg start_transfer wrote 21 already
        rig.slots.free_resources(req)
        rig.kv.release_index_slot(1)

        rig.executor._terminate_request(req)  # from release_transfer -> effects.terminate_request

        assert rig.terminations() == 0
        assert rig.coord.held_request_ids() == {1}
        assert rig.kv.count("release_index_slot") == 2  # the wrapper makes the second a no-op
        attempt.deliver_all()
        rig.advance()
        rig.executor._do_terminate_request.assert_called_once_with(req)
        assert rig.records() == [] and not rig.hooks.owns(req)

    def test_case_d_publish_landed_first_send_in_flight(self, rig):
        """The publish landed while the request was still running: the record is released and
        nothing is owed. The disagg send finishing later terminates through the gate: True."""
        req = make_request(1, 100)
        attempt = rig.publish(req)
        attempt.deliver_all()
        rig.advance()
        assert rig.records() == []
        assert rig.terminations() == 0  # _conclude_publish did nothing for an unfinished request
        req.state = KV_HELD_FOR_TRANSFER  # the disagg send still holds it

        rig.executor._terminate_request(req)  # release_transfer, send complete

        rig.executor._do_terminate_request.assert_called_once_with(req)
        assert not rig.hooks.owns(req)
        assert rig.slots.freed == []  # the gate did not hold, so no slot work of ours
        assert rig.coord.status_dump()["finished_pending"] == []

    @pytest.mark.parametrize("publish_landed_first", [True, False])
    def test_case_e_force_terminate_for_partial_reuse(self, rig, publish_landed_first):
        """``_handle_responses`` terminates on prefill completion while the disagg send is still
        in flight, and ``release_transfer`` never calls ``terminate_request`` in this mode. The gate
        answers from its own table only: exactly one termination, by the engine or by this layer,
        and no request left in any table."""
        req = make_request(1, 100)
        attempt = rig.publish(req)
        if publish_landed_first:
            attempt.deliver_all()
            rig.advance()

        rig.executor._terminate_request(req)  # the send is still in flight; nobody else will come

        if publish_landed_first:
            rig.executor._do_terminate_request.assert_called_once_with(req)
        else:
            assert rig.terminations() == 0 and req.state == KV_HELD_FOR_TRANSFER
            attempt.deliver_all()
            rig.advance()
            rig.executor._do_terminate_request.assert_called_once_with(req)
        # Nothing waits on a disagg release that is never coming.
        for _ in range(3):
            rig.advance()
        assert rig.terminations() == 1
        assert not rig.hooks.owns(req) and not rig.hooks.has_pending_work()
        assert rig.records() == []
        assert rig.coord.owned_requests() == []
        assert rig.coord.status_dump()["finished_pending"] == []

    def test_publish_rejected_outright_is_held_one_round_then_terminated_exactly_once(self, rig):
        """Every publisher refused the request's blocks here, but a peer's may have taken them:
        the record votes once so the ranks agree, and the layer terminates the request with the
        agreement, exactly once."""
        req = make_request(1, 100)
        finish_prefill(req)
        rig.kv.kv_cache_map[1] = FakeKVCache(history_length=100)
        rig.publisher.reject_next_calls = 1
        rig.hooks.publish_committed_blocks([req])
        assert rig.publisher.attempts == []

        rig.executor._terminate_request(req)
        assert rig.terminations() == 0 and rig.coord.held_request_ids() == {1}

        rig.advance()
        rig.executor._do_terminate_request.assert_called_once_with(req)
        assert rig.records() == [] and not rig.hooks.owns(req)
        assert rig.coord.status_dump()["finished_pending"] == []
        for _ in range(2):
            rig.advance()
        assert rig.terminations() == 1

    def test_gate_is_asked_once_per_request_even_if_terminate_is_reentered(self, rig):
        req = make_request(1, 100)
        attempt = rig.publish(req)
        rig.executor._terminate_request(req)
        rig.executor._terminate_request(req)  # a second release point (e.g. disagg after engine)
        assert rig.terminations() == 0 and rig.coord.held_request_ids() == {1}
        attempt.deliver_all()
        rig.advance()
        assert rig.terminations() == 1

    def test_request_finished_while_its_fetch_is_in_flight_is_held_then_terminated(self, rig):
        """A cancelled parked request: the fetch is abandoned, the request held until the
        outcome arrives, then terminated by this layer with no unpark and no give-back."""
        req = make_request(1, 100)
        plan, attempt = rig.plan_reserve_launch(req)
        rig.executor._terminate_request(req)
        assert rig.terminations() == 0
        assert req.state == KV_HELD_FOR_TRANSFER  # held
        assert rig.coord.parked_request_ids() == frozenset()
        assert rig.coord.held_request_ids() == {1}
        attempt.deliver_all()
        rig.advance()
        rig.executor._do_terminate_request.assert_called_once_with(req)
        rig.executor._revert_ctx_alloc.assert_not_called()
        assert req.context_current_position == 0  # never unparked
        assert rig.records() == [] and not rig.hooks.owns(req)

    def test_request_finished_with_a_plan_but_no_launch_is_terminated_with_the_agreement(self, rig):
        """No pages here, but a peer may have launched the same plan: the record votes once so
        every rank terminates the request in the same round."""
        req = make_request(1, 100)
        rig.plan(req)
        assert rig.records()[0]["state"] == "PLANNED"
        rig.executor._terminate_request(req)
        assert rig.terminations() == 0 and rig.coord.held_request_ids() == {1}
        rig.advance()
        rig.executor._do_terminate_request.assert_called_once_with(req)
        assert rig.records() == [] and rig.store.count("quiesce") == 0

    def test_dummy_requests_bypass_the_gate(self, rig):
        req = make_request(1, 100)
        req.is_dummy_request = True
        finish_prefill(req)
        rig.kv.kv_cache_map[1] = FakeKVCache(history_length=100)
        rig.hooks.publish_committed_blocks([req])  # never offered: dummies are not published ...
        assert rig.publisher.count("publish") == 0
        notify = Mock(wraps=rig.coord.holds_finished_request)
        rig.coord.holds_finished_request = notify
        rig.executor._terminate_request(req)  # ... and the gate is not even asked
        notify.assert_not_called()
        rig.executor._do_terminate_request.assert_called_once_with(req)
        assert rig.reader.forgotten == []

    def test_gate_does_not_hold_a_request_this_layer_never_saw(self, rig):
        req = make_request(9, 100)
        assert rig.hooks.holds_finished_request(req) is False
        assert rig.reader.forgotten == [9]
        assert rig.coord.status_dump()["finished_pending"] == []


# =============================================================================================
# The cancel path
# =============================================================================================


class TestCancel:
    """``_try_cancel_request`` asks ``owns`` first: a parked request is the layer's until it
    lands, a held one until it is terminated; a running request with a publish in flight
    stays cancellable."""

    def test_parked_request_cannot_be_cancelled_until_it_lands(self, rig):
        req = make_request(1, 100)
        plan, attempt = rig.plan_reserve_launch(req)
        assert rig.executor._try_cancel_request(req) is False
        attempt.deliver_all()
        rig.advance(req)
        assert rig.executor._try_cancel_request(req) is True

    def test_held_request_cannot_be_cancelled(self, rig):
        req = make_request(1, 100)
        rig.publish(req)
        rig.executor._terminate_request(req)
        assert rig.executor._try_cancel_request(req) is False

    def test_owns_reads_the_record_table_not_the_state(self, rig):
        req = make_request(1, 100)
        req.state = KV_FETCH_IN_PROGRESS  # a disagg gen-init in transmission looks like this
        assert rig.hooks.owns(req) is False
        assert rig.executor._try_cancel_request(req) is True  # no transceiver: cancellable
        dummy = make_request(2, 100)
        dummy.is_dummy_request = True
        assert rig.hooks.owns(dummy) is False

    def test_running_request_with_a_publish_in_flight_is_not_owned_but_is_in_flight(self, rig):
        """A publish does not own a running request: it stays cancellable (``owns`` is
        False) while its pages stay protected (``inflight_request_ids`` names it)."""
        req = make_request(1, 100)
        rig.publish(req)  # still running: not finished, publish IN_FLIGHT
        assert rig.hooks.owns(req) is False
        assert rig.hooks.inflight_request_ids() == {1}
        assert rig.executor._try_cancel_request(req) is True


# =============================================================================================
# Through the engine's own error, cancel and response paths
# =============================================================================================


class TestHeldRequestFailurePaths:
    """A held request whose publish fails or expires has already answered its client: the
    store's trouble is logged as a WARNING and the layer terminates the request exactly once,
    without the engine's error path."""

    def test_publish_failure_on_a_held_request_warns_and_terminates_once(self, rig, caplog):
        rig.wire_engine_error_path()
        req = make_request(1, 100)
        attempt = rig.publish(req)
        rig.executor._terminate_request(req)
        assert rig.coord.held_request_ids() == {1}

        attempt.finish(Failed("store down"))
        with caplog.at_level("WARNING"):
            rig.advance()

        rig.executor._handle_errors.assert_not_called()
        rig.executor._do_terminate_request.assert_called_once_with(req)
        assert any("kv publish failed" in r.getMessage() for r in caplog.records)
        assert req.state != LlmRequestState.GENERATION_COMPLETE  # not failed
        assert rig.coord.held_request_ids() == frozenset()
        assert not rig.hooks.owns(req)
        assert rig.records() == [] and rig.coord.status_dump()["finished_pending"] == []
        for _ in range(2):  # nothing left that could terminate it again
            rig.advance()
        assert rig.terminations() == 1

    def test_publish_expiry_on_a_held_request_warns_and_terminates_once_on_the_outcome(
        self, clock, caplog
    ):
        """Past the deadline the attempt may still be writing the store; it is never quiesced
        while live. The warning is given, the hold stays, and the outcome settles the record."""
        rig = EngineRig(publish_timeout_s=10.0)
        rig.wire_engine_error_path()
        req = make_request(1, 100)
        attempt = rig.publish(req)  # deadline: 1010
        rig.executor._terminate_request(req)
        assert rig.coord.held_request_ids() == {1}
        clock["t"] += 9.0
        rig.advance()
        assert rig.terminations() == 0 and rig.coord.held_request_ids() == {1}

        clock["t"] += 1.5  # past the deadline, no outcome
        with caplog.at_level("WARNING"):
            rig.advance()

        rig.executor._handle_errors.assert_not_called()
        assert any("kv publish past its deadline" in r.getMessage() for r in caplog.records)
        assert rig.terminations() == 0 and rig.publisher.count("quiesce") == 0
        assert rig.coord.held_request_ids() == {1}
        assert [r["expired"] for r in rig.records()] == [True]

        attempt.deliver_all()
        rig.advance()
        rig.executor._do_terminate_request.assert_called_once_with(req)
        assert rig.coord.held_request_ids() == frozenset()
        assert rig.records() == [] and rig.coord.status_dump()["finished_pending"] == []
        assert rig.publisher.count("quiesce") == 1  # the pages were vouched for first


class TestFetchExpiry:
    """A normal fetch past ``fetch_timeout_s``: the request fails through the engine's error
    path, but its pages stay held until the backend's outcome arrives (or ``close`` frees them);
    a Delivered that arrives after the expiry no longer lands anything."""

    def _expire_parked_fetch(self, rig, clock):
        rig.wire_engine_error_path()
        req = make_request(1, 100)
        rig.executor.active_requests = [req]
        plan, attempt = rig.plan_reserve_launch(req)  # deadline: now + 10
        clock["t"] += 10.5
        rig.advance(req)
        rig.executor._handle_errors.assert_called_once()
        assert rig.executor._handle_errors.call_args.args[0] == "kv fetch timed out"
        assert rig.executor._handle_errors.call_args.kwargs["requests"] == [req]
        # The engine error path reached the gate, which held the request: its pages may still
        # be written by the backend.
        assert rig.terminations() == 0
        assert rig.coord.held_request_ids() == {1}
        assert req.state == KV_HELD_FOR_TRANSFER
        assert 1 in rig.kv.kv_cache_map
        rig.executor._revert_ctx_alloc.assert_not_called()
        return req, attempt

    def test_expired_fetch_fails_the_request_and_holds_its_pages(self, clock):
        rig = EngineRig(fetch_timeout_s=10.0)
        req, attempt = self._expire_parked_fetch(rig, clock)
        attempt.deliver_all()  # late: the request is gone
        rig.advance()
        rig.executor._do_terminate_request.assert_called_once_with(req)
        assert req.context_current_position == 0  # never unparked
        assert rig.kv.count("try_commit_blocks") == 0
        assert rig.records() == [] and not rig.hooks.owns(req)

    def test_expired_fetch_without_an_outcome_is_freed_by_close(self, clock, tmp_path):
        rig = EngineRig(fetch_timeout_s=10.0, status_dump_path=str(tmp_path / "kvt.json"))
        req, _ = self._expire_parked_fetch(rig, clock)
        rig.hooks.close()
        rig.executor._free_request_resources.assert_called_once_with(req)
        assert rig.terminations() == 0


class TestEngineErrorPathWithParkedRequests:
    """Scenario: ``_handle_errors(requests=None)`` (fatal) while a fetch is parked. The engine
    terminates every active request through the gate; a parked request's fetch is abandoned and
    the request held until its outcome arrives -- or freed by ``close`` if none ever does."""

    def _park_then_fail_fatally(self, rig):
        rig.wire_engine_error_path()
        req = make_request(1, 100)
        rig.executor.active_requests = [req]
        plan, attempt = rig.plan_reserve_launch(req)
        rig.effects.fail_fatal(RuntimeError("cuda died"))
        rig.executor._handle_errors.assert_called_once()
        assert rig.executor._handle_errors.call_args.kwargs["requests"] is None
        assert rig.executor.active_requests == []
        # The gate held it: fetch in flight, abandoned; terminated when the outcome arrives.
        assert rig.terminations() == 0
        assert rig.held_and_parked() == ({1}, frozenset())
        assert req.state == KV_HELD_FOR_TRANSFER
        return req, attempt

    def test_parked_request_is_held_then_terminated_once_when_the_outcome_arrives(self, rig):
        req, attempt = self._park_then_fail_fatally(rig)
        attempt.deliver_all()
        rig.advance()
        rig.executor._do_terminate_request.assert_called_once_with(req)
        rig.executor._revert_ctx_alloc.assert_not_called()  # a gone request gets no give-back
        assert not rig.hooks.owns(req) and rig.records() == []

    def test_parked_request_without_an_outcome_is_freed_by_close(self, tmp_path):
        rig = EngineRig(status_dump_path=str(tmp_path / "kvt.json"))
        req, _ = self._park_then_fail_fatally(rig)
        rig.hooks.close()  # backends close in time: the pages are safe to free
        rig.executor._free_request_resources.assert_called_once_with(req)
        assert rig.terminations() == 0
        with open(tmp_path / "kvt.json", encoding="utf-8") as f:
            dump = json.load(f)
        assert dump["coordinator"]["finished_pending"] == [1]
        assert [r["state"] for r in dump["coordinator"]["records"]] == ["IN_FLIGHT"]


class TestDeferredEngineTermination:
    """Under attention DP the engine defers a failed request's termination to a later lockstep
    flush (``_handle_errors`` buffers it). The layer may have terminated the held request on
    its outcome by then; the late ``_terminate_request`` must not free it a second time."""

    def test_late_gate_after_the_layer_terminated_does_not_terminate_twice(self, clock):
        rig = EngineRig(
            fetch_timeout_s=10.0
        )  # the ``_handle_errors`` mock defers: it terminates nothing
        req = make_request(1, 100)
        rig.executor.active_requests = [req]
        plan, attempt = rig.plan_reserve_launch(req)
        clock["t"] += 10.5
        rig.advance(req)
        rig.executor._handle_errors.assert_called_once()
        assert rig.coord.held_request_ids() == {1} and req.state == KV_HELD_FOR_TRANSFER

        attempt.deliver_all()
        rig.advance()
        rig.executor._do_terminate_request.assert_called_once_with(req)
        assert not rig.hooks.owns(req)

        rig.executor._terminate_request(req)  # the deferred flush reaches the gate now
        rig.executor._do_terminate_request.assert_called_once_with(req)
        assert rig.hooks.holds_finished_request(req) is False  # and once only: a fresh id after
        assert rig.terminations() == 1

    def test_outcome_in_the_same_round_as_the_expiry_terminates_once_before_the_deferred_gate(
        self, clock
    ):
        """The attempt's outcome is already in when the deadline passes: one ``advance`` fails
        the request (the engine defers its termination), holds it, and terminates it on the
        outcome. The deferred flush then reaches the gate and must not terminate it again."""
        rig = EngineRig(
            fetch_timeout_s=10.0
        )  # the ``_handle_errors`` mock defers: it terminates nothing
        req = make_request(1, 100)
        rig.executor.active_requests = [req]
        plan, attempt = rig.plan_reserve_launch(req)
        attempt.deliver_all()
        clock["t"] += 10.5
        rig.advance(req)
        rig.executor._handle_errors.assert_called_once()
        rig.executor._do_terminate_request.assert_called_once_with(req)
        assert req.context_current_position == 0  # failed, not landed
        assert rig.records() == [] and not rig.hooks.owns(req)

        rig.executor._terminate_request(req)  # the deferred flush
        rig.executor._do_terminate_request.assert_called_once_with(req)
        assert rig.hooks.holds_finished_request(req) is False
        assert rig.terminations() == 1


class TestCancelWhileParked:
    def _cancel_pass(self, rig, req) -> bool:
        """What ``_handle_canceled_requests`` does for one request: ask, then terminate."""
        if not rig.executor._try_cancel_request(req):
            return False
        rig.executor._terminate_request(req)
        return True

    def test_cancel_waits_for_the_landing_then_terminates_once(self, rig):
        req = make_request(1, 100)
        plan, attempt = rig.plan_reserve_launch(req)
        assert self._cancel_pass(rig, req) is False  # parked: this layer's
        assert rig.terminations() == 0
        attempt.deliver_all()
        rig.advance(req)
        assert req.state == CONTEXT_INIT
        assert self._cancel_pass(rig, req) is True
        rig.executor._do_terminate_request.assert_called_once_with(req)
        assert rig.records() == [] and not rig.hooks.owns(req)
        rig.advance()
        assert rig.terminations() == 1

    def test_cancel_waits_for_the_give_back_then_terminates_once(self, rig):
        req = make_request(1, 100)
        plan, attempt = rig.plan_reserve_launch(req)
        assert self._cancel_pass(rig, req) is False
        rig.executor._revert_ctx_alloc.side_effect = lambda reqs: [
            rig.kv.kv_cache_map.pop(r.py_request_id, None) for r in reqs
        ]
        attempt.finish(Failed("boom"))
        rig.advance(req)  # given back, planned again (retry budget)
        assert req.state == CONTEXT_INIT and not rig.hooks.owns(req)
        assert self._cancel_pass(rig, req) is True
        # The PLANNED retry record votes once more so the ranks release it together; the layer
        # then terminates the request it held meanwhile.
        assert rig.terminations() == 0 and rig.coord.held_request_ids() == {1}
        rig.advance()
        rig.executor._do_terminate_request.assert_called_once_with(req)
        assert rig.records() == []
        assert rig.coord.status_dump()["finished_pending"] == []


class TestResponsePass:
    def test_case_b_through_a_real_handle_responses_pass(self, rig, monkeypatch):
        """``_handle_responses``: the request finished with its first token, its response goes
        out, ``_terminate_request`` hits the gate, the publish in flight holds it, and it leaves
        ``active_requests``. When the publish lands the layer terminates it once."""
        rig.wire_response_pass(monkeypatch)
        req = make_request(1, 100, max_new_tokens=1)
        attempt = rig.publish(req)
        req.py_decoding_iter = 1
        req.state = LlmRequestState.GENERATION_COMPLETE
        assert req.is_finished
        rig.executor.active_requests = [req]

        terminated = rig.executor._handle_responses()

        assert terminated == [req]
        assert rig.executor.active_requests == []
        rig.executor._enqueue_responses.assert_called_once()
        (responses,), _ = rig.executor._enqueue_responses.call_args
        assert [rid for rid, _ in responses] == [1]
        assert rig.terminations() == 0  # held
        assert req.state == KV_HELD_FOR_TRANSFER
        assert rig.coord.held_request_ids() == {1}

        attempt.deliver_all()
        rig.advance()
        rig.executor._do_terminate_request.assert_called_once_with(req)
        assert rig.records() == [] and not rig.hooks.owns(req)

    def test_case_a_through_a_real_handle_responses_pass(self, rig, monkeypatch):
        rig.wire_response_pass(monkeypatch)
        req = make_request(1, 100, max_new_tokens=1)
        attempt = rig.publish(req)
        attempt.deliver_all()
        rig.advance()
        req.py_decoding_iter = 1
        req.state = LlmRequestState.GENERATION_COMPLETE
        rig.executor.active_requests = [req]
        rig.executor._handle_responses()
        rig.executor._do_terminate_request.assert_called_once_with(req)
        assert not rig.hooks.owns(req)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``EngineKVTransferEffects``: every effect of design §7.3 in table order, driven through the
real coordinator and hooks over an executor built with ``object.__new__``; the two alias
request states; ``EngineRequestView``; ``EngineWorkQueue`` and ``SingleRankCollective``.
"""

import types
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from engine_fakes import (
    CONTEXT_INIT,
    TOKEN_END,
    TPB,
    SingleRankCollective,
    extent_names,
    make_request,
)

from tensorrt_llm._torch.disaggregation.base.cache_backend import Failed
from tensorrt_llm._torch.disaggregation.remote_cache import DEFER
from tensorrt_llm._torch.pyexecutor.kv_transfer.effects import (
    KV_FETCH_IN_PROGRESS,
    KV_HELD_FOR_TRANSFER,
    EngineRequestView,
    EngineWorkQueue,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

pytestmark = pytest.mark.cpu_only


# =============================================================================================
# Alias states and the request view
# =============================================================================================


def test_alias_states_are_the_disagg_transfer_states_and_outside_the_schedulable_range():
    assert KV_FETCH_IN_PROGRESS is LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
    assert KV_HELD_FOR_TRANSFER is LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
    assert KV_FETCH_IN_PROGRESS.value == 9 and KV_HELD_FOR_TRANSFER.value == 21
    # The V2 scheduler schedules [CONTEXT_INIT, GENERATION_COMPLETE); neither alias is inside.
    lo, hi = LlmRequestState.CONTEXT_INIT.value, LlmRequestState.GENERATION_COMPLETE.value
    assert not lo <= KV_FETCH_IN_PROGRESS.value < hi
    assert not lo <= KV_HELD_FOR_TRANSFER.value < hi


def test_request_view_has_constant_plan_inputs_and_passes_everything_else_through():
    req = make_request(5, 100)
    view = EngineRequestView(req)
    assert view.is_disagg_generation_init is False and view.is_generation_first_context is False
    assert view.route_hints == {}
    assert view.py_request_id == 5 and view.prompt_len == 100
    assert view.is_first_context_chunk and view.context_remaining_length == 100
    assert view.request is req
    assert repr(view) == "EngineRequestView(5)"
    with pytest.raises(AttributeError):
        _ = view.no_such_attribute


def test_work_queue_and_single_rank_collective_complete_the_contract():
    queue = EngineWorkQueue()
    ran = []
    for i in range(3):
        queue.post(lambda i=i: ran.append(i))
    assert queue.drain(2) == 2 and ran == [0, 1]
    assert queue.drain(5) == 1 and ran == [0, 1, 2]
    assert queue.drain(5) == 0
    assert SingleRankCollective().allgather({"x": 1}) == [{"x": 1}]


# =============================================================================================
# Effects in design §7.3 order
# =============================================================================================


class TestParkForFetch:
    def test_launch_parks_the_request_in_the_fetch_state(self, rig):
        req = make_request(1, 100)  # 3 nameable blocks
        plan, attempt = rig.plan_reserve_launch(req)
        assert plan.token_end == 96 and plan.source == "store"
        assert req.state == KV_FETCH_IN_PROGRESS
        assert rig.coord.parked_request_ids() == {1}
        assert rig.coord.held_request_ids() == frozenset()
        assert rig.hooks.owns(req)
        assert rig.hooks.has_pending_work()
        # prepare_fetch_resources went first, with the engine request, not the view.
        rig.executor._prepare_disagg_gen_resources.assert_called_once()
        (prepared,), _ = rig.executor._prepare_disagg_gen_resources.call_args
        assert prepared == [req]
        assert extent_names(attempt.payload) == rig.reader.unit_names(req, range(3))

    def test_a_parked_request_is_not_planned_again(self, rig):
        req = make_request(1, 100)
        rig.plan_reserve_launch(req)
        assert rig.hooks.fetch_answer(req) is None
        rig.advance(req)  # still in flight, still not a candidate
        assert rig.store.count("fetch") == 1


class TestUnpark:
    def test_landing_settles_the_cursor_commits_and_returns_to_context_init(self, rig):
        req = make_request(1, 100)
        plan, attempt = rig.plan_reserve_launch(req)
        attempt.deliver_all()
        rig.advance(req)
        assert req.state == CONTEXT_INIT
        assert req.context_current_position == plan.token_end == TOKEN_END
        assert req.context_remaining_length == 4
        assert req.context_chunk_size == 4  # settled chunk spans to the prompt end
        assert req.py_ctx_pre_resize_cap is None  # the pages keep their size
        assert rig.kv.calls == [("try_commit_blocks", 1)]
        assert rig.kv.kv_cache_map[1].num_committed_tokens == TOKEN_END
        assert not rig.hooks.owns(req)
        assert not rig.coord.has_pending_work()
        assert rig.hooks.fetch_answer(req) is None  # decided: compute the rest locally
        # A landed fetch record reaches its release point when the request ends (design §4.3).
        assert [r["state"] for r in rig.records()] == ["DELIVERED"]
        rig.executor._terminate_request(req)
        rig.executor._do_terminate_request.assert_called_once_with(req)
        assert rig.records() == []

    def test_landing_below_the_committed_depth_settles_at_the_commit(self, rig):
        """Local reuse already committed further than the fetch target (the ask was empty):
        the cursor settles at the committed depth and the commit is a no-op."""
        req = make_request(1, 200)
        rig.store.probe_answer = rig.reader.unit_names(req, range(3))  # the store holds 3 blocks
        plan = rig.plan(req)
        assert plan.token_end == TOKEN_END
        rig.reserve(req, plan.token_end, committed=128)  # local reuse went further
        rig.kv.kv_cache_map[1].history_length = 128
        attempt = rig.launch(req)
        attempt.deliver_all()
        rig.advance(req)
        assert req.state == CONTEXT_INIT
        assert req.context_current_position == 128
        assert rig.kv.kv_cache_map[1].num_committed_tokens == 128
        assert not rig.hooks.owns(req)

    def test_landing_below_the_declared_history_is_a_wiring_error(self, rig):
        req = make_request(1, 100)
        plan = rig.plan(req)
        rig.reserve(req, plan.token_end - TPB)  # the scheduler declared less than planned
        attempt = rig.launch(req)
        attempt.deliver_all()
        with pytest.raises(RuntimeError, match="history_length=64 < token_end=96"):
            rig.advance(req)

    def test_landing_without_a_cache_is_a_wiring_error(self, rig):
        req = make_request(1, 100)
        plan, attempt = rig.plan_reserve_launch(req)
        del rig.kv.kv_cache_map[1]
        attempt.deliver_all()
        with pytest.raises(RuntimeError, match="history_length=None"):
            rig.advance(req)

    def test_short_commit_warns_and_continues(self, rig, effects_logger):
        req = make_request(1, 100)
        plan, attempt = rig.plan_reserve_launch(req)
        rig.kv.commit_to[1] = 64  # the commit falls short of the fetch target
        attempt.deliver_all()
        rig.advance(req)
        assert req.state == CONTEXT_INIT and req.context_current_position == TOKEN_END
        effects_logger.warning.assert_called_once()
        assert "committed 64 tokens after a fetch to 96" in (
            effects_logger.warning.call_args.args[0] % effects_logger.warning.call_args.args[1:]
        )

    def test_gen_init_landing_has_no_place_here(self, rig):
        req = make_request(1, 100)
        with pytest.raises(RuntimeError, match="gen-init landing"):
            rig.effects.unpark(EngineRequestView(req), TOKEN_END, True, None)


class TestRevertFetchPages:
    def test_failed_fetch_reverts_the_allocation_and_replans(self, rig):
        req = make_request(1, 100)
        plan, attempt = rig.plan_reserve_launch(req)
        # The revert drops the cache, as the wrapper does when history outran pre-resize capacity.
        rig.executor._revert_ctx_alloc.side_effect = lambda reqs: [
            rig.kv.kv_cache_map.pop(r.py_request_id, None) for r in reqs
        ]
        attempt.finish(Failed("store outage"))
        rig.advance(req)
        rig.executor._revert_ctx_alloc.assert_called_once_with([req])
        assert rig.kv.count("free_resources") == 0
        assert req.state == CONTEXT_INIT
        assert not rig.hooks.owns(req)
        assert rig.store.count("quiesce") == 1  # quiesced before the pages were given back
        # One retry: the request is deferred again and the record is kept for the budget.
        assert rig.hooks.fetch_answer(req) is DEFER
        assert rig.records()[0]["state"] == "PLANNED" and rig.records()[0]["try_index"] == 0
        plan2 = rig.plan(req)
        assert plan2.token_end == TOKEN_END

    def test_a_cache_that_survives_the_revert_is_dropped_and_the_cursor_rewound(self, rig):
        """``revert_allocate_context`` returns True without touching a cache whose
        ``py_ctx_pre_resize_cap`` is None; the pages then hold no data but the history says they do."""
        req = make_request(1, 100)
        plan, attempt = rig.plan_reserve_launch(req)
        req.py_ctx_pre_resize_cap = None
        req.context_current_position = 32  # a local reuse match the request came in with
        attempt.finish(Failed("boom"))
        rig.advance(req)
        assert rig.kv.count("free_resources") == 1
        assert 1 not in rig.kv.kv_cache_map
        assert req.context_current_position == 0
        assert req.context_chunk_size == req.prompt_len
        assert req.py_ctx_pre_resize_cap is None
        assert req.state == CONTEXT_INIT

    def test_rejected_submission_gives_pages_back_without_a_quiesce(self, rig):
        req = make_request(1, 100)
        plan = rig.plan(req)
        rig.reserve(req, plan.token_end)
        rig.store.reject_next_calls = 1
        rig.hooks.launch_reserved_fetches([req])
        assert rig.store.attempts == []
        rig.executor._revert_ctx_alloc.assert_called_once_with([req])
        assert rig.store.count("quiesce") == 0
        assert req.state == CONTEXT_INIT and not rig.hooks.owns(req)
        # The plan stands: the scheduler reserves for the same fetch next round, no retry spent.
        assert rig.hooks.fetch_answer(req) is plan

    def test_second_failure_lands_the_request_on_the_local_path(self, rig):
        req = make_request(1, 100)
        for _ in range(2):
            plan, attempt = rig.plan_reserve_launch(req)
            attempt.finish(Failed("boom"))
            rig.advance(req)
        assert rig.hooks.fetch_answer(req) is None
        assert rig.records() == []
        assert req.state == CONTEXT_INIT


class TestPrepareFetchResources:
    def test_forwards_engine_requests_to_the_gen_init_preparation_without_the_latch(self, rig):
        a, b = make_request(1, 100), make_request(2, 100)
        rig.effects.prepare_fetch_resources([EngineRequestView(a), EngineRequestView(b)])
        rig.executor._prepare_disagg_gen_resources.assert_called_once_with(
            [a, b], latch_cached_tokens=False
        )

    def test_the_engine_preparation_leaves_cached_tokens_to_the_first_forward_for_a_fetch(self):
        """``cached_tokens`` is written once. Latched at launch it would report the local reuse
        depth; left alone, the first forward after the landing reports the fetched depth. A
        gen-init request, which never runs a context forward here, is latched as before."""
        executor = SimpleNamespace(resource_manager=SimpleNamespace(resource_managers={}))
        fetched, gen_init = make_request(1, 100), make_request(2, 100)
        PyExecutor._prepare_disagg_gen_resources(executor, [fetched], latch_cached_tokens=False)
        PyExecutor._prepare_disagg_gen_resources(executor, [gen_init])
        # Not latched by the preparation: the first write afterwards (the context forward's)
        # is the one that sticks, and a later write does not move it.
        assert fetched.cached_tokens == 0
        fetched.cached_tokens = TOKEN_END
        fetched.cached_tokens = 50
        assert fetched.cached_tokens == TOKEN_END
        # Latched by the preparation: a write afterwards changes nothing.
        gen_init.cached_tokens = TOKEN_END
        assert gen_init.cached_tokens == gen_init.prepopulated_prompt_len

    def test_launch_prepares_resources_without_latching_cached_tokens(self, rig):
        """Through the fetch flow with the engine's real preparation: the launch prepares the
        resource managers and leaves ``cached_tokens`` untouched (latched there it would read
        0: nothing was matched locally), and so does the landing; the latch is still open for
        the context forward that follows."""
        executor = rig.executor
        executor._prepare_disagg_gen_resources = types.MethodType(
            PyExecutor._prepare_disagg_gen_resources, executor
        )
        req = make_request(1, 100)
        plan, attempt = rig.plan_reserve_launch(req)
        assert rig.kv.calls[-1] == ("prepare_resources", 1)
        assert req.cached_tokens == 0
        attempt.deliver_all()
        rig.advance(req)
        assert req.context_current_position == plan.token_end == TOKEN_END
        assert req.cached_tokens == 0
        req.cached_tokens = (
            TOKEN_END  # the first write, the context forward's, is the one that sticks
        )
        req.cached_tokens = 50
        assert req.cached_tokens == TOKEN_END


class TestHoldForTransfer:
    def test_held_request_keeps_pages_and_releases_slots(self, rig):
        req = make_request(1, 100)
        rig.publish(req)
        assert rig.slots.slots == {1}
        assert rig.executor._terminate_request(req) is None
        assert rig.terminations() == 0
        assert req.state == KV_HELD_FOR_TRANSFER
        assert rig.coord.held_request_ids() == {1}
        assert rig.coord.parked_request_ids() == frozenset()
        assert rig.slots.freed == [1] and rig.slots.slots == set()
        assert rig.kv.count("release_index_slot") == 1
        assert 1 in rig.kv.kv_cache_map  # the pages stay for the backend
        assert rig.kv.count("free_resources") == 0
        assert rig.hooks.owns(req)
        assert rig.hooks.has_pending_work()
        assert rig.reader.forgotten == []  # forgotten once the held request is terminated

    def test_hold_reads_nothing_of_disagg_and_tolerates_slots_already_released(self, rig):
        """A seq slot the disagg send freed is a no-op to free again, and ``release_index_slot``
        is idempotent."""
        req = make_request(1, 100)
        rig.publish(req)
        rig.slots.free_resources(req)  # disagg start_transfer did this
        rig.kv.release_index_slot(1)  # and this
        rig.executor._terminate_request(req)
        assert rig.slots.freed == [1, 1] and rig.slots.slots == set()
        assert rig.kv.count("release_index_slot") == 2  # the wrapper makes the second a no-op
        assert req.state == KV_HELD_FOR_TRANSFER

    def test_hold_without_a_cache_skips_the_index_slot(self, rig):
        req = make_request(1, 100)
        rig.publish(req)
        del rig.kv.kv_cache_map[1]  # already freed by another owner
        rig.effects.hold_for_transfer([EngineRequestView(req)])
        assert rig.kv.count("release_index_slot") == 0


class TestTerminateRequest:
    def test_goes_straight_to_do_terminate_with_the_engine_request(self, rig):
        req = make_request(1, 100)
        rig.publish(req)
        rig.executor._terminate_request(req)
        assert rig.coord.held_request_ids() == {1}
        rig.effects.terminate_request(EngineRequestView(req))
        rig.executor._do_terminate_request.assert_called_once_with(req)

    def test_goes_through_the_pp_termination_handler_when_the_engine_has_one(self, rig):
        """Under disaggregated PP every termination must pass the ranks' ring, or one rank frees
        the request while its peers still count on it."""
        handler = Mock()
        rig.executor._disagg_pp_termination_handler = handler
        req = make_request(1, 100)
        rig.effects.terminate_request(EngineRequestView(req))
        handler.terminate.assert_called_once_with(req)
        rig.executor._do_terminate_request.assert_not_called()

    def test_a_held_request_terminates_through_the_pp_termination_handler_once(self, rig):
        """The whole way with the handler present: the engine's ``_terminate_request`` reaches
        the gate, which holds the request; when the publish lands the layer terminates it
        through the handler, exactly once, and never through ``_do_terminate_request``."""
        handler = Mock()
        rig.executor._disagg_pp_termination_handler = handler
        req = make_request(1, 100)
        attempt = rig.publish(req)
        rig.executor._terminate_request(req)
        handler.terminate.assert_not_called()
        assert rig.coord.held_request_ids() == {1}
        attempt.deliver_all()
        rig.advance()
        handler.terminate.assert_called_once_with(req)
        rig.executor._do_terminate_request.assert_not_called()
        assert rig.records() == [] and not rig.hooks.owns(req)
        rig.advance()
        handler.terminate.assert_called_once_with(req)


class TestFailRequests:
    def test_fail_requests_uses_the_request_scoped_error_path(self, rig):
        a, b = make_request(1, 100), make_request(2, 100)
        rig.effects.fail_requests([EngineRequestView(a), EngineRequestView(b)], "kv fetch failed")
        rig.executor._handle_errors.assert_called_once_with(
            "kv fetch failed", requests=[a, b], charge_budget=False
        )

    def test_fail_fatal_marks_the_executor_and_takes_the_rank_local_fatal_path(self, rig):
        """A refused quiesce is observed by one rank; the collective-aligned fatal path is for
        a fatal every rank agreed on, and would enter a gather the peers are not in."""
        rig.effects.fail_fatal(RuntimeError("cannot vouch for memory"))
        assert rig.executor.is_shutdown is True
        assert isinstance(rig.executor._fatal_error, RuntimeError)
        assert "cannot vouch for memory" in str(rig.executor._fatal_error)
        rig.executor._handle_errors.assert_called_once_with(
            "cannot vouch for memory", requests=None, charge_budget=False
        )

    def test_quiesce_refusal_reaches_fail_fatal(self, rig):
        req = make_request(1, 100)
        plan, attempt = rig.plan_reserve_launch(req)
        rig.store.quiesce_answers.append(False)
        attempt.finish(Failed("boom"))
        rig.advance(req)
        assert rig.executor.is_shutdown is True
        assert "cannot confirm memory of request 1" in str(rig.executor._fatal_error)
        rig.executor._revert_ctx_alloc.assert_not_called()  # pages are poisoned, not given back

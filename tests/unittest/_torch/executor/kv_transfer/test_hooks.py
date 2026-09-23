# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""U2 (integration plan §9, §11): the engine-side effects and the loop hooks.

A ``PyExecutor`` built with ``object.__new__`` and the attributes the effects read, a real
``KVTransferCoordinator`` and ``Planner``, and fakes for the reader and the backends. Covered:
every effect of design §7.3 in table order; the two alias states; the release gate's five cases
A-E of plan §9 (E synthesized: the disagg send never comes back); the dummy bypass; ``is_tracking``
on the cancel path; ``has_transfer_in_flight`` for idle detection; ``pace_idle``; ``close()`` order
and its timeout against a hanging backend; and the status dump's JSON schema.
"""

import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from engine_fakes import (
    TPB,
    FakeFetches,
    FakeKVCache,
    FakeKVCacheManager,
    FakeLandsOnHost,
    FakePublishes,
    FakeReader,
    FakeSlotManager,
    HangingClose,
    SingleRankDist,
    extent_names,
    finish_prefill,
    make_executor,
    make_request,
)

from tensorrt_llm._torch.disaggregation.backends.config import BackendEntry
from tensorrt_llm._torch.disaggregation.backends.registry import BackendHandle
from tensorrt_llm._torch.disaggregation.base.cache_backend import Failed
from tensorrt_llm._torch.disaggregation.orchestration.kv_transfer.coordinator import (
    KVTransferCoordinator,
)
from tensorrt_llm._torch.disaggregation.remote_cache import DEFER, FetchPlan, FetchSource, Planner
from tensorrt_llm._torch.pyexecutor.kv_transfer import effects, hooks
from tensorrt_llm._torch.pyexecutor.kv_transfer.effects import (
    KV_FETCH_IN_PROGRESS,
    KV_PUBLISH_IN_PROGRESS,
    EngineRequestView,
    EngineWorkQueue,
    PyExecutorKVTransferEffects,
)
from tensorrt_llm._torch.pyexecutor.kv_transfer.hooks import KVTransferHooks
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

pytestmark = pytest.mark.cpu_only

CONTEXT_INIT = LlmRequestState.CONTEXT_INIT


class Rig:
    """One executor with the real coordinator, planner, effects and hooks over fakes. With
    ``host_first`` the store lands in its own memory first (``LandsOnHost``)."""

    def __init__(
        self,
        *,
        status_dump_path=None,
        close_timeout_s: float = 5.0,
        backend_close=None,
        probe_answer="all",
        publish: bool = True,
        fetch_timeout_s=None,
        publish_timeout_s=None,
        host_first: bool = False,
    ) -> None:
        self.kv = FakeKVCacheManager(TPB)
        self.slots = FakeSlotManager()
        self.executor = make_executor(self.kv, self.slots)
        self.reader = FakeReader(self.kv)
        self.store = (
            FakeLandsOnHost(name="store")
            if host_first
            else FakeFetches(name="store", probe_answer=probe_answer)
        )
        self.publisher = FakePublishes()
        sources = [FetchSource("store", self.store, None)]
        self.planner = Planner(sources, self.reader, TPB, probe_timeout_s=0.05)
        self.effects = PyExecutorKVTransferEffects(self.executor)
        self.coord = KVTransferCoordinator(
            sources,
            [self.publisher] if publish else [],
            self.planner,
            self.reader,
            self.effects,
            EngineWorkQueue(),
            SingleRankDist(),
            fetch_timeout_s=fetch_timeout_s,
            publish_timeout_s=publish_timeout_s,
        )
        self.trace: list = []
        self.backend_close = backend_close or (lambda: self.trace.append("backend.close"))
        handle = BackendHandle(
            name="store",
            hint_key=None,
            fetcher=self.store,
            publisher=self.publisher if publish else None,
            pool_registrar=None,
            close=self.backend_close,
            counters=lambda: {
                "fetch_hits": self.store.count("fetch"),
                "publish_stored": self.publisher.count("publish"),
            },
        )
        entry = BackendEntry.from_dict(
            {
                "name": "store",
                "type": "fake",
                "roles": ["fetch", "publish"] if publish else ["fetch"],
            }
        )
        self.hooks = KVTransferHooks(
            self.executor,
            self.coord,
            self.effects,
            [handle],
            [entry],
            close_timeout_s=close_timeout_s,
            status_dump_path=status_dump_path,
            started_at=123.5,
        )
        self.executor.kv_transfer = self.hooks
        self.executor._free_request_resources.side_effect = lambda req: self.trace.append(
            ("free", req.py_request_id)
        )

    # -- the loop, one hook at a time --

    def advance(self, *active) -> None:
        self.hooks.advance_round(list(active))

    def plan(self, req) -> FetchPlan:
        self.advance(req)
        plan = self.hooks.plan_fetch(req)
        assert isinstance(plan, FetchPlan), f"expected a plan, got {plan!r}"
        return plan

    def reserve(self, req, token_end: int, *, committed: int = 0) -> FakeKVCache:
        """What the scheduler's ``reserve_transfer_pages(req, token_end)`` leaves behind."""
        kv_cache = FakeKVCache(history_length=token_end, num_committed_tokens=committed)
        self.kv.kv_cache_map[req.py_request_id] = kv_cache
        req.py_ctx_pre_resize_cap = 0
        self.slots.add(req)
        return kv_cache

    def launch(self, req):
        before = len(self.store.attempts)
        self.hooks.launch_reserved_fetches([req])
        assert len(self.store.attempts) == before + 1, "launch did not create a store attempt"
        return self.store.attempts[-1]

    def plan_reserve_launch(self, req):
        plan = self.plan(req)
        self.reserve(req, plan.token_end)
        return plan, self.launch(req)

    def publish(self, req):
        """Prefill ended this step; the loop offers the request's blocks."""
        finish_prefill(req)
        self.kv.kv_cache_map.setdefault(
            req.py_request_id, FakeKVCache(history_length=req.prompt_len)
        )
        self.slots.add(req)
        before = len(self.publisher.attempts)
        self.hooks.publish_committed_blocks([req])
        assert len(self.publisher.attempts) == before + 1, "no publish attempt was made"
        return self.publisher.attempts[-1]

    def records(self) -> list[dict]:
        return self.coord.status_dump()["records"]

    def terminations(self) -> int:
        return self.executor._do_terminate_request.call_count

    def wire_engine_error_path(self) -> None:
        """Make the ``_handle_errors`` mock do what the engine's does to the failed requests:
        mark them complete, drop them from ``active_requests``, terminate each through
        ``_terminate_request`` (whose release gate asks this layer)."""
        executor = self.executor

        def handle_errors(error_msg=None, *, requests=None, charge_budget=True, **_):
            failed = list(executor.active_requests) if requests is None else list(requests)
            for request in failed:
                request.state = LlmRequestState.GENERATION_COMPLETE
            executor.active_requests = [r for r in executor.active_requests if r not in failed]
            for request in failed:
                executor._terminate_request(request)

        executor._handle_errors.side_effect = handle_errors

    def wire_response_pass(self, monkeypatch) -> None:
        """What a real ``_handle_responses`` reads besides the requests themselves."""
        executor = self.executor
        executor.perf_manager = Mock()
        executor.iter_counter = 3
        executor.stream_interval = 1
        executor.disable_overlap_scheduler = True
        disagg = Mock()
        disagg.inflight_cancel_active.return_value = False
        # ``PyExecutor.disagg`` is a read-only property over the disagg coordinator.
        monkeypatch.setattr(PyExecutor, "disagg", property(lambda self: disagg))
        executor.model_engine = SimpleNamespace(route_capture=None)
        executor.force_terminate_ctx_for_partial_reuse = False
        executor._enqueue_responses = Mock()
        executor.dist = SimpleNamespace(rank=0)
        executor.gather_all_responses = False


@pytest.fixture
def rig():
    return Rig()


@pytest.fixture
def effects_logger(monkeypatch):
    fake = Mock()
    monkeypatch.setattr(effects, "logger", fake)
    return fake


@pytest.fixture
def hooks_logger(monkeypatch):
    fake = Mock()
    monkeypatch.setattr(hooks, "logger", fake)
    return fake


# =============================================================================================
# Alias states and the request view
# =============================================================================================


def test_alias_states_are_the_disagg_transfer_states_and_outside_the_schedulable_range():
    assert KV_FETCH_IN_PROGRESS is LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
    assert KV_PUBLISH_IN_PROGRESS is LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
    assert KV_FETCH_IN_PROGRESS.value == 9 and KV_PUBLISH_IN_PROGRESS.value == 21
    # The V2 scheduler schedules [CONTEXT_INIT, GENERATION_COMPLETE); neither alias is inside.
    lo, hi = LlmRequestState.CONTEXT_INIT.value, LlmRequestState.GENERATION_COMPLETE.value
    assert not lo <= KV_FETCH_IN_PROGRESS.value < hi
    assert not lo <= KV_PUBLISH_IN_PROGRESS.value < hi


def test_request_view_has_constant_plan_inputs_and_passes_everything_else_through():
    req = make_request(5, 100)
    view = EngineRequestView(req)
    assert view.is_gen_init is False and view.is_gen_first_context is False
    assert view.route_hints == {}
    assert view.py_request_id == 5 and view.prompt_len == 100
    assert view.is_first_context_chunk and view.context_remaining_length == 100
    assert view.request is req
    assert repr(view) == "EngineRequestView(5)"
    with pytest.raises(AttributeError):
        _ = view.no_such_attribute


def test_work_queue_and_single_rank_dist_complete_the_contract():
    queue = EngineWorkQueue()
    ran = []
    for i in range(3):
        queue.post(lambda i=i: ran.append(i))
    assert queue.drain(2) == 2 and ran == [0, 1]
    assert queue.drain(5) == 1 and ran == [0, 1, 2]
    assert queue.drain(5) == 0
    assert SingleRankDist().allgather({"x": 1}) == [{"x": 1}]


# =============================================================================================
# Effects in design §7.3 order
# =============================================================================================


class TestParkForFetch:
    def test_launch_parks_the_request_in_the_fetch_state(self, rig):
        req = make_request(1, 100)  # 3 nameable blocks -> token_end 96
        plan, attempt = rig.plan_reserve_launch(req)
        assert plan.token_end == 96 and plan.source == "store"
        assert req.state == KV_FETCH_IN_PROGRESS
        assert rig.coord.parked_request_ids() == {1}
        assert rig.coord.held_request_ids() == frozenset()
        assert rig.hooks.is_tracking(req)
        assert rig.hooks.has_transfer_in_flight()
        # prepare_fetch_resources went first, with the engine request, not the view.
        rig.executor._prepare_disagg_gen_resources.assert_called_once()
        (prepared,), _ = rig.executor._prepare_disagg_gen_resources.call_args
        assert prepared == [req]
        assert extent_names(attempt.payload) == rig.reader.unit_names(req, range(3))

    def test_a_parked_request_is_not_planned_again(self, rig):
        req = make_request(1, 100)
        rig.plan_reserve_launch(req)
        assert rig.hooks.plan_fetch(req) is None
        rig.advance(req)  # still in flight, still not a candidate
        assert rig.store.count("fetch") == 1


class TestUnpark:
    def test_landing_settles_the_cursor_commits_and_returns_to_context_init(self, rig):
        req = make_request(1, 100)
        plan, attempt = rig.plan_reserve_launch(req)
        attempt.deliver_all()
        rig.advance(req)
        assert req.state == CONTEXT_INIT
        assert req.context_current_position == plan.token_end == 96
        assert req.context_remaining_length == 4
        assert req.context_chunk_size == 4  # settled chunk spans to the prompt end
        assert req.py_ctx_pre_resize_cap is None  # plan §10 #11
        assert rig.kv.calls == [("try_commit_blocks", 1)]
        assert rig.kv.kv_cache_map[1].num_committed_tokens == 96
        assert not rig.hooks.is_tracking(req)
        assert not rig.hooks.has_transfer_in_flight()
        assert rig.hooks.plan_fetch(req) is None  # decided: compute the rest locally
        # A landed fetch record reaches its release point when the request ends (design §4.3).
        assert [r["state"] for r in rig.records()] == ["LANDED"]
        rig.executor._terminate_request(req)
        rig.executor._do_terminate_request.assert_called_once_with(req)
        assert rig.records() == []

    def test_landing_below_the_committed_depth_settles_at_the_commit(self, rig):
        """Local reuse already committed further than the fetch target (the ask was empty):
        the cursor settles at the committed depth and the commit is a no-op."""
        req = make_request(1, 200)
        rig.store.probe_answer = rig.reader.unit_names(req, range(3))  # the store holds 3 blocks
        plan = rig.plan(req)
        assert plan.token_end == 96
        rig.reserve(req, plan.token_end, committed=128)  # local reuse went further
        rig.kv.kv_cache_map[1].history_length = 128
        attempt = rig.launch(req)
        attempt.deliver_all()
        rig.advance(req)
        assert req.state == CONTEXT_INIT
        assert req.context_current_position == 128
        assert rig.kv.kv_cache_map[1].num_committed_tokens == 128
        assert not rig.hooks.is_tracking(req)

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
        rig.kv.commit_to[1] = 64  # plan §10 #14
        attempt.deliver_all()
        rig.advance(req)
        assert req.state == CONTEXT_INIT and req.context_current_position == 96
        effects_logger.warning.assert_called_once()
        assert "committed 64 tokens after a fetch to 96" in (
            effects_logger.warning.call_args.args[0] % effects_logger.warning.call_args.args[1:]
        )

    def test_gen_init_landing_has_no_place_here(self, rig):
        req = make_request(1, 100)
        with pytest.raises(RuntimeError, match="gen-init landing"):
            rig.effects.unpark(EngineRequestView(req), 96, True, None)


class TestGiveBackFetchPages:
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
        assert not rig.hooks.is_tracking(req)
        assert rig.store.count("quiesce") == 1  # quiesced before the pages were given back
        # One retry: the request is undecided again and the record is kept for the budget.
        assert rig.hooks.plan_fetch(req) is DEFER
        assert rig.records()[0]["state"] == "PLANNED" and rig.records()[0]["try_index"] == 0
        plan2 = rig.plan(req)
        assert plan2.token_end == 96

    def test_a_cache_that_survives_the_revert_is_dropped_and_the_cursor_rewound(self, rig):
        """Plan §10 #13: ``revert_allocate_context`` returns True without touching a cache whose
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
        rig.store.reject_next = 1
        rig.hooks.launch_reserved_fetches([req])
        assert rig.store.attempts == []
        rig.executor._revert_ctx_alloc.assert_called_once_with([req])
        assert rig.store.count("quiesce") == 0
        assert req.state == CONTEXT_INIT and not rig.hooks.is_tracking(req)
        # The plan stands: the scheduler reserves for the same fetch next round, no retry spent.
        assert rig.hooks.plan_fetch(req) is plan

    def test_second_failure_lands_the_request_on_the_local_path(self, rig):
        req = make_request(1, 100)
        for _ in range(2):
            plan, attempt = rig.plan_reserve_launch(req)
            attempt.finish(Failed("boom"))
            rig.advance(req)
        assert rig.hooks.plan_fetch(req) is None
        assert rig.records() == []
        assert req.state == CONTEXT_INIT


class TestPrepareFetchResources:
    def test_forwards_engine_requests_to_the_gen_init_preparation(self, rig):
        a, b = make_request(1, 100), make_request(2, 100)
        rig.effects.prepare_fetch_resources([EngineRequestView(a), EngineRequestView(b)])
        rig.executor._prepare_disagg_gen_resources.assert_called_once_with([a, b])


class TestHoldForTransfer:
    def test_held_request_keeps_pages_and_releases_slots(self, rig):
        req = make_request(1, 100)
        rig.publish(req)
        assert rig.slots.slots == {1}
        assert rig.executor._terminate_request(req) is None
        assert rig.terminations() == 0
        assert req.state == KV_PUBLISH_IN_PROGRESS
        assert rig.coord.held_request_ids() == {1}
        assert rig.coord.parked_request_ids() == frozenset()
        assert rig.slots.freed == [1] and rig.slots.slots == set()
        assert rig.kv.count("release_index_slot") == 1
        assert 1 in rig.kv.kv_cache_map  # the pages stay for the backend
        assert rig.kv.count("free_resources") == 0
        assert rig.hooks.is_tracking(req)
        assert rig.hooks.has_transfer_in_flight()
        assert rig.reader.forgotten == []  # forgotten once the held request is terminated

    def test_hold_reads_nothing_of_disagg_and_tolerates_slots_already_released(self, rig):
        """Plan §9: a seq slot the disagg send freed is a no-op to free again, and
        ``release_index_slot`` is idempotent."""
        req = make_request(1, 100)
        rig.publish(req)
        rig.slots.free_resources(req)  # disagg start_transfer did this
        rig.kv.release_index_slot(1)  # and this
        rig.executor._terminate_request(req)
        assert rig.slots.freed == [1, 1] and rig.slots.slots == set()
        assert rig.kv.count("release_index_slot") == 2  # the wrapper makes the second a no-op
        assert req.state == KV_PUBLISH_IN_PROGRESS

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


class TestFailRequests:
    def test_fail_requests_uses_the_request_scoped_error_path(self, rig):
        a, b = make_request(1, 100), make_request(2, 100)
        rig.effects.fail_requests([EngineRequestView(a), EngineRequestView(b)], "kv fetch failed")
        rig.executor._handle_errors.assert_called_once_with(
            "kv fetch failed", requests=[a, b], charge_budget=False
        )

    def test_fail_fatal_marks_the_executor_and_fails_collectively(self, rig):
        rig.effects.fail_fatal(RuntimeError("cannot vouch for memory"))
        assert rig.executor.is_shutdown is True
        assert isinstance(rig.executor._fatal_error, RuntimeError)
        assert "cannot vouch for memory" in str(rig.executor._fatal_error)
        rig.executor._handle_errors.assert_called_once_with(
            "cannot vouch for memory",
            requests=None,
            charge_budget=False,
            fatal_is_collective_aligned=True,
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


# =============================================================================================
# Publish selection (plan §9 rule 4)
# =============================================================================================


class TestPublishSelection:
    def test_only_finished_prefills_with_pages_are_offered(self, rig):
        done = make_request(1, 100)
        finish_prefill(done)
        rig.kv.kv_cache_map[1] = FakeKVCache(history_length=100)
        mid = make_request(2, 100)  # prefill not over
        rig.kv.kv_cache_map[2] = FakeKVCache(history_length=64)
        failed = make_request(3, 100)  # failed: its pages are gone
        finish_prefill(failed)
        failed.state = LlmRequestState.GENERATION_COMPLETE
        first_token_done = make_request(4, 100, max_new_tokens=1)  # finished with prefill
        finish_prefill(first_token_done)
        rig.kv.kv_cache_map[4] = FakeKVCache(history_length=100)
        first_token_done.state = LlmRequestState.GENERATION_COMPLETE
        dummy = make_request(5, 100)
        finish_prefill(dummy)
        rig.kv.kv_cache_map[5] = FakeKVCache(history_length=100)
        dummy.is_dummy_request = True

        rig.hooks.publish_committed_blocks([done, mid, failed, first_token_done, dummy])

        assert [c for c in rig.reader.calls if c[0] == "publish_description"] == [
            ("publish_description", 1),
            ("publish_description", 4),
        ]
        assert rig.publisher.count("publish") == 2
        assert all(a.payload.is_last for a in rig.publisher.attempts)

    def test_generation_only_request_is_not_offered(self, rig):
        from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestType

        req = make_request(1, 100)
        req.py_llm_request_type = LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY
        assert req.is_generation_only_request
        finish_prefill(req)
        rig.kv.kv_cache_map[1] = FakeKVCache(history_length=100)
        rig.hooks.publish_committed_blocks([req])
        assert rig.publisher.count("publish") == 0

    def test_a_suspended_cache_is_not_offered(self, rig):
        """A request whose cache was suspended (evicted to a lower tier) has no pages on the
        device to read; it is skipped rather than published from memory that is not there."""
        req = make_request(1, 100)
        finish_prefill(req)
        cache = FakeKVCache(history_length=100)
        cache.is_active = False
        rig.kv.kv_cache_map[1] = cache
        rig.hooks.publish_committed_blocks([req])
        assert rig.publisher.count("publish") == 0
        assert rig.reader.calls == []

    def test_no_publisher_configured_offers_nothing(self):
        rig = Rig(publish=False)
        req = make_request(1, 100)
        finish_prefill(req)
        rig.kv.kv_cache_map[1] = FakeKVCache(history_length=100)
        rig.hooks.publish_committed_blocks([req])
        assert rig.reader.calls == []
        assert rig.executor._terminate_request(req) is None
        assert rig.terminations() == 1  # nothing to hold it


# =============================================================================================
# The release gate: plan §9 cases A-E
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
        assert req.state != KV_PUBLISH_IN_PROGRESS
        assert not rig.hooks.is_tracking(req)
        assert rig.kv.count("free_resources") == 0  # _do_terminate_request owns that
        assert rig.coord.status_dump()["finished_pending"] == []

    def test_case_b_publish_in_flight_layer_holds_then_terminates(self, rig):
        req = make_request(1, 100)
        attempt = rig.publish(req)

        rig.executor._terminate_request(req)  # _handle_responses' release point

        assert rig.terminations() == 0
        assert req.state == KV_PUBLISH_IN_PROGRESS
        assert rig.hooks.is_tracking(req) and rig.hooks.has_transfer_in_flight()
        rig.advance()  # still in flight
        assert rig.terminations() == 0
        attempt.deliver_all()
        rig.advance()
        rig.executor._do_terminate_request.assert_called_once_with(req)
        assert not rig.hooks.is_tracking(req) and not rig.hooks.has_transfer_in_flight()
        assert rig.records() == [] and rig.coord.status_dump()["finished_pending"] == []

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
        assert not rig.hooks.is_tracking(req)
        assert rig.records() == [] and rig.coord.status_dump()["finished_pending"] == []

    def test_case_c_ctx_only_send_done_first_publish_in_flight(self, rig):
        """The disagg ``release_transfer`` reaches ``_terminate_request`` first, with the seq and
        index slots already released by the send; the gate holds until the publish lands."""
        req = make_request(1, 100)
        attempt = rig.publish(req)
        req.state = KV_PUBLISH_IN_PROGRESS  # disagg start_transfer wrote 21 already
        rig.slots.free_resources(req)
        rig.kv.release_index_slot(1)

        rig.executor._terminate_request(req)  # from release_transfer -> effects.terminate_request

        assert rig.terminations() == 0
        assert rig.coord.held_request_ids() == {1}
        assert rig.kv.count("release_index_slot") == 2  # the wrapper makes the second a no-op
        attempt.deliver_all()
        rig.advance()
        rig.executor._do_terminate_request.assert_called_once_with(req)
        assert rig.records() == [] and not rig.hooks.is_tracking(req)

    def test_case_d_publish_landed_first_send_in_flight(self, rig):
        """The publish landed while the request was still running: the record is released and
        nothing is owed. The disagg send finishing later terminates through the gate: True."""
        req = make_request(1, 100)
        attempt = rig.publish(req)
        attempt.deliver_all()
        rig.advance()
        assert rig.records() == []
        assert rig.terminations() == 0  # _finish_publish did nothing for an unfinished request
        req.state = KV_PUBLISH_IN_PROGRESS  # the disagg send still holds it

        rig.executor._terminate_request(req)  # release_transfer, send complete

        rig.executor._do_terminate_request.assert_called_once_with(req)
        assert not rig.hooks.is_tracking(req)
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
            assert rig.terminations() == 0 and req.state == KV_PUBLISH_IN_PROGRESS
            attempt.deliver_all()
            rig.advance()
            rig.executor._do_terminate_request.assert_called_once_with(req)
        # Nothing waits on a disagg release that is never coming.
        for _ in range(3):
            rig.advance()
        assert rig.terminations() == 1
        assert not rig.hooks.is_tracking(req) and not rig.hooks.has_transfer_in_flight()
        assert rig.records() == []
        assert rig.coord.tracked_requests() == []
        assert rig.coord.status_dump()["finished_pending"] == []

    def test_publish_rejected_outright_terminates_exactly_once(self, rig):
        """Every publisher refused the request's blocks and the request then finished: nothing is
        in flight, nothing is held, and the engine's own ``_terminate_request`` is the one
        termination."""
        req = make_request(1, 100)
        finish_prefill(req)
        rig.kv.kv_cache_map[1] = FakeKVCache(history_length=100)
        rig.publisher.reject_next = 1
        rig.hooks.publish_committed_blocks([req])
        assert rig.publisher.attempts == []

        rig.executor._terminate_request(req)

        rig.executor._do_terminate_request.assert_called_once_with(req)
        assert rig.records() == [] and not rig.hooks.is_tracking(req)
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
        assert req.state == KV_PUBLISH_IN_PROGRESS  # held
        assert rig.coord.parked_request_ids() == frozenset()
        assert rig.coord.held_request_ids() == {1}
        attempt.deliver_all()
        rig.advance()
        rig.executor._do_terminate_request.assert_called_once_with(req)
        rig.executor._revert_ctx_alloc.assert_not_called()
        assert req.context_current_position == 0  # never unparked
        assert rig.records() == [] and not rig.hooks.is_tracking(req)

    def test_request_finished_with_a_plan_but_no_launch_terminates_now(self, rig):
        req = make_request(1, 100)
        rig.plan(req)
        assert rig.records()[0]["state"] == "PLANNED"
        rig.executor._terminate_request(req)
        rig.executor._do_terminate_request.assert_called_once_with(req)
        assert rig.records() == []

    def test_dummy_requests_bypass_the_gate(self, rig):
        req = make_request(1, 100)
        req.is_dummy_request = True
        finish_prefill(req)
        rig.kv.kv_cache_map[1] = FakeKVCache(history_length=100)
        rig.hooks.publish_committed_blocks([req])  # never offered (rule 4) ...
        assert rig.publisher.count("publish") == 0
        notify = Mock(wraps=rig.coord.notify_request_finished)
        rig.coord.notify_request_finished = notify
        rig.executor._terminate_request(req)  # ... and the gate is not even asked
        notify.assert_not_called()
        rig.executor._do_terminate_request.assert_called_once_with(req)
        assert rig.reader.forgotten == []

    def test_gate_answers_true_for_a_request_this_layer_never_saw(self, rig):
        req = make_request(9, 100)
        assert rig.hooks.on_request_finished(req) is True
        assert rig.reader.forgotten == [9]
        assert rig.coord.status_dump()["finished_pending"] == []


# =============================================================================================
# Cancel path, idle detection, pacing
# =============================================================================================


class TestCancelAndIdle:
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

    def test_is_tracking_reads_the_record_table_not_the_state(self, rig):
        req = make_request(1, 100)
        req.state = KV_FETCH_IN_PROGRESS  # a disagg gen-init in transmission looks like this
        assert rig.hooks.is_tracking(req) is False
        assert rig.executor._try_cancel_request(req) is True  # no transceiver: cancellable
        dummy = make_request(2, 100)
        dummy.is_dummy_request = True
        assert rig.hooks.is_tracking(dummy) is False

    def test_running_request_with_a_publish_in_flight_is_not_tracked_but_is_in_flight(self, rig):
        """A publish does not own a running request: it stays cancellable (``is_tracking`` is
        False) while its pages stay protected (``inflight_request_ids`` names it)."""
        req = make_request(1, 100)
        rig.publish(req)  # still running: not finished, publish IN_FLIGHT
        assert rig.hooks.is_tracking(req) is False
        assert rig.hooks.inflight_request_ids() == {1}
        assert rig.executor._try_cancel_request(req) is True

    def test_has_transfer_in_flight_reports_fetches_and_publishes(self, rig):
        assert rig.hooks.has_transfer_in_flight() is False
        req = make_request(1, 100)
        rig.plan(req)  # planned, not launched
        assert rig.hooks.has_transfer_in_flight() is False
        rig.reserve(req, 96)
        attempt = rig.launch(req)
        assert rig.hooks.has_transfer_in_flight() is True
        attempt.deliver_all()
        rig.advance(req)
        assert rig.hooks.has_transfer_in_flight() is False
        pub = rig.publish(req)
        assert rig.hooks.has_transfer_in_flight() is True
        pub.deliver_all()
        rig.advance()
        assert rig.hooks.has_transfer_in_flight() is False

    def test_pace_idle_sleeps_only_when_a_backend_can_make_progress(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
        rig = Rig(probe_answer=None)  # the store never answers: the request is deferred
        rig.hooks.pace_idle()
        assert sleeps == []
        req = make_request(1, 100)
        rig.advance(req)
        assert rig.hooks.plan_fetch(req) is DEFER
        rig.hooks.pace_idle()
        assert sleeps == [0.001]
        # An in-flight transfer paces too.
        rig2 = Rig()
        req2 = make_request(2, 100)
        rig2.plan_reserve_launch(req2)
        rig2.hooks.pace_idle()
        assert sleeps == [0.001, 0.001]
        rig2.store.attempts[-1].deliver_all()
        rig2.advance(req2)
        rig2.hooks.pace_idle()
        assert sleeps == [0.001, 0.001]

    def test_candidates_are_first_chunk_context_init_non_dummy_only(self, rig):
        first = make_request(1, 100)
        dummy = make_request(2, 100)
        dummy.is_dummy_request = True
        continuation = make_request(3, 100)
        continuation.context_chunk_size = 32
        continuation.move_to_next_context_chunk()
        assert not continuation.is_first_context_chunk
        gen_init = make_request(4, 100)
        gen_init.state = LlmRequestState.DISAGG_GENERATION_INIT
        short = make_request(5, 20)  # nothing nameable: planned None, never probed

        rig.advance(first, dummy, continuation, gen_init, short)

        assert isinstance(rig.hooks.plan_fetch(first), FetchPlan)
        assert rig.hooks.plan_fetch(short) is None
        assert rig.hooks.plan_fetch(gen_init) is None  # plan §9 rule 1
        probed = {name for _, (name, _) in ((m, a) for m, a in rig.store.calls if m == "probe")}
        assert len(probed) == 1  # only ``first`` reached the store
        for other in (dummy, continuation):
            assert rig.coord.plan_fetch(EngineRequestView(other)) is DEFER  # never planned


# =============================================================================================
# close(): order, timeout, status dump
# =============================================================================================


def wait_for_closer_thread_to_exit(timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while any(t.name == "kv-transfer-close" for t in threading.enumerate()):
        assert time.monotonic() < deadline, "the close thread did not exit"
        time.sleep(0.01)


class TestClose:
    def test_close_stops_backends_then_frees_held_and_parked_then_writes_the_dump(
        self, tmp_path, monkeypatch
    ):
        dump_path = tmp_path / "kvt.json"
        rig = Rig(status_dump_path=str(dump_path))
        held = make_request(1, 100)
        rig.publish(held)
        rig.executor._terminate_request(held)
        parked = make_request(2, 100)
        rig.plan_reserve_launch(parked)
        assert rig.coord.held_request_ids() == {1} and rig.coord.parked_request_ids() == {2}
        real_write = rig.hooks._write_status_dump
        monkeypatch.setattr(
            rig.hooks,
            "_write_status_dump",
            lambda: (rig.trace.append("dump"), real_write())[1],
        )

        rig.hooks.close()

        assert rig.trace[0] == "backend.close"
        assert sorted(rig.trace[1:3]) == [("free", 1), ("free", 2)]
        assert rig.trace[3] == "dump"
        assert rig.executor._do_terminate_request.call_count == 0  # not through terminate
        assert dump_path.exists()
        # Idempotent: a second close does nothing more.
        rig.hooks.close()
        assert rig.trace.count("backend.close") == 1
        assert rig.executor._free_request_resources.call_count == 2

    def test_close_with_nothing_held_frees_nothing(self, rig):
        rig.hooks.close()
        assert rig.trace == ["backend.close"]
        rig.executor._free_request_resources.assert_not_called()

    def test_close_gives_up_a_hanging_backend_after_the_timeout(self, tmp_path, hooks_logger):
        """The backends did not stop: a request whose record is still in flight keeps its pages
        (a backend may still be writing them); one whose transfers are all over is freed; the
        dump is still written."""
        hanging = HangingClose()
        dump_path = tmp_path / "kvt.json"
        rig = Rig(close_timeout_s=0.2, backend_close=hanging, status_dump_path=str(dump_path))
        held = make_request(1, 100)
        rig.publish(held)
        rig.executor._terminate_request(held)
        parked = make_request(2, 100)
        rig.plan_reserve_launch(parked)
        try:
            started = time.monotonic()
            rig.hooks.close()
            elapsed = time.monotonic() - started
            assert hanging.entered.is_set()
            assert 0.2 <= elapsed < 5.0
            hooks_logger.error.assert_called_once()
            assert "did not close within" in hooks_logger.error.call_args.args[0]
            kept = sorted(c.args[1] for c in hooks_logger.warning.call_args_list)
            assert kept == [1, 2]
            rig.executor._free_request_resources.assert_not_called()
            assert dump_path.exists()
            assert 1 in rig.kv.kv_cache_map and 2 in rig.kv.kv_cache_map
        finally:
            hanging.gate.set()
            wait_for_closer_thread_to_exit()

    def test_close_within_the_timeout_logs_no_error(self, hooks_logger):
        rig = Rig(close_timeout_s=5.0)
        rig.hooks.close()
        hooks_logger.error.assert_not_called()
        wait_for_closer_thread_to_exit()

    def test_status_dump_schema(self, tmp_path):
        dump_path = tmp_path / "kvt.json"
        rig = Rig(status_dump_path=str(dump_path))
        held = make_request(1, 100)
        rig.publish(held)
        rig.executor._terminate_request(held)
        parked = make_request(2, 100)
        rig.plan_reserve_launch(parked)

        rig.hooks.close()

        with open(dump_path, encoding="utf-8") as f:
            dump = json.load(f)
        assert set(dump) == {"started_at", "pid", "rank", "coordinator", "backends"}
        assert dump["started_at"] == 123.5
        import os

        assert dump["pid"] == os.getpid()
        assert dump["rank"] is None  # the rig does not name a rank; the assembly does
        coordinator = dump["coordinator"]
        assert set(coordinator) == {
            "plan_authority",
            "records",
            "decided_plans",
            "finished_pending",
        }
        assert coordinator["plan_authority"] == "VOTED"
        assert coordinator["finished_pending"] == [1]
        assert isinstance(coordinator["decided_plans"], int)
        records = {(r["request_id"], r["direction"]): r for r in coordinator["records"]}
        assert set(records) == {(1, "publish"), (2, "fetch")}
        for record in records.values():
            assert set(record) == {
                "request_id",
                "direction",
                "state",
                "try_index",
                "attempts",
                "outcomes",
                "deadline",
                "abandoned",
                "token_end",
                "launch_gave_up",
                "peer_launched_at",
                "has_landing",
                "waiting_since",
            }
        assert records[(1, "publish")]["state"] == "IN_FLIGHT"
        assert records[(2, "fetch")]["state"] == "IN_FLIGHT"
        assert records[(2, "fetch")]["token_end"] == 96
        assert records[(1, "publish")]["token_end"] is None
        assert dump["backends"] == [
            {
                "name": "store",
                "type": "fake",
                "roles": ["fetch", "publish"],
                "landing": "device",  # the handle's default; a host-first backend says "host"
                "counters": {"fetch_hits": 1, "publish_stored": 1},  # read at dump time
            }
        ]

    def test_no_dump_path_writes_nothing(self, tmp_path):
        rig = Rig(status_dump_path=None)
        rig.hooks.close()
        assert list(tmp_path.iterdir()) == []

    def test_status_dump_is_readable_before_close(self, rig):
        dump = rig.hooks.status_dump()
        assert dump["coordinator"] == {
            "plan_authority": "VOTED",
            "records": [],
            "decided_plans": 0,
            "finished_pending": [],
        }
        assert dump["backends"][0]["name"] == "store"


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
        assert not rig.hooks.is_tracking(req)
        assert rig.records() == [] and rig.coord.status_dump()["finished_pending"] == []
        for _ in range(2):  # nothing left that could terminate it again
            rig.advance()
        assert rig.terminations() == 1

    def test_publish_expiry_on_a_held_request_warns_and_terminates_once(self, monkeypatch, caplog):
        now = [5000.0]
        monkeypatch.setattr(time, "monotonic", lambda: now[0])
        rig = Rig(publish_timeout_s=10.0)
        rig.wire_engine_error_path()
        req = make_request(1, 100)
        rig.publish(req)  # deadline: 5010
        rig.executor._terminate_request(req)
        assert rig.coord.held_request_ids() == {1}
        now[0] += 9.0
        rig.advance()
        assert rig.terminations() == 0 and rig.coord.held_request_ids() == {1}

        now[0] += 1.5  # past the deadline, no outcome
        with caplog.at_level("WARNING"):
            rig.advance()

        rig.executor._handle_errors.assert_not_called()
        assert any("kv publish timed out" in r.getMessage() for r in caplog.records)
        rig.executor._do_terminate_request.assert_called_once_with(req)
        assert rig.coord.held_request_ids() == frozenset()
        assert rig.records() == [] and rig.coord.status_dump()["finished_pending"] == []
        assert rig.publisher.count("quiesce") == 1  # the pages were vouched for first


class TestFetchExpiry:
    """A normal fetch past ``fetch_timeout_s``: the request fails through the engine's error
    path, but its pages stay held until the backend's outcome arrives (or ``close`` frees them);
    a Delivered that arrives after the expiry no longer lands anything."""

    def _expire_parked_fetch(self, rig, now):
        rig.wire_engine_error_path()
        req = make_request(1, 100)
        rig.executor.active_requests = [req]
        plan, attempt = rig.plan_reserve_launch(req)  # deadline: now + 10
        now[0] += 10.5
        rig.advance(req)
        rig.executor._handle_errors.assert_called_once()
        assert rig.executor._handle_errors.call_args.args[0] == "kv fetch timed out"
        assert rig.executor._handle_errors.call_args.kwargs["requests"] == [req]
        # The engine error path reached the gate, which held the request: its pages may still
        # be written by the backend.
        assert rig.terminations() == 0
        assert rig.coord.held_request_ids() == {1}
        assert req.state == KV_PUBLISH_IN_PROGRESS
        assert 1 in rig.kv.kv_cache_map
        rig.executor._revert_ctx_alloc.assert_not_called()
        return req, attempt

    def test_expired_fetch_fails_the_request_and_holds_its_pages(self, monkeypatch):
        now = [5000.0]
        monkeypatch.setattr(time, "monotonic", lambda: now[0])
        rig = Rig(fetch_timeout_s=10.0)
        req, attempt = self._expire_parked_fetch(rig, now)
        attempt.deliver_all()  # late: the request is gone
        rig.advance()
        rig.executor._do_terminate_request.assert_called_once_with(req)
        assert req.context_current_position == 0  # never unparked
        assert rig.kv.count("try_commit_blocks") == 0
        assert rig.records() == [] and not rig.hooks.is_tracking(req)

    def test_expired_fetch_without_an_outcome_is_freed_by_close(self, monkeypatch, tmp_path):
        now = [5000.0]
        monkeypatch.setattr(time, "monotonic", lambda: now[0])
        rig = Rig(fetch_timeout_s=10.0, status_dump_path=str(tmp_path / "kvt.json"))
        req, _ = self._expire_parked_fetch(rig, now)
        rig.hooks.close()
        rig.executor._free_request_resources.assert_called_once_with(req)
        assert rig.terminations() == 0


class TestRejectedSubmissions:
    def test_consecutive_rejections_are_capped_then_the_request_computes_locally(self, rig):
        """Back-pressure is a reason to try again, not forever: after a bounded run of
        ``SubmissionRejected`` the rank gives the plan up and answers the scheduler DEFER; the
        next round's agreement spends the retry on a fresh plan, a second run gives that up
        too, and the agreement after it settles on local compute."""
        rig.store.reject_next = 100
        req = make_request(1, 100)
        for _ in range(12):
            rig.advance(req)
            plan = rig.hooks.plan_fetch(req)
            if plan is None:
                break
            if plan is DEFER:
                continue  # given up: waiting for the agreement, nothing reserved this round
            assert isinstance(plan, FetchPlan)
            rig.reserve(req, plan.token_end)
            rig.hooks.launch_reserved_fetches([req])
        assert rig.hooks.plan_fetch(req) is None, "rejections were never capped"
        assert rig.store.count("fetch") == 6  # 3 per plan, two plans
        assert rig.store.attempts == []  # nothing escaped
        assert rig.executor._revert_ctx_alloc.call_count == rig.store.count("fetch")
        assert req.state == CONTEXT_INIT and not rig.hooks.is_tracking(req)
        assert rig.records() == []

    def test_publish_expiry_of_a_running_request_only_warns(self, monkeypatch):
        now = [5000.0]
        monkeypatch.setattr(time, "monotonic", lambda: now[0])
        rig = Rig(publish_timeout_s=10.0)
        rig.wire_engine_error_path()
        req = make_request(1, 100)
        rig.publish(req)
        now[0] += 11.0
        rig.advance()
        rig.executor._handle_errors.assert_not_called()
        assert rig.records() == []
        rig.executor._terminate_request(req)  # ends normally later
        assert rig.terminations() == 1


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
        assert rig.coord.held_request_ids() == {1} and rig.coord.parked_request_ids() == frozenset()
        assert req.state == KV_PUBLISH_IN_PROGRESS
        return req, attempt

    def test_parked_request_is_held_then_terminated_once_when_the_outcome_arrives(self, rig):
        req, attempt = self._park_then_fail_fatally(rig)
        attempt.deliver_all()
        rig.advance()
        rig.executor._do_terminate_request.assert_called_once_with(req)
        rig.executor._revert_ctx_alloc.assert_not_called()  # a gone request gets no give-back
        assert not rig.hooks.is_tracking(req) and rig.records() == []

    def test_parked_request_without_an_outcome_is_freed_by_close(self, tmp_path):
        rig = Rig(status_dump_path=str(tmp_path / "kvt.json"))
        req, _ = self._park_then_fail_fatally(rig)
        rig.hooks.close()  # backends close in time: the pages are safe to free
        rig.executor._free_request_resources.assert_called_once_with(req)
        assert rig.terminations() == 0
        with open(tmp_path / "kvt.json", encoding="utf-8") as f:
            dump = json.load(f)
        assert dump["coordinator"]["finished_pending"] == [1]
        assert [r["state"] for r in dump["coordinator"]["records"]] == ["IN_FLIGHT"]


class TestRecomputePauseProtection:
    """A request whose publish is in flight must keep its pages: the executor's recompute-pause
    teardown skips it (the scheduler never picks it as a victim either; see the scheduler seam
    tests)."""

    def test_terminate_recompute_paused_requests_skips_a_request_with_a_publish_in_flight(
        self, rig
    ):
        publishing = make_request(1, 100)
        rig.publish(publishing)  # still running: not finished, publish IN_FLIGHT
        assert 1 in rig.coord.inflight_request_ids()
        plain = make_request(2, 100)
        batch = SimpleNamespace(recompute_paused_requests=[publishing, plain])

        rig.executor._terminate_recompute_paused_requests(batch)

        rig.executor._free_request_resources.assert_called_once_with(plain)
        assert 1 in rig.kv.kv_cache_map

    def test_terminate_recompute_paused_requests_frees_once_the_publish_landed(self, rig):
        req = make_request(1, 100)
        attempt = rig.publish(req)
        attempt.deliver_all()
        rig.advance()
        assert rig.coord.inflight_request_ids() == frozenset()
        rig.executor._terminate_recompute_paused_requests(
            SimpleNamespace(recompute_paused_requests=[req])
        )
        rig.executor._free_request_resources.assert_called_once_with(req)


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
        assert rig.records() == [] and not rig.hooks.is_tracking(req)
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
        assert req.state == CONTEXT_INIT and not rig.hooks.is_tracking(req)
        assert self._cancel_pass(rig, req) is True
        rig.executor._do_terminate_request.assert_called_once_with(req)
        assert rig.records() == []  # the PLANNED retry record is released with the request
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
        assert req.state == KV_PUBLISH_IN_PROGRESS
        assert rig.coord.held_request_ids() == {1}

        attempt.deliver_all()
        rig.advance()
        rig.executor._do_terminate_request.assert_called_once_with(req)
        assert rig.records() == [] and not rig.hooks.is_tracking(req)

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
        assert not rig.hooks.is_tracking(req)


class TestCandidatesAndPublishers:
    def test_plan_fetch_answers_none_for_every_non_candidate(self, rig):
        """Scenario: a planner is attached, yet dummies, chunk continuations, gen-init and
        DISAGG_CONTEXT_INIT_AND_TRANS requests take the ordinary path: ``plan_fetch`` is None."""
        dummy = make_request(1, 100)
        dummy.is_dummy_request = True
        continuation = make_request(2, 100)
        continuation.context_chunk_size = 32
        continuation.move_to_next_context_chunk()
        gen_init = make_request(3, 100)
        gen_init.state = LlmRequestState.DISAGG_GENERATION_INIT
        ctx_and_trans = make_request(4, 100)
        ctx_and_trans.state = LlmRequestState.DISAGG_CONTEXT_INIT_AND_TRANS
        for request in (dummy, continuation, gen_init, ctx_and_trans):
            assert rig.hooks.plan_fetch(request) is None
            rig.advance(request)
            assert rig.hooks.plan_fetch(request) is None
        assert rig.store.count("probe") == 0
        assert rig.coord.status_dump()["decided_plans"] == 0

    def test_max_tokens_one_request_is_published(self, rig):
        """A request that finishes with its first token still publishes: its prefill ended and
        its pages are there (rule 4 does not look at the state)."""
        req = make_request(1, 100, max_new_tokens=1)
        finish_prefill(req)
        req.py_decoding_iter = 1
        req.state = LlmRequestState.GENERATION_COMPLETE
        rig.kv.kv_cache_map[1] = FakeKVCache(history_length=100)
        rig.hooks.publish_committed_blocks([req])
        assert rig.publisher.count("publish") == 1
        extent = rig.publisher.attempts[0].payload
        assert extent.is_last and len(extent.units) == 3  # (100 - 1) // 32 nameable blocks


# =============================================================================================
# Host-first fetch through the scheduler seam
# =============================================================================================


class TestHostFirstSchedulerSeam:
    """A ``LandsOnHost`` store: the landing starts when the plan is decided, with no pages; the
    scheduler is asked to reserve only once the landing is complete, and keeps being asked
    while it cannot; the placement into the pages then parks the request as a fetch does."""

    def scheduler_round(self, rig: Rig, req) -> bool:
        """The scheduler's fetch path for one request in one round: ask for the plan, reserve
        pages for it, queue the request for launch when the reservation went through."""
        plan = rig.hooks.plan_fetch(req)
        if not isinstance(plan, FetchPlan):
            return False
        if not rig.kv.reserve_transfer_pages(req, plan.token_end):
            return False
        req.py_ctx_pre_resize_cap = 0
        rig.slots.add(req)
        rig.hooks.launch_reserved_fetches([req])
        return True

    def test_landing_starts_with_the_plan_and_wants_no_pages_yet(self):
        rig = Rig(host_first=True)
        req = make_request(1, 100)
        rig.advance(req)
        assert rig.store.count("fetch_to_host") == 1 and rig.hooks.plan_fetch(req) is DEFER
        assert rig.kv.count("reserve_transfer_pages") == 0
        assert req.state == CONTEXT_INIT and not rig.hooks.is_tracking(req)
        assert rig.hooks.has_transfer_in_flight()  # a landing paces the idle loop ...
        assert rig.hooks.inflight_request_ids() == frozenset()  # ... but protects no page
        assert [r["state"] for r in rig.records()] == ["STAGING"]
        assert rig.executor._try_cancel_request(req) is True  # not parked: cancellable

    def test_staged_request_is_asked_for_pages_every_round_and_placed_once_they_come(self):
        rig = Rig(host_first=True)
        req = make_request(1, 100)
        rig.advance(req)
        landing = rig.store.landings[0]
        landing.deliver_all()
        rig.advance(req)
        assert [r["state"] for r in rig.records()] == ["STAGED"]

        rig.kv.reserve_answer = False
        for _ in range(3):
            assert self.scheduler_round(rig, req) is False
            rig.advance(req)
        assert rig.kv.count("reserve_transfer_pages") == 3
        assert [r["state"] for r in rig.records()] == ["STAGED"]
        assert rig.store.count("place") == 0 and req.state == CONTEXT_INIT
        assert landing.releases == 0

        rig.kv.reserve_answer = True
        assert self.scheduler_round(rig, req) is True
        assert rig.store.count("place") == 1
        assert req.state == KV_FETCH_IN_PROGRESS and rig.coord.parked_request_ids() == {1}
        assert rig.hooks.is_tracking(req) and rig.hooks.inflight_request_ids() == {1}
        (placed,) = rig.store.attempts
        assert extent_names(placed.payload) == rig.reader.unit_names(req, range(3))

        placed.deliver_all()
        rig.advance(req)
        assert req.state == CONTEXT_INIT and req.context_current_position == 96
        assert landing.releases == 1 and not rig.hooks.is_tracking(req)
        assert [r["state"] for r in rig.records()] == ["LANDED"]

    def test_request_finished_while_staging_is_not_held_and_releases_at_once(self):
        rig = Rig(host_first=True)
        req = make_request(1, 100)
        rig.advance(req)
        landing = rig.store.landings[0]
        rig.executor._terminate_request(req)
        rig.executor._do_terminate_request.assert_called_once_with(req)  # not held
        assert rig.coord.held_request_ids() == frozenset()
        assert landing.releases == 1 and rig.records() == []
        assert rig.reader.forgotten == [1]
        rig.advance()
        assert rig.terminations() == 1

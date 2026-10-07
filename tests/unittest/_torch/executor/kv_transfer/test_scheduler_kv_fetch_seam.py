# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The KV fetch seam of ``KVCacheV2Scheduler``.

The scheduler asks a duck-typed ``kv_transfer`` about every first-chunk context request
before it prepares any cache for it. ``DEFER`` skips the request this round at no cost; a plan
reserves pages with ``reserve_transfer_pages(req, token_end)`` and puts the request on
``fetch_launch_queue`` outside the forward-pass budget; ``None`` takes the ordinary path. The two
pre-existing ``continue`` statements of the pending-context loop still come first, and the module
imports nothing of the coordination layer.

Mocks in the style of ``tests/unittest/_torch/executor/kv_cache/test_kv_cache_v2_scheduler.py``.
"""

import os
import subprocess
import sys
import textwrap
from unittest.mock import Mock, patch

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import BlockReusePolicy
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import SchedulerOutput
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler
from tensorrt_llm.llmapi.llm_args import CapacitySchedulerPolicy, ContextChunkingPolicy

pytestmark = pytest.mark.cpu_only

CONTEXT_INIT = LlmRequestState.CONTEXT_INIT.value
DISAGG_GEN_INIT = LlmRequestState.DISAGG_GENERATION_INIT.value
GEN_IN_PROGRESS = LlmRequestState.GENERATION_IN_PROGRESS.value
TPB = 32


# ---------------------------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------------------------


def make_ctx_request(request_id: int, prompt_len: int, *, is_first_context_chunk: bool = True):
    req = Mock()
    req.request_id = request_id
    req.py_request_id = request_id
    req.state_value = CONTEXT_INIT
    req.state = LlmRequestState.CONTEXT_INIT
    req.prompt_len = prompt_len
    req.context_remaining_length = prompt_len
    req.context_current_position = 0
    req.py_connector_served_position = 0
    req.expect_snapshot_points = []
    req.num_draft_tokens = 0
    req.has_draft_tokens = False
    req.py_draft_tokens = []
    req.is_first_context_chunk = is_first_context_chunk
    req.is_last_context_chunk = True
    req.context_chunk_size = 0
    req.lora_task_id = None
    req.is_context_init_state = True
    req.is_generation_in_progress_state = False
    req.is_disagg_generation_init_state = False
    req.is_dummy_request = False
    req.encoder_output_len = None
    req.py_encoder_output_ready_event = None
    req.py_skip_cross_kv_projection = False
    req.py_multimodal_data = None
    return req


def make_gen_request(request_id: int):
    req = Mock()
    req.request_id = request_id
    req.py_request_id = request_id
    req.state_value = GEN_IN_PROGRESS
    req.get_beam_width_by_iter.return_value = 1
    req.num_draft_tokens = 0
    req.has_draft_tokens = False
    req.py_draft_tokens = []
    req.lora_task_id = None
    req.is_context_init_state = False
    req.is_generation_in_progress_state = True
    req.is_first_context_chunk = False
    req.py_encoder_output_ready_event = None
    req.py_multimodal_data = None
    return req


def make_disagg_gen_init_request(request_id: int, prompt_len: int):
    req = Mock()
    req.request_id = request_id
    req.py_request_id = request_id
    req.state_value = DISAGG_GEN_INIT
    req.context_remaining_length = prompt_len
    req.prompt_len = prompt_len
    req.is_context_init_state = False
    req.is_generation_in_progress_state = False
    req.is_first_context_chunk = True
    req.is_disagg_generation_init_state = True
    req.lora_task_id = None
    req.num_draft_tokens = 0
    req.has_draft_tokens = False
    req.py_draft_tokens = []
    return req


class _KVCacheMap(dict):
    def __missing__(self, key):
        entry = Mock()
        entry.is_active = True
        self[key] = entry
        return entry


def make_kv_cache_manager(
    *,
    reserve_transfer_pages_fn=None,
    enable_block_reuse: bool = False,
    block_reuse_policy=BlockReusePolicy.PER_REQUEST,
    first_new_block_fn=None,
):
    mgr = Mock()
    mgr.tokens_per_block = TPB
    mgr.block_reuse_policy = block_reuse_policy
    mgr.enable_partial_reuse = True
    mgr.enable_joint_kv_cache_reuse = False
    mgr.enable_block_reuse = enable_block_reuse
    mgr.num_extra_kv_tokens = 0
    mgr.can_evict = False
    mgr._has_cp_helix = False
    mgr.is_vswa = False
    # No KV connector: a Mock here would read as a pending load on every request and exempt
    # it from eviction. No tier below GPU: the context path's page preemption stays off.
    mgr.kv_connector_manager = None
    mgr.has_cache_tier_below_gpu = True
    mgr.probe_first_new_block_key.side_effect = first_new_block_fn or (lambda req: None)
    mgr.kv_cache_map = _KVCacheMap()
    mgr.prepare_context.side_effect = lambda req, reuse_limit=None: True
    mgr.prepare_context_cache.side_effect = lambda req, reuse_limit=None: 0
    mgr.reuse_match_backoff = 0
    mgr.probe_context_reuse.side_effect = lambda req: None
    mgr.resize_context.side_effect = lambda req, n: True
    mgr.try_allocate_draft_context.side_effect = lambda req, n: True
    mgr.reserve_transfer_pages.side_effect = reserve_transfer_pages_fn or (
        lambda req, token_end: True
    )
    mgr.prepare_disagg_gen_init.side_effect = lambda req: True
    mgr.try_allocate_generation.side_effect = lambda req: True
    mgr._resume_and_restore.return_value = True
    mgr.is_request_active.side_effect = lambda req_id: mgr.kv_cache_map[req_id].is_active
    return mgr


def make_scheduler(
    kv_cache_manager,
    *,
    max_batch_size: int = 100,
    max_num_tokens=1024,
    ctx_chunk_config=None,
    enable_prefix_aware_scheduling: bool = True,
) -> KVCacheV2Scheduler:
    with patch(
        "tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2.KVCacheManagerV2",
        new=type(kv_cache_manager),
    ):
        return KVCacheV2Scheduler(
            max_batch_size=max_batch_size,
            max_num_tokens=max_num_tokens,
            kv_cache_manager=kv_cache_manager,
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
            ctx_chunk_config=ctx_chunk_config,
            enable_prefix_aware_scheduling=enable_prefix_aware_scheduling,
            enable_recompute_pause=True,
        )


class FakePlan:
    def __init__(self, token_end: int) -> None:
        self.token_end = token_end


class FakePlanner:
    """Duck-typed ``kv_transfer``: ``DEFER`` is its own sentinel, compared by identity."""

    DEFER = object()

    def __init__(self, answers=None) -> None:
        self.answers = dict(answers or {})
        self.asked: list[int] = []

    def fetch_answer(self, req):
        self.asked.append(req.py_request_id)
        return self.answers.get(req.py_request_id)


def ids(requests) -> list[int]:
    return [r.py_request_id for r in requests]


# ---------------------------------------------------------------------------------------------
# No planner / the output field
# ---------------------------------------------------------------------------------------------


def test_scheduler_output_has_an_empty_fetch_launch_queue_by_default():
    out = SchedulerOutput([], [], [], [], [], 0)
    assert out.fetch_launch_queue == []
    assert out.recompute_paused_requests == []
    # A fresh list each time: V1 schedulers must not share one mutable default.
    assert out.fetch_launch_queue is not SchedulerOutput([], [], [], [], [], 0).fetch_launch_queue


def test_without_a_planner_every_context_request_takes_the_normal_path():
    mgr = make_kv_cache_manager()
    sched = make_scheduler(mgr)
    assert sched.kv_transfer is None
    req = make_ctx_request(1, 100)
    out = sched.schedule_request([req], set())
    assert ids(out.context_requests) == [1]
    assert out.fetch_launch_queue == []
    mgr.prepare_context.assert_called_once()
    mgr.reserve_transfer_pages.assert_not_called()


# ---------------------------------------------------------------------------------------------
# DEFER / plan / None
# ---------------------------------------------------------------------------------------------


def test_defer_skips_the_request_this_round_without_preparing_a_cache():
    mgr = make_kv_cache_manager()
    sched = make_scheduler(mgr)
    planner = FakePlanner({1: FakePlanner.DEFER})
    sched.kv_transfer = planner
    deferred, other = make_ctx_request(1, 100), make_ctx_request(2, 100)

    out = sched.schedule_request([deferred, other], set())

    assert planner.asked == [1, 2]
    assert ids(out.context_requests) == [2]  # the deferred one stays in CONTEXT_INIT
    assert out.fetch_launch_queue == []
    # No cache was created, resumed or resized for the deferred request.
    assert 1 not in mgr.kv_cache_map
    for call_args in mgr.prepare_context.call_args_list:
        assert call_args.args[0] is not deferred
    mgr.reserve_transfer_pages.assert_not_called()


def test_plan_reserves_pages_to_token_end_and_queues_the_request_for_launch():
    mgr = make_kv_cache_manager()
    sched = make_scheduler(mgr)
    planner = FakePlanner({1: FakePlan(token_end=96)})
    sched.kv_transfer = planner
    req = make_ctx_request(1, 100)

    out = sched.schedule_request([req], set())

    mgr.reserve_transfer_pages.assert_called_once_with(req, 96)
    assert ids(out.fetch_launch_queue) == [1]
    assert out.context_requests == [] and out.generation_requests == []
    # The ordinary context admission never ran for it.
    mgr.prepare_context.assert_not_called()
    mgr.resize_context.assert_not_called()


def test_planned_request_is_exempt_from_the_request_and_token_budgets():
    """Like a disagg gen-init: a reservation for a fetch joins no forward pass this round, so
    it counts toward neither ``num_requests`` nor ``num_tokens``."""
    mgr = make_kv_cache_manager()
    # Room for exactly one request and exactly the other request's tokens.
    sched = make_scheduler(mgr, max_batch_size=1, max_num_tokens=100)
    planner = FakePlanner({1: FakePlan(token_end=64)})
    sched.kv_transfer = planner
    fetching, normal = make_ctx_request(1, 100), make_ctx_request(2, 100)

    out = sched.schedule_request([fetching, normal], set())

    assert ids(out.fetch_launch_queue) == [1]
    assert ids(out.context_requests) == [2]
    assert out.num_fitting_requests == 1


def test_failed_reservation_skips_the_request_and_drops_its_cache():
    mgr = make_kv_cache_manager(reserve_transfer_pages_fn=lambda req, token_end=None: False)
    sched = make_scheduler(mgr)
    planner = FakePlanner({1: FakePlan(token_end=64)})
    sched.kv_transfer = planner
    req = make_ctx_request(1, 100)
    req.context_current_position = 32  # a reuse match the failed reservation left behind
    mgr.kv_cache_map[1]  # the cache it created

    out = sched.schedule_request([req], set())

    mgr.reserve_transfer_pages.assert_called_once_with(req, 64)
    assert out.fetch_launch_queue == [] and out.context_requests == []
    mgr.free_resources.assert_called_once_with(req)
    # rewind_context_after_cache_drop: the cursor is back at the start of the prompt.
    req.set_prepopulated_prompt_len.assert_called_with(0, TPB)
    assert req.context_current_position == 0
    assert req.py_ctx_pre_resize_cap is None
    assert req.state == LlmRequestState.CONTEXT_INIT


def test_failed_reservation_without_a_cache_only_rewinds():
    mgr = make_kv_cache_manager(reserve_transfer_pages_fn=lambda req, token_end=None: False)
    sched = make_scheduler(mgr)
    sched.kv_transfer = FakePlanner({1: FakePlan(token_end=64)})
    req = make_ctx_request(1, 100)

    out = sched.schedule_request([req], set())

    assert out.fetch_launch_queue == [] and out.context_requests == []
    mgr.free_resources.assert_not_called()
    req.set_prepopulated_prompt_len.assert_called_with(0, TPB)


def test_none_answer_takes_the_normal_context_path():
    mgr = make_kv_cache_manager()
    sched = make_scheduler(mgr)
    planner = FakePlanner({1: None})
    sched.kv_transfer = planner
    req = make_ctx_request(1, 100)

    out = sched.schedule_request([req], set())

    assert planner.asked == [1]
    assert ids(out.context_requests) == [1]
    assert out.fetch_launch_queue == []
    mgr.prepare_context.assert_called_once()
    mgr.reserve_transfer_pages.assert_not_called()


def test_planner_is_asked_once_per_request_per_round():
    mgr = make_kv_cache_manager()
    sched = make_scheduler(mgr)
    planner = FakePlanner({1: None, 2: FakePlanner.DEFER, 3: FakePlan(token_end=32)})
    sched.kv_transfer = planner
    reqs = [make_ctx_request(i, 100) for i in (1, 2, 3)]
    sched.schedule_request(reqs, set())
    assert sorted(planner.asked) == [1, 2, 3]
    planner.asked.clear()
    sched.schedule_request(reqs, set())
    assert sorted(planner.asked) == [1, 2, 3]  # a deferred request is asked again next round


# ---------------------------------------------------------------------------------------------
# The deadlock detector
# ---------------------------------------------------------------------------------------------

ROUNDS_PAST_THE_STALL_LIMIT = KVCacheV2Scheduler._DEADLOCK_STALL_ITERS + 1


def test_a_request_deferred_by_the_transfer_layer_is_not_a_scheduling_stall():
    """A deferred request waits on the transfer layer (a store lookup, a landing, the ranks'
    agreement), each bounded by that layer's own clocks: progress in the making, not a stall."""
    mgr = make_kv_cache_manager()
    sched = make_scheduler(mgr)
    sched.kv_transfer = FakePlanner({1: FakePlanner.DEFER})
    req = make_ctx_request(1, 100)
    for _ in range(ROUNDS_PAST_THE_STALL_LIMIT):
        out = sched.schedule_request([req], set())
    assert out.context_requests == [] and out.fetch_launch_queue == []
    assert sched._stalled_schedules == 0


def test_a_reserved_fetch_is_progress_for_the_deadlock_detector():
    mgr = make_kv_cache_manager()
    sched = make_scheduler(mgr)
    sched.kv_transfer = FakePlanner({1: FakePlan(token_end=64)})
    req = make_ctx_request(1, 100)
    for _ in range(ROUNDS_PAST_THE_STALL_LIMIT):
        out = sched.schedule_request([req], set())
    assert ids(out.fetch_launch_queue) == [1] and sched._stalled_schedules == 0


def test_a_fetch_reservation_that_finds_no_pages_is_not_a_scheduling_stall_either():
    """A planned fetch the scheduler finds no pages for is bounded by the transfer layer's wait
    clock (tens of seconds), after which the fetch is given up and the request computes locally;
    the detector's thousand full-speed passes would fire within a second, long before that."""
    mgr = make_kv_cache_manager(reserve_transfer_pages_fn=lambda req, token_end=None: False)
    sched = make_scheduler(mgr)
    sched.kv_transfer = FakePlanner({1: FakePlan(token_end=64)})
    req = make_ctx_request(1, 100)
    for _ in range(ROUNDS_PAST_THE_STALL_LIMIT):
        out = sched.schedule_request([req], set())
    assert out.fetch_launch_queue == [] and out.context_requests == []
    assert sched._stalled_schedules == 0


def test_pool_exhaustion_is_detected_once_the_request_is_back_on_the_local_path():
    """When the transfer layer has given a fetch up (``None``: compute locally), a context
    admission that keeps failing is the stall the detector exists for, exactly as without the
    seam."""
    mgr = make_kv_cache_manager()
    mgr.prepare_context.side_effect = lambda req, reuse_limit=None: False
    sched = make_scheduler(mgr)
    sched.kv_transfer = FakePlanner({1: None})
    req = make_ctx_request(1, 100)
    with pytest.raises(RuntimeError, match="V2 scheduler deadlock"):
        for _ in range(ROUNDS_PAST_THE_STALL_LIMIT):
            sched.schedule_request([req], set())


# ---------------------------------------------------------------------------------------------
# Who is never a candidate
# ---------------------------------------------------------------------------------------------


class FilteringPlanner(FakePlanner):
    """A planner that applies the hooks' real candidate filter before answering a plan."""

    def fetch_answer(self, req):
        from tensorrt_llm._torch.pyexecutor.kv_transfer.hooks import _is_fetch_candidate

        self.asked.append(req.py_request_id)
        if not _is_fetch_candidate(req):
            return None
        return self.answers.get(req.py_request_id)


def test_dummy_request_with_a_planner_attached_is_scheduled_normally():
    """The hooks' candidate filter answers None for a dummy even when a plan would exist; the
    scheduler then takes the ordinary context path for it."""
    mgr = make_kv_cache_manager()
    sched = make_scheduler(mgr)
    planner = FilteringPlanner({1: FakePlan(token_end=64), 2: FakePlan(token_end=64)})
    sched.kv_transfer = planner
    dummy = make_ctx_request(1, 100)
    dummy.is_dummy_request = True
    real = make_ctx_request(2, 100)
    out = sched.schedule_request([dummy, real], set())
    assert planner.asked == [1, 2]
    assert ids(out.context_requests) == [1]  # the dummy computed locally
    assert ids(out.fetch_launch_queue) == [2]  # the real request fetches
    mgr.reserve_transfer_pages.assert_called_once_with(real, 64)


# ---------------------------------------------------------------------------------------------
# A request with a publish in flight keeps its pages (design §4.3)
# ---------------------------------------------------------------------------------------------

PROTECT_99 = dict(protected_from_eviction_request_ids=frozenset({99}))
"""What the executor passes: ``kv_transfer.inflight_request_ids()`` at scheduling time."""


def test_generation_request_with_a_publish_in_flight_is_not_evicted():
    """Pool pressure would evict the started request at the tail; its blocks are being read by
    a backend, so the scheduler must pick nobody and the asking request self-evicts instead."""

    def alloc_fn(req):
        return req.request_id in (0, 99)  # gen1 never fits; the victim itself does

    mgr = make_kv_cache_manager()
    mgr.try_allocate_generation.side_effect = alloc_fn
    sched = make_scheduler(mgr, max_num_tokens=100)
    victim = make_gen_request(99)  # started before gen1: the ordinary eviction victim
    out = sched.schedule_request(
        [make_gen_request(0), victim, make_gen_request(1)], set(), **PROTECT_99
    )
    assert ids(out.generation_requests) == [0, 99]
    assert 99 not in ids(out.paused_requests)
    assert 99 not in ids(out.recompute_paused_requests)
    for call_args in mgr.suspend_request.call_args_list:
        assert call_args.args[0] is not victim
    assert ids(out.paused_requests) == [1]  # gen1 evicted itself


def test_generation_request_with_a_publish_in_flight_is_not_recompute_paused():
    """With a secondary tier the fallback is a full recompute teardown; a request whose pages a
    backend is still reading is not a candidate for that either."""

    def alloc_fn(req):
        return req.request_id in (0, 99)

    mgr = make_kv_cache_manager()
    mgr.can_evict = True
    mgr.try_allocate_generation.side_effect = alloc_fn
    sched = make_scheduler(mgr, max_num_tokens=100)
    victim = make_gen_request(99)
    out = sched.schedule_request(
        [make_gen_request(0), victim, make_gen_request(1)], set(), **PROTECT_99
    )
    assert 99 not in ids(out.recompute_paused_requests)
    assert 99 not in ids(out.paused_requests)
    for call_args in mgr.free_resources.call_args_list:
        assert call_args.args[0] is not victim
    assert ids(out.generation_requests) == [0, 99]


def test_protected_request_that_cannot_allocate_is_progress_for_the_deadlock_detector():
    """Nothing scheduled and nothing evicted, but the one generation request is protected by a
    transfer in flight: that transfer is what will free pages, so this is not a deadlock."""
    mgr = make_kv_cache_manager()
    mgr.try_allocate_generation.side_effect = lambda req: False
    sched = make_scheduler(mgr, max_num_tokens=100)
    protected = make_gen_request(99)
    out = sched.schedule_request([protected], set(), **PROTECT_99)  # no deadlock error
    assert out.generation_requests == []
    assert 99 not in ids(out.paused_requests) and 99 not in ids(out.recompute_paused_requests)
    mgr.suspend_request.assert_not_called()


def test_generation_request_without_a_publish_in_flight_is_evicted_as_before():
    def alloc_fn(req):
        return req.request_id == 0

    mgr = make_kv_cache_manager()
    mgr.try_allocate_generation.side_effect = alloc_fn
    sched = make_scheduler(mgr, max_num_tokens=100)
    victim = make_gen_request(99)
    out = sched.schedule_request([make_gen_request(0), make_gen_request(1), victim], set())
    assert 99 in ids(out.paused_requests)


def test_context_chunk_continuation_is_asked_and_takes_the_normal_path():
    """The candidate rule lives in the planner: the scheduler asks about every pending context
    request, and a chunk continuation is answered None."""
    mgr = make_kv_cache_manager()
    sched = make_scheduler(mgr)
    planner = FakePlanner()
    sched.kv_transfer = planner
    continuation = make_ctx_request(1, 100, is_first_context_chunk=False)
    continuation.context_remaining_length = 40

    out = sched.schedule_request([continuation], set())

    assert planner.asked == [1]
    assert ids(out.context_requests) == [1] and out.fetch_launch_queue == []
    mgr.reserve_transfer_pages.assert_not_called()


def test_disagg_gen_init_request_never_reaches_the_planner():
    mgr = make_kv_cache_manager()
    sched = make_scheduler(mgr)
    planner = FakePlanner()
    sched.kv_transfer = planner
    gen_init = make_disagg_gen_init_request(1, 100)
    out = sched.schedule_request([gen_init], set())
    assert planner.asked == []
    assert ids(out.fitting_disagg_gen_init_requests) == [1]
    mgr.prepare_disagg_gen_init.assert_called_once_with(gen_init)  # the original call shape


def test_generation_request_is_not_asked():
    mgr = make_kv_cache_manager()
    sched = make_scheduler(mgr)
    planner = FakePlanner()
    sched.kv_transfer = planner
    out = sched.schedule_request([make_gen_request(1)], set())
    assert planner.asked == []
    assert ids(out.generation_requests) == [1]


# ---------------------------------------------------------------------------------------------
# The two pre-existing ``continue`` statements come first
# ---------------------------------------------------------------------------------------------


def test_exhausted_chunk_token_budget_skips_the_request_before_the_planner_is_asked():
    mgr = make_kv_cache_manager()
    # Chunking on; one generation request eats the whole token budget in phase 1.
    sched = make_scheduler(
        mgr,
        max_num_tokens=1,
        ctx_chunk_config=(ContextChunkingPolicy.FIRST_COME_FIRST_SERVED, TPB),
    )
    planner = FakePlanner({2: FakePlan(token_end=64)})
    sched.kv_transfer = planner
    gen, ctx = make_gen_request(1), make_ctx_request(2, 100)

    out = sched.schedule_request([gen, ctx], set())

    assert ids(out.generation_requests) == [1]
    assert planner.asked == []  # the chunk-budget ``continue`` came first
    assert out.fetch_launch_queue == [] and out.context_requests == []
    mgr.reserve_transfer_pages.assert_not_called()


def test_contributed_first_block_skips_the_duplicate_before_the_planner_is_asked():
    """Prefix-aware skip (ALL_REUSABLE + reuse): two first-chunk requests share their first new
    block. The first is admitted (planner says compute locally); the duplicate is deferred behind
    it by the pre-existing ``continue``, so the planner never hears about it this round."""
    shared_key = b"block-0"
    mgr = make_kv_cache_manager(
        enable_block_reuse=True,
        block_reuse_policy=BlockReusePolicy.ALL_REUSABLE,
        first_new_block_fn=lambda req: shared_key,
    )
    sched = make_scheduler(mgr)
    assert sched._prefix_skip_enabled
    planner = FakePlanner({1: None, 2: FakePlan(token_end=64)})
    sched.kv_transfer = planner
    first, duplicate = make_ctx_request(1, 100), make_ctx_request(2, 100)

    out = sched.schedule_request([first, duplicate], set())

    assert ids(out.context_requests) == [1]
    assert planner.asked == [1]
    assert out.fetch_launch_queue == []
    mgr.reserve_transfer_pages.assert_not_called()


def test_planner_is_asked_after_the_prefix_probe_but_before_any_cache_work():
    """Order inside the pending-context loop: probe_first_new_block_key (skip check), then the
    planner, then peft/prepare_context. A deferred request pays for the probe and nothing else."""
    order = []
    mgr = make_kv_cache_manager(
        enable_block_reuse=True,
        block_reuse_policy=BlockReusePolicy.ALL_REUSABLE,
        first_new_block_fn=lambda req: order.append(("probe", req.py_request_id)) or None,
    )
    mgr.prepare_context.side_effect = lambda req, reuse_limit=None: (
        order.append(("prepare_context", req.py_request_id)) or True
    )
    sched = make_scheduler(mgr)

    class OrderedPlanner(FakePlanner):
        def fetch_answer(self, req):
            order.append(("fetch_answer", req.py_request_id))
            return super().fetch_answer(req)

    sched.kv_transfer = OrderedPlanner({1: FakePlanner.DEFER, 2: None})
    reqs = [make_ctx_request(1, 100), make_ctx_request(2, 100)]
    sched.schedule_request(reqs, set())
    assert order == [
        ("probe", 1),
        ("fetch_answer", 1),
        ("probe", 2),
        ("fetch_answer", 2),
        ("prepare_context", 2),
    ]


# ---------------------------------------------------------------------------------------------
# Import hygiene
# ---------------------------------------------------------------------------------------------


def test_scheduler_and_executor_modules_do_not_import_the_coordination_layer():
    code = textwrap.dedent(
        """
        import sys
        import tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2
        import tensorrt_llm._torch.pyexecutor.py_executor
        import tensorrt_llm._torch.pyexecutor.py_executor_creator
        banned = (
            "tensorrt_llm._torch.disaggregation.orchestration.kv_transfer",
            "tensorrt_llm._torch.disaggregation.remote_cache",
            "tensorrt_llm._torch.disaggregation.backends",
            "tensorrt_llm._torch.disaggregation.resource.kv_v2",
            "tensorrt_llm._torch.pyexecutor.kv_transfer",
        )
        loaded = sorted(m for m in sys.modules if m.startswith(banned))
        assert not loaded, f"coordination layer imported at module load: {loaded}"
        """
    )
    env = {k: v for k, v in os.environ.items() if k != "TRTLLM_KV_TRANSFER_CONFIG"}
    subprocess.run([sys.executable, "-c", code], check=True, env=env, timeout=600)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The hook points of integration plan §5, read off the engine source.

Each hook is one guarded call in a shared engine file, placed relative to a named neighbour
(after ``poll_gen_transfers``, before ``_send_kv_async``, ...). Running ``_executor_loop`` in a
unit test would need the model engine, sampler and hang detector; the order the plan requires is
a property of the source, so this checks the source: delete or move a hook and a test here fails
and names it. Behaviour of each hook is covered by ``test_effects_binding.py``.
"""

import ast
import inspect
import re

import pytest

from tensorrt_llm._torch.pyexecutor import py_executor, py_executor_creator
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler import scheduler_v2

pytestmark = pytest.mark.cpu_only

GUARD = "self.kv_transfer is not None"


def source_of(method) -> str:
    return inspect.getsource(method)


def ordered(text: str, *needles: str) -> None:
    """Every needle occurs in ``text``, in the given order."""
    position = -1
    for needle in needles:
        found = text.find(needle, position + 1)
        assert found > position, f"{needle!r} missing or out of order"
        position = found


def hook_calls(text: str, method_name: str) -> list[str]:
    return re.findall(rf"self\.kv_transfer\.{method_name}\(", text)


# ---- the loop: advance -> schedule -> launch -> (forward) -> publish -> send ----


def test_advance_runs_at_the_loop_head_after_the_disagg_poll_and_before_scheduling():
    text = source_of(PyExecutor._prepare_and_schedule_batch)
    ordered(
        text,
        "self.disagg.poll_gen_transfers()",
        GUARD,
        "self.kv_transfer.advance_round(self.active_requests)",
        "self.disagg.check_transfer_timeouts()",
        "self._schedule(",
        GUARD,
        "self.kv_transfer.launch_reserved_fetches(",
        "self._kv_fetch_launch_queue",
    )
    assert len(hook_calls(text, "advance_round")) == 1
    assert len(hook_calls(text, "launch_reserved_fetches")) == 1


def test_schedule_hands_the_fetch_launch_queue_to_the_executor():
    text = source_of(PyExecutor._schedule)
    assert "self._kv_fetch_launch_queue = scheduler_output.fetch_launch_queue" in text


def test_schedule_protects_requests_with_a_transfer_in_flight_from_eviction():
    text = source_of(PyExecutor._schedule)
    ordered(
        text,
        GUARD,
        "protected_from_eviction_request_ids=self.kv_transfer.",
        "inflight_request_ids()",
        "self.scheduler.schedule_request(",
    )
    text = source_of(PyExecutor._terminate_recompute_paused_requests)
    ordered(text, GUARD, "self.kv_transfer.inflight_request_ids()", "continue")


def test_publish_follows_the_context_commit_and_precedes_the_disagg_send_in_the_plain_loop():
    text = source_of(PyExecutor._executor_loop)
    ordered(
        text,
        "self._update_requests(sample_state",
        "self._update_v2_context_resources(scheduled_batch)",
        GUARD,
        "self.kv_transfer.publish_committed_blocks(",
        "scheduled_batch.context_requests",
        "self._send_kv_async(scheduled_batch.all_requests())",
        "self.disagg.pace_idle()",
        "self.kv_transfer.pace_idle()",
    )
    assert len(hook_calls(text, "publish_committed_blocks")) == 1


def test_overlap_loop_publishes_only_the_previous_batch():
    """Only ``previous_batch`` is committed and forward-complete at that point (plan §5 #4)."""
    text = source_of(PyExecutor._executor_loop_overlap)
    ordered(
        text,
        "self._update_requests(",
        "self.previous_batch.sample_state",
        GUARD,
        "self.kv_transfer.publish_committed_blocks(",
        "self.previous_batch.scheduled_requests",
        "self._send_kv_async(",
        "self.previous_batch.scheduled_requests.all_requests()",
        "self.disagg.pace_idle()",
        "self.kv_transfer.pace_idle()",
    )
    publish = re.search(
        r"self\.kv_transfer\.publish_committed_blocks\(\s*self\.previous_batch\.scheduled_requests\.\s*context_requests\)",
        text,
    )
    assert publish is not None, "the overlap loop must publish previous_batch only"
    assert len(hook_calls(text, "publish_committed_blocks")) == 1


# ---- release gate, cancel, idle, shutdown ----


def test_release_gate_is_the_first_statement_of_terminate_request():
    tree = ast.parse(source_of(PyExecutor._terminate_request).strip())
    body = tree.body[0].body
    first = body[0]
    text = ast.unparse(first)
    assert isinstance(first, ast.If)
    ordered(
        text,
        GUARD,
        "not request.is_dummy_request",
        "not self.kv_transfer.on_request_finished(request)",
        "return",
    )


def test_cancel_asks_is_tracking_before_the_transceiver():
    text = source_of(PyExecutor._try_cancel_request)
    ordered(
        text,
        GUARD,
        "self.kv_transfer.is_tracking(",
        "return False",
        "if self.kv_cache_transceiver is None:",
    )


def test_idle_detection_counts_a_transfer_in_flight_as_live():
    text = source_of(PyExecutor._fetch_and_enqueue_requests)
    idle = re.search(r"idle = \((.*?)\)\n", text, re.S)
    assert idle is not None
    ordered(
        idle.group(1),
        "total_num_live_requests == 0",
        "not self.is_shutdown",
        GUARD,
        "self.kv_transfer.has_transfer_in_flight()",
    )


def test_shutdown_closes_the_layer_after_the_device_sync_and_before_the_managers():
    text = source_of(PyExecutor.shutdown)
    ordered(
        text,
        "torch.cuda.synchronize()",
        GUARD,
        "self.kv_transfer.close()",
        "for manager in self.resource_manager.resource_managers.values()",
        "manager.shutdown()",
    )


def test_executor_declares_the_class_default_and_the_launch_queue():
    assert PyExecutor.kv_transfer is None
    init = source_of(PyExecutor.__init__)
    assert "self._kv_fetch_launch_queue: List[LlmRequest] = []" in init


def test_every_hook_call_is_guarded():
    """No ``self.kv_transfer.<hook>(`` call without a ``kv_transfer is not None`` guard nearby."""
    text = inspect.getsource(py_executor)
    for match in re.finditer(r"self\.kv_transfer\.\w+\(", text):
        window = text[max(0, match.start() - 400) : match.start()]
        assert GUARD in window, f"unguarded hook at offset {match.start()}: {match.group(0)}"


# ---- assembly and the scheduler seam ----


def test_creator_assembles_lazily_from_the_environment_variable():
    text = source_of(py_executor_creator._create_py_executor_impl)
    ordered(
        text,
        'os.environ.get("TRTLLM_KV_TRANSFER_CONFIG")',
        "from .kv_transfer.assembly import attach_kv_transfer",
        "attach_kv_transfer(",
        "py_executor.start_worker()",
    )
    module_text = inspect.getsource(py_executor_creator)
    assert "kv_transfer.assembly" not in module_text.split("def _create_py_executor_impl")[0]


def test_scheduler_asks_the_planner_after_the_prefix_probe_and_before_any_cache_work():
    text = source_of(scheduler_v2.KVCacheV2Scheduler._schedule_loop)
    ordered(
        text,
        "_has_context_chunk_budget(budget)",
        "probe_first_new_block_key(req)",
        "self._try_take_fetch_path(req)",
        "fetch_launch_queue.append(req)",
        "budget.peft_pages_needed(req)",
        "self._try_schedule_context(req, budget)",
    )
    assert "kv_transfer_planner = None" in source_of(scheduler_v2.KVCacheV2Scheduler.__init__)

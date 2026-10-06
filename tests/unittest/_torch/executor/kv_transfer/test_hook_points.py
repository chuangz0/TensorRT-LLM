# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The KV transfer hook points of the engine loop, read off the engine source.

Each hook is one guarded call in a shared engine file, placed relative to a named neighbour
(after ``poll_gen_transfers``, before ``_send_kv_async``, ...). Running ``_executor_loop`` in a
unit test would need the model engine, sampler and hang detector; the order the design requires
is a property of the source, so this checks the source: delete or move a hook and a test here
fails and names it. The source-text assertions are deliberate, not a stopgap: they pin the
*placement* of each hook, which no behavioural test of the hooks can see, and they are written
against whitespace-collapsed source so that reformatting does not break them. Behaviour of each
hook is covered by ``test_hooks.py``.
"""

import ast
import inspect
import re
import textwrap

import pytest

from tensorrt_llm._torch.pyexecutor import py_executor, py_executor_creator
from tensorrt_llm._torch.pyexecutor.kv_transfer.effects import PyExecutorKVTransferEffects
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler import scheduler_v2

pytestmark = pytest.mark.cpu_only

GUARD = "self.kv_transfer is not None"


def source_of(method) -> str:
    """The method's source with every whitespace run collapsed to one space, so a marker keeps
    matching however the formatter wraps a call."""
    return re.sub(r"\s+", " ", inspect.getsource(method))


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
    text = source_of(PyExecutor._protected_from_eviction_ids)
    ordered(text, GUARD, "return self.kv_transfer.inflight_request_ids()", "return frozenset()")
    text = source_of(PyExecutor._schedule)
    ordered(
        text,
        "protected = self._protected_from_eviction_ids()",
        "self.scheduler.schedule_request(",
        "protected_from_eviction_request_ids=protected)",
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
    """Only ``previous_batch`` is committed and forward-complete at that point."""
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


# ---- the pipeline-parallel loop ----


def test_pp_loop_advances_at_the_head_launches_after_stage_0_and_paces_idle():
    text = source_of(PyExecutor._executor_loop_pp)
    ordered(
        text,
        "if self.should_stop_processing:",
        "break",
        "self.disagg.poll_gen_transfers()",
        GUARD,
        "self.kv_transfer.advance_round(self.active_requests)",
        "self._pad_attention_dp_dummy_request()",
        "self._pp_schedule_and_propagate(microbatch_id)",
        "if self.dist.rank != 0:",
        "protected = self._protected_from_eviction_ids()",
        "protected_from_eviction_request_ids=protected)",
        "self.disagg.revert_deferred_gen_init(",
        GUARD,
        "self.kv_transfer.launch_reserved_fetches( self._kv_fetch_launch_queue if self.dist.rank "
        "== 0 else local_scheduler_output.fetch_launch_queue)",
        "self.disagg.pace_idle()",
        GUARD,
        "self.kv_transfer.pace_idle()",
    )
    assert len(hook_calls(text, "advance_round")) == 1
    assert len(hook_calls(text, "launch_reserved_fetches")) == 1
    assert len(hook_calls(text, "pace_idle")) == 1
    # The one advance sits before any ``continue``/``break`` other than the stop check's.
    head, _, _ = inspect.getsource(PyExecutor._executor_loop_pp).partition(
        "self.kv_transfer.advance_round("
    )
    assert re.findall(r"^\s*(break|continue)\b", head, re.M) == ["break"]


def test_pp_schedule_propagation_exports_on_the_owner_and_adopts_on_the_followers():
    text = source_of(PyExecutor._pp_schedule_and_propagate)
    ordered(
        text,
        "self._schedule(",
        "self.disagg.admit(",
        GUARD,
        "kv_fetch_answers = self.kv_transfer.export_plan_answers()",
        "SerializableSchedulerOutput.from_scheduler_result(",
        "kv_fetch_answers=kv_fetch_answers",
        "if scheduled_batch is None:",
        GUARD,
        "self.kv_transfer.adopt_plan_answers(",
        "self.active_requests, serializable_schedule.kv_fetch_answers",
        "serializable_schedule.to_scheduler_result(",
    )


def test_pp_executed_batch_publishes_after_the_context_commit_and_before_the_disagg_send():
    text = source_of(PyExecutor._handle_executed_batch)
    ordered(
        text,
        "self._update_requests(executed_batch.sample_state)",
        "self._update_v2_context_resources(scheduled_requests)",
        GUARD,
        "self.kv_transfer.publish_committed_blocks(",
        "scheduled_requests.context_requests)",
        "self._send_kv_async(finished_ctx_reqs)",
    )
    assert len(hook_calls(text, "publish_committed_blocks")) == 1


# ---- release gate, cancel, idle, shutdown ----


def test_release_gate_precedes_everything_that_frees_the_request():
    """The gate is the first ``kv_transfer`` statement of ``_terminate_request`` and nothing
    before it terminates. The KV connector block ahead of it only defers: both of its early
    returns re-enter ``_terminate_request`` later (``_finish_connector_load_termination``,
    ``_release_transfer``), so the gate still runs for every termination. The layer's own
    deferred release (``effects.terminate_request``) takes the exits after the gate without
    re-entering it, so a connector statement after the gate would be skipped for a held
    request: the connector block stays ahead of it."""
    source = textwrap.dedent(inspect.getsource(PyExecutor._terminate_request))
    body = ast.parse(source).body[0].body
    statements = [ast.unparse(statement) for statement in body]
    gate_index = next(i for i, text in enumerate(statements) if "kv_transfer" in text)
    assert isinstance(body[gate_index], ast.If)
    ordered(
        statements[gate_index],
        GUARD,
        "not request.is_dummy_request",
        "not self.kv_transfer.on_request_finished(request)",
        "return",
    )
    before = " ".join(statements[:gate_index])
    after = " ".join(statements[gate_index + 1 :])
    for terminator in ("_do_terminate_request", "_disagg_pp_termination_handler"):
        assert terminator not in before, f"{terminator} runs before the release gate"
        assert terminator in after
    assert "kv_connector_manager" not in after, "connector deferral after the gate is skipped"


def test_deferred_release_takes_the_same_exits_as_the_statements_after_the_gate():
    """A held request is terminated by the layer through ``effects.terminate_request``, which
    must choose between the pipeline-parallel termination handler and ``_do_terminate_request``
    exactly as ``_terminate_request`` does after its gate; otherwise a held request under PP
    would be freed on one rank while its peers wait for the ring."""
    gate_exits = source_of(PyExecutor._terminate_request).split("on_request_finished")[1]
    effect = source_of(PyExecutorKVTransferEffects.terminate_request)
    for text in (gate_exits, effect):
        ordered(
            text,
            "_disagg_pp_termination_handler",
            "is not None",
            "is_dummy_request",
            ".terminate(",
            "_do_terminate_request(",
        )


def test_cancel_asks_owns_before_the_transceiver():
    text = source_of(PyExecutor._try_cancel_request)
    ordered(
        text,
        GUARD,
        "self.kv_transfer.owns(",
        "return False",
        "if self.kv_cache_transceiver is None:",
    )


def test_idle_detection_counts_a_transfer_in_flight_as_live():
    text = inspect.getsource(PyExecutor._fetch_and_enqueue_requests)
    idle = re.search(r"idle = \((.*?)\)\n\s*if idle:", text, re.S)
    assert idle is not None
    ordered(
        idle.group(1),
        "total_num_live_requests == 0",
        "not self.is_shutdown",
        "not self._has_pending_connector_transfers()",
        GUARD,
        "self.kv_transfer.has_pending_work()",
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
    """Attached once the final executor exists (after any profiling re-creation) and before its
    worker starts; the assembly module is imported there and nowhere at module load."""
    text = source_of(py_executor_creator._create_py_executor)
    ordered(
        text,
        "py_executor = create_py_executor_instance(",
        "start_worker=False",
        'os.environ.get("TRTLLM_KV_TRANSFER_CONFIG")',
        "from .kv_transfer.assembly import attach_kv_transfer",
        "attach_kv_transfer(",
        "py_executor.start_worker()",
    )
    assert text.count("py_executor.start_worker()") == 1
    module = ast.parse(inspect.getsource(py_executor_creator))
    module_imports = [
        ast.unparse(statement)
        for statement in module.body
        if isinstance(statement, (ast.Import, ast.ImportFrom))
    ]
    assert not [text for text in module_imports if "kv_transfer" in text]


def test_scheduler_asks_the_planner_after_the_prefix_probe_and_before_any_cache_work():
    text = source_of(scheduler_v2.KVCacheV2Scheduler._schedule_loop)
    ordered(
        text,
        "_has_context_chunk_budget(budget)",
        "probe_first_new_block_key(req)",
        "self._try_take_fetch_path(req)",
        "fetch_launch_queue.append(req)",
        "budget.peft_pages_needed(req)",
        "self._try_schedule_context(",
    )
    assert "kv_transfer_hooks = None" in source_of(scheduler_v2.KVCacheV2Scheduler.__init__)

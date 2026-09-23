# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The KV transfer layer as the executor loop and the scheduler see it (design §3.2, plan §4).

``KVTransferEngineBinding`` is the one object the loop calls: it selects the requests each hook
concerns, wraps them in ``EngineRequestView``, forwards to ``KVTransferCoordinator``, answers the
release gate and the cancel path from this layer's own tables, and owns the shutdown order. All
logic lives here or below; the shared engine files hold one guarded call per hook.

Known follow-ups:

1. An empty-ask plan (local reuse already covers every nameable block) still costs one
   park/unpark round, because the decision may not depend on rank-local reuse (design §7.2
   step 6). Follow-up: a consensus-safe short-circuit.
2. On a context-only worker the loop sleeps in ``_fetch_and_enqueue_requests`` once the last
   context-only request has left ``active_requests``, so disagg's release and this layer's
   ``LANDED`` record wait for the next wake. This is disagg's pre-existing idle-wake gap; it is
   not widened here.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import TYPE_CHECKING, Sequence

from tensorrt_llm.logger import logger

from ...disaggregation.backends.config import BackendEntry
from ...disaggregation.backends.registry import BackendHandle, close_backends
from ...disaggregation.orchestration.kv_transfer.coordinator import (
    KVTransferCoordinator,
    PlanAnswers,
)
from ...disaggregation.orchestration.kv_transfer.interfaces import DEFER, PlanAuthority
from ...disaggregation.resource.kv_v2_reader import KVv2ResourceReader
from ..llm_request import LlmRequest, LlmRequestState
from .effects import EngineRequestView, PyExecutorKVTransferEffects

if TYPE_CHECKING:
    from ..py_executor import PyExecutor

__all__ = ["KVTransferEngineBinding"]

_IDLE_BACKEND_WAIT_S = 0.001
"""How long an idle loop pass sleeps while only a backend can make progress."""

_CONTEXT_INIT_STATE_VALUE = LlmRequestState.CONTEXT_INIT.value


class KVTransferEngineBinding:
    """One call per hook point of plan §5; see the naming table of plan §4.

    Args:
        executor: The engine; read for its request lists and its resource release.
        coordinator: The record table and its four phases.
        effects: The engine-side effects; also the parked / held tables the gate reads.
        reader: The resource view, told to forget finished requests.
        backends: The built backends, in config order; closed by ``close``.
        backend_entries: Their config entries, for the status dump.
        close_timeout_s: How long ``close`` waits for the backends before giving them up.
        status_dump_path: Where ``close`` writes the JSON status dump; ``None`` writes nothing.
        started_at: ``time.time()`` at assembly; recorded in the dump so a test can order
            several engines by creation.
        rank: This rank's index in the world, recorded in the dump; ``None`` when unknown.
    """

    DEFER = DEFER
    """``plan_fetch`` answer meaning "skip this round"; the scheduler compares against it."""

    def __init__(
        self,
        executor: PyExecutor,
        coordinator: KVTransferCoordinator,
        effects: PyExecutorKVTransferEffects,
        reader: KVv2ResourceReader,
        backends: Sequence[BackendHandle],
        backend_entries: Sequence[BackendEntry],
        *,
        close_timeout_s: float,
        status_dump_path: str | None = None,
        started_at: float | None = None,
        rank: int | None = None,
    ) -> None:
        self._executor = executor
        self.coordinator = coordinator
        self.effects = effects
        self.reader = reader
        self.backends = tuple(backends)
        self._backend_entries = tuple(backend_entries)
        self._close_timeout_s = close_timeout_s
        self._status_dump_path = status_dump_path
        self._started_at = time.time() if started_at is None else started_at
        self._rank = rank
        self._is_closed = False
        self._num_deferred_requests = 0
        """Requests still undecided after the last advance (or, on a follower, after the last
        adoption); each is waiting on a store lookup."""

    # ---- loop entry points ----

    def advance_round(self, active_requests: Sequence[LlmRequest]) -> None:
        """Loop head (design §3.2 step 1), once per round: pick the undecided candidates, then
        ``coordinator.advance`` reaps landed transfers and plans them. A follower plans nothing
        here; its undecided count is set when it adopts the owner's answers."""
        candidates = self._undecided_candidates(active_requests)
        self.coordinator.advance(candidates, time.monotonic())
        if self.coordinator.plan_authority is not PlanAuthority.FOLLOWER:
            self._num_deferred_requests = sum(
                1 for candidate in candidates if self.coordinator.plan_fetch(candidate) is DEFER
            )

    def export_plan_answers(self) -> PlanAnswers:
        """Owner of the pipeline-parallel loop, after ``advance_round``: this round's decided plan
        answers, to travel with the schedule to the other ranks."""
        return self.coordinator.export_plan_answers()

    def adopt_plan_answers(
        self, active_requests: Sequence[LlmRequest], answers: PlanAnswers
    ) -> None:
        """Follower of the pipeline-parallel loop, before it runs its local scheduler: take the
        owner's answers; every undecided candidate without one stays deferred."""
        candidates = self._undecided_candidates(active_requests)
        self.coordinator.adopt_plan_answers(candidates, answers)
        answered = {request_id for request_id, _ in answers}
        self._num_deferred_requests = sum(
            1 for candidate in candidates if candidate.py_request_id not in answered
        )

    def plan_fetch(self, request: LlmRequest):
        """Scheduler hook (design §5): ``FetchPlan``, ``None`` or ``DEFER`` for ``request``.

        A request that is not a fetch candidate (plan §9 rule 3) computes locally: ``None``.
        """
        if not _is_fetch_candidate(request):
            return None
        return self.coordinator.plan_fetch(EngineRequestView(request))

    def launch_reserved_fetches(self, fetch_launch_queue: Sequence[LlmRequest]) -> None:
        """After scheduling (design §3.2 step 3): start the fetches the scheduler reserved for."""
        if fetch_launch_queue:
            views = [EngineRequestView(request) for request in fetch_launch_queue]
            self.coordinator.launch_fetches(views, time.monotonic())

    def publish_committed_blocks(self, context_requests: Sequence[LlmRequest]) -> None:
        """After a context step whose forward has completed and whose blocks are committed
        (design §3.2 step 4, plan §5 #3-#4): offer the blocks of requests whose prefill ended."""
        completed = self._publishable_completed_contexts(context_requests)
        if completed:
            self.coordinator.publish_committed_blocks(completed, finished=(), now=time.monotonic())

    # ---- release gate, cancel path, idle pacing ----

    def on_request_finished(self, request: LlmRequest) -> bool:
        """The release gate (design §4.3, plan §9): ``True`` when the engine may terminate the
        request now, ``False`` when this layer holds it and will terminate it later."""
        request_id = request.py_request_id
        self.coordinator.notify_request_finished(EngineRequestView(request))
        self.reader.forget_request(request_id)
        return request_id not in self.effects.held_request_ids

    def is_tracking(self, request: LlmRequest) -> bool:
        """Whether this layer has a live record of the request: parked for a fetch, or held for a
        publish. Ownership is read from the tables, never from the request state."""
        request_id = request.py_request_id
        return (
            request_id in self.effects.parked_request_ids
            or request_id in self.effects.held_request_ids
        )

    def has_transfer_in_flight(self) -> bool:
        return self.coordinator.has_inflight()

    def inflight_request_ids(self) -> frozenset[int]:
        """Requests whose pages a backend may still read or write. They stay schedulable, but the
        scheduler must neither evict nor recompute-pause them (``schedule_request``'s
        ``protected_from_eviction_request_ids``), and the engine must not free them behind the
        release gate."""
        return self.coordinator.inflight_request_ids()

    def pace_idle(self) -> None:
        """An idle loop pass that only a backend can unblock (a transfer in flight, or a lookup a
        request is deferred on) yields briefly instead of spinning through the probe budget."""
        if self._num_deferred_requests or self.coordinator.has_inflight():
            time.sleep(_IDLE_BACKEND_WAIT_S)

    # ---- shutdown ----

    def status_dump(self) -> dict:
        return {
            "started_at": self._started_at,
            "pid": os.getpid(),
            "rank": self._rank,
            "coordinator": self.coordinator.status_dump(),
            "backends": [
                {
                    "name": handle.name,
                    "type": entry.type,
                    "roles": sorted(entry.roles),
                    "counters": dict(handle.counters()),
                }
                for handle, entry in zip(self.backends, self._backend_entries)
            ],
        }

    def close(self) -> None:
        """Stop the backends within ``close_timeout_s``, free every request this layer still
        holds or has parked, then write the status dump (plan §5 #9, §9).

        When the backends do not stop in time, a request whose transfer record is still in flight
        keeps its pages: a backend that may still be writing them must not see them freed. Its
        resources are then left to the manager shutdown that follows.
        """
        if self._is_closed:
            return
        self._is_closed = True
        backends_closed = self._close_backends_within_timeout()
        still_in_flight = (
            frozenset() if backends_closed else self.coordinator.inflight_request_ids()
        )
        for request in self.effects.requests_to_release_on_close():
            if request.py_request_id in still_in_flight:
                logger.warning(
                    "kv transfer: request %d keeps its pages, its transfer is still in flight",
                    request.py_request_id,
                )
                continue
            self._executor._free_request_resources(request)
        self._write_status_dump()

    def _close_backends_within_timeout(self) -> bool:
        """A backend's ``close`` waits for its deliveries in flight; nothing below it has a
        timeout of its own, so this bound is the only one. Returns whether they all closed."""
        closer = threading.Thread(
            target=close_backends, args=(self.backends,), name="kv-transfer-close", daemon=True
        )
        closer.start()
        closer.join(self._close_timeout_s)
        if closer.is_alive():
            logger.error(
                "kv transfer: backends did not close within %.1f s; giving them up. The KV pools "
                "are about to be freed while a backend operation may still be in flight; the "
                "backend threads are daemons, so they cannot keep the process from exiting.",
                self._close_timeout_s,
            )
            return False
        return True

    def _write_status_dump(self) -> None:
        if self._status_dump_path is None:
            return
        dump = self.status_dump()
        with open(self._status_dump_path, "w", encoding="utf-8") as dump_file:
            json.dump(dump, dump_file, indent=2, default=str)
        logger.info("kv transfer: status dump written to %s", self._status_dump_path)

    # ---- request selection ----

    def _undecided_candidates(
        self, active_requests: Sequence[LlmRequest]
    ) -> list[EngineRequestView]:
        """Design §3.2 candidates: fetch candidates whose plan is not decided yet."""
        candidates = []
        for request in active_requests:
            if not _is_fetch_candidate(request):
                continue
            view = EngineRequestView(request)
            if self.coordinator.plan_fetch(view) is DEFER:
                candidates.append(view)
        return candidates

    def _publishable_completed_contexts(
        self, context_requests: Sequence[LlmRequest]
    ) -> list[EngineRequestView]:
        """Plan §9 rule 4: prefill ended, pages present and active on the GPU (a failed request
        has none; a suspended cache has left the GPU and the store must not read it), not
        generation-only (its prefix was published by the context worker that computed it), not
        dummy. A request that finished with its first token still publishes."""
        kv_cache_manager = self._executor.kv_cache_manager
        completed = []
        for request in context_requests:
            has_prefill_ended = request.context_remaining_length == 0
            has_active_pages = kv_cache_manager.is_request_active(request.py_request_id)
            is_generation_side = request.is_generation_only_request
            if not has_prefill_ended or not has_active_pages or is_generation_side:
                continue
            if request.is_dummy_request:
                continue
            completed.append(EngineRequestView(request))
        return completed


def _is_fetch_candidate(request: LlmRequest) -> bool:
    """Plan §9 rule 3: exactly ``CONTEXT_INIT`` (not ``DISAGG_CONTEXT_INIT_AND_TRANS``), first
    chunk, not dummy, not gen-init (never in ``CONTEXT_INIT``; the explicit fallback of rule 1)."""
    return (
        request.state_value == _CONTEXT_INIT_STATE_VALUE
        and request.is_first_context_chunk
        and not request.is_dummy_request
        and not request.is_disagg_generation_init_state
    )

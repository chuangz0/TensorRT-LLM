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
"""The engine side of the KV transfer coordination layer's protocols (design §7.3, §12.2).

``EngineRequestView`` is the ``RequestView`` over one ``LlmRequest``; ``PyExecutorKVTransferEffects``
is the ``KVTransferEffects`` over one ``PyExecutor`` and the only place that writes a request's
transfer state; ``EngineWorkQueue`` and ``EngineDist`` complete the contract. The two state
constants spell the coordination layer's states with the alias table of design §12.2.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import TYPE_CHECKING, Callable, Mapping, Sequence

from tensorrt_llm.logger import logger

from ..kv_cache.kv_cache_manager_v2 import settle_context_cursor
from ..llm_request import LlmRequest, LlmRequestState, rewind_context_after_cache_drop
from ..resource_manager import ResourceManagerType

if TYPE_CHECKING:
    from tensorrt_llm.mapping import Mapping as ParallelMapping

    from ..py_executor import PyExecutor

__all__ = [
    "KV_FETCH_IN_PROGRESS",
    "KV_PUBLISH_IN_PROGRESS",
    "EngineDist",
    "EngineRequestView",
    "EngineWorkQueue",
    "PyExecutorKVTransferEffects",
]

# Both aliases lie outside the V2 scheduler's schedulable range. The disagg transceiver writes
# the same two values for its own transfers; which layer owns a request in one of them is decided
# by each layer's own record table (plan §9 rule 5), never by reading the state.
KV_FETCH_IN_PROGRESS = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
"""A context request whose KV prefix is being fetched; the scheduler does not touch it."""
KV_PUBLISH_IN_PROGRESS = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
"""A finished request kept alive because a transfer still touches its pages."""


class EngineRequestView:
    """The ``RequestView`` the coordination layer reads, over one ``LlmRequest``.

    The three attributes the planner branches on are constants: this assembly never plans for a
    disaggregated generation-init request (they are filtered out before planning) and carries no
    routing hints. Every other attribute read falls through to the request, so ``resource/`` can
    hand the view to the cache manager in place of the request. Effects unwrap ``request`` to write.
    """

    __slots__ = ("request",)

    is_gen_init = False
    is_gen_first_context = False
    route_hints: Mapping[str, Mapping[str, object]] = {}

    def __init__(self, request: LlmRequest) -> None:
        self.request = request

    def __getattr__(self, name: str):
        return getattr(self.request, name)

    def __repr__(self) -> str:
        return f"EngineRequestView({self.request.py_request_id})"


def _engine_request(request_or_view) -> LlmRequest:
    """The ``LlmRequest`` behind a view; a bare request passes through."""
    if isinstance(request_or_view, EngineRequestView):
        return request_or_view.request
    return request_or_view


class EngineWorkQueue:
    """``EngineQueue``: callables posted from backend threads, run on the engine thread.

    No backend in this assembly posts to it yet; it exists so the contract is complete.
    """

    def __init__(self) -> None:
        self._posted: deque[Callable[[], None]] = deque()
        self._lock = threading.Lock()

    def post(self, work: Callable[[], None]) -> None:
        with self._lock:
            self._posted.append(work)

    def drain(self, budget: int) -> int:
        ran = 0
        while ran < budget:
            with self._lock:
                if not self._posted:
                    break
                work = self._posted.popleft()
            work()
            ran += 1
        return ran


class EngineDist:
    """``DistLike`` over the executor's ``dist``, for the ranks that plan together.

    Without attention DP every rank plans together: the whole world, through ``allgather``. Under
    attention DP each replica plans on its own: this rank's pipeline group, through
    ``pp_allgather``. A group of one rank answers with its own payload and enters no collective,
    as the disagg transceiver's sync policy does for a world of one.
    """

    def __init__(self, dist, mapping: ParallelMapping) -> None:
        if mapping.enable_attention_dp:
            self._group_size = mapping.pp_size
            self._allgather = dist.pp_allgather
        else:
            self._group_size = mapping.world_size
            self._allgather = dist.allgather

    def allgather(self, payload: object) -> list:
        if self._group_size == 1:
            return [payload]
        return self._allgather(payload)


class PyExecutorKVTransferEffects:
    """``KVTransferEffects`` over a ``PyExecutor``, in the order of the design §7.3 table.

    Stateless: which requests are parked or held is the coordinator's record table to answer.
    """

    def __init__(self, executor: PyExecutor) -> None:
        self._executor = executor

    # ---- effects ----

    def park_for_fetch(self, requests: Sequence) -> None:
        """State -> ``KV_FETCH_IN_PROGRESS``; the scheduler skips the request until it lands."""
        for request in requests:
            _engine_request(request).state = KV_FETCH_IN_PROGRESS
        logger.debug("kv transfer: parked %s for fetch", [r.py_request_id for r in requests])

    def unpark(self, request, token_end: int, no_local_fallback: bool, aux) -> None:
        """A fetch landed: settle the cursor at ``max(token_end, num_committed_tokens)``, commit,
        back to ``CONTEXT_INIT``. The maximum because local reuse may already exceed the
        block-aligned target (design §7.2 step 6) and the cursor never moves below a commit.

        Narrower than the protocol allows, by the scope guard: only the primary KV cache manager
        commits (no draft manager), ``aux`` has no consumer (no worker backend, so nothing rides
        along), and a gen-init landing (``no_local_fallback``) raises because gen-init requests
        belong to the disagg transceiver, never to this assembly."""
        engine_request = _engine_request(request)
        if no_local_fallback:
            raise RuntimeError(
                f"request {engine_request.py_request_id}: a gen-init landing has no place in "
                "this assembly"
            )
        kv_cache_manager = self._executor.kv_cache_manager
        self._check_history_declared(engine_request, token_end)
        # CONTEXT_INIT first: the C++ request only lets a context-phase request move its cursor
        # (plan §10 #1).
        engine_request.state = LlmRequestState.CONTEXT_INIT
        # Fetched content always ends short of the prompt, so the cursor is settled the way a
        # local reuse hit is (design §7.3 step 1). Local reuse may already reach past
        # ``token_end`` (design §7.2 step 6: a plan trimmed to an empty ask is still legal); the
        # cursor never moves below what is committed, and the commit below is then a no-op for
        # that part.
        kv_cache = kv_cache_manager.kv_cache_map[engine_request.py_request_id]
        settle_at = max(token_end, int(kv_cache.num_committed_tokens))
        settle_context_cursor(engine_request, settle_at, kv_cache_manager.tokens_per_block)
        # The pages now hold real content and are committed next; a later revert must not shrink
        # them away (plan §10 #11).
        engine_request.py_ctx_pre_resize_cap = None
        # Only the primary manager commits: a draft KV cache manager is refused at assembly, so
        # there is no paired pool whose commit would have to agree with this one.
        kv_cache_manager.try_commit_blocks(engine_request)
        self._warn_if_commit_fell_short(engine_request, token_end)
        logger.debug(
            "kv transfer: request %d landed at token_end=%d, cursor=%d",
            engine_request.py_request_id,
            token_end,
            engine_request.context_current_position,
        )

    def give_back_fetch_pages(self, requests: Sequence) -> None:
        """Revert the pages the scheduler reserved for a fetch; back to ``CONTEXT_INIT``.

        A cache that survives the revert with history declared to the fetch target but no data
        in its pages is dropped, so the request re-enters as a fresh first chunk (plan §10 #13).
        """
        engine_requests = [_engine_request(request) for request in requests]
        kv_cache_manager = self._executor.kv_cache_manager
        # CONTEXT_INIT first, as in ``unpark``: reverting and rewinding move the context cursor,
        # which the C++ request only allows in the context phase (plan §10 #1).
        for engine_request in engine_requests:
            engine_request.state = LlmRequestState.CONTEXT_INIT
        self._executor._revert_ctx_alloc(engine_requests)
        for engine_request in engine_requests:
            if engine_request.py_request_id in kv_cache_manager.kv_cache_map:
                kv_cache_manager.free_resources(engine_request)
                rewind_context_after_cache_drop(engine_request, kv_cache_manager.tokens_per_block)
        logger.debug(
            "kv transfer: gave back fetch pages of %s",
            [request.py_request_id for request in engine_requests],
        )

    def prepare_fetch_resources(self, requests: Sequence) -> None:
        """Prepare the non-KV resource managers, as for a gen-init receive."""
        self._executor._prepare_disagg_gen_resources(
            [_engine_request(request) for request in requests]
        )

    def hold_for_transfer(self, requests: Sequence) -> None:
        """A finished request with a transfer in flight: keep its pages, release the rest.

        Reads nothing of the disagg transfer manager (plan §9): a seq slot the disagg send already
        released is a no-op to release again, and ``release_index_slot`` is idempotent. The pages
        themselves stay allocated for the transfer.
        """
        for request in requests:
            engine_request = _engine_request(request)
            self._release_seq_slot(engine_request)
            self._release_index_slot(engine_request)
            engine_request.state = KV_PUBLISH_IN_PROGRESS

    def terminate_request(self, request) -> None:
        """Final release of a held request once every transfer of it is done.

        Goes straight to ``_do_terminate_request``: the request has already left
        ``active_requests`` (only unfinished requests are kept there), and re-entering
        ``_terminate_request`` would ask the release gate again (plan §9).
        """
        self._executor._do_terminate_request(_engine_request(request))

    def fail_requests(self, requests: Sequence, reason: str) -> None:
        """Fail requests through the engine's error path.

        The engine terminates a failed request through ``_terminate_request``, whose release gate
        asks the coordinator whether it holds the request (a fetch still in flight does).
        """
        engine_requests = [_engine_request(request) for request in requests]
        self._executor._handle_errors(reason, requests=engine_requests, charge_budget=False)

    def fail_fatal(self, error: BaseException) -> None:
        executor = self._executor
        executor._fatal_error = RuntimeError(f"Fatal error: {error}")
        executor.is_shutdown = True
        executor._handle_errors(
            str(error), requests=None, charge_budget=False, fatal_is_collective_aligned=True
        )

    # ---- helpers ----

    def _check_history_declared(self, engine_request: LlmRequest, token_end: int) -> None:
        """The scheduler reserved the fetch with ``reserve_transfer_pages(token_end)``, which
        declares history up to ``token_end``; a landing below that is a wiring error."""
        history = self._executor.kv_cache_manager.get_history_length(engine_request)
        if history is None or history < token_end:
            raise RuntimeError(
                f"request {engine_request.py_request_id}: kv_cache.history_length={history} < "
                f"token_end={token_end} when the fetch landed"
            )

    def _warn_if_commit_fell_short(self, engine_request: LlmRequest, token_end: int) -> None:
        """A commit short of ``token_end`` makes the scheduler re-settle the cursor at the local
        reuse depth and recompute the fetched pages: slow but correct (plan §10 #14)."""
        kv_cache = self._executor.kv_cache_manager.kv_cache_map.get(engine_request.py_request_id)
        committed = 0 if kv_cache is None else int(kv_cache.num_committed_tokens)
        if committed < token_end:
            logger.warning(
                "kv transfer: request %d committed %d tokens after a fetch to %d; the fetched "
                "prefix beyond the commit will be recomputed",
                engine_request.py_request_id,
                committed,
                token_end,
            )

    def _release_seq_slot(self, engine_request: LlmRequest) -> None:
        """Speculative-decoding resources need no release: the guard refuses spec-decode."""
        resource_managers = self._executor.resource_manager.resource_managers
        seq_slot_manager = resource_managers.get(ResourceManagerType.SEQ_SLOT_MANAGER)
        if seq_slot_manager is not None:
            seq_slot_manager.free_resources(engine_request)

    def _release_index_slot(self, engine_request: LlmRequest) -> None:
        """Free the page-table slot early while the pages stay allocated for the transfer."""
        kv_cache_manager = self._executor.kv_cache_manager
        if engine_request.py_request_id in kv_cache_manager.kv_cache_map:
            kv_cache_manager.release_index_slot(engine_request.py_request_id)

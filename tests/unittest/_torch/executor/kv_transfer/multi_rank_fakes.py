# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One real ``KVTransferHooks`` per rank, over the fakes of ``engine_fakes``, for the threaded
multi-rank tests: ``RankRig`` is one rank; its collective is a ``FakeDistRank`` of the
``FakeDistGroup`` all the rigs of a world share, reached through the real ``EngineCollective``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence

from engine_fakes import (
    TPB,
    FakeFetches,
    FakeKVCache,
    FakeKVCacheManager,
    FakePublishes,
    FakeReader,
    FakeSlotManager,
    finish_prefill,
    make_executor,
)

from tensorrt_llm._torch.disaggregation.backends.config import (
    DEFAULT_LANDING_WAIT_TIMEOUT_S,
    BackendEntry,
)
from tensorrt_llm._torch.disaggregation.backends.registry import BackendHandle
from tensorrt_llm._torch.disaggregation.orchestration.kv_transfer.coordinator import (
    KVTransferCoordinator,
)
from tensorrt_llm._torch.disaggregation.orchestration.kv_transfer.interfaces import PlanAuthority
from tensorrt_llm._torch.disaggregation.remote_cache import FetchPlan, FetchSource, Planner
from tensorrt_llm._torch.disaggregation.resource.region import parallel_shard_tag
from tensorrt_llm._torch.pyexecutor.kv_transfer.effects import (
    EngineCollective,
    EngineWorkQueue,
    PyExecutorKVTransferEffects,
)
from tensorrt_llm._torch.pyexecutor.kv_transfer.hooks import KVTransferHooks
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest

__extra_import_path__ = ["../../disaggregation"]
from fake_dist import FakeDistGroup, FakeDistRank  # noqa: E402

__all__ = ["CLOSE_TIMEOUT_S", "CountingEffects", "FakeDistGroup", "RankRig", "rank_mapping"]

CLOSE_TIMEOUT_S = 5.0
"""Every rig's ``close_timeout_s``: how long its shutdown drain counts its own pending work."""


class CountingEffects(PyExecutorKVTransferEffects):
    """The real engine effects, counting the ones a multi-rank scenario asserts on."""

    def __init__(self, executor) -> None:
        super().__init__(executor)
        self.unparks = 0
        self.give_backs = 0
        self.failed: list[tuple[tuple[int, ...], str]] = []

    def unpark(self, request, token_end, no_local_fallback, aux) -> None:
        self.unparks += 1
        super().unpark(request, token_end, no_local_fallback, aux)

    def give_back_fetch_pages(self, requests) -> None:
        self.give_backs += 1
        super().give_back_fetch_pages(requests)

    def fail_requests(self, requests, reason: str) -> None:
        self.failed.append((tuple(r.py_request_id for r in requests), reason))
        super().fail_requests(requests, reason)


def rank_mapping(dist: FakeDistRank, *, enable_attention_dp: bool = False) -> SimpleNamespace:
    """The ``Mapping`` attributes ``EngineCollective`` and ``parallel_shard_tag`` read, for one rank."""
    return SimpleNamespace(
        rank=dist.rank,
        world_size=dist.world_size,
        tp_size=dist.tp_size,
        tp_rank=dist.tp_rank,
        pp_size=dist.pp_size,
        pp_rank=dist.pp_rank,
        cp_size=1,
        enable_attention_dp=enable_attention_dp,
    )


class RankRig:
    """One rank of a world: real coordinator, planner, effects and hooks over the fakes, with
    ``schedule_round`` standing in for what the engine loop and the V2 scheduler do per round.

    ``unlaunched_timeout_s``, ``fetch_timeout_s`` and ``plan_authority`` go to the coordinator,
    ``close_timeout_s`` to the hooks (a per-rig value plays clock skew between ranks); the store
    answers every probe and every fetch stays in flight until ``deliver_all``. The executor's
    ``_free_request_resources`` frees the fake cache, so an eviction is observable in
    ``kv.kv_cache_map``.
    """

    def __init__(
        self,
        group: FakeDistGroup,
        rank: int,
        *,
        enable_attention_dp: bool = False,
        unlaunched_timeout_s: float | None = 10.0,
        fetch_timeout_s: float | None = None,
        plan_authority: PlanAuthority = PlanAuthority.VOTED,
        close_timeout_s: float = CLOSE_TIMEOUT_S,
    ) -> None:
        self.rank = rank
        self.dist = group.rank(rank)
        self.mapping = rank_mapping(self.dist, enable_attention_dp=enable_attention_dp)
        self.shard_tag = parallel_shard_tag(self.mapping)
        self.kv = FakeKVCacheManager(TPB)
        self.slots = FakeSlotManager()
        self.executor = make_executor(self.kv, self.slots)
        self.reader = FakeReader(self.kv)
        self.store = FakeFetches(name="store")
        self.publisher = FakePublishes()
        sources = [FetchSource("store", self.store, None)]
        self.planner = Planner(sources, self.reader, TPB, probe_timeout_s=0.05)
        self.effects = CountingEffects(self.executor)
        self.coord = KVTransferCoordinator(
            sources,
            [self.publisher],
            self.planner,
            self.reader,
            self.effects,
            EngineWorkQueue(),
            EngineCollective(self.dist, self.mapping),
            fetch_timeout_s=fetch_timeout_s,
            unlaunched_timeout_s=unlaunched_timeout_s,
            landing_wait_timeout_s=DEFAULT_LANDING_WAIT_TIMEOUT_S,
            plan_authority=plan_authority,
        )
        handle = BackendHandle(
            name="store",
            hint_key=None,
            fetcher=self.store,
            publisher=self.publisher,
            pool_registrar=None,
            close=lambda: None,
        )
        entry = BackendEntry.from_dict({"name": "store", "type": "fake"})
        self.hooks = KVTransferHooks(
            self.executor,
            self.coord,
            self.effects,
            [handle],
            [entry],
            close_timeout_s=close_timeout_s,
        )
        self.executor.kv_transfer = self.hooks
        self.executor._free_request_resources.side_effect = self.kv.free_resources

    # -- one round of the loop on this rank --

    def schedule_round(
        self, active: Sequence[LlmRequest], *, adopt=None, recompute_pause: Sequence = ()
    ) -> None:
        """``advance_round``, then, for every request the coordinator planned, reserve its pages
        the way the scheduler's ``reserve_transfer_pages`` does and launch the fetch.

        ``adopt`` plays Stage 0 for a follower: called after ``advance_round`` with the active
        requests, it must return the owner's exported answers, which are adopted before the
        local scheduling below.

        ``recompute_pause`` plays pool pressure: the requests torn down for a recompute this
        round, handed unfiltered to the executor's own teardown
        (``_terminate_recompute_paused_requests``). This rig does not model the scheduler's
        protected-set filter (the scheduler seam tests cover it); what keeps a request a
        transfer still touches out of the teardown here is the executor's own guard, which
        reads ``inflight_request_ids`` itself.
        """
        self.hooks.advance_round(list(active))
        if adopt is not None:
            self.hooks.adopt_plan_answers(list(active), adopt(list(active)))
        launch_queue = []
        for request in active:
            plan = self.hooks.plan_fetch(request)
            if isinstance(plan, FetchPlan) and self.reserve(request, plan.token_end):
                launch_queue.append(request)
        self.hooks.launch_reserved_fetches(launch_queue)
        if recompute_pause:
            self.executor._terminate_recompute_paused_requests(
                SimpleNamespace(recompute_paused_requests=list(recompute_pause))
            )

    def reserve(self, request: LlmRequest, token_end: int) -> bool:
        if not self.kv.reserve_transfer_pages(request, token_end):
            return False
        request.py_ctx_pre_resize_cap = 0
        self.slots.add(request)
        return True

    def deliver_all(self) -> None:
        """Every fetch and publish attempt of this rank's backends still in flight completes."""
        for attempt in (*self.store.attempts, *self.publisher.attempts):
            if attempt.poll() is None:
                attempt.deliver_all()

    def publish(self, request: LlmRequest) -> None:
        """Prefill ended this step: offer the request's blocks."""
        finish_prefill(request)
        self.kv.kv_cache_map.setdefault(
            request.py_request_id, FakeKVCache(history_length=request.prompt_len)
        )
        self.hooks.publish_committed_blocks([request])

    # -- what a scenario reads --

    def gathers(self) -> list[str]:
        """The collectives this rank entered, in order."""
        return [name for name, _ in self.dist.calls]

    def records(self) -> list[dict]:
        return self.coord.status_dump()["records"]

    def record(self, request_id: int, direction: str = "fetch") -> dict | None:
        for rec in self.records():
            if rec["request_id"] == request_id and rec["direction"] == direction:
                return rec
        return None

    def published_unit_names(self) -> frozenset[bytes]:
        return frozenset(
            unit.name for attempt in self.publisher.attempts for unit in attempt.payload.units
        )

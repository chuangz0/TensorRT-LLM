# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One real ``KVTransferHooks`` per rank for the threaded multi-rank tests: ``RankEngineRig`` is
an ``EngineRig`` whose collective is a ``FakeDistRank`` of the ``FakeDistGroup`` all the rigs of a
world share, reached through the real ``EngineCollective``, with ``schedule_round`` standing in for
what the engine loop and the V2 scheduler do per round.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence

from engine_fakes import EngineRig

from tensorrt_llm._torch.disaggregation.orchestration.kv_transfer.engine_protocols import (
    PlanAuthority,
)
from tensorrt_llm._torch.disaggregation.remote_cache import FetchPlan
from tensorrt_llm._torch.disaggregation.resource.region import parallel_shard_tag
from tensorrt_llm._torch.pyexecutor.kv_transfer.effects import (
    EngineCollective,
    EngineKVTransferEffects,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest

__extra_import_path__ = ["../../disaggregation"]
from fake_dist import FakeDistGroup, FakeDistRank  # noqa: E402

__all__ = ["CLOSE_TIMEOUT_S", "CountingEffects", "FakeDistGroup", "RankEngineRig", "rank_mapping"]

CLOSE_TIMEOUT_S = 5.0
"""Every rig's ``close_timeout_s``: how long its shutdown drain counts its own pending work."""


class CountingEffects(EngineKVTransferEffects):
    """The real engine effects, counting the ones a multi-rank scenario asserts on."""

    def __init__(self, executor) -> None:
        super().__init__(executor)
        self.unparks = 0
        self.give_backs = 0
        self.failed: list[tuple[tuple[int, ...], str]] = []

    def unpark(self, request, token_end, no_local_fallback, aux) -> None:
        self.unparks += 1
        super().unpark(request, token_end, no_local_fallback, aux)

    def revert_fetch_pages(self, requests) -> None:
        self.give_backs += 1
        super().revert_fetch_pages(requests)

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


class RankEngineRig(EngineRig):
    """One rank of a world: an ``EngineRig`` over ``CountingEffects`` and the real
    ``EngineCollective`` of this rank's ``FakeDistRank``.

    ``unlaunched_timeout_s``, ``fetch_timeout_s`` and ``plan_authority`` go to the coordinator,
    ``close_timeout_s`` to the hooks (a per-rig value plays clock skew between ranks); the store
    answers every probe and every fetch stays in flight until ``deliver_all``.
    """

    def __init__(
        self,
        group: FakeDistGroup,
        rank: int,
        *,
        enable_attention_dp: bool = False,
        unlaunched_timeout_s: float | None = 10.0,
        fetch_timeout_s: float | None = None,
        plan_authority: PlanAuthority = PlanAuthority.ALL_RANKS,
        close_timeout_s: float = CLOSE_TIMEOUT_S,
    ) -> None:
        self.rank = rank
        self.dist = group.rank(rank)
        self.mapping = rank_mapping(self.dist, enable_attention_dp=enable_attention_dp)
        self.shard_tag = parallel_shard_tag(self.mapping)
        super().__init__(
            collective=EngineCollective(self.dist, self.mapping),
            effects_class=CountingEffects,
            unlaunched_timeout_s=unlaunched_timeout_s,
            fetch_timeout_s=fetch_timeout_s,
            plan_authority=plan_authority,
            close_timeout_s=close_timeout_s,
        )

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
            plan = self.hooks.fetch_answer(request)
            if isinstance(plan, FetchPlan) and self.try_reserve(request, plan.token_end):
                launch_queue.append(request)
        self.hooks.launch_reserved_fetches(launch_queue)
        if recompute_pause:
            self.executor._terminate_recompute_paused_requests(
                SimpleNamespace(recompute_paused_requests=list(recompute_pause))
            )

    def try_reserve(self, request: LlmRequest, token_end: int) -> bool:
        """The scheduler's reservation through the fake manager, which ``reserve_answer`` can
        refuse (unlike ``reserve``, which leaves the pages behind unconditionally)."""
        if not self.kv.reserve_transfer_pages(request, token_end):
            return False
        request.py_ctx_pre_resize_cap = 0
        self.slots.add(request)
        return True

    # -- what a scenario reads --

    def gathers(self) -> list[str]:
        """The collectives this rank entered, in order."""
        return [name for name, _ in self.dist.calls]

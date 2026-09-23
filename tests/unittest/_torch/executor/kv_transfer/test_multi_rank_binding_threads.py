# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``N`` real engine bindings on ``N`` threads, one per TP rank, whose coordinators meet in the
``FakeDistGroup`` collective through the real ``EngineDist`` (plan S3 (b)).

Each round every rank runs what the loop runs: ``advance_round``, ``plan_fetch``, the scheduler's
page reservation, ``launch_reserved_fetches``; then delivers its own store's attempts; later
``publish_committed_blocks``. Asserted: the protocol is symmetric (one ``allgather`` per round
with the same payload on every rank), every rank lands once, TP shards publish the same unit
names under different shard tags, and a rank that cannot reserve pages holds the landing until
``unlaunched_timeout_s`` makes every rank start over in the same round.
"""

import time

import pytest
from engine_fakes import make_request
from multi_rank_fakes import FakeDistGroup, RankRig

from tensorrt_llm._torch.disaggregation.orchestration.kv_transfer.interfaces import DEFER
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState

pytestmark = pytest.mark.cpu_only

PROMPT_LEN = 100
TOKEN_END = 96  # 3 nameable blocks of 32
UNLAUNCHED_TIMEOUT_S = 10.0


@pytest.fixture
def clock(monkeypatch):
    """Freeze the binding's ``time.monotonic``; the fake collective's barriers keep their own
    wall clock, so a frozen clock never stalls them."""
    now = {"t": 1000.0}
    monkeypatch.setattr(time, "monotonic", lambda: now["t"])
    return now


def make_world(world_size: int, **rig_kwargs):
    group = FakeDistGroup(world_size=world_size, tp_size=world_size)
    rigs = [RankRig(group, rank, **rig_kwargs) for rank in range(world_size)]
    return group, rigs


def rank_local_requests(world_size: int, request_id: int = 1) -> list:
    """The same request as each rank's engine holds it: one object per rank, one id."""
    return [make_request(request_id, PROMPT_LEN) for _ in range(world_size)]


@pytest.mark.parametrize("world_size", [2, 4])
def test_tp_ranks_land_once_with_one_gather_per_round_and_distinct_shard_tags(world_size):
    group, rigs = make_world(world_size)
    requests = rank_local_requests(world_size)

    def engine_loop(rank):
        rig, request = rigs[rank], requests[rank]
        rig.schedule_round([request])  # round 1: plan, reserve, launch
        rig.deliver_all()
        rig.schedule_round([request])  # round 2: every rank TERMINAL -> unpark
        rig.publish(request)
        rig.deliver_all()
        rig.schedule_round([])  # round 3: every publish TERMINAL -> released

    group.run(engine_loop)

    for rig, request in zip(rigs, requests):
        assert rig.effects.unparks == 1 and rig.effects.give_backs == 0
        assert request.state == LlmRequestState.CONTEXT_INIT
        assert rig.kv.kv_cache_map[1].num_committed_tokens == TOKEN_END  # landed there ...
        assert request.context_remaining_length == 0  # ... and prefill ran to the end
        assert rig.gathers() == ["allgather"] * 3
        assert rig.dist.calls == rigs[0].dist.calls  # the same word from every rank, every round
        assert [r["state"] for r in rig.records()] == ["LANDED"]  # the publish is released
        assert rig.publisher.count("publish") == 1 and rig.publisher.count("quiesce") == 1
    # Every shard publishes the same unit names; only the shard tag in the layout fingerprint
    # keeps their store entries apart.
    assert len({rig.published_unit_names() for rig in rigs}) == 1
    assert len({rig.shard_tag for rig in rigs}) == world_size


def test_a_rank_without_pages_holds_the_landing_until_the_timeout_restarts_every_rank(clock):
    world_size = 2
    group, rigs = make_world(world_size, unlaunched_timeout_s=UNLAUNCHED_TIMEOUT_S)
    requests = rank_local_requests(world_size)
    laggard, launched = rigs[1], rigs[0]
    laggard.kv.reserve_answer = False  # its scheduler finds no pages for the fetch

    def one_round(rank):
        rigs[rank].schedule_round([requests[rank]])
        rigs[rank].deliver_all()

    group.run(one_round)  # t = 1000: rank 0 launches and delivers; rank 1 stays unlaunched
    clock["t"] += 1.0
    group.run(one_round)  # rank 0 TERMINAL, rank 1 UNLAUNCHED: no landing
    assert [rig.effects.unparks for rig in rigs] == [0, 0]
    assert [r["state"] for r in launched.records()] == ["IN_FLIGHT"]
    assert [r["state"] for r in laggard.records()] == ["PLANNED"]
    assert laggard.fetch_record(1).peer_launched_at == 1001.0

    clock["t"] += UNLAUNCHED_TIMEOUT_S - 0.1
    group.run(one_round)  # still within the budget
    assert [rig.effects.unparks for rig in rigs] == [0, 0]

    clock["t"] += 0.1
    group.run(one_round)  # rank 1 votes FAILED: every rank starts over in this round
    assert [rig.effects.unparks for rig in rigs] == [0, 0]
    assert launched.effects.give_backs == 1 and launched.store.count("quiesce") == 1
    launched.executor._revert_ctx_alloc.assert_called_once()
    assert laggard.effects.give_backs == 0
    for rig, request in zip(rigs, requests):
        assert rig.binding.plan_fetch(request) is DEFER
        assert request.state == LlmRequestState.CONTEXT_INIT

    laggard.kv.reserve_answer = True
    clock["t"] += 1.0
    group.run(one_round)  # both plan again, reserve and launch
    clock["t"] += 1.0
    group.run(one_round)  # both TERMINAL: the landing
    for rig, request in zip(rigs, requests):
        assert rig.effects.unparks == 1 and rig.effects.failed == []
        assert request.context_current_position == TOKEN_END
        assert rig.gathers() == ["allgather"] * 6


def test_attention_dp_replicas_run_without_a_collective_and_share_the_shard_tag():
    group, rigs = make_world(2, enable_attention_dp=True)  # PP groups of one
    requests = [make_request(10 + rank, PROMPT_LEN) for rank in range(2)]  # per-replica sets

    def engine_loop(rank):
        rig, request = rigs[rank], requests[rank]
        rig.schedule_round([request])
        rig.deliver_all()
        rig.schedule_round([request])

    group.run(engine_loop)

    for rig, request in zip(rigs, requests):
        assert rig.effects.unparks == 1
        assert request.context_current_position == TOKEN_END
        assert rig.gathers() == []  # a group of one enters no collective
        assert rig.shard_tag == "heads=all"

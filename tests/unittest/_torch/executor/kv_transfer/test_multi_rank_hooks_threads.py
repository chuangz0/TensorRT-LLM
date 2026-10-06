# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``N`` real ``KVTransferHooks`` on ``N`` threads, one per TP rank, whose coordinators meet in the
``FakeDistGroup`` collective through the real ``EngineCollective`` (plan S3 (b)).

Each round every rank runs what the loop runs: ``advance_round``, ``plan_fetch``, the scheduler's
page reservation, ``launch_reserved_fetches``; then delivers its own store's attempts; later
``publish_committed_blocks``. Asserted: the protocol is symmetric (one ``allgather`` per round
with the same payload on every rank), every rank lands once, TP shards publish the same unit
names under different shard tags, and a rank that cannot reserve pages holds the landing until
``unlaunched_timeout_s`` makes every rank start over in the same round.

Under ``pp_size=2`` (plan S4 (b)) the test plays Stage 0 of the PP loop: the owner's exported
answers cross to the follower through a queue, pickled as the schedule would carry them.
"""

import pickle
import queue
import threading
import time

import pytest
from engine_fakes import make_request
from multi_rank_fakes import FakeDistGroup, RankRig

from tensorrt_llm._torch.disaggregation.orchestration.kv_transfer.interfaces import PlanAuthority
from tensorrt_llm._torch.disaggregation.remote_cache import DEFER
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState

pytestmark = pytest.mark.cpu_only

PROMPT_LEN = 100
TOKEN_END = 96  # 3 nameable blocks of 32
UNLAUNCHED_TIMEOUT_S = 10.0


@pytest.fixture
def clock(monkeypatch):
    """Freeze the hooks' ``time.monotonic``; the fake collective's barriers keep their own
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
        assert rig.hooks.plan_fetch(request) is DEFER
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


class Stage0:
    """The schedule's path from the owner to the follower, as the test plays it: the owner
    exports after its ``advance_round``; the follower blocks until that round's answers arrive.
    Answers cross pickled, as they do inside ``SerializableSchedulerOutput``."""

    def __init__(self, owner: RankRig) -> None:
        self._owner = owner
        self._wire: queue.Queue = queue.Queue()
        self.sent: list = []

    def owner_round(self, active) -> None:
        self._owner.schedule_round(active)
        answers = self._owner.hooks.export_plan_answers()
        self.sent.append(answers)
        self._wire.put(pickle.dumps(answers))

    def adopt(self, active):
        return pickle.loads(self._wire.get(timeout=5.0))


def test_pp_follower_adopts_the_owners_plans_and_lands_in_the_same_round():
    group = FakeDistGroup(world_size=2, tp_size=1, pp_size=2)
    owner = RankRig(group, 0, plan_authority=PlanAuthority.OWNER)
    follower = RankRig(group, 1, plan_authority=PlanAuthority.FOLLOWER)
    requests = rank_local_requests(2)
    stage0 = Stage0(owner)

    def engine_loop(rank):
        if rank == 0:
            stage0.owner_round([requests[0]])  # round 1: plan, export, reserve, launch
            owner.deliver_all()
            stage0.owner_round([requests[0]])  # round 2: both TERMINAL -> unpark
        else:
            follower.schedule_round([requests[1]], adopt=stage0.adopt)  # adopt, reserve, launch
            follower.deliver_all()
            follower.schedule_round([requests[1]], adopt=stage0.adopt)

    group.run(engine_loop)

    assert stage0.sent == [[(1, (TOKEN_END, "store"))], []]
    assert follower.store.count("probe") == 0  # the follower never plans
    assert follower.hooks._num_deferred_requests == 0
    for rig, request in zip((owner, follower), requests):
        assert rig.effects.unparks == 1 and rig.effects.give_backs == 0
        assert request.context_current_position == TOKEN_END
        assert rig.gathers() == ["allgather"] * 2
        assert rig.coord.status_dump()["plan_authority"] == rig.coord.plan_authority.value
    # Both ranks speak in every round, and neither carries plans in the collective.
    assert [payload[2] for _, payload in owner.dist.calls] == [[], []]
    assert owner.dist.calls == follower.dist.calls


def test_pp_follower_without_pages_launches_a_round_late_and_both_land_together():
    group = FakeDistGroup(world_size=2, tp_size=1, pp_size=2)
    owner = RankRig(group, 0, plan_authority=PlanAuthority.OWNER)
    follower = RankRig(group, 1, plan_authority=PlanAuthority.FOLLOWER)
    follower.kv.reserve_answer = False  # round 1: adopted, but its scheduler finds no pages
    requests = rank_local_requests(2)
    stage0 = Stage0(owner)
    after_round = [threading.Barrier(2) for _ in range(3)]

    def engine_loop(rank):
        if rank == 0:
            stage0.owner_round([requests[0]])  # round 1: plan, export, reserve, launch
            owner.deliver_all()
            after_round[0].wait()
            stage0.owner_round([requests[0]])  # round 2: owner TERMINAL, follower UNLAUNCHED
            after_round[1].wait()
            stage0.owner_round([requests[0]])  # round 3: both TERMINAL -> unpark
            after_round[2].wait()
        else:
            follower.schedule_round([requests[1]], adopt=stage0.adopt)  # adopted, not launched
            after_round[0].wait()
            follower.kv.reserve_answer = True
            follower.schedule_round([requests[1]], adopt=stage0.adopt)  # reserves and launches
            after_round[1].wait()
            assert [owner.effects.unparks, follower.effects.unparks] == [0, 0]
            assert follower.fetch_record(1).peer_launched_at is None  # cleared by its launch
            follower.deliver_all()
            follower.schedule_round([requests[1]], adopt=stage0.adopt)
            after_round[2].wait()

    group.run(engine_loop)

    assert stage0.sent == [[(1, (TOKEN_END, "store"))], [], []]
    assert follower.kv.count("reserve_transfer_pages") == 2
    for rig, request in zip((owner, follower), requests):
        assert rig.effects.unparks == 1 and rig.effects.give_backs == 0
        assert request.context_current_position == TOKEN_END
        assert rig.gathers() == ["allgather"] * 3
    votes_by_round = [payload[0] for _, payload in follower.dist.calls]
    assert [vote[1] for vote in votes_by_round[1]] == ["UNLAUNCHED"]  # round 2 held the landing
    assert [vote[1] for vote in votes_by_round[2]] == ["TERMINAL"]


def test_pp_follower_without_an_answer_counts_the_candidate_as_deferred():
    group = FakeDistGroup(world_size=2, tp_size=1, pp_size=2)
    owner = RankRig(group, 0, plan_authority=PlanAuthority.OWNER)
    follower = RankRig(group, 1, plan_authority=PlanAuthority.FOLLOWER)
    owner.store.probe_answer = None  # the owner's store has not answered: it defers
    requests = rank_local_requests(2)
    stage0 = Stage0(owner)

    def engine_loop(rank):
        if rank == 0:
            stage0.owner_round([requests[0]])
        else:
            follower.schedule_round([requests[1]], adopt=stage0.adopt)

    group.run(engine_loop)

    assert stage0.sent == [[]]
    assert owner.hooks.plan_fetch(requests[0]) is DEFER
    assert follower.hooks.plan_fetch(requests[1]) is DEFER
    assert follower.hooks._num_deferred_requests == 1
    assert owner.hooks._num_deferred_requests == 1


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

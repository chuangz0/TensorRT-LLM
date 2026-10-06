# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``EngineCollective``: the coordinator's ``Collective`` over the executor's ``dist`` and its ``Mapping``.

Without attention DP the ranks that plan together are the world (``dist.allgather``); under
attention DP they are this rank's pipeline group (``dist.pp_allgather``). A group of one rank
answers with its own payload and enters no collective at all.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.kv_transfer.effects import EngineCollective

pytestmark = pytest.mark.cpu_only


class RecordingDist:
    """A ``dist`` that records which collective was entered and answers a fixed gathered list."""

    def __init__(self, world_size: int, pp_size: int) -> None:
        self.world_size = world_size
        self.pp_size = pp_size
        self.calls: list[tuple[str, object]] = []

    def allgather(self, obj) -> list:
        self.calls.append(("allgather", obj))
        return [obj] * self.world_size

    def pp_allgather(self, obj) -> list:
        self.calls.append(("pp_allgather", obj))
        return [obj] * self.pp_size


def mapping(*, world_size: int, pp_size: int, enable_attention_dp: bool) -> SimpleNamespace:
    return SimpleNamespace(
        world_size=world_size, pp_size=pp_size, enable_attention_dp=enable_attention_dp
    )


def test_without_attention_dp_the_world_gathers():
    dist = RecordingDist(world_size=4, pp_size=2)
    collective = EngineCollective(dist, mapping(world_size=4, pp_size=2, enable_attention_dp=False))
    payload = ([], [], [(1, "DEFER")])

    assert collective.allgather(payload) == [payload] * 4
    assert dist.calls == [("allgather", payload)]


def test_under_attention_dp_the_pipeline_group_gathers():
    dist = RecordingDist(world_size=4, pp_size=2)
    collective = EngineCollective(dist, mapping(world_size=4, pp_size=2, enable_attention_dp=True))
    payload = ([], [], [(1, "DEFER")])

    assert collective.allgather(payload) == [payload] * 2
    assert dist.calls == [("pp_allgather", payload)]


@pytest.mark.parametrize(
    "world_size, pp_size, enable_attention_dp",
    [(1, 1, False), (4, 1, True)],
    ids=["single_rank", "attention_dp_without_pp"],
)
def test_a_group_of_one_answers_locally_without_a_collective(
    world_size, pp_size, enable_attention_dp
):
    dist = RecordingDist(world_size=world_size, pp_size=pp_size)
    collective = EngineCollective(
        dist,
        mapping(world_size=world_size, pp_size=pp_size, enable_attention_dp=enable_attention_dp),
    )
    payload = {"x": 1}

    gathered = collective.allgather(payload)

    assert gathered == [payload] and gathered[0] is payload
    assert dist.calls == []

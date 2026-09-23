# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``EngineDist``: the coordinator's ``DistLike`` over the executor's ``dist`` and its ``Mapping``.

The ``"world"`` scope is ``dist.allgather``, the ``"pp"`` scope ``dist.pp_allgather``; a group
of one rank answers with its own payload and enters no collective at all.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.kv_transfer.effects import EngineDist, collective_group_size

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


def mapping(*, world_size: int, pp_size: int) -> SimpleNamespace:
    return SimpleNamespace(world_size=world_size, pp_size=pp_size)


def test_group_size_is_the_world_or_the_pp_group():
    m = mapping(world_size=8, pp_size=2)
    assert collective_group_size(m, "world") == 8
    assert collective_group_size(m, "pp") == 2


def test_world_scope_enters_allgather_and_pp_scope_enters_pp_allgather():
    dist = RecordingDist(world_size=4, pp_size=2)
    engine_dist = EngineDist(dist, mapping(world_size=4, pp_size=2))
    payload = ([], [], [(1, "DEFER")])

    assert engine_dist.allgather(payload, "world") == [payload] * 4
    assert engine_dist.allgather(payload, "pp") == [payload] * 2
    assert dist.calls == [("allgather", payload), ("pp_allgather", payload)]


@pytest.mark.parametrize(
    "scope, world_size, pp_size",
    [("world", 1, 1), ("pp", 4, 1)],
    ids=["single_rank_world", "attention_dp_without_pp"],
)
def test_a_group_of_one_answers_locally_without_a_collective(scope, world_size, pp_size):
    dist = RecordingDist(world_size=world_size, pp_size=pp_size)
    engine_dist = EngineDist(dist, mapping(world_size=world_size, pp_size=pp_size))
    payload = {"x": 1}

    gathered = engine_dist.allgather(payload, scope)

    assert gathered == [payload] and gathered[0] is payload
    assert dist.calls == []

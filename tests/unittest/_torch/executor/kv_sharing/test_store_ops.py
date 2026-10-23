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
"""``decide`` turns the reduced reports on the open store operations into the same decisions on
every rank. Two ranks are reduced here as the round reduces them: the minimum of each value."""

from types import SimpleNamespace

import numpy as np
import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import MemoryKind
from tensorrt_llm._torch.pyexecutor.kv_sharing import store_ops
from tensorrt_llm._torch.pyexecutor.kv_sharing.store_ops import (
    Action,
    OpKind,
    OpState,
    StoreOp,
    decide,
    report,
)

pytestmark = pytest.mark.cpu_only

FETCH, PUBLISH = OpKind.FETCH, OpKind.PUBLISH
PARKED, SETTLING, OPEN = OpState.PARKED, OpState.SETTLING, OpState.OPEN
HOST, DEVICE = MemoryKind.HOST, MemoryKind.DEVICE
DONE_READY = (1, 1, 1)
NOT_DONE = (0, 0, 1)


def op(seq: int, kind=FETCH, state=PARKED, memory_kind=HOST, end=64) -> StoreOp:
    request = SimpleNamespace(py_request_id=100 + seq)
    return StoreOp(seq, kind, request, memory_kind, state, start=0, end=end)


def reduced(entries, *ranks, ok=(True, True)) -> np.ndarray:
    """The minimum over ranks of their reports, each rank's ``[done, ready, on_time]`` per entry."""
    reports = [
        report(entries, len(entries), rank_ok, np.array(columns, dtype=np.int64))
        for rank_ok, columns in zip(ok, ranks)
    ]
    return np.minimum.reduce(reports)


def actions(decisions) -> list[str]:
    """Each decision as "action seq", a drop with its reason."""
    described = []
    for d in decisions:
        parts = [d.action.value] + ([str(d.op.seq)] if d.op else [])
        described.append(" ".join(parts + ([d.reason] if d.action is Action.DROP else [])))
    return described


CASES = {
    # (entries as (seq, kind, state, memory kind), rank 0's and rank 1's columns) -> decisions
    "every row arrived in time, host staging: settle the copy first": (
        [(1, FETCH, PARKED, HOST)],
        [DONE_READY],
        [DONE_READY],
        ["return_all 1", "settle 1"],
    ),
    "every row arrived in time, GPU Direct: resume at once": (
        [(1, FETCH, PARKED, DEVICE)],
        [DONE_READY],
        [DONE_READY],
        ["return_all 1", "resume 1"],
    ),
    "the copy settled on every rank: resume": (
        [(1, FETCH, SETTLING, HOST)],
        [DONE_READY],
        [DONE_READY],
        ["resume 1"],
    ),
    "the copy settled on one rank only: wait": (
        [(1, FETCH, SETTLING, HOST)],
        [DONE_READY],
        [(0, 1, 1)],
        [],
    ),
    "a short delivery, or a lend refused, on one rank: drop": (
        [(1, FETCH, PARKED, HOST)],
        [DONE_READY],
        [(1, 0, 1)],
        ["return_none 1", "drop 1 short"],
    ),
    "one rank still reading: wait": (
        [(1, FETCH, PARKED, HOST)],
        [DONE_READY],
        [NOT_DONE],
        [],
    ),
    "one rank late while another still reads: abandon, drop, keep the lease": (
        [(1, FETCH, PARKED, HOST)],
        [(0, 0, 0)],
        [DONE_READY],
        ["abandon 1", "drop 1 late"],
    ),
    "late and done in one round: no resume": (
        [(1, FETCH, PARKED, HOST)],
        [(1, 1, 0)],
        [DONE_READY],
        ["return_none 1", "drop 1 late"],
    ),
    "an open lease goes back once every rank's access ended": (
        [(1, PUBLISH, OPEN, HOST), (2, FETCH, OPEN, HOST)],
        [DONE_READY, DONE_READY],
        [NOT_DONE, DONE_READY],
        ["return_none 2"],
    ),
}


@pytest.mark.parametrize("case", CASES)
def test_reduced_reports_decide_every_entry_alike(case: str) -> None:
    specs, rank0, rank1, expected = CASES[case]
    entries = [op(seq, kind, state, kind_of_memory) for seq, kind, state, kind_of_memory in specs]
    assert actions(decide(entries, reduced(entries, rank0, rank1))) == expected


def test_a_stopped_lender_under_host_staging_stops_sharing_and_drops_every_fetch() -> None:
    entries = [op(1, state=PARKED), op(2, state=SETTLING), op(3, PUBLISH, OPEN)]
    columns = [NOT_DONE, DONE_READY, DONE_READY]
    decisions = decide(entries, reduced(entries, columns, columns, ok=(True, False)))
    assert actions(decisions) == [
        "stop_sharing",
        "abandon 1",
        "drop 1 stopped",
        "drop 2 stopped",
        "return_none 3",
    ]


def test_a_stopped_lender_with_gpu_direct_leases_out_is_fatal() -> None:
    entries = [op(1, memory_kind=DEVICE)]
    decisions = decide(entries, reduced(entries, [NOT_DONE], [NOT_DONE], ok=(False, True)))
    assert actions(decisions) == ["fatal"]


def own_report(entries, announced: int) -> np.ndarray:
    """A rank's report with every entry done and ready."""
    return report(entries, announced, True, np.array([DONE_READY] * len(entries), dtype=np.int64))


@pytest.mark.parametrize(
    "rank1",
    [[op(1, state=SETTLING)], [op(1, end=96)], []],
    ids=["another state", "another range", "fewer entries"],
)
def test_tables_that_differ_across_ranks_are_fatal(rank1) -> None:
    rank0 = [op(1)]
    reports = own_report(rank0, 1), own_report(rank1, 1)
    assert len(reports[0]) == len(reports[1])
    assert actions(decide(rank0, np.minimum(*reports))) == ["fatal"]


def test_a_round_covering_entries_closed_since_its_announcement_agrees() -> None:
    """Two entries were announced; a cancel closed the first on every rank before the round."""
    tables = [[op(2)], [op(2)]]
    reports = [own_report(table, 2) for table in tables]
    assert all(len(r) == store_ops.HEADER + 2 * store_ops.COLUMNS for r in reports)
    assert reports[0][1] == reports[1][1]
    assert actions(decide(tables[0], np.minimum(*reports))) == ["return_all 2", "settle 2"]

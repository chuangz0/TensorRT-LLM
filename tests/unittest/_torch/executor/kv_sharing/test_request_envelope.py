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
"""What kv_sharing sends with the executor's request broadcast reaches every tensor-parallel rank:
rank 0's round size, and its store update (the store hits it measured again for waiting requests
and a publishing switch), applied on every rank before any request is activated. Two ranks run the
executor's own fetch and broadcast path over a fake communicator."""

import collections
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
from tensorrt_llm._torch.pyexecutor.kv_sharing.orchestrator import SharedKvOrchestrator
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.request_utils import RequestBroadcaster
from tensorrt_llm._torch.pyexecutor.scheduler import FCFSWaitingQueue

pytestmark = pytest.mark.cpu_only

WAITING_ID = 7
ROUND_SIZE = 3


class Probe:
    """Rank 0's probe: it measured the waiting request's hit again; records the order of calls."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def take_refreshed_hits(self, waiting):
        self.calls.append("take_refreshed_hits")
        return ((WAITING_ID, 64),)

    def hold_until_probed(self, fetched):
        self.calls.append("hold_until_probed")
        return fetched


class Sharing:
    """The executor's kv_sharing hooks on one rank: rank 0 announces a round, has measured the
    waiting request's hit again and pauses publishing; building and applying the update is the
    orchestrator's own code."""

    _shared_table = True

    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.counters = collections.Counter()
        self.announced = None
        self._admitting = True
        # Only rank 0 probes and decides publishing; here a pause began.
        self._probe = Probe() if rank == 0 else None
        self._publication_breaker = SimpleNamespace(allows=lambda now: False) if rank == 0 else None
        self._publishing = True
        self._unpublished = {7: object()}  # a range waiting for room, dropped at the pause

    def request_queue_wait(self, timeout):
        return timeout

    def round_to_announce(self) -> int:
        return ROUND_SIZE if self.rank == 0 else 0

    def round_announced(self, entries: int) -> None:
        self.announced = entries

    take_rank0_items = SharedKvOrchestrator.take_rank0_items
    apply_store_update = SharedKvOrchestrator.apply_store_update


def rank(index: int, wire: dict) -> PyExecutor:
    def broadcast(value, root):
        if index == root:
            wire["payload"] = deepcopy(value)
        return deepcopy(wire["payload"])

    def broadcast_int64(values, root, prefer_cpu):
        if index == root:
            wire["count"] = list(values)
        return np.asarray(wire["count"], dtype=np.int64)

    executor = object.__new__(PyExecutor)
    executor.kv_connector_manager = None
    executor.control_requests = []
    executor.canceled_req_ids = []
    executor.request_accumulated = []
    executor.is_shutdown = False
    executor._disable_mpi = False
    executor.hang_detector = SimpleNamespace(pause=nullcontext)
    executor.dist = SimpleNamespace(
        rank=index,
        world_size=2,
        tp_size=2,
        cp_size=1,
        has_pp=False,
        broadcast=broadcast,
        broadcast_int64=broadcast_int64,
    )
    executor.request_broadcaster = RequestBroadcaster(executor.dist, executor.hang_detector)
    executor.executor_request_queue = Mock()
    executor.executor_request_queue.get_from_request_queue.return_value = []
    executor.kv_sharing = Sharing(index)
    executor.waiting_queue = FCFSWaitingQueue()
    executor.waiting_queue.add_request(RequestQueueItem(WAITING_ID, store_hit_tokens=512))
    return executor


def test_the_round_size_and_the_store_update_reach_every_rank_before_activation() -> None:
    wire: dict = {}
    executors = [rank(index, wire) for index in range(2)]
    for executor in executors:  # rank 0 broadcasts first, as the root does
        executor._fetch_and_enqueue_requests(executor.waiting_queue, total_num_live_requests=1)
    for executor in executors:
        assert executor.kv_sharing.announced == ROUND_SIZE
        (waiting,) = executor.waiting_queue
        assert waiting.store_hit_tokens == 64
        assert executor.kv_sharing.counters["hits_refreshed"] == 1
        assert executor.kv_sharing._publishing is False and not executor.kv_sharing._unpublished
    # The hits are taken before the hold releases requests: a refresh scan forgets hits not in
    # the waiting queue, which a released request joins only after the broadcast.
    assert executors[0].kv_sharing._probe.calls == ["take_refreshed_hits", "hold_until_probed"]

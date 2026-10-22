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
"""The orchestrator over a fake lender, a fake store and the real transfer manager holding leases:
what the rounds do with fetches and publications, and what a cancel, a stopped lender and the
shutdown leave behind."""

import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.disaggregation._assembly import RegionNamespace
from tensorrt_llm._torch.disaggregation.base.backend import Capabilities, Outcome
from tensorrt_llm._torch.disaggregation.lifecycle.retirement import RetirementWatchdog
from tensorrt_llm._torch.disaggregation.transfer_manager import LifecycleBinding, TransferManager
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import (
    LeaseView,
    LentLayerGroupRows,
    MemoryKind,
    MemoryRegion,
    ResumeBounds,
)
from tensorrt_llm._torch.pyexecutor.kv_sharing.orchestrator import (
    BINDING_ID,
    DOMAIN_ID,
    PARKED_STATE,
    PROFILE_ID,
    SharedKvOrchestrator,
    SharingPolicy,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState

pytestmark = pytest.mark.cpu_only

TPB = 4
REGION = MemoryRegion("layout", 4096, 64, 16, 4, (0,), MemoryKind.HOST)  # four one-block slots
PROMPT = 13  # three whole blocks before the last token


class Lease:
    def __init__(self, rows: int, failure: str | None) -> None:
        addresses = REGION.address + REGION.slot_bytes * np.arange(rows, dtype=np.int64)
        keys = np.zeros((rows, 54), dtype=np.uint8)
        group = LentLayerGroupRows(0, np.arange(rows, dtype=np.int64), keys, addresses, 0)
        self.view = LeaseView((group,))
        self.failure = failure
        self.released = False

    def poll(self):
        return None if self.failure else self.view

    def release(self) -> None:
        self.released = True


class Lender:
    """Lends every range at once, or refuses it; a returned fetch settles when ``settle`` says."""

    regions = (REGION,)

    def __init__(self) -> None:
        self.stopped = False
        self.refusal: str | None = None
        self.lend_error: Exception | None = None
        self.settled: set[int] = set()
        self.leases: list[Lease] = []

    def _lend(self, request, start, end, *, memory_kind, rows, ready_event) -> Lease:
        if self.lend_error is not None:
            raise self.lend_error
        self.leases.append(Lease((end - start) // TPB, self.refusal))
        return self.leases[-1]

    lend_read = lend_write = _lend

    def return_writes(self, entries):
        for _, lease, _ in entries:
            lease.release()
        return [None] * len(entries)

    def progress(self) -> None:
        pass

    def poll_resume_bounds(self, request):
        if request.py_request_id not in self.settled:
            return None
        return ResumeBounds(max_position=request.py_store_hit_tokens, min_position=0)

    def deferred_close_page_counts(self) -> dict[int, int]:
        return {}


class Store:
    """The backend: each operation ends when the test says, delivering every row or none."""

    capabilities = Capabilities(fetch=True, publish=True)

    def __init__(self) -> None:
        self.attempts: list[SimpleNamespace] = []

    def _submit(self, view, hint):
        attempt = SimpleNamespace(view=view, outcome=None, ended=False)
        attempt.poll = lambda: attempt.outcome
        attempt.quiesced = lambda: attempt.ended
        self.attempts.append(attempt)
        return attempt

    fetch = publish = _submit

    def end(self, index: int, delivered: bool = True) -> None:
        attempt = self.attempts[index]
        attempt.outcome = Outcome.for_view(attempt.view, attempt.view.row_masks(delivered))
        attempt.ended = True

    def shutdown(self, timeout: float) -> frozenset:
        """Ends what still runs, as a store client's teardown does."""
        for attempt in self.attempts:
            if attempt.outcome is None:
                attempt.outcome = Outcome.for_view(attempt.view, fault="shutdown")
                attempt.ended = True
        return frozenset()


class Manager:
    """The KV cache manager's side of a fetch: claims, frees and resumes."""

    tokens_per_block = TPB
    max_seq_len = 64

    def __init__(self) -> None:
        self.kv_cache_map: dict[int, object] = {}
        self.resumed: dict[int, int] = {}

    def probe_context_reuse(self, request) -> int:
        return 0

    def prepare_context_cache(self, request, reuse_limit) -> int:
        self.kv_cache_map[request.py_request_id] = object()
        return reuse_limit

    def free_resources(self, request) -> None:
        del self.kv_cache_map[request.py_request_id]

    def get_primary_block_counts(self) -> tuple[int, int]:
        return 0, 1000

    def resume_served_prefix(self, request, position: int) -> None:
        request.state = LlmRequestState.CONTEXT_INIT
        self.resumed[request.py_request_id] = position


class Event:
    def record(self, stream) -> None:
        pass


def request(request_id: int, stored: int = 8, computed: int = 0) -> SimpleNamespace:
    """A new text request whose first ``stored`` prompt tokens the store holds."""
    return SimpleNamespace(
        py_request_id=request_id,
        state=LlmRequestState.CONTEXT_INIT,
        is_first_context_chunk=True,
        is_finished=False,
        is_dummy=False,
        py_multimodal_data=None,
        prompt_len=PROMPT,
        py_store_hit_tokens=stored,
        py_published_end=None,
        prepopulated_prompt_len=0,
        context_current_position=computed,
        set_prepopulated_prompt_len=lambda length, tokens_per_block: None,
    )


class Server:
    """One rank: the orchestrator, the transfer manager in hold mode, and an iteration."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(torch.cuda, "Event", Event)
        self.lender, self.store, self.manager = Lender(), Store(), Manager()
        self.transfers = TransferManager(wake=lambda: None, hold_returns=True)
        self.transfers.assemble(
            BINDING_ID,
            RegionNamespace((REGION,), MemoryKind.HOST),
            PROFILE_ID,
            lambda regions, wake, config, post: self.store,
            domain_id=DOMAIN_ID,
            lender=self.lender,
            lifecycle=LifecycleBinding(RetirementWatchdog(lambda event: None), 60.0),
            provider_cleanup_complete=lambda: True,
        )
        coordinator = SimpleNamespace(
            take_shared_domain_failures=self.transfers.take_domain_failures
        )
        policy = SharingPolicy(
            fetch=True,
            publish=True,
            min_fetch_tokens=TPB,
            max_fetch_tokens=16 * TPB,
            fetch_timeout_s=60.0,
            publication_tokens=3 * TPB,
            max_util_for_resume=1.0,
        )
        self.orchestrator = SharedKvOrchestrator(
            policy=policy,
            kv_cache_manager=self.manager,
            lender=self.lender,
            transfers=self.transfers,
            coordinator=coordinator,
            execution_stream=None,
            watchdog=SimpleNamespace(stop=lambda: None),
            probe=None,
        )

    def park(self, *requests) -> None:
        self.orchestrator.add_fetch_candidates(list(requests))
        assert self.orchestrator.start_fetches() == {r.py_request_id for r in requests}
        assert all(r.state == PARKED_STATE for r in requests)

    def iterate(self) -> None:
        """A loop head (the transfer manager advances) and the start of scheduling (a round)."""
        self.transfers.advance(time.monotonic())
        self.orchestrator.resolve_store_operations()

    @property
    def counters(self):
        return self.orchestrator.counters

    def shut_down(self) -> None:
        self.transfers.begin_shutdown(time.monotonic() + 5)
        self.orchestrator.shutdown()
        assert self.transfers.shutdown_status().complete
        for binding in self.transfers.bindings.values():
            if binding.shutdown_thread is not None:
                binding.shutdown_thread.join(1)


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch):
    server = Server(monkeypatch)
    yield server
    server.shut_down()


def test_a_fetch_whose_every_row_arrived_resumes_once_its_copy_settled(server):
    fetched = request(1)
    server.park(fetched)
    server.iterate()  # the store starts reading
    server.store.end(0)
    server.iterate()  # every row arrived: the lease goes back, the copy into the cache is queued
    assert server.lender.leases[0].released and not server.manager.resumed
    server.lender.settled.add(1)
    server.iterate()
    assert server.manager.resumed == {1: 8} and fetched.py_published_end == 8
    assert server.counters["fetches_resumed"] == 1 and server.counters["fetched_blocks"] == 2


def test_the_shutdown_drops_parked_and_settling_fetches_and_returns_their_leases(server):
    parked, settling = request(1), request(2)
    server.park(parked, settling)
    server.iterate()
    server.store.end(1)
    server.iterate()  # the second fetch settles; the first still reads
    server.orchestrator.begin_shutdown()
    for dropped in (parked, settling):
        assert dropped.state == LlmRequestState.CONTEXT_INIT
        assert dropped.py_request_id not in server.manager.kv_cache_map
    assert server.counters["fetches_dropped_shutdown"] == 2
    # The read still running keeps its lease until its access ends.
    assert not server.lender.leases[0].released
    server.store.end(0)
    server.iterate()
    assert server.lender.leases[0].released and not server.manager.resumed


def test_a_cancel_while_settling_closes_the_fetch_without_a_resume(server):
    cancelled = request(1)
    server.park(cancelled)
    server.iterate()
    server.store.end(0)
    server.iterate()
    server.orchestrator.cancel_fetch(cancelled)
    assert cancelled.state == LlmRequestState.CONTEXT_INIT  # no longer reads as in transfer
    server.lender.settled.add(1)
    server.iterate()
    assert not server.manager.resumed and not server.transfers.operations


def test_a_refused_lend_drops_its_fetch_and_sharing_goes_on(server):
    server.lender.refusal = "no room"
    refused = request(1)
    server.park(refused)
    server.iterate()
    assert server.lender.leases[0].released and refused.state == LlmRequestState.CONTEXT_INIT
    assert server.counters["fetches_not_started"] == server.counters["fetches_dropped_short"] == 1
    server.lender.refusal = None
    server.park(request(2))  # still admitting


@pytest.mark.parametrize(
    "failure,dropped", [("a lend raises", 2), ("the lender stops on its own", 1)]
)
def test_a_stopped_lender_stops_sharing_and_drops_fetches(server, failure, dropped):
    settling, later = request(1), request(2)
    server.park(settling)
    server.iterate()
    server.store.end(0)
    server.iterate()
    if failure == "a lend raises":
        server.lender.lend_error = RuntimeError("the lender stopped")
        server.park(later)
    else:
        server.lender.stopped = True
    server.iterate()
    assert settling.state == LlmRequestState.CONTEXT_INIT and not server.manager.resumed
    # The settling fetch, and the one that never started where the lend raised.
    assert server.counters["fetches_dropped_stopped"] == dropped
    server.orchestrator.add_fetch_candidates([request(3)])
    assert server.orchestrator.start_fetches() == set()  # sharing stopped
    # A stopped lender domain tears its store binding down on a thread, which ends here.
    server.shut_down()


def test_a_publication_that_does_not_fit_waits_for_the_open_one(server):
    first, second = request(1, stored=0, computed=PROMPT), request(2, stored=0, computed=PROMPT)
    server.manager.kv_cache_map.update({1: object(), 2: object()})
    server.orchestrator.publish_committed_blocks([first, second])
    assert len(server.lender.leases) == 1 and second.py_published_end is None
    assert server.counters["publications_waited"] == 1
    server.iterate()  # the store starts writing the first
    server.store.end(0)
    server.iterate()  # the first closes; the second takes its room
    assert server.lender.leases[0].released and second.py_published_end == 3 * TPB
    server.iterate()
    server.store.end(1)
    server.iterate()
    assert server.counters["published_blocks"] == 6

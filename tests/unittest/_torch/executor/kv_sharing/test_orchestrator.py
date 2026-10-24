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
what the rounds do with fetches and publications, what a cancel, a stopped lender and the shutdown
leave behind, what bounds publications on each data path, and where a model with a sliding window
fetches to and what it publishes."""

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
from tensorrt_llm._torch.pyexecutor.kv_sharing import orchestrator as orchestrator_module
from tensorrt_llm._torch.pyexecutor.kv_sharing.orchestrator import (
    BINDING_ID,
    DOMAIN_ID,
    MAX_OPEN_PUBLICATIONS,
    PARKED_STATE,
    PROFILE_ID,
    SharedKvOrchestrator,
    SharingPolicy,
    StagingBudget,
)
from tensorrt_llm._torch.pyexecutor.kv_sharing.startup import _staging_budget
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState

pytestmark = pytest.mark.cpu_only

TPB = 4
REGION = MemoryRegion("layout", 4096, 64, 16, 4, (0,), MemoryKind.HOST)  # four one-block slots
POOL = MemoryRegion("pool", 1 << 20, 1024, 16, 64, (0,), MemoryKind.DEVICE)  # the KV cache's
PROMPT = 13  # three whole blocks before the last token
PROMPT_PAGES = 5  # the prompt's four pages and one step
WINDOW_ROWS = 2  # the rows a windowed layer group lends: its window of two blocks
# Staging for three blocks of open publications beside a fetch, one layer group.
THREE_BLOCKS = StagingBudget(free_rows=(3,), rows_per_block=(1,), window_rows=(0,))


class Lease:
    def __init__(
        self, rows: list[int], memory_kind: MemoryKind, failure: str | None, ready_event
    ) -> None:
        """Lends ``rows[lg]`` rows of each layer group ``lg``."""
        groups = []
        for lg, count in enumerate(rows):
            ordinals = np.arange(count, dtype=np.int64)
            keys = np.zeros((count, 54), dtype=np.uint8)
            if memory_kind is MemoryKind.HOST:
                addresses = REGION.address + REGION.slot_bytes * ordinals
                groups.append(LentLayerGroupRows(lg, ordinals, keys, addresses, 0))
            else:
                groups.append(
                    LentLayerGroupRows(lg, ordinals, keys, slots=ordinals, region_indices=(1,))
                )
        self.view = LeaseView(tuple(groups))
        self.failure = failure
        self.ready_event = ready_event
        self.released = False

    def poll(self):
        return None if self.failure else self.view

    def release(self) -> None:
        self.released = True


class Lender:
    """Lends every range at once, or refuses it; a returned fetch settles when ``settle`` says.
    Its layer group keeps the whole context; with ``windowed`` a second one keeps a window of
    WINDOW_ROWS blocks, whose every lease lends WINDOW_ROWS rows."""

    regions = (REGION, POOL)

    def __init__(self, windowed: bool = False) -> None:
        self.stopped = False
        self.refusal: str | None = None
        self.lend_error: Exception | None = None
        self.settled: set[int] = set()
        self.leases: list[Lease] = []
        self.ranges: list[tuple] = []  # (start, end) of each lend
        groups = [SimpleNamespace(kept_window=None, num_sink_blocks=0, pool_group=0)]
        if windowed:
            window = SimpleNamespace(kept_window=WINDOW_ROWS * TPB, num_sink_blocks=0, pool_group=0)
            groups.append(window)
        self.page_layout = SimpleNamespace(
            layer_groups=tuple(groups), pool_slot_bytes={0: (REGION.slot_bytes,)}
        )

    def _lend(self, request, start, end, *, memory_kind, rows, ready_event) -> Lease:
        if self.lend_error is not None:
            raise self.lend_error
        self.ranges.append((start, end))
        full_start = start[0] if isinstance(start, list) else start
        windows = len(self.page_layout.layer_groups) - 1
        counts = [(end - full_start) // TPB] + [WINDOW_ROWS] * windows
        self.leases.append(Lease(counts, memory_kind, self.refusal, ready_event))
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

    capabilities = Capabilities(fetch=True, publish=True, slot_rows=True)

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
        self.total_pages = 1000

    def probe_context_reuse(self, request) -> int:
        return 0

    def prepare_context_cache(self, request, reuse_limit) -> int:
        self.kv_cache_map[request.py_request_id] = object()
        return reuse_limit

    def free_resources(self, request) -> None:
        del self.kv_cache_map[request.py_request_id]

    def get_primary_block_counts(self) -> tuple[int, int]:
        return 0, self.total_pages

    def resume_served_prefix(self, request, position: int) -> None:
        request.state = LlmRequestState.CONTEXT_INIT
        self.resumed[request.py_request_id] = position


# The pages the largest request's resume needs, which the room for pinned pages leaves aside.
RESUME_PAGES = Manager.max_seq_len // TPB + 1


class Event:
    def __init__(self) -> None:
        self.records = 0

    def record(self, stream) -> None:
        self.records += 1


def request(request_id: int, stored: int = 8, computed: int = 0) -> SimpleNamespace:
    """A new text request whose first ``stored`` prompt tokens the store holds."""
    return SimpleNamespace(
        py_request_id=request_id,
        state=LlmRequestState.CONTEXT_INIT,
        is_first_context_chunk=True,
        is_finished=False,
        is_dummy=False,
        is_context_only_request=False,
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

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        memory_kind: MemoryKind = MemoryKind.HOST,
        min_over_ranks=None,
        windowed: bool = False,
        max_fetch_tokens: int = 16 * TPB,
    ):
        monkeypatch.setattr(torch.cuda, "Event", Event)
        self.lender, self.store, self.manager = Lender(windowed), Store(), Manager()
        self.transfers = TransferManager(wake=lambda: None, hold_returns=True)
        self.transfers.assemble(
            BINDING_ID,
            RegionNamespace((REGION, POOL), memory_kind),
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
            memory_kind=memory_kind,
            fetch=True,
            publish=True,
            min_fetch_tokens=TPB,
            max_fetch_tokens=max_fetch_tokens,
            fetch_timeout_s=60.0,
            staging=THREE_BLOCKS if memory_kind is MemoryKind.HOST else None,
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
            min_over_ranks=min_over_ranks,
        )

    def park(self, *requests) -> None:
        self.orchestrator.add_fetch_candidates(list(requests))
        assert self.orchestrator.start_fetches() == {r.py_request_id for r in requests}
        assert all(r.state == PARKED_STATE for r in requests)

    def iterate(self) -> None:
        """A loop head (the transfer manager advances), the start of scheduling (a round) and
        the request broadcast, which announces the next round to every rank."""
        self.transfers.advance(time.monotonic())
        self.orchestrator.resolve_store_operations()
        self.orchestrator.round_announced(self.orchestrator.round_to_announce())

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


@pytest.fixture
def gpu_direct_server(monkeypatch: pytest.MonkeyPatch):
    server = Server(monkeypatch, MemoryKind.DEVICE)
    yield server
    server.shut_down()


def computed(*request_ids: int) -> list[SimpleNamespace]:
    """Requests that computed their whole prompts, none of which the store held."""
    return [request(request_id, stored=0, computed=PROMPT) for request_id in request_ids]


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
    assert server.counters["fetched_objects"] == 2
    assert server.counters["fetched_bytes"] == 2 * REGION.slot_bytes


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
    assert server.counters["publication_waits"] == 1
    server.iterate()  # the store starts writing the first
    server.store.end(0)
    server.iterate()  # the first closes; the second takes its room
    assert server.lender.leases[0].released and second.py_published_end == 3 * TPB
    server.iterate()
    server.store.end(1)
    server.iterate()
    assert server.counters["published_objects"] == 6


def test_blocks_another_server_is_still_putting_are_skipped_not_failed(server):
    """The store took 1 of the 3 blocks: the other 2 were another server's open puts (Mooncake:
    exists, uncommitted), which that server completes."""
    server.manager.kv_cache_map[1] = object()
    server.orchestrator.publish_committed_blocks(computed(1))
    server.iterate()  # the store starts writing
    attempt = server.store.attempts[0]
    attempt.outcome = Outcome.for_view(attempt.view, (np.array([True, False, False]),))
    attempt.ended = True
    server.iterate()
    assert server.lender.leases[0].released
    assert server.counters["published_objects"] == 1
    assert server.counters["publication_objects_skipped"] == 2
    assert server.counters["publications_failed"] == 0


def test_a_publication_the_store_faulted_counts_as_failed(server):
    server.manager.kv_cache_map[1] = object()
    server.orchestrator.publish_committed_blocks(computed(1))
    server.iterate()  # the store starts writing
    attempt = server.store.attempts[0]
    delivered = (np.array([True, False, False]),)
    attempt.outcome = Outcome.for_view(attempt.view, delivered, "the store failed")
    attempt.ended = True
    server.iterate()
    assert server.lender.leases[0].released
    assert server.counters["publications_failed"] == 1
    assert server.counters["published_objects"] == 0
    assert server.counters["publication_objects_skipped"] == 0


def test_a_gpu_direct_publication_waits_while_its_locked_pages_would_crowd_out_resumes(
    gpu_direct_server,
):
    server = gpu_direct_server
    # Room for two requests' locked pages beside the largest request's resume.
    server.manager.total_pages = RESUME_PAGES + 2 * PROMPT_PAGES
    requests = computed(1, 2, 3)
    server.manager.kv_cache_map.update({r.py_request_id: object() for r in requests})
    server.orchestrator.publish_committed_blocks(requests)
    assert len(server.lender.leases) == 2 and requests[2].py_published_end is None
    # A fetch would pin pages too: it waits as well.
    server.orchestrator.add_fetch_candidates([request(4)])
    assert server.orchestrator.start_fetches() == set()
    assert server.counters["fetches_skipped_for_pages"] == 1
    server.iterate()
    server.store.end(1)
    server.iterate()  # the second request's pages are free to go again
    assert len(server.lender.leases) == 3 and requests[2].py_published_end == 3 * TPB


def test_no_later_lending_phase_records_the_event_a_lease_waits_for(server):
    """A lease in place asks its event's latest record: phases of later iterations record other
    events, so no later forward holds back an earlier lease."""
    for request_id in (1, 2, 3):
        server.manager.kv_cache_map[request_id] = object()
        server.orchestrator.publish_committed_blocks([request(request_id, stored=0, computed=TPB)])
        server.iterate()
    events = [lease.ready_event for lease in server.lender.leases]
    assert len({id(event) for event in events}) == 3
    assert [event.records for event in events] == [1, 1, 1]


def test_gpu_direct_publications_wait_at_the_cap_unless_they_end_a_context_only_request(
    gpu_direct_server,
):
    """A context-only request is freed soon after its context: its last range cannot wait for
    the cap. Under GPU Direct a waiting range holds back no range behind it, and a chunked
    context-only request's earlier range waits until it merges into the last."""
    server = gpu_direct_server
    full = computed(*range(1, MAX_OPEN_PUBLICATIONS + 1))
    waiting = request(MAX_OPEN_PUBLICATIONS + 1, stored=0, computed=PROMPT)
    chunked = request(MAX_OPEN_PUBLICATIONS + 2, stored=0, computed=TPB)  # its first chunk
    chunked.is_context_only_request = True
    for each in (*full, waiting, chunked):
        server.manager.kv_cache_map[each.py_request_id] = object()
    server.orchestrator.publish_committed_blocks([*full, waiting, chunked])
    assert len(server.lender.leases) == MAX_OPEN_PUBLICATIONS
    assert waiting.py_published_end is None and chunked.py_published_end is None
    chunked.context_current_position = PROMPT  # its last chunk
    server.orchestrator.publish_committed_blocks([chunked])
    assert len(server.lender.leases) == MAX_OPEN_PUBLICATIONS + 1
    assert chunked.py_published_end == 3 * TPB and waiting.py_published_end is None
    server.iterate()  # the store starts writing
    server.store.end(0)
    server.store.end(1)
    server.iterate()  # two locks go back: below the cap again
    assert waiting.py_published_end == 3 * TPB


def test_a_gpu_direct_publication_that_can_never_fit_is_dropped_once_and_logged(
    gpu_direct_server, monkeypatch
):
    server = gpu_direct_server
    warnings: list[str] = []
    monkeypatch.setattr(
        orchestrator_module.logger, "warning_once", lambda message, key: warnings.append(message)
    )
    server.manager.total_pages = RESUME_PAGES  # no room at all beside the largest resume
    dropped = request(1, stored=0, computed=TPB)
    server.manager.kv_cache_map[1] = object()
    server.orchestrator.publish_committed_blocks([dropped])
    dropped.context_current_position = PROMPT  # a later chunk publishes nothing either
    server.orchestrator.publish_committed_blocks([dropped])
    server.iterate()
    assert not server.lender.leases and dropped.py_published_end == PROMPT
    assert server.counters["publications_dropped_for_pages"] == 1
    assert len(warnings) == 1 and "does not fit the 0 pages left" in warnings[0]


def test_under_tensor_parallelism_a_stopped_store_binding_ends_the_server_before_the_round(
    monkeypatch,
):
    """One rank's store binding stops: that rank's shares would go unread and unwritten, so the
    server ends at the next round, before its collective."""

    def collective(report):
        pytest.fail("the round entered its collective")

    server = Server(monkeypatch, min_over_ranks=collective)
    try:
        server.park(request(1))
        server.orchestrator.round_announced(1)  # the next round covers the parked fetch
        # As the transfer manager does when a store call fails in a way it cannot contain.
        server.transfers._stop_binding(BINDING_ID, "a store call failed")
        with pytest.raises(RuntimeError, match="a store call failed; .* ends the server"):
            server.iterate()
    finally:
        server.shut_down()


def test_refreshed_hits_update_only_waiting_requests(server):
    waiting = [SimpleNamespace(id=1, store_hit_tokens=8), SimpleNamespace(id=2, store_hit_tokens=8)]
    server.orchestrator.apply_refreshed_hits(waiting, ((1, 16), (9, 4)))  # 9 no longer waits
    assert [item.store_hit_tokens for item in waiting] == [16, 8]
    assert server.counters["hits_refreshed"] == 1


def test_a_windowed_publication_lends_each_window_from_block_0_and_counts_objects_from_its_view(
    monkeypatch,
):
    """A request whose first block the store held publishes blocks 1 and 2: the full group from
    the range's start, the windowed one from block 0. The view holds 2 plus 2 rows; the store takes
    3 of the 4 objects, the fourth was another server's open put."""
    server = Server(monkeypatch, windowed=True)
    try:
        server.manager.kv_cache_map[1] = object()
        server.orchestrator.publish_committed_blocks([request(1, stored=TPB, computed=PROMPT)])
        assert server.lender.ranges == [([TPB, 0], 3 * TPB)]
        server.iterate()  # the store starts writing
        attempt = server.store.attempts[0]
        assert [len(group) for group in attempt.view.groups] == [2, WINDOW_ROWS]
        delivered = (np.array([True, True]), np.array([True, False]))
        attempt.outcome = Outcome.for_view(attempt.view, delivered)
        attempt.ended = True
        server.iterate()
        assert server.counters["published_objects"] == 3
        assert server.counters["publication_objects_skipped"] == 1
    finally:
        server.shut_down()


def test_a_windowed_fetch_longer_than_max_fetch_tokens_computes_instead_of_being_cut(monkeypatch):
    """A windowed group's rows are stored only at the ends publications reached: a fetch cut
    shorter would miss them, so the request computes its prompt."""
    server = Server(monkeypatch, windowed=True, max_fetch_tokens=TPB)
    try:
        server.orchestrator.add_fetch_candidates([request(1, stored=2 * TPB)])
        assert server.orchestrator.start_fetches() == set() and not server.lender.ranges
        assert server.counters["fetches_skipped_too_long"] == 1
    finally:
        server.shut_down()


def test_host_staging_holds_one_window_per_publication_beside_a_fetch():
    """Staging less one fetch's rows, per region. gemma-3's shape: a region of its windowed group
    (512 tokens, 16 blocks of 32) with two fetches' slots, and a region of its full group; a fetch
    of 256 blocks."""
    windowed = SimpleNamespace(kept_window=512, num_sink_blocks=0)
    full = SimpleNamespace(kept_window=None, num_sink_blocks=0)
    staging = (
        MemoryRegion("windowed", 1 << 30, 32, 1, 32, (0,), MemoryKind.HOST),
        MemoryRegion("full", 2 << 30, 512, 1, 512, (1,), MemoryKind.HOST),
    )
    lender = SimpleNamespace(
        page_layout=SimpleNamespace(layer_groups=(windowed, full)), regions=(*staging, POOL)
    )
    budget = _staging_budget(lender, fetch_tokens=256 * 32, tokens_per_block=32)
    assert budget == StagingBudget(free_rows=(16, 256), rows_per_block=(0, 1), window_rows=(16, 0))
    # One publication at a time, of up to 256 blocks: the windowed region holds one window.
    assert budget.longest == 256 and budget.fits(256, [])
    assert not budget.fits(1, [1])
    # gpt-oss's shape: one region serves a full group and a window of 4 blocks.
    shared = StagingBudget(free_rows=(260,), rows_per_block=(1,), window_rows=(4,))
    assert shared.longest == 256 and shared.fits(150, [100]) and not shared.fits(153, [100])
    # Full attention alone: the budget counts blocks, as a token budget did.
    assert THREE_BLOCKS.longest == 3 and THREE_BLOCKS.fits(2, [1]) and not THREE_BLOCKS.fits(1, [3])

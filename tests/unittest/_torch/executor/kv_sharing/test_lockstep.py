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
"""Tensor-parallel ranks in lockstep, without MPI: one thread per rank, each with its own
orchestrator, transfer manager holding leases, lender, KV cache manager and store. The ranks run
the same script of arrivals, cancels, finishes and a shutdown, as the request broadcast makes them;
only their stores differ (when each access ends, and whether it delivers every row) and their
clocks. The announced round must keep their tables, decisions, locks, pages and request states
equal in every pass, release no lease before every rank's access ended, and drop a fetch on every
rank in the first round some rank finds it late."""

import io
import math
import random
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional

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
    PARKED_STATE,
    PROFILE_ID,
    SharedKvOrchestrator,
    SharingPolicy,
)
from tensorrt_llm._torch.pyexecutor.kv_sharing.store_ops import OpKind, OpState, StoreOp
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState

pytestmark = pytest.mark.cpu_only

TPB = 4
SLOTS = 64
REGION = MemoryRegion("staging", 4096, 16 * SLOTS, 16, SLOTS, (0,), MemoryKind.HOST)
POOL = MemoryRegion("pool", 1 << 20, 16 * SLOTS, 16, SLOTS, (0,), MemoryKind.DEVICE)
REQUESTS = 14
SHUTDOWN_AT = 30
ITERATIONS = 38
STEP_S = 1.0  # one iteration of every rank's clock; each rank adds its own jitter below one step
FETCH_TIMEOUT_S = 5.5
RANKS = 4
# A fetch appended in pass i is covered by pass i+2's round at the latest (pass i+1 announces it).
COVERED_WITHIN_PASSES = 2
_clock = threading.local()
_real_monotonic = time.monotonic


def rank_clock() -> float:
    """The orchestrator's clock: the rank thread's logical time, or real time after the loop."""
    return getattr(_clock, "now", None) or _real_monotonic()


class Group:
    """The collectives of a tensor-parallel group of threads. A rank entering another collective,
    or none, fails the test instead of hanging it."""

    def __init__(self, ranks: int) -> None:
        self._barrier = threading.Barrier(ranks, timeout=10)
        self._slots: list = [None] * ranks

    def _exchange(self, rank: int, kind: str, value) -> list:
        self._slots[rank] = (kind, value)
        self._barrier.wait()
        entries = list(self._slots)
        self._barrier.wait()  # every rank read the slots before any writes them again
        kinds = {entry[0] for entry in entries}
        assert kinds == {kind}, f"ranks entered different collectives: {entries}"
        return [entry[1] for entry in entries]

    def min(self, rank: int, vector: np.ndarray) -> np.ndarray:
        vectors = self._exchange(rank, "min", np.array(vector))
        assert len({len(v) for v in vectors}) == 1, f"reports of different lengths: {vectors}"
        return np.minimum.reduce(vectors)

    def broadcast(self, rank: int, value: int) -> int:
        return self._exchange(rank, "broadcast", value)[0]


class Lease:
    def __init__(self, lender: "Lender", request, start: int, end: int, memory_kind, kind) -> None:
        ordinals = np.arange((end - start) // TPB, dtype=np.int64)
        keys = np.zeros((len(ordinals), 54), dtype=np.uint8)
        if memory_kind is MemoryKind.HOST:
            addresses = REGION.address + REGION.slot_bytes * ordinals
            group = LentLayerGroupRows(0, ordinals, keys, addresses, 0)
        else:
            group = LentLayerGroupRows(0, ordinals, keys, slots=ordinals, region_indices=(1,))
        self.view = LeaseView((group,))
        self.failure = None
        self.key = (request.py_request_id, kind, start)
        lender.keys[id(self.view)] = self.key
        self._lender, self._request_id, self._end = lender, request.py_request_id, end
        self._locks = memory_kind is MemoryKind.DEVICE  # a host lease takes no page lock
        self.released = False
        if self._locks:
            lender.locks[self._request_id] += 1

    def poll(self):
        return self.view

    def release(self, settles: bool = False) -> None:
        if self.released:
            return
        self.released = True
        self._lender.released[self.key] = self._lender.iteration
        if settles:
            # GPU Direct rows are in place; host staging copies them into the cache later.
            later = self._lender.rng.randint(0, 3) if not self._locks else 0
            self._lender.settling[self._request_id] = (self._end, later)
        if self._locks:
            self._lender.locks[self._request_id] -= 1
            self._lender.manager.unlocked(self._request_id)


class Lender:
    """Lends every range; GPU Direct leases lock their request's pages until they go back. A
    returned host staging fetch settles after a random number of progress calls."""

    regions = (REGION, POOL)
    stopped = False

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.manager: Manager = None
        self.locks: Counter[int] = Counter()
        self.settling: dict[int, tuple[int, int]] = {}  # request id -> (end, progress calls left)
        self.released: dict[tuple, int] = {}  # lease key -> iteration
        self.keys: dict[int, tuple] = {}  # id of a lease's view -> the lease's key
        self.iteration = 0

    def lend_write(self, request, start, end, *, memory_kind, rows, ready_event) -> Lease:
        return Lease(self, request, start, end, memory_kind, OpKind.FETCH)

    def lend_read(self, request, start, end, *, memory_kind, rows, ready_event) -> Lease:
        return Lease(self, request, start, end, memory_kind, OpKind.PUBLISH)

    def return_writes(self, entries):
        for _, lease, masks in entries:
            lease.release(settles=all(mask.all() for mask in masks))
        return [None] * len(entries)

    def progress(self) -> None:
        for request_id, (end, left) in list(self.settling.items()):
            self.settling[request_id] = (end, max(left - 1, 0))

    def poll_resume_bounds(self, request):
        end, left = self.settling.get(request.py_request_id, (0, 1))
        return ResumeBounds(max_position=end, min_position=0) if left == 0 else None

    def deferred_close_page_counts(self) -> dict[int, int]:
        return dict(self.manager.deferred)


class Manager:
    """Pages: a cache holds its whole prompt and one step; a cache freed while a lease locks it
    keeps its pages until the last lock goes."""

    tokens_per_block = TPB
    max_seq_len = 64

    def __init__(self, lender: Lender) -> None:
        self.lender = lender
        lender.manager = self
        self.kv_cache_map: dict[int, object] = {}
        self.pages: dict[int, int] = {}
        self.deferred: dict[int, int] = {}
        self.used, self.total = 0, 400

    def probe_context_reuse(self, request):
        return None if request.py_request_id in self.kv_cache_map else 0

    def prepare_context_cache(self, request, reuse_limit) -> int:
        self.claim(request)
        return reuse_limit

    def claim(self, request) -> None:
        pages = math.ceil(request.prompt_len / TPB) + 1
        self.kv_cache_map[request.py_request_id] = object()
        self.pages[request.py_request_id] = pages
        self.used += pages

    def free_resources(self, request) -> None:
        del self.kv_cache_map[request.py_request_id]
        pages = self.pages.pop(request.py_request_id)
        if self.lender.locks[request.py_request_id]:
            self.deferred[request.py_request_id] = pages
        else:
            self.used -= pages

    def unlocked(self, request_id: int) -> None:
        if request_id in self.deferred and not self.lender.locks[request_id]:
            self.used -= self.deferred.pop(request_id)

    def get_primary_block_counts(self) -> tuple[int, int]:
        return self.used, self.total

    def resume_served_prefix(self, request, position: int) -> None:
        request.state = LlmRequestState.CONTEXT_INIT
        request.context_current_position = position


class Store:
    """Each access ends after a random number of iterations, delivering every row, all but the
    last, or none with a fault."""

    capabilities = Capabilities(fetch=True, publish=True, slot_rows=True)

    def __init__(self, rng: random.Random, keys: dict[int, tuple]) -> None:
        self.rng = rng
        self.keys = keys
        self.running: list[SimpleNamespace] = []
        self.ended: dict[tuple, int] = {}  # lease key -> iteration

    def _submit(self, view, hint):
        attempt = SimpleNamespace(view=view, outcome=None, ended=False, left=self.rng.randint(0, 4))
        attempt.poll = lambda: attempt.outcome
        attempt.quiesced = lambda: attempt.ended
        self.running.append(attempt)
        return attempt

    fetch = publish = _submit

    def tick(self, iteration: int) -> None:
        for attempt in list(self.running):
            if attempt.left:
                attempt.left -= 1
                continue
            masks, fault = attempt.view.row_masks(True), None
            draw = self.rng.random()
            if draw < 0.15:
                masks, fault = attempt.view.row_masks(False), "the store failed"
            elif draw < 0.3:
                masks[0][-1] = False
            attempt.outcome, attempt.ended = Outcome.for_view(attempt.view, masks, fault), True
            self.ended[self.keys[id(attempt.view)]] = iteration
            self.running.remove(attempt)

    def shutdown(self, timeout: float) -> frozenset:
        for attempt in self.running:
            attempt.outcome, attempt.ended = Outcome.for_view(attempt.view, fault="shutdown"), True
        return frozenset()


@dataclass
class Script:
    """The events every rank sees alike, from one seed."""

    arrivals: dict[int, list[tuple[int, int, int]]]  # iteration -> (request id, prompt, hit)
    cancels: dict[int, int]  # iteration -> request id
    finish_after: dict[int, int]  # request id -> iterations from its context to its end

    @classmethod
    def of(cls, seed: int) -> "Script":
        rng = random.Random(seed)
        arrivals: dict[int, list] = {}
        for request_id in range(1, REQUESTS + 1):
            prompt = rng.randint(2 * TPB + 1, 10 * TPB)
            hit = rng.choice([0, rng.randint(1, prompt // TPB) * TPB])
            arrivals.setdefault(rng.randint(0, SHUTDOWN_AT - 4), []).append(
                (request_id, prompt, hit)
            )
        cancels = {rng.randint(1, SHUTDOWN_AT): rng.randint(1, REQUESTS) for _ in range(3)}
        finish_after = {r: rng.randint(1, 5) for r in range(1, REQUESTS + 1)}
        return cls(arrivals, cancels, finish_after)


@dataclass
class RankRecord:
    """What one rank saw, per iteration and at the end."""

    trace: str = ""
    passes: list[tuple] = field(default_factory=list)  # equal on every rank
    parked: list[set[int]] = field(default_factory=list)  # fetch seqs parked after each round
    late_from: dict[int, int] = field(default_factory=dict)  # fetch seq -> first late iteration
    appended: dict[int, int] = field(default_factory=dict)  # fetch seq -> iteration
    released: dict[tuple, int] = field(default_factory=dict)
    ended: dict[tuple, int] = field(default_factory=dict)
    drained: bool = False


class _Trace(io.StringIO):
    def close(self) -> None:  # the orchestrator closes its trace at shutdown; keep the text
        pass


def new_request(request_id: int, prompt: int, hit: int) -> SimpleNamespace:
    return SimpleNamespace(
        py_request_id=request_id,
        state=LlmRequestState.CONTEXT_INIT,
        is_first_context_chunk=True,
        is_finished=False,
        is_dummy=False,
        is_context_only_request=False,
        py_multimodal_data=None,
        prompt_len=prompt,
        py_store_hit_tokens=hit or None,
        py_published_end=None,
        prepopulated_prompt_len=0,
        context_current_position=0,
        set_prepopulated_prompt_len=lambda length, tokens_per_block: None,
        computed_at=None,
    )


def run_rank(
    rank: int,
    group: Group,
    script: Script,
    memory_kind,
    seed: int,
    trace: _Trace,
    diverge_at: Optional[int],
) -> RankRecord:
    """One rank's run. With ``diverge_at``, rank 1 adds an entry of its own at the head of its
    table in the first pass from then on that has entries."""
    rng = random.Random(seed * 1000 + rank)  # this rank's store and clock
    lender = Lender(rng)
    store = Store(rng, lender.keys)
    manager = Manager(lender)
    transfers = TransferManager(wake=lambda: None, hold_returns=True)
    transfers.assemble(
        BINDING_ID,
        RegionNamespace((REGION, POOL), memory_kind),
        PROFILE_ID,
        lambda regions, wake, config, post: store,
        domain_id=DOMAIN_ID,
        lender=lender,
        lifecycle=LifecycleBinding(RetirementWatchdog(lambda event: None), 600.0),
        provider_cleanup_complete=lambda: True,
    )
    orchestrator = SharedKvOrchestrator(
        policy=SharingPolicy(
            memory_kind=memory_kind,
            fetch=True,
            publish=True,
            min_fetch_tokens=TPB,
            max_fetch_tokens=16 * TPB,
            fetch_timeout_s=FETCH_TIMEOUT_S,
            publication_tokens=12 * TPB if memory_kind is MemoryKind.HOST else None,
            max_util_for_resume=1.0,
        ),
        kv_cache_manager=manager,
        lender=lender,
        transfers=transfers,
        coordinator=SimpleNamespace(take_shared_domain_failures=transfers.take_domain_failures),
        execution_stream=None,
        watchdog=SimpleNamespace(stop=lambda: None),
        probe=None,
        min_over_ranks=lambda report: group.min(rank, report),
        trace=trace,
    )
    requests: dict[int, SimpleNamespace] = {}
    record = RankRecord()
    for iteration in range(ITERATIONS):
        _clock.now = 1000.0 + iteration * STEP_S + rng.random() * 0.9 * STEP_S
        lender.iteration = iteration
        transfers.advance(_clock.now)
        for op in orchestrator._ops:
            if op.state is OpState.PARKED and _clock.now >= op.deadline:
                record.late_from.setdefault(op.seq, iteration)
        orchestrator.resolve_store_operations()
        record.parked.append({op.seq for op in orchestrator._ops if op.state is OpState.PARKED})
        orchestrator.round_announced(group.broadcast(rank, orchestrator.round_to_announce()))
        if rank == 1 and diverge_at is not None and iteration >= diverge_at and orchestrator._ops:
            head = orchestrator._ops[0]
            extra = StoreOp(0, OpKind.PUBLISH, head.request, memory_kind, OpState.OPEN, 0, TPB)
            orchestrator._ops.insert(0, extra)
            diverge_at = None
        if iteration == SHUTDOWN_AT:
            orchestrator.begin_shutdown()
        cancelled = requests.get(script.cancels.get(iteration))
        if cancelled is not None and not cancelled.is_finished:
            orchestrator.cancel_fetch(cancelled)
            if cancelled.py_request_id in manager.kv_cache_map:
                manager.free_resources(cancelled)
            cancelled.is_finished = True
        arrived = [new_request(*args) for args in script.arrivals.get(iteration, [])]
        if iteration < SHUTDOWN_AT:
            requests.update((r.py_request_id, r) for r in arrived)
            orchestrator.add_fetch_candidates(arrived)
            orchestrator.start_fetches()
        context = []
        for request in requests.values():
            computes = request.state == LlmRequestState.CONTEXT_INIT and request.computed_at is None
            if computes and not request.is_finished:
                if request.py_request_id not in manager.kv_cache_map:
                    manager.claim(request)
                request.context_current_position = request.prompt_len
                request.computed_at = iteration
                context.append(request)
        orchestrator.publish_committed_blocks(context)
        for op in orchestrator._ops:
            if op.kind is OpKind.FETCH:
                record.appended.setdefault(op.seq, iteration)
        for request in requests.values():
            due = request.computed_at is not None and not request.is_finished
            if (
                due
                and request.computed_at + script.finish_after[request.py_request_id] <= iteration
            ):
                manager.free_resources(request)
                request.is_finished = True
        store.tick(iteration)
        record.passes.append(
            (
                iteration,
                tuple(
                    (op.seq, op.kind.name, op.state.name, op.start, op.end)
                    for op in orchestrator._ops
                ),
                sum(lender.locks.values()),
                manager.used,
                tuple(manager.deferred.items()),
                tuple(
                    (r.py_request_id, r.state, r.is_finished, r.computed_at)
                    for r in requests.values()
                ),
            )
        )
    record.released, record.ended = dict(lender.released), dict(store.ended)
    _clock.now = None  # the drain runs on real time
    transfers.begin_shutdown(_real_monotonic() + 5)
    orchestrator.shutdown()
    record.drained = transfers.shutdown_status().complete
    for binding in transfers.bindings.values():
        if binding.shutdown_thread is not None:
            binding.shutdown_thread.join(5)
    record.trace = trace.getvalue()
    return record


def run_ranks(
    memory_kind: MemoryKind, seed: int, diverge_at: Optional[int] = None
) -> tuple[list[RankRecord | BaseException], list[_Trace]]:
    """Every rank's record, or what it raised, and its trace."""
    group, script = Group(RANKS), Script.of(seed)
    traces = [_Trace() for _ in range(RANKS)]
    outcomes: list = [None] * RANKS

    def worker(rank: int) -> None:
        try:
            outcomes[rank] = run_rank(
                rank, group, script, memory_kind, seed, traces[rank], diverge_at
            )
        except BaseException as error:  # handed to the test's thread; peers left waiting
            outcomes[rank] = error  # in a collective break out at the barrier's timeout

    threads = [threading.Thread(target=worker, args=(rank,)) for rank in range(RANKS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return outcomes, traces


@pytest.fixture(autouse=True)
def rank_clocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """No CUDA events; the orchestrator reads its rank thread's clock."""
    monkeypatch.setattr(torch.cuda, "Event", lambda: SimpleNamespace(record=lambda stream: None))
    monkeypatch.setattr(orchestrator_module, "time", SimpleNamespace(monotonic=rank_clock))


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("memory_kind", [MemoryKind.HOST, MemoryKind.DEVICE], ids=["host", "gpu"])
def test_ranks_whose_stores_differ_agree_in_every_round(memory_kind: MemoryKind, seed: int) -> None:
    outcomes, _ = run_ranks(memory_kind, seed)
    errors = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
    # A rank's own error first: its peers then time out in the collective it left.
    for error in sorted(errors, key=lambda error: isinstance(error, threading.BrokenBarrierError)):
        raise error
    records = outcomes
    first = records[0]
    assert first.trace.count("round") > 10, first.trace  # the script exercises rounds
    for record in records[1:]:
        # The same tables, decisions, locks, pages and request states in every pass.
        assert record.trace == first.trace
        for mine, theirs in zip(record.passes, first.passes):
            assert mine == theirs
    for record in records:
        assert record.drained
        # No lease went back before every rank's access ended (the loop's releases).
        for key, released in record.released.items():
            ends = [other.ended[key] for other in records if key in other.ended]
            assert all(released > end for end in ends), (key, released, ends)
    # A fetch leaves PARKED on every rank in the first round some rank finds it late, once that
    # round covers it (announced the pass after it was appended).
    for seq in {seq for record in records for seq in record.late_from}:
        late = min(record.late_from[seq] for record in records if seq in record.late_from)
        covered = first.appended[seq] + COVERED_WITHIN_PASSES
        decided = max(late, covered)
        for record in records:
            assert all(seq not in parked for parked in record.parked[decided:]), (seq, decided)
    # The shutdown left no request parked.
    *_, last_states = first.passes[-1]
    assert all(state != PARKED_STATE for _, state, _, _ in last_states)


def test_a_table_that_differs_on_one_rank_ends_every_rank_in_the_same_round() -> None:
    outcomes, traces = run_ranks(MemoryKind.DEVICE, seed=0, diverge_at=5)
    for outcome in outcomes:  # a raise, not a collective left waiting
        assert isinstance(outcome, RuntimeError) and "tables differ" in str(outcome), outcome
    last_rounds = [trace.getvalue().splitlines()[-1].split()[1:] for trace in traces]
    fields = [dict(field.split("=", 1) for field in line) for line in last_rounds]
    assert {(f["pass"], f["decisions"]) for f in fields} == {(fields[0]["pass"], "fatal")}

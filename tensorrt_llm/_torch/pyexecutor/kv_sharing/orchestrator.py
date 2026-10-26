# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The executor's one entry to KV sharing.

Its methods run on the executor thread, the only one the lender and the transfer manager accept
calls on once the executor started.
"""

from __future__ import annotations

import collections
import datetime
import itertools
import math
import re
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, NoReturn, Optional, TextIO

import numpy as np
import torch

from tensorrt_llm.logger import logger
from tensorrt_llm.math_utils import ceil_div

from ...disaggregation.transfer_manager import Accepted, Direction, TransferIntent, TransferManager
from ..executor_request_queue import STORE_UPDATE_REQUEST_ID, RequestQueueItem
from ..kv_cache.sharing import Lender, MemoryKind, RowSelection
from ..llm_request import LlmRequestState, rewind_context_after_cache_drop
from .breaker import Breaker
from .replicas import ReplicaSet
from .store_ops import Action, Decision, OpKind, OpState, StoreOp, decide, report, table_hash

if TYPE_CHECKING:
    from ...disaggregation.blob.mooncake import MooncakeDriver
    from ...disaggregation.lifecycle.retirement import RetirementWatchdog
    from ..kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
    from ..llm_request import LlmRequest
    from .probe import ProbeService

# Transfer manager identifiers of the one store binding and its lender.
BINDING_ID = "mooncake"
DOMAIN_ID = "kv-cache-lender"
PROFILE_ID = "kv-sharing"
# A parked request waits in the state the executor already treats as "KV cache in transit": the
# scheduler skips it, and nothing evicts or preempts it.
PARKED_STATE = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
# Rank 0's longest wait on an empty request queue while requests wait for their store probe.
HELD_REQUEST_POLL = datetime.timedelta(milliseconds=1)
NO_WAIT = datetime.timedelta(0)
# Under GPU Direct, publications open at once at most: each locks its request's pages, which can
# then be neither evicted nor suspended, until a round returns its lease. A context-only request's
# last range is exempt: its request is freed soon after, and the range is then lost.
MAX_OPEN_PUBLICATIONS = 8
# Lends wait for an event recorded on the execution stream, and a lease in place waits for its
# event's latest record: recording that event again would hold the lease behind a newer forward.
# So each lending phase (at most three per iteration) takes the next event of this ring, and an
# event is recorded again two iterations later at the earliest.
READY_EVENTS = 8


@dataclass
class _PublicationLocks:
    """Under GPU Direct, what one publishing pass may still lock: open publications against
    MAX_OPEN_PUBLICATIONS, and pages kept from eviction against the room the manager's resumes
    leave.

    Attributes:
        room: Pages that may be pinned at all.
        pinned: Pages pinned now.
        open: Open publications.
        locked: The requests whose pages an open publication pins already.
    """

    room: int
    pinned: int
    open: int
    locked: set[int]

    def take(self, request_id: int, pages: int, *, exempt_from_cap: bool) -> bool:
        """Locks the request's ``pages`` if they fit, past the open-publication cap when
        ``exempt_from_cap``; whether they did."""
        pages = 0 if request_id in self.locked else pages
        at_cap = self.open >= MAX_OPEN_PUBLICATIONS and not exempt_from_cap
        if at_cap or self.pinned + pages > self.room:
            return False
        self.open += 1
        self.pinned += pages
        self.locked.add(request_id)
        return True


@dataclass(frozen=True)
class StagingBudget:
    """Host staging rows open publications may hold, per staging region (one per pool group): its
    slots less one fetch's. A publication of n blocks takes n rows of each full-attention layer
    group the region serves and at most one window's rows of each windowed one, whatever n.

    Attributes:
        free_rows: Per region, its slots less one fetch's.
        rows_per_block: Per region, the rows one published block takes: one per full-attention
            layer group it serves.
        window_rows: Per region, sinks plus ``ceil(window / tokens_per_block)`` blocks, summed over
            the windowed layer groups it serves.
    """

    free_rows: tuple[int, ...]
    rows_per_block: tuple[int, ...]
    window_rows: tuple[int, ...]

    def fits(self, blocks: int, open_blocks: Sequence[int]) -> bool:
        """Whether a publication of ``blocks`` blocks fits beside open ones of ``open_blocks``."""
        publications, total = 1 + len(open_blocks), blocks + sum(open_blocks)
        return all(
            per_block * total + window * publications <= free
            for free, per_block, window in zip(
                self.free_rows, self.rows_per_block, self.window_rows
            )
        )

    @property
    def longest(self) -> int:
        """The most blocks one publication may hold alone; unbounded (``sys.maxsize``) where no
        region serves a full-attention layer group."""
        return min(
            (
                (free - window) // per_block
                for free, per_block, window in zip(
                    self.free_rows, self.rows_per_block, self.window_rows
                )
                if per_block
            ),
            default=sys.maxsize,
        )


@dataclass(frozen=True)
class SharingPolicy:
    """What a server does with the store, and its limits.

    Attributes:
        memory_kind: DEVICE under GPU Direct, the store reading and writing KV cache pages in
            place; HOST under host staging.
        fetch: Whether requests read stored prompt prefixes.
        publish: Whether the server stores the prompt blocks it computes.
        min_fetch_tokens: Shorter stored prefixes are computed.
        max_fetch_tokens: Most tokens one fetch reads past the local prefix. With a sliding window
            a longer stored prefix is computed instead of cut.
        fetch_timeout_s: A fetch still running this long after admission is abandoned.
        staging: Under host staging, the rows open publications may hold: staging less one
            fetch, so publications never keep a fetch waiting for staging. None under GPU Direct.
        max_util_for_resume: The KV cache manager's own page budget for resumes.
        max_parked_fetches: Under attention DP, the most requests parked for a fetch at once:
            ``max_num_active_requests - 1``, so a rank whose requests all wait for the store
            keeps a seat for the padding request its peers' forwards need. None elsewhere.
    """

    memory_kind: MemoryKind
    fetch: bool
    publish: bool
    min_fetch_tokens: int
    max_fetch_tokens: int
    fetch_timeout_s: float
    staging: Optional[StagingBudget]
    max_util_for_resume: float
    max_parked_fetches: Optional[int] = None


class SharedKvOrchestrator:
    """Fetches stored prompt prefixes for new requests and publishes the whole prompt blocks each
    context step commits.

    Publishing pauses after failed publications in a row (a ``Breaker``). A rank with a table of
    its own decides by its own breaker; with a shared table, rank 0 decides and every rank applies
    its switch from the store-update item before activation, so all ranks stop and start appending
    publications in the same pass.

    Every store operation is an entry of one table (``store_ops``). Entries are appended and
    closed only where every rank runs the same code with the same inputs: the round
    (``resolve_store_operations``) closes them by ``decide``, and a cancel or the shutdown request
    closes a settling fetch at once. Under attention DP each rank's table holds its own requests'
    operations.

    Args:
        policy: What the server does with the store.
        kv_cache_manager: The executor's KV cache manager.
        lender: The KV cache manager's lender.
        transfers: The transfer manager, holding leases until a decision returns them.
        execution_stream: The stream the forward passes run on.
        watchdog: The watchdog of the transfer manager's store accesses, stopped at shutdown.
        driver: The store driver, whose stats join the counters at shutdown.
        probe: Rank 0's probe service, or None where this server does not fetch.
        min_over_ranks: The elementwise minimum of a report over the tensor-parallel ranks, a
            collective every rank enters; None where this rank's table is its own (one rank, or
            attention DP, where each rank serves requests of its own).
        replicas: The ranks holding this rank's KV cache bytes, which share its publications'
            puts.
        decides_publishing: Whether this rank's publication breaker decides when publishing
            pauses: on its own table, or rank 0 of a shared one.
        trace: Where each round and the exit are written, one line each, for tests to compare
            across ranks; None writes nothing.
    """

    def __init__(
        self,
        *,
        policy: SharingPolicy,
        kv_cache_manager: KVCacheManagerV2,
        lender: Lender,
        transfers: TransferManager,
        execution_stream: torch.cuda.Stream,
        watchdog: RetirementWatchdog,
        driver: Optional[MooncakeDriver],
        probe: Optional[ProbeService],
        min_over_ranks: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        replicas: ReplicaSet = ReplicaSet(),
        decides_publishing: bool = True,
        trace: Optional[TextIO] = None,
    ) -> None:
        self.counters: collections.Counter[str] = collections.Counter()
        self._policy = policy
        self._tokens_per_block = kv_cache_manager.tokens_per_block
        self._kv_cache_manager = kv_cache_manager
        self._lender = lender
        self._transfers = transfers
        self._execution_stream = execution_stream
        self._watchdog = watchdog
        self._driver = driver
        self._probe = probe
        self._min_over_ranks = min_over_ranks
        self._replicas = replicas
        self._trace = trace
        # The entries the next round covers, as rank 0 announced them; read once.
        self._announced = 0
        self._passes = 0
        # Made once: freeing an event lets go of the GIL.
        self._ready_events = itertools.cycle([torch.cuda.Event() for _ in range(READY_EVENTS)])
        self._ops: list[StoreOp] = []
        self._seq = itertools.count(1)
        # Requests with a stored prefix, until the scheduler's next pass decides on each.
        self._candidates: dict[int, LlmRequest] = {}
        # Requests with committed prompt blocks not handed to the store yet, oldest first.
        self._unpublished: dict[int, LlmRequest] = {}
        # Whether new store operations are admitted; alike on every rank.
        self._admitting = True
        self._publication_breaker = Breaker("publications") if decides_publishing else None
        # With a shared table: whether publications are appended, as rank 0 last switched it;
        # alike on every rank.
        self._publishing = True
        # Whether the transfer manager took an operation since this pass's round: the loop head
        # handed the store everything accepted before it.
        self._submitted_this_pass = False
        # Rank 0: when a fetch last resumed (monotonic).
        self._last_resume_at = -math.inf
        page_layout = lender.page_layout
        layer_groups = page_layout.layer_groups
        # Per layer group: whether it keeps a sliding window, the most pages it holds of one
        # request (None for full attention), and the bytes of one of its stored objects. A window
        # holds its sinks and up to ceil(W / tpb) + 1 pages: W tokens ending inside a block span
        # one block more than at a block boundary, where fetches and publications end. That is its
        # steady state; while a context chunk computes it also holds the chunk's pages, which these
        # page checks leave out (they are heuristics: a refused grow takes the normal path).
        self._windowed_groups = tuple(group.kept_window is not None for group in layer_groups)
        self._windowed = any(self._windowed_groups)
        self._window_pages = tuple(
            None
            if group.kept_window is None
            else group.num_sink_blocks + ceil_div(group.kept_window, self._tokens_per_block) + 1
            for group in layer_groups
        )
        self._object_bytes = tuple(
            sum(page_layout.pool_slot_bytes[group.pool_group]) for group in layer_groups
        )
        # The lender's admission rule: parked pages must leave room for the largest request's
        # resume window and its next step.
        self._resume_window_pages = self._pages_held(
            kv_cache_manager.max_seq_len + self._tokens_per_block
        )
        if probe is not None:  # rank 0 of a server that fetches
            self._warn_if_no_fetch_fits()

    @property
    def memory_kind(self) -> MemoryKind:
        """DEVICE under GPU Direct, HOST under host staging."""
        return self._policy.memory_kind

    # Rank 0, around the request queue.

    def take_rank0_items(
        self, fetched: list[RequestQueueItem], waiting: Iterable[RequestQueueItem]
    ) -> tuple[list[RequestQueueItem], Optional[RequestQueueItem]]:
        """Rank 0, after reading the request queue: the fetched items to broadcast now (a new
        request waits for its probe's answer), and the store-update item every rank applies before
        activation, or None when no waiting request's hit and no publishing switch changed."""
        hits: tuple[tuple[int, int], ...] = ()
        if self._probe is not None:
            # Hits first: the refresh forgets hits not in ``waiting``, and a request the hold
            # releases joins ``waiting`` only after this broadcast.
            if self._admitting:
                hits = self._probe.take_refreshed_hits(waiting)
            fetched = self._probe.hold_until_probed(fetched)
        switch = None
        if self._shared_table:  # rank 0 of a shared table holds the publication breaker
            allowed = self._publication_breaker.allows(time.monotonic())
            switch = None if allowed == self._publishing else allowed
        if not hits and switch is None:
            return fetched, None
        return fetched, RequestQueueItem(
            STORE_UPDATE_REQUEST_ID, store_hits=hits, store_publishing=switch
        )

    def take_held_requests(self) -> list[RequestQueueItem]:
        """Rank 0: removes and returns the requests its probe holds, which are in no queue, for the
        fatal error drain to answer."""
        return [] if self._probe is None else self._probe.take_held()

    def request_queue_wait(
        self, timeout: Optional[datetime.timedelta]
    ) -> Optional[datetime.timedelta]:
        """How long rank 0 may wait on an empty request queue: not at all while store operations
        are open, since scheduling passes close them; 1 ms while requests wait for their probe;
        otherwise ``timeout`` (None waits for the next request)."""
        if self.has_store_work():
            return NO_WAIT
        if self._probe is not None and self._probe.holds_requests():
            return HELD_REQUEST_POLL if timeout is None else min(timeout, HELD_REQUEST_POLL)
        return timeout

    def round_to_announce(self) -> int:
        """Rank 0, before the request broadcast: how many open entries the next round covers.
        Every one with a shared table; on its own table, 0, as a rank's own rounds need no
        announcement."""
        return len(self._ops) if self._shared_table else 0

    def has_store_work(self) -> bool:
        """Whether this rank has store work only executor passes advance: open entries, or
        transfer-manager work left after their decisions. Under attention DP it keeps rank 0 from
        sleeping on an empty request queue while another rank has some."""
        return bool(self._ops) or self._transfers.pending()

    # Every rank, around the request broadcast.

    def round_announced(self, round_size: int) -> None:
        """Every rank, right after each request broadcast: rank 0's round size. The next
        ``resolve_store_operations`` enters the round's collective exactly when it is not zero,
        so every rank makes these two calls once per pass, in this order."""
        self._announced = round_size

    def apply_store_update(
        self, waiting: Iterable[RequestQueueItem], update: RequestQueueItem
    ) -> None:
        """Applies rank 0's store-update item: sets the store hit of each waiting request rank 0
        measured again, and switches publishing as rank 0 decided (None: unchanged). Runs before
        any request of the broadcast is activated, so every rank admits from the same hit and
        appends the same publications."""
        by_id = dict(update.store_hits)
        for item in waiting:
            if item.id in by_id:
                item.store_hit_tokens = by_id[item.id]
                self.counters["hits_refreshed"] += 1
        publishing = update.store_publishing
        if publishing is not None:
            self._publishing = publishing
            if not publishing:  # the ranges waiting to be published are dropped
                self._unpublished.clear()

    # Each scheduling pass.

    def resolve_store_operations(self) -> None:
        """The round: returns the leases whose store access ended, resumes or drops parked
        requests, then publishes what waited for room. Call it first in each scheduling pass,
        after the overlap loop's input-copy wait: returns and drops change host page tables.

        Raises:
            RuntimeError: The server must end: this rank's lender or store binding stopped, or the
                ranks' tables differ.
        """
        started = time.perf_counter()
        self._passes += 1
        self._submitted_this_pass = False
        failure = self._store_failure()
        if failure is not None:
            self._raise_store_failure(failure)
        announced, self._announced = self._announced, 0
        # With a shared table, rank 0's announcement (alike on every rank) decides whether all
        # enter the round; on its own table, a rank covers the whole table.
        round_size = announced if self._shared_table else len(self._ops)
        if round_size:
            self._close_ops(round_size)
        if self._unpublished and not self._publications_paused():
            self._publish_waiting()
        # Under tensor parallelism this includes the round's wait for the slowest rank.
        self.counters["round_seconds"] += time.perf_counter() - started

    def add_fetch_candidates(self, new_requests: list[LlmRequest]) -> None:
        """Remembers the new requests whose prompt prefix the store holds."""
        if not (self._policy.fetch and self._admitting):
            return
        for request in new_requests:
            hit = request.py_store_hit_tokens
            if hit is not None and hit >= self._policy.min_fetch_tokens:
                self._candidates[request.py_request_id] = request

    def start_fetches(self) -> set[int]:
        """The scheduler's admission hook: parks each candidate that fetches, and returns their
        ids. A parked request takes no batch slot and no token budget. Every candidate is tried
        once, except that a generation-first request still waiting for its generation server
        stays a candidate until the transceiver promotes it (alike on every rank). Its hit is not
        probed again meanwhile: if the store evicted the objects, its fetch delivers short, drops
        and computes the prompt."""
        waiting = LlmRequestState.DISAGG_CONTEXT_WAIT_SCHEDULER
        candidates = list(self._candidates.values())
        self._candidates = {r.py_request_id: r for r in candidates if r.state == waiting}
        tried = [r for r in candidates if r.state != waiting]
        if not tried:
            return set()
        started = time.perf_counter()
        ready = next(self._ready_events)  # the pass's fetches share one
        parked = {r.py_request_id for r in tried if self._park_for_fetch(r, ready)}
        self.counters["admission_seconds"] += time.perf_counter() - started
        return parked

    def has_store_transfers_to_start(self) -> bool:
        """Whether ``start_store_transfers`` has work: operations this pass handed the transfer
        manager under GPU Direct. Host staging leases need the lender's progress for their slots,
        so they start at the next pass's loop head."""
        return self._submitted_this_pass and self._policy.memory_kind is MemoryKind.DEVICE

    def start_store_transfers(self) -> None:
        """Hands the store the operations this pass lent whose pages are ready, so a fetch
        admitted in this pass starts while the forward runs. Call it after the iteration's
        input-copy wait: a lease that failed unlocks its pages, which rewrites the host
        page-index entries the input copy reads."""
        if self.has_store_transfers_to_start():
            self._submitted_this_pass = False
            self._transfers.start_submitted()

    def publish_committed_blocks(self, context_requests: list[LlmRequest]) -> None:
        """Hands the store the prompt blocks this step's context requests committed. Call it after
        the forward launch and the context commits, before anything can free these requests."""
        if not (self._policy.publish and self._admitting) or self._publications_paused():
            return
        started = time.perf_counter()
        for request in context_requests:
            # Readers name a prompt by its tokens alone, which multimodal prompts are not.
            if not request.is_dummy and request.py_multimodal_data is None:
                self._unpublished[request.py_request_id] = request
        if self._unpublished:
            self._publish_waiting()
        self.counters["publication_seconds"] += time.perf_counter() - started

    def cancel_fetch(self, request: LlmRequest) -> bool:
        """Lets a request being cancelled go: its fetch is abandoned, and it no longer reads as
        in transfer. Its cache may be freed at once: under host staging the store writes into
        staging only, and under GPU Direct the lease's lock keeps the pages from reuse, until a
        decision returns the lease. It also stops being a fetch candidate, so it never parks
        later. Returns whether the request was parked for a fetch or its fetch was settling."""
        self._candidates.pop(request.py_request_id, None)
        fetches = [op for op in self._ops if op.request_waits and op.request is request]
        for op in fetches:
            self._let_go(op)
        self._ops = [op for op in self._ops if not op.closed]
        self.counters["fetches_cancelled"] += len(fetches)
        return bool(fetches)

    def begin_shutdown(self) -> None:
        """Admits no new store operation, and drops parked fetches: their requests compute their
        prompts. Call it at the shutdown request, on every rank."""
        self._stop_admitting()
        for op in self._ops:
            if op.request_waits:
                self._let_go(op)
                self._drop(op, "shutdown")
        self._ops = [op for op in self._ops if not op.closed]

    def shutdown(self) -> None:
        """Returns every lease once its store access ended, within the transfer manager's
        shutdown deadline, adds the probe's, the transfer manager's and the store driver's counts
        to the counters, logs them and stops the access watchdog. Call it when the executor loop
        ended, after the transfer manager's shutdown began: no rank schedules any more, so each
        returns its leases on its own."""
        if self._trace is not None:
            self._trace.write(f"exit passes={self._passes} open={len(self._ops)}\n")
            self._trace.close()
            self._trace = None
        self._stop_admitting()
        # Counted here; the transfer manager returns the rest uncounted.
        for op in self._ops:
            if op.state is OpState.SETTLING:
                continue  # its lease went back already
            if op.operation_id is None or self._transfers.ended(op.operation_id):
                self._return_none(op)
        self._transfers.stop_holding()
        deadline = time.monotonic() + self._transfers.shutdown_timeout_s
        while not self._transfers.shutdown_status().complete and time.monotonic() < deadline:
            self._transfers.advance(time.monotonic())
            self._transfers.wait_for_progress()
        if self._probe is not None:
            self._probe.stop_and_wait()
            self.counters.update(self._probe.counters)
        self.counters.update({f"transfers_{n}": c for n, c in self._transfers.counters.items()})
        if self._driver is not None:
            self.counters.update({f"driver_{n}": c for n, c in self._driver.stats().items()})
        counters = {n: round(c, 3) if isinstance(c, float) else c for n, c in self.counters.items()}
        logger.info(f"kv_sharing counters: {counters}")
        # The watchdog refuses to stop while a store access may still touch KV cache memory.
        if self._transfers.shutdown_status().complete:
            self._watchdog.stop()

    # The round.

    @property
    def _shared_table(self) -> bool:
        """Every tensor-parallel rank keeps this same table; rounds reduce reports over them."""
        return self._min_over_ranks is not None

    def _publications_paused(self) -> bool:
        """Whether publishing pauses now: with a shared table as rank 0 last switched it, on its
        own table by this rank's breaker."""
        if self._shared_table:
            return not self._publishing
        return not self._publication_breaker.allows(time.monotonic())

    def _store_failure(self) -> Optional[str]:
        """Why this rank's lender, its transfer-manager domain or its store binding stopped, or
        None."""
        failures = self._transfers.take_domain_failures()
        if failures:
            return failures[0].fault
        if self._lender.stopped:
            return "the KV cache lender stopped"
        return self._transfers.bindings[BINDING_ID].fault

    def _close_ops(self, round_size: int) -> None:
        """Reports on the first ``round_size`` open operations (fewer where some closed since the
        announcement), reduces the reports over the ranks and applies the decisions."""
        # After closures this may cover entries appended since the announcement: every rank's
        # table changed alike, so every rank covers the same entries.
        entries = self._ops[:round_size]
        now = time.monotonic()
        columns = np.array([self._report_on(op, now) for op in entries], dtype=np.int64)
        own = report(entries, round_size, columns)
        # On its own table, the minimum over ranks is this rank's own report.
        reduced = self._min_over_ranks(own) if self._shared_table else own
        gpu_direct = self._policy.memory_kind is MemoryKind.DEVICE
        decisions = decide(entries, reduced, gpu_direct=gpu_direct)
        if self._trace is not None:  # before a FATAL raises
            self._trace_round(round_size, entries, decisions)
        for decision in decisions:
            self._apply(decision)
        if self._probe is not None:
            self._tell_probe_about_fetches(decisions)
        self._ops = [op for op in self._ops if not op.closed]

    def _trace_round(
        self, round_size: int, entries: list[StoreOp], decisions: list[Decision]
    ) -> None:
        """One line per round, equal on every rank of a tensor-parallel group: what it covered,
        the page counts every rank must share, and the decisions."""
        used, total = self._kv_cache_manager.get_primary_block_counts()
        deferred = sum(self._lender.held_page_counts().deduct.values())
        actions = ",".join(
            d.action.value if d.op is None else f"{d.action.value}:{d.op.seq}" for d in decisions
        )
        self._trace.write(
            f"round pass={self._passes} size={round_size} covered={len(entries)} "
            f"hash={table_hash(entries)} pages={used}/{total} deferred={deferred} "
            f"decisions={actions}\n"
        )

    def _report_on(self, op: StoreOp, now: float) -> tuple[int, int, int]:
        """``(done, ready, on_time)`` on this rank: whether the operation's lease may go back, a
        fetch's every row arrived, and the fetch is within its deadline."""
        if op.state is OpState.PARKED:
            on_time = int(now < op.deadline)
            if op.operation_id is None:  # it never started here: nothing arrives
                return 1, 0, on_time
            return (
                int(self._transfers.ended(op.operation_id)),
                int(self._transfers.ready(op.operation_id)),
                on_time,
            )
        if op.state is OpState.SETTLING:
            return int(self._lender.poll_resume_bounds(op.request) is not None), 1, 1
        return int(op.operation_id is None or self._transfers.ended(op.operation_id)), 1, 1

    def _apply(self, decision: Decision) -> None:
        op = decision.op
        match decision.action:
            case Action.RETURN_ALL:
                self._count_fetched(op)
                self._transfers.return_now(op.operation_id, deliver=True)
            case Action.RETURN_NONE:
                self._return_none(op)
            case Action.SETTLE:
                op.state = OpState.SETTLING
            case Action.RESUME:
                self._resume(op)
            case Action.DROP:
                self._drop(op, decision.reason)
            case Action.ABANDON:
                self._abandon(op)
            case Action.FATAL:
                raise RuntimeError(f"kv_sharing: {decision.reason}")

    def _tell_probe_about_fetches(self, decisions: list[Decision]) -> None:
        """Rank 0 tells its probe how the store served this round's fetches: well where one
        resumed; badly where one came back short with a fault on this rank (a failed store call;
        short without a fault is an evicted object), or where one went late and none resumed
        within ``fetch_timeout_s`` (a slow data plane: a degraded link, a fallback to TCP). A long
        queue on a healthy store also makes fetches late, but others keep resuming meanwhile. A
        transfer that never ends is caught by the access watchdog, which ends the process."""
        now = time.monotonic()
        resumed = faulted = late = False
        for decision in decisions:
            op = decision.op
            if decision.action is Action.RESUME:
                resumed = True
            elif decision.action is Action.DROP and decision.reason == "late":
                late = True
            elif decision.action is Action.DROP and decision.reason == "short":
                # A store fault shows in the Outcome; a lend the lender refused, at the call or at
                # its first poll, never reached the store.
                if op.operation_id is not None:
                    outcome = self._transfers.operations[op.operation_id].outcome
                    faulted |= outcome is not None and outcome.fault is not None
        if resumed:
            self._last_resume_at = now
            self._probe.record_fetch(failed=False)
        elif faulted:
            self._probe.record_fetch(failed=True, why="a fetch failed in the store")
        elif late and now - self._last_resume_at > self._policy.fetch_timeout_s:
            why = f"fetches went late and none resumed in {self._policy.fetch_timeout_s} s"
            self._probe.record_fetch(failed=True, why=why)

    def _return_none(self, op: StoreOp) -> None:
        """The lease goes back with nothing delivered; the entry closes."""
        if op.operation_id is not None:
            if op.kind is OpKind.PUBLISH:
                self._record_publication_outcome(op)
            self._transfers.return_now(op.operation_id, deliver=False)
        op.closed = True

    def _record_publication_outcome(self, op: StoreOp) -> None:
        """Counts this rank's puts of a publication whose access ended: objects stored, objects
        skipped (another writer, a full store; the driver's stats say which), and a failure where
        the store or transfer failed, which feeds the publication breaker. A pause it starts drops
        the ranges waiting to be published (with a shared table, at rank 0's broadcast switch
        instead)."""
        failed = self._transfers.delivered_masks(op.operation_id) is None
        # A failed publication's rows stored before it failed are in the store all the same.
        outcome = self._transfers.operations[op.operation_id].outcome
        masks = () if outcome is None else outcome.delivered_masks
        if not failed and not any(len(mask) for mask in masks):
            return  # this rank's stripe of the rows was empty: it put nothing
        if self._publication_breaker is not None:
            paused = self._publication_breaker.record(
                time.monotonic(), failed=failed, why="a publication failed"
            )
            self.counters["publication_pauses"] += int(paused)
            if paused and not self._shared_table:
                self._unpublished.clear()
        stored = sum(int(mask.sum()) for mask in masks)
        self.counters["published_objects"] += stored
        if failed:
            self.counters["publications_failed"] += 1
        else:
            self.counters["publication_objects_skipped"] += sum(len(m) for m in masks) - stored

    def _count_fetched(self, op: StoreOp) -> None:
        """Counts the objects and bytes a fetch whose every row arrived read from the store: its
        view holds one group per layer group, in order, empty or not."""
        masks = self._transfers.delivered_masks(op.operation_id)
        if masks is None:  # its lender domain stopped earlier in this round
            return
        rows = [int(mask.sum()) for mask in masks]
        self.counters["fetched_objects"] += sum(rows)
        self.counters["fetched_bytes"] += sum(r * b for r, b in zip(rows, self._object_bytes))

    def _resume(self, op: StoreOp) -> None:
        """The request computes on from the fetch's end, which every rank holds; the entry
        closes."""
        bounds = self._lender.poll_resume_bounds(op.request)
        if bounds is None or not bounds.min_position <= op.end <= bounds.max_position:
            raise RuntimeError(
                f"kv_sharing: request {op.request_id} fetched its prompt to {op.end}, outside its "
                f"resume bounds {bounds}"
            )
        self._kv_cache_manager.resume_served_prefix(op.request, op.end)
        op.request.py_published_end = op.end  # the store holds what was fetched
        op.closed = True
        self.counters["fetches_resumed"] += 1
        self.counters["fetched_blocks"] += (op.end - op.start) // self._tokens_per_block

    def _drop(self, op: StoreOp, reason: str) -> None:
        """The request computes its prompt from its local prefix: the scheduler admits it again
        in this pass. A settling entry closes; a parked one's lease goes back by a later
        decision."""
        request = op.request
        request.state = LlmRequestState.CONTEXT_INIT  # the cursor setters check for it
        self._free_cache(request)
        rewind_context_after_cache_drop(request, self._tokens_per_block)
        # What the store holds of the prompt is in doubt: publish from the local prefix.
        request.py_store_hit_tokens = None
        if op.state is OpState.SETTLING:
            op.closed = True
        # fetches_dropped_late, _short or _shutdown.
        self.counters[f"fetches_dropped_{reason}"] += 1

    def _abandon(self, op: StoreOp) -> None:
        """No request waits for the fetch any more; its lease goes back once its access ended."""
        if op.operation_id is not None:
            self._transfers.abandon(op.operation_id, "no request waits for the fetch")
        op.state = OpState.OPEN

    def _let_go(self, op: StoreOp) -> None:
        """The fetch's request no longer waits for it: a parked fetch is abandoned, a settling
        one, whose lease went back already, closes."""
        if op.state is OpState.PARKED:
            self._abandon(op)
        else:
            op.closed = True
        if op.request.state == PARKED_STATE:
            op.request.state = LlmRequestState.CONTEXT_INIT

    def _stop_admitting(self) -> None:
        """Admits no new store operation; open ones close by rounds as before."""
        self._admitting = False
        self._candidates.clear()
        self._unpublished.clear()
        if self._probe is not None:
            self._probe.stop()

    # Fetch admission.

    def _park_for_fetch(self, request: LlmRequest, ready: torch.cuda.Event) -> bool:
        """Claims the local prefix, lends the rows up to the stored end for writing, parks the
        request and appends a parked fetch, whatever the lend or submit did on this rank."""
        if request.state != LlmRequestState.CONTEXT_INIT or not request.is_first_context_chunk:
            return False
        if self._parked_fetches_at_cap():
            return False
        matched = self._kv_cache_manager.probe_context_reuse(request)
        if matched is None:  # it holds a cache already
            return False
        tpb = self._tokens_per_block
        claim = matched // tpb * tpb
        hit = request.py_store_hit_tokens
        # A named fetch reads whole blocks, at most max_fetch_tokens of them, and ends before the
        # last prompt token.
        longest = self._policy.max_fetch_tokens // tpb * tpb
        end = min(hit, claim + longest, (request.prompt_len - 1) // tpb * tpb)
        if self._windowed and end != hit:
            # Window rows are held only at the ends publications reached: a cut fetch would miss
            # them, so the request computes its prompt.
            self.counters["fetches_skipped_too_long"] += 1
            return False
        if end - claim < self._policy.min_fetch_tokens:
            return False
        if not self._pages_allow_fetch(request):
            self.counters["fetches_skipped_for_pages"] += 1
            return False
        if self._kv_cache_manager.prepare_context_cache(request, reuse_limit=claim) != claim:
            self._free_cache(request)
            self.counters["fetches_skipped_for_claim"] += 1
            return False
        # After the claim's own work; the lender adds its grow's under GPU Direct.
        ready.record(self._execution_stream)
        op = StoreOp(
            seq=next(self._seq),
            kind=OpKind.FETCH,
            request=request,
            state=OpState.PARKED,
            start=claim,
            end=end,
            deadline=time.monotonic() + self._policy.fetch_timeout_s,
        )
        if not self._start(op, ready):
            self.counters["fetches_not_started"] += 1
        request.state = PARKED_STATE
        self._ops.append(op)
        self.counters["fetches_started"] += 1
        return True

    def _parked_fetches_at_cap(self) -> bool:
        """Under attention DP, whether parked fetches take every seat the padding request does
        not need."""
        cap = self._policy.max_parked_fetches
        if cap is None:
            return False
        parked = sum(op.request_waits for op in self._ops)
        return parked >= cap

    def _pages_allow_fetch(self, request: LlmRequest) -> bool:
        """Whether the request's whole prompt plus one step fits in free pages, and the pages
        nothing may evict stay within the manager's resume budget with it."""
        used, total = self._kv_cache_manager.get_primary_block_counts()
        need = self._pages_of(request.prompt_len)
        return need <= total - used and self._pinned_pages() + need <= self._resume_room(total)

    def _pinned_pages(self) -> int:
        """Pages no eviction or suspension may take now: what parked fetches lent up to their
        ends; under GPU Direct the prompt and one step of each request an open publication locks,
        once each (one freed meanwhile counts among the deferred closes); and the pages of caches
        whose close waits for a lease."""
        pinned = sum(self._pages_held(op.end) for op in self._ops if op.request_waits)
        if self._policy.memory_kind is MemoryKind.DEVICE:
            holders = self._kv_cache_manager.kv_cache_map
            locked = {
                op.request_id: op.request
                for op in self._ops
                if op.kind is OpKind.PUBLISH and op.request_id in holders
            }
            pinned += sum(self._pages_of(request.prompt_len) for request in locked.values())
        return pinned + sum(self._lender.held_page_counts().deduct.values())

    def _pages_of(self, prompt_len: int) -> int:
        """The pages of a whole prompt of ``prompt_len`` tokens and one step."""
        return self._pages_held(prompt_len + self._tokens_per_block)

    def _pages_held(self, tokens: int) -> int:
        """The pages one request holding ``tokens`` tokens takes over every layer group: a
        windowed group at most its window's."""
        blocks = ceil_div(tokens, self._tokens_per_block)
        return sum(blocks if cap is None else min(blocks, cap) for cap in self._window_pages)

    def _resume_room(self, total_pages: int) -> int:
        """The pages the manager may hold pinned and still resume its largest request."""
        return int(self._policy.max_util_for_resume * total_pages) - self._resume_window_pages

    def _warn_if_no_fetch_fits(self) -> None:
        """At startup: where the resume room cannot hold the shortest fetch, no request ever
        fetches."""
        _, total_pages = self._kv_cache_manager.get_primary_block_counts()
        room = self._resume_room(total_pages)
        # The shortest prompt that fetches: min_fetch_tokens, then the last token it computes.
        needed = self._pages_of(self._policy.min_fetch_tokens + 1)
        if room < needed:
            logger.warning(
                f"kv_sharing: no request can fetch from the store: one fetch needs {needed} KV "
                f"cache pages, and max_util_for_resume ({self._policy.max_util_for_resume}) of "
                f"the {total_pages} pages leaves {max(room, 0)} beside the resume of one "
                f"max_seq_len ({self._kv_cache_manager.max_seq_len}) request; set max_seq_len "
                "lower or give the KV cache more memory"
            )

    # Publication.

    def _publish_waiting(self) -> None:
        """Lends and submits the waiting ranges in arrival order while they fit beside the open
        publications."""
        holders = self._kv_cache_manager.kv_cache_map
        host_staging = self._policy.memory_kind is MemoryKind.HOST
        locks = None if host_staging else self._publication_locks()
        ready = None
        for request in list(self._unpublished.values()):
            start, end, committed_end = self._publication_range(request)
            # A request freed meanwhile has nothing left to read.
            if request.py_request_id not in holders or end <= start:
                del self._unpublished[request.py_request_id]
                continue
            if host_staging:
                if not self._staging_fits((end - start) // self._tokens_per_block):
                    self.counters["publication_waits"] += 1
                    return  # staging frees in order: the ranges behind wait too
            else:  # under GPU Direct a range waits on its own
                pages = self._pages_of(request.prompt_len)
                if pages > locks.room:  # even with nothing else pinned
                    request.py_published_end = request.prompt_len  # its later chunks too
                    del self._unpublished[request.py_request_id]
                    self.counters["publications_dropped_for_pages"] += 1
                    logger.warning_once(
                        f"kv_sharing: a request of {pages} KV cache pages does not fit the "
                        f"{locks.room} pages left beside the largest request's resume, so GPU "
                        "Direct publishes nothing of it; give the KV cache more memory or "
                        "max_util_for_resume, or use data_path 'host_staging'",
                        key="kv_sharing: a publication never fits",
                    )
                    continue
                exempt = self._ends_context(request, end)
                if not locks.take(request.py_request_id, pages, exempt_from_cap=exempt):
                    self.counters["publication_waits"] += 1
                    continue
            if ready is None:
                ready = next(self._ready_events)
                ready.record(self._execution_stream)  # after the forward and the commits
            if end == committed_end:
                del self._unpublished[request.py_request_id]
            self._publish(request, start, end, ready)

    def _staging_fits(self, blocks: int) -> bool:
        """Host staging: whether a range of ``blocks`` blocks fits in what open publications leave
        of the budget."""
        # Every open publication counts, also a replica's that lent nothing here: all ranks gate
        # alike, so their tables stay equal.
        tpb = self._tokens_per_block
        open_blocks = [(op.end - op.start) // tpb for op in self._ops if op.kind is OpKind.PUBLISH]
        return self._policy.staging.fits(blocks, open_blocks)

    def _publication_locks(self) -> _PublicationLocks:
        """GPU Direct: what one publishing pass may still lock."""
        _, total_pages = self._kv_cache_manager.get_primary_block_counts()
        publications = [op for op in self._ops if op.kind is OpKind.PUBLISH]
        return _PublicationLocks(
            room=self._resume_room(total_pages),
            pinned=self._pinned_pages(),
            open=len(publications),
            locked={op.request_id for op in publications},
        )

    def _ends_context(self, request: LlmRequest, end: int) -> bool:
        """Whether the range is a context-only request's last: its request is freed soon after."""
        tpb = self._tokens_per_block
        prompt_blocks_end = request.prompt_len // tpb * tpb
        return request.is_context_only_request and end == prompt_blocks_end

    def _publication_range(self, request: LlmRequest) -> tuple[int, int, int]:
        """``(start, end, committed_end)``: the request's committed whole prompt blocks from where
        publication stopped, past what the store held at the probe, ``[start, end)`` at most the
        host staging budget long."""
        tpb = self._tokens_per_block
        committed_end = min(request.context_current_position, request.prompt_len) // tpb * tpb
        published = request.py_published_end
        if published is None:  # blocks reused locally are not put again
            published = request.prepopulated_prompt_len // tpb * tpb
        # Nor those the store held at the probe, a resumed fetch's past its end included.
        start = max(published, min(request.py_store_hit_tokens or 0, committed_end))
        if self._policy.memory_kind is MemoryKind.HOST:
            longest = self._policy.staging.longest * tpb
            return start, min(committed_end, start + longest), committed_end
        return start, committed_end, committed_end

    def _publish(self, request: LlmRequest, start: int, end: int, ready: torch.cuda.Event) -> None:
        op = StoreOp(
            seq=next(self._seq),
            kind=OpKind.PUBLISH,
            request=request,
            state=OpState.OPEN,
            start=start,
            end=end,
        )
        request.py_published_end = end
        # Under GPU Direct every replica lends and submits the range, so its pages lock alike on
        # every rank; under host staging a replica that puts nothing lends nothing (a host lease
        # locks no page). A range that does not start (a lend refused) closes like one that put
        # nothing.
        if self._replicas.lends(self._policy.memory_kind) and not self._start(op, ready):
            self.counters["publications_not_started"] += 1
        self._ops.append(op)
        self.counters["publications_started"] += 1

    # Helpers.

    def _start(self, op: StoreOp, ready: torch.cuda.Event) -> bool:
        """Lends the operation's rows by name, in place or in host staging, and hands the lease
        to the transfer manager, which puts this rank's stripe of a publication's rows. Returns
        False where the lender refused the lend: the entry reports nothing delivered.

        Raises:
            RuntimeError: The lender lends no more, or the transfer manager refused the lease.
        """
        fetch = op.kind is OpKind.FETCH
        memory_kind = self._policy.memory_kind
        lend = self._lender.lend_write if fetch else self._lender.lend_read
        try:
            lease = lend(
                op.request,
                op.start if fetch else self._read_start(op.start),
                op.end,
                memory_kind=memory_kind,
                rows=RowSelection.NAMED,
                ready_event=ready,
            )
        except RuntimeError as error:  # the lender lends no more
            self._raise_store_failure(f"a lend failed: {error}", cause=error)
        if lease.failure is not None:
            lease.release()  # a lend refused at the call holds no lock, slot or fetch
            # One warning per kind of reason: positions and request ids do not make a new one.
            logger.warning_once(
                f"kv_sharing: the lender refused a {op.kind.name.lower()}: {lease.failure}",
                key=re.sub(r"\d+", "#", lease.failure),
            )
            return False
        intent = TransferIntent(
            BINDING_ID,
            PROFILE_ID,
            Direction.FETCH if fetch else Direction.PUBLISH,
            memory_kind,
            put_stripe=None if fetch else self._replicas.put_stripe(memory_kind),
        )
        result = self._transfers.submit(lease, intent)
        if not isinstance(result, Accepted):
            lease.release()  # nothing else holds it
            self._raise_store_failure(f"the transfer manager refused an operation: {result.reason}")
        op.operation_id = result.operation_id
        self._submitted_this_pass = True
        return True

    def _read_start(self, start: int) -> int | list[int]:
        """A publication's lend start: ``start`` under full attention; per layer group with
        windows, a windowed group from block 0, so the lender lends its sinks and every window row
        needed at the range's end, the rows below ``start`` included (fetched, reused locally or
        computed by an earlier step)."""
        if not self._windowed:
            return start
        return [0 if windowed else start for windowed in self._windowed_groups]

    def _raise_store_failure(self, reason: str, cause: Optional[Exception] = None) -> NoReturn:
        """Raises, ending the server: this rank's lender or store binding stopped, so this rank's
        KV cache and locks may no longer follow its store operations, nor under tensor parallelism
        its peers'. With several ranks the executor's rank-crash path then kills every rank
        (``hard_kill_on_rank_crash``), so peers waiting in a collective do not hang."""
        failure = RuntimeError(
            f"kv_sharing: {reason}; a store failure ends the server, as this rank's KV cache may no "
            "longer match its store operations"
        )
        if cause is None:
            raise failure
        raise failure from cause

    def _free_cache(self, request: LlmRequest) -> None:
        if request.py_request_id in self._kv_cache_manager.kv_cache_map:
            self._kv_cache_manager.free_resources(request)

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
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from tensorrt_llm.logger import logger
from tensorrt_llm.math_utils import ceil_div

from ...disaggregation.transfer_manager import Accepted, Direction, TransferIntent, TransferManager
from ..kv_cache.sharing import Lender, MemoryKind, RowSelection
from ..llm_request import LlmRequestState, rewind_context_after_cache_drop
from .store_ops import Action, Decision, OpKind, OpState, StoreOp, decide, report

if TYPE_CHECKING:
    from ...disaggregation.lifecycle.retirement import RetirementWatchdog
    from ...disaggregation.orchestration.coordinator import DisaggTransferCoordinator
    from ..executor_request_queue import RequestQueueItem
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
# Events lends wait for, recorded on the execution stream. A lease in place asks its event's latest
# record, so one event recorded every iteration would keep a waiting lease behind each newer
# forward: each lending phase takes the next event of a ring instead. Three phases may take one
# per iteration (the round's retried publications, fetch admission, the step's publications), so
# an event comes round after 2.7 iterations at the earliest; a lease still holding it then only
# waits for a later forward on the same stream.
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

    def take(self, request_id: int, pages: int, past_cap: bool) -> bool:
        """Locks the request's ``pages`` if they fit, past the open-publication cap when
        ``past_cap``; whether they did."""
        pages = 0 if request_id in self.locked else pages
        at_cap = self.open >= MAX_OPEN_PUBLICATIONS and not past_cap
        if at_cap or self.pinned + pages > self.room:
            return False
        self.open += 1
        self.pinned += pages
        self.locked.add(request_id)
        return True


@dataclass(frozen=True)
class SharingPolicy:
    """What a server does with the store, and its limits.

    Attributes:
        memory_kind: DEVICE under GPU Direct, the store reading and writing KV cache pages in
            place; HOST under host staging.
        fetch: Whether requests read stored prompt prefixes.
        publish: Whether the server stores the prompt blocks it computes.
        min_fetch_tokens: Shorter stored prefixes are computed.
        max_fetch_tokens: Longest prefix one fetch reads.
        fetch_timeout_s: A fetch still running this long after admission is abandoned.
        publication_tokens: Under host staging, the most prompt tokens open publications may
            hold, whole blocks: staging less one fetch, so publications never keep a fetch
            waiting for staging. Unused under GPU Direct (None).
        max_util_for_resume: The KV cache manager's own page budget for resumes.
    """

    memory_kind: MemoryKind
    fetch: bool
    publish: bool
    min_fetch_tokens: int
    max_fetch_tokens: int
    fetch_timeout_s: float
    publication_tokens: Optional[int]
    max_util_for_resume: float


class SharedKvOrchestrator:
    """Fetches stored prompt prefixes for new requests and publishes the whole prompt blocks each
    context step commits.

    Every store operation is an entry of one table (``store_ops``). Entries are appended where
    every rank runs the same code with the same inputs: a fetch's admission inside the scheduler
    and a publication after the context commit, whatever this rank's own lend or submit did. Only
    ``resolve_store_operations``, at the start of each scheduling pass, closes them: there every
    lease goes back and every parked request resumes or computes its prompt, by ``decide``.

    Args:
        policy: What the server does with the store.
        kv_cache_manager: The executor's KV cache manager.
        lender: The KV cache manager's lender.
        transfers: The transfer manager, holding leases until a decision returns them.
        coordinator: The executor's coordinator, which advances the transfer manager.
        execution_stream: The stream the forward passes run on.
        watchdog: The watchdog of the transfer manager's store accesses, stopped at shutdown.
        probe: Rank 0's probe service, or None where this server does not fetch.
    """

    def __init__(
        self,
        *,
        policy: SharingPolicy,
        kv_cache_manager: KVCacheManagerV2,
        lender: Lender,
        transfers: TransferManager,
        coordinator: DisaggTransferCoordinator,
        execution_stream: torch.cuda.Stream,
        watchdog: RetirementWatchdog,
        probe: Optional[ProbeService],
    ) -> None:
        self.counters: collections.Counter[str] = collections.Counter()
        self._policy = policy
        self._tokens_per_block = kv_cache_manager.tokens_per_block
        self._kv_cache_manager = kv_cache_manager
        self._lender = lender
        self._transfers = transfers
        self._coordinator = coordinator
        self._execution_stream = execution_stream
        self._watchdog = watchdog
        self._probe = probe
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
        # This rank's lender and store binding still serve; reported in each round.
        self._lender_ok = True
        self._bytes_per_block = sum(
            region.slot_bytes for region in lender.regions if region.memory_kind is MemoryKind.HOST
        )
        # The lender's admission rule: parked pages must leave room for the largest request's
        # resume window and its next step.
        self._resume_window_pages = ceil_div(kv_cache_manager.max_seq_len, self._tokens_per_block)
        self._resume_window_pages += 1

    @property
    def memory_kind(self) -> MemoryKind:
        """DEVICE under GPU Direct, HOST under host staging."""
        return self._policy.memory_kind

    # Rank 0, around the request queue.

    def hold_until_probed(self, fetched: list[RequestQueueItem]) -> list[RequestQueueItem]:
        """The fetched items to broadcast now; new requests wait for their probe's answer."""
        return fetched if self._probe is None else self._probe.hold_until_probed(fetched)

    def request_queue_wait(
        self, timeout: Optional[datetime.timedelta]
    ) -> Optional[datetime.timedelta]:
        """How long rank 0 may wait on an empty request queue: not at all while store operations
        are open, since scheduling passes close them; 1 ms while requests wait for their probe;
        otherwise ``timeout`` (None waits for the next request)."""
        if self._ops or self._transfers.pending():
            return NO_WAIT
        if self._probe is not None and self._probe.holds_requests():
            return HELD_REQUEST_POLL if timeout is None else min(timeout, HELD_REQUEST_POLL)
        return timeout

    # Each scheduling pass.

    def resolve_store_operations(self) -> None:
        """The round: returns the leases whose store access ended, resumes or drops parked
        requests, then publishes what waited for room. Call it first in each scheduling pass,
        after the overlap loop's input-copy wait: returns and drops change host page tables."""
        if self._coordinator.take_shared_domain_failures() or self._lender.stopped:
            self._lender_ok = False
        if self._ops:
            self._close_ops()
        if self._unpublished:
            self._publish_waiting()

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
        ids. A parked request takes no batch slot and no token budget."""
        if not self._candidates:
            return set()
        ready = next(self._ready_events)  # the pass's fetches share one
        parked = set()
        for request in self._candidates.values():
            if self._park_for_fetch(request, ready):
                parked.add(request.py_request_id)
        self._candidates.clear()
        return parked

    def publish_committed_blocks(self, context_requests: list[LlmRequest]) -> None:
        """Hands the store the prompt blocks this step's context requests committed. Call it after
        the forward launch and the context commits, before anything can free these requests."""
        if not (self._policy.publish and self._admitting):
            return
        for request in context_requests:
            # Readers name a prompt by its tokens alone, which multimodal prompts are not.
            if not request.is_dummy and request.py_multimodal_data is None:
                self._unpublished[request.py_request_id] = request
        self._publish_waiting()

    def cancel_fetch(self, request: LlmRequest) -> None:
        """Lets a request being cancelled go: its fetch is abandoned, and it no longer reads as
        in transfer. Its cache may be freed at once: under host staging the store writes into
        staging only, and under GPU Direct the lease's lock keeps the pages from reuse, until a
        decision returns the lease."""
        self._candidates.pop(request.py_request_id, None)
        for op in self._ops:
            if op.kind is OpKind.FETCH and op.state is not OpState.OPEN and op.request is request:
                self._let_go(op)
        self._ops = [op for op in self._ops if not op.closed]

    def begin_shutdown(self) -> None:
        """Admits no new store operation, and drops parked fetches: their requests compute their
        prompts. Call it at the shutdown request, on every rank."""
        self._stop_admitting()
        for op in self._ops:
            if op.kind is OpKind.FETCH and op.state is not OpState.OPEN:
                self._let_go(op)
                self._drop(op, "shutdown")
        self._ops = [op for op in self._ops if not op.closed]

    def shutdown(self) -> None:
        """Returns every lease once its store access ended, within the transfer manager's
        shutdown deadline, logs the counters and stops the access watchdog. Call it when the
        executor loop ended, after the transfer manager's shutdown began: no rank schedules any
        more, so each returns its leases on its own."""
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
            self._transfers.wait_for_progress(0.001)
        probe = {}
        if self._probe is not None:
            self._probe.stop(wait_s=1.0)
            probe = self._probe.counters
        transfers = {f"transfers_{name}": count for name, count in self._transfers.counters.items()}
        logger.info(f"kv_sharing counters: {dict(self.counters, **probe, **transfers)}")
        # The watchdog refuses to stop while a store access may still touch KV cache memory.
        if self._transfers.shutdown_status().complete:
            self._watchdog.stop()

    # The round.

    def _close_ops(self) -> None:
        """Reports on every open operation and applies the decisions."""
        for op in self._ops:
            if op.kind is OpKind.FETCH and op.state is not OpState.OPEN and op.request.is_finished:
                self._let_go(op)  # the request ended while parked, as by an error
        entries = [op for op in self._ops if not op.closed]
        now = time.monotonic()
        columns = np.array([self._report_on(op, now) for op in entries], dtype=np.int64)
        # On one rank the minimum over ranks is this rank's own report.
        reduced = report(entries, len(entries), self._lender_ok, columns)
        for decision in decide(entries, reduced):
            self._apply(decision)
        self._ops = [op for op in self._ops if not op.closed]

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
            # A stopped lender is queried no more; the round drops the request.
            settled = not self._lender_ok or self._lender.poll_resume_bounds(op.request) is not None
            return int(settled), 1, 1
        return int(op.operation_id is None or self._transfers.ended(op.operation_id)), 1, 1

    def _apply(self, decision: Decision) -> None:
        op = decision.op
        match decision.action:
            case Action.RETURN_ALL:
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
            case Action.STOP_SHARING:
                if self._admitting:
                    logger.error(
                        "kv_sharing stops: a lender or store binding stopped; requests compute "
                        "their prompts locally from now on"
                    )
                self._stop_admitting()
            case Action.FATAL:
                raise RuntimeError(f"kv_sharing: {decision.reason}")

    def _return_none(self, op: StoreOp) -> None:
        """The lease goes back with nothing delivered; the entry closes."""
        if op.kind is OpKind.PUBLISH:
            delivered = op.operation_id is not None and self._transfers.ready(op.operation_id)
            if delivered:
                self.counters["published_blocks"] += (op.end - op.start) // self._tokens_per_block
            else:
                self.counters["publications_failed"] += 1
        if op.operation_id is not None:
            self._transfers.return_now(op.operation_id, deliver=False)
        elif op.lease is not None:
            op.lease.release()  # a lease the transfer manager did not take is still ours
        op.closed = True

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
        blocks = (op.end - op.start) // self._tokens_per_block
        self.counters["fetches_resumed"] += 1
        self.counters["fetched_blocks"] += blocks
        self.counters["fetched_bytes"] += blocks * self._bytes_per_block

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
        # fetches_dropped_late, _short, _stopped or _shutdown.
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
        matched = self._kv_cache_manager.probe_context_reuse(request)
        if matched is None:  # it holds a cache already
            return False
        tpb = self._tokens_per_block
        claim = matched // tpb * tpb
        # A named fetch ends at the whole blocks before the last prompt token.
        end = min(
            request.py_store_hit_tokens,
            claim + self._policy.max_fetch_tokens,
            (request.prompt_len - 1) // tpb * tpb,
        )
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
            memory_kind=self._policy.memory_kind,
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

    def _pages_allow_fetch(self, request: LlmRequest) -> bool:
        """Whether the request's whole prompt plus one step fits in free pages, and the pages
        nothing may evict stay within the manager's resume budget with it."""
        used, total = self._kv_cache_manager.get_primary_block_counts()
        need = self._pages_of(request)
        return need <= total - used and self._pinned_pages() + need <= self._resume_room(total)

    def _pinned_pages(self) -> int:
        """Pages no eviction or suspension may take now: what parked fetches lent up to their
        ends; under GPU Direct the prompt and one step of each request an open publication locks,
        once each (one freed meanwhile counts among the deferred closes); and the pages of caches
        whose close waits for a lease."""
        tpb = self._tokens_per_block
        pinned = sum(
            ceil_div(op.end, tpb)
            for op in self._ops
            if op.kind is OpKind.FETCH and op.state is not OpState.OPEN
        )
        if self._policy.memory_kind is MemoryKind.DEVICE:
            holders = self._kv_cache_manager.kv_cache_map
            locked = {
                op.request_id: op.request
                for op in self._ops
                if op.kind is OpKind.PUBLISH and op.request_id in holders
            }
            pinned += sum(self._pages_of(request) for request in locked.values())
        return pinned + sum(self._lender.deferred_close_page_counts().values())

    def _pages_of(self, request: LlmRequest) -> int:
        """The request's whole prompt and one step."""
        return ceil_div(request.prompt_len, self._tokens_per_block) + 1

    def _resume_room(self, total_pages: int) -> int:
        """The pages the manager may hold pinned and still resume its largest request."""
        return int(self._policy.max_util_for_resume * total_pages) - self._resume_window_pages

    # Publication.

    def _publish_waiting(self) -> None:
        """Lends and submits the waiting ranges in arrival order while they fit beside the open
        publications."""
        holders = self._kv_cache_manager.kv_cache_map
        locks = None if self._policy.memory_kind is MemoryKind.HOST else self._publication_locks()
        ready = None
        for request in list(self._unpublished.values()):
            start, end, committed_end = self._publication_range(request)
            # A request freed meanwhile has nothing left to read.
            if request.py_request_id not in holders or end <= start:
                del self._unpublished[request.py_request_id]
                continue
            if locks is None:
                if not self._staging_fits(end - start):
                    self.counters["publication_waits"] += 1
                    return  # staging frees in order: the ranges behind wait too
            else:
                pages = self._pages_of(request)
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
                    continue  # under GPU Direct a range waits on its own
                if not locks.take(request.py_request_id, pages, self._ends_context(request, end)):
                    self.counters["publication_waits"] += 1
                    continue  # under GPU Direct a range waits on its own
            if ready is None:
                ready = next(self._ready_events)
                ready.record(self._execution_stream)  # after the forward and the commits
            if end == committed_end:
                del self._unpublished[request.py_request_id]
            self._publish(request, start, end, ready)

    def _staging_fits(self, tokens: int) -> bool:
        """Host staging: whether the range fits in what open publications leave of the budget."""
        open_tokens = sum(op.end - op.start for op in self._ops if op.kind is OpKind.PUBLISH)
        return tokens <= self._policy.publication_tokens - open_tokens

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
            return start, min(committed_end, start + self._policy.publication_tokens), committed_end
        return start, committed_end, committed_end

    def _publish(self, request: LlmRequest, start: int, end: int, ready: torch.cuda.Event) -> None:
        op = StoreOp(
            seq=next(self._seq),
            kind=OpKind.PUBLISH,
            request=request,
            memory_kind=self._policy.memory_kind,
            state=OpState.OPEN,
            start=start,
            end=end,
        )
        request.py_published_end = end
        if not self._start(op, ready):
            self.counters["publications_not_started"] += 1
            if op.lease is not None and op.lease.failure is not None:
                # The lender refused the range by the request's own state, which every rank
                # shares: its next range would be refused too.
                request.py_published_end = request.prompt_len
                self._unpublished.pop(request.py_request_id, None)
        self._ops.append(op)
        self.counters["publications_started"] += 1

    # Helpers.

    def _start(self, op: StoreOp, ready: torch.cuda.Event) -> bool:
        """Lends the operation's rows by name, in place or in host staging, and hands the lease
        to the transfer manager. Returns whether the store access started; where it did not, the
        entry reports nothing delivered and a decision returns any lease it holds."""
        if not self._lender_ok:
            return False
        fetch = op.kind is OpKind.FETCH
        lend = self._lender.lend_write if fetch else self._lender.lend_read
        try:
            lease = lend(
                op.request,
                op.start,
                op.end,
                memory_kind=op.memory_kind,
                rows=RowSelection.NAMED,
                ready_event=ready,
            )
        except RuntimeError as error:  # the lender lends no more
            self._lender_failed(f"a lend failed: {error}")
            return False
        if lease.failure is not None:
            op.lease = lease
            # One warning per kind of reason: positions and request ids do not make a new one.
            logger.warning_once(
                f"kv_sharing: the lender refused a {op.kind.name.lower()}: {lease.failure}",
                key=re.sub(r"\d+", "#", lease.failure),
            )
            return False
        intent = TransferIntent(
            BINDING_ID, PROFILE_ID, Direction.FETCH if fetch else Direction.PUBLISH, op.memory_kind
        )
        result = self._transfers.submit(lease, intent)
        if not isinstance(result, Accepted):
            op.lease = lease
            self._lender_failed(f"the transfer manager refused an operation: {result.reason}")
            return False
        op.operation_id = result.operation_id
        return True

    def _lender_failed(self, reason: str) -> None:
        """This rank's lender or store binding stopped: report it in the next round, which stops
        sharing on every rank, and stop the transfer manager's lender calls."""
        if self._lender_ok:
            logger.error(f"kv_sharing: {reason}")
            self._transfers.stop_lender_domain(DOMAIN_ID, reason)
        self._lender_ok = False

    def _free_cache(self, request: LlmRequest) -> None:
        if request.py_request_id in self._kv_cache_manager.kv_cache_map:
            self._kv_cache_manager.free_resources(request)

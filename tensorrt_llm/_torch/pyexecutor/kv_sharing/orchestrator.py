# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The executor's one entry to KV sharing.

Its methods run on the executor thread, the only one the lender and the transfer manager accept
calls on once the executor started.
"""

from __future__ import annotations

import collections
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from tensorrt_llm.logger import logger

from ...disaggregation.transfer_manager import (
    Accepted,
    Direction,
    OperationObservation,
    TransferIntent,
    TransferManager,
)
from ..kv_cache.sharing import Lender, MemoryKind, RowSelection

if TYPE_CHECKING:
    from ...disaggregation.lifecycle.retirement import RetirementWatchdog
    from ...disaggregation.orchestration.coordinator import DisaggTransferCoordinator
    from ..llm_request import LlmRequest

# Transfer manager identifiers of the one store binding and its lender.
BINDING_ID = "mooncake"
DOMAIN_ID = "kv-cache-lender"
PROFILE_ID = "kv-sharing"


@dataclass
class _Publication:
    """A prompt range handed to the store, until its operation retires."""

    request: LlmRequest
    tokens: int


class SharedKvOrchestrator:
    """Publishes the whole prompt blocks each context step commits to the store, and counts what
    moved.

    Publications in flight hold at most ``publication_tokens`` tokens of host staging: a fetch
    waits behind them for staging slots, and a store call counts toward the access deadline from
    the moment it is handed to the store. A range that does not fit waits for earlier ones to
    retire; a range that did not fully land ends its request's publication, since blocks after a
    hole cannot be reached by a reader.

    Args:
        publish: Whether this server stores the prompt blocks it computes.
        tokens_per_block: The KV cache's block size.
        publication_tokens: Most prompt tokens publications in flight may hold, whole blocks.
        lender: The KV cache manager's lender.
        transfers: The transfer manager that owns every lease handed to the store.
        coordinator: The executor's coordinator, which collects the transfer manager's observations.
        execution_stream: The stream the forward passes run on.
        watchdog: The watchdog of the transfer manager's store accesses, stopped at shutdown.
    """

    def __init__(
        self,
        *,
        publish: bool,
        tokens_per_block: int,
        publication_tokens: int,
        lender: Lender,
        transfers: TransferManager,
        coordinator: DisaggTransferCoordinator,
        execution_stream: torch.cuda.Stream,
        watchdog: RetirementWatchdog,
    ) -> None:
        self.counters: collections.Counter[str] = collections.Counter()
        self._publishing = publish
        self._tokens_per_block = tokens_per_block
        self._publication_tokens = publication_tokens
        self._lender = lender
        self._transfers = transfers
        self._coordinator = coordinator
        self._execution_stream = execution_stream
        self._watchdog = watchdog
        # Recorded after the forward launches and handed to every lend; one reused event, since
        # constructing one drops the GIL.
        self._forward_done = torch.cuda.Event()
        # Requests with committed prompt blocks not handed to the store yet, oldest first.
        self._unpublished: dict[int, LlmRequest] = {}
        self._publications: dict[int, _Publication] = {}  # by operation id
        self._publication_tokens_in_flight = 0
        self._stopped = False

    def publish_committed_blocks(self, context_requests: list[LlmRequest]) -> None:
        """Hands the store the prompt blocks this step's context requests committed. Call it after
        the forward launch and the context commits, before anything can free these requests."""
        if not self._publishing:
            return
        for request in context_requests:
            # Readers name a prompt by its tokens alone, which multimodal prompts are not.
            if not request.is_dummy and request.py_multimodal_data is None:
                self._unpublished[request.py_request_id] = request
        self._publish_unpublished()

    def before_scheduling(self) -> None:
        """Takes the transfer manager's news of this iteration's loop head and publishes what
        waited for room. Call it once per iteration, before scheduling."""
        for update in self._coordinator.take_shared_updates():
            if update.retired and update.operation_id in self._publications:
                self._retire_publication(self._publications.pop(update.operation_id), update)
        failures = self._coordinator.take_shared_domain_failures()
        if failures:
            self._stop_sharing(failures[0].fault)
        if self._publishing and self._unpublished:
            self._publish_unpublished()

    def has_waiting_publications(self) -> bool:
        """Whether prompt ranges wait for room to be published: they are retried before scheduling,
        so the executor loop must not idle on its request queue meanwhile."""
        return bool(self._unpublished)

    def begin_shutdown(self) -> None:
        """Publishes nothing more: the transfer manager accepts no work once shutdown began. Call
        it at the shutdown request."""
        self._publishing = False
        self._unpublished.clear()

    def shutdown(self) -> None:
        """Logs the counters and stops the access watchdog once every store access ended. Call it
        when the executor loop ended, after the transfer manager's shutdown began."""
        self.before_scheduling()
        transfers = {f"transfers_{name}": count for name, count in self._transfers.counters.items()}
        logger.info(f"kv_sharing counters: {dict(self.counters, **transfers)}")
        # The watchdog refuses to stop while a store access may still touch KV cache memory.
        if self._transfers.shutdown_status().complete:
            self._watchdog.stop()

    def _publish_unpublished(self) -> None:
        """Lends and submits the waiting ranges in arrival order, each at most the publication
        budget, until one does not fit next to the publications in flight."""
        tokens_per_block = self._tokens_per_block
        forward_recorded = False
        for request_id, request in list(self._unpublished.items()):
            start = request.py_published_end
            if start is None:  # blocks this request reused locally are not published again
                start = request.prepopulated_prompt_len // tokens_per_block * tokens_per_block
            computed = min(request.context_current_position, request.prompt_len)
            committed_end = computed // tokens_per_block * tokens_per_block
            end = min(committed_end, start + self._publication_tokens)
            if end <= start:
                del self._unpublished[request_id]
                continue
            if self._publication_tokens_in_flight + end - start > self._publication_tokens:
                return  # the range waits, unpublished, for earlier publications to retire
            if not forward_recorded:
                self._forward_done.record(self._execution_stream)
                forward_recorded = True
            try:
                lease = self._lender.lend_read(
                    request,
                    start,
                    end,
                    memory_kind=MemoryKind.HOST,
                    rows=RowSelection.NAMED,
                    ready_event=self._forward_done,
                )
            except RuntimeError as error:  # the lender lends no more
                self._stop_sharing(f"lend_read failed: {error}", stop_lender=True)
                return
            if lease.failure is not None:
                lease.release()  # a failed lease is still ours to return
                _warn_once_per_reason("a publication was refused", lease.failure)
                self._end_publication(request)
                self.counters["publications_refused"] += 1
                continue
            intent = TransferIntent(BINDING_ID, PROFILE_ID, Direction.PUBLISH, MemoryKind.HOST)
            result = self._transfers.submit(lease, intent)
            if not isinstance(result, Accepted):
                # Outside shutdown, which stops publishing first, the binding has stopped.
                lease.release()  # a refused lease stays ours
                self._stop_sharing(f"the transfer manager refused a publication: {result.reason}")
                return
            request.py_published_end = end
            if end == committed_end:
                del self._unpublished[request_id]
            self._publications[result.operation_id] = _Publication(request, end - start)
            self._publication_tokens_in_flight += end - start
            self.counters["publications_started"] += 1

    def _retire_publication(self, publication: _Publication, update: OperationObservation) -> None:
        self._publication_tokens_in_flight -= publication.tokens
        outcome = update.outcome
        delivered = 0 if outcome is None else sum(int(m.sum()) for m in outcome.delivered_masks)
        rows = 0 if outcome is None else sum(len(m) for m in outcome.delivered_masks)
        self.counters["published_blocks"] += delivered
        failed = outcome is None or outcome.fault is not None or update.fault is not None
        if failed or delivered < rows:
            self._end_publication(publication.request)
            self.counters["publications_failed"] += 1

    def _end_publication(self, request: LlmRequest) -> None:
        """Publishes nothing more of ``request``: a reader stops at the first block the store
        lacks, so the blocks after it would be unreachable."""
        request.py_published_end = request.prompt_len
        self._unpublished.pop(request.py_request_id, None)

    def _stop_sharing(self, reason: str, *, stop_lender: bool = False) -> None:
        """Turns sharing off for good; requests compute their prompts locally from now on."""
        if self._stopped:
            return
        self._stopped = True
        logger.error(f"kv_sharing stops: {reason}; requests compute their prompts locally")
        self._publishing = False
        self._unpublished.clear()
        if stop_lender:
            self._transfers.stop_lender_domain(DOMAIN_ID, reason)


def _warn_once_per_reason(what: str, reason: str) -> None:
    """One warning per kind of reason: positions and request ids in it do not make a new one."""
    logger.warning_once(f"kv_sharing: {what}: {reason}", key=re.sub(r"\d+", "#", reason))

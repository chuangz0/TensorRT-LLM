# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rank 0 asks the store how much of each new prompt it holds before the request is scheduled,
and asks again for requests that still wait."""

from __future__ import annotations

import collections
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from tensorrt_llm.bindings import executor as tllm_executor
from tensorrt_llm.logger import logger

from ...disaggregation.blob.mooncake import PROBE_KEYS_PER_CALL
from ...disaggregation.blob.store import ProbeStatus

if TYPE_CHECKING:
    from ...disaggregation.blob.store import StoreDriver
    from ..executor_request_queue import RequestQueueItem
    from ..kv_cache.sharing import Lender

# A waiting request's hit is measured again this long after its last probe: each existence check
# leases the objects for reading for Mooncake's default_kv_lease_ttl (10 s), so a request admitted
# later still finds what it was told about.
REFRESH_AFTER_S = 4.0
# How often rank 0 looks for waiting hits due for a refresh.
REFRESH_SCAN_S = 1.0
# A refresh round stops adding prompts once it reached this many keys, one store call's worth, so a
# probe of new requests that carries a refresh makes about one store call more.
REFRESH_KEYS_PER_ROUND = PROBE_KEYS_PER_CALL
# TODO(mooncake): a refresh asks about every share of every block. Puts that carried each block's
# shares as one group (0.3.13's ReplicateConfig.group_ids) would let one key's existence check
# renew every share's read lease, but 0.3.13 does not answer whether the group's other members
# exist; an answer per group would let a refresh ask one key per block.


@dataclass(frozen=True)
class _Prompt:
    """A request to probe: its tokens are copied at enqueue, so the probe thread never reads the
    request itself."""

    request_id: int
    tokens: np.ndarray  # int32
    cache_salt: Optional[str]
    num_blocks: int
    # The largest answer: see ``_prompt_to_probe``.
    answer_limit: int


@dataclass
class _Hit:
    """A waiting request with a stored prefix worth fetching, as rank 0 last measured it.

    Attributes:
        prompt: What to probe again.
        store_hit_tokens: The hit the request carries.
        probed_at: When the store last answered for it; its read leases run from then.
        refreshing: A refresh of it is out.
    """

    prompt: _Prompt
    store_hit_tokens: int
    probed_at: float
    refreshing: bool = False


class ProbeService:
    """Asks the store, on its own thread, how much of each new prompt a fetch can read from it, and
    keeps rank 0's requests out of the broadcast until the answer came or the hold limit passed. A
    request leaves with ``store_hit_tokens`` set to the answer, or ``None`` unanswered. The answer
    is the lender's ``usable_end`` of the blocks the store holds per layer group, a block counting
    where the store holds every share of it (each rank reads its own shares): the leading run of
    whole blocks under full attention; with a sliding window, an end a publication reached. Requests
    that left with a hit worth fetching are probed again every REFRESH_AFTER_S while they wait, and
    ``take_refreshed_hits`` reports the hits that changed.

    Args:
        lender: Names the blocks as a publication names them.
        driver: The store driver; only this service's thread calls its ``probe``.
        key_stem: How every store key of this server's objects starts.
        tokens_per_block: The KV cache's block size.
        min_fetch_tokens: Shorter stored prefixes are not fetched; a prompt must exceed it to be
            probed, as a fetch leaves the last prompt token to compute.
        shares_per_group: How many distinct shares each layer group's content is split into
            (one per tensor-parallel rank with KV heads of its own; 1 for MLA and attention DP).
        windowed: Whether a layer group keeps a sliding window.
        hold_s: Longest a request waits for its answer.
        clock: Monotonic seconds.
    """

    def __init__(
        self,
        lender: Lender,
        driver: StoreDriver,
        key_stem: str,
        *,
        tokens_per_block: int,
        min_fetch_tokens: int,
        shares_per_group: Sequence[int],
        windowed: bool,
        hold_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.counters: collections.Counter[str] = collections.Counter()
        self._lender = lender
        self._driver = driver
        self._key_stem = key_stem
        self._tokens_per_block = tokens_per_block
        self._min_fetch_tokens = min_fetch_tokens
        self._shares_per_group = tuple(shares_per_group)
        self._keys_per_block = sum(shares_per_group)  # one key per (layer group, share) per block
        # The first key row of each layer group among a prompt's rows, group-major.
        self._group_rows = np.cumsum((0, *self._shares_per_group[:-1]))
        self._windowed = windowed
        self._hold_s = hold_s
        self._clock = clock
        # Guards the six fields below and wakes the probe thread.
        self._cond = threading.Condition()
        self._work: list[_Prompt] = []
        self._deadlines: dict[int, float] = {}  # by request id, while held and unanswered
        # By request id: the prompt, the answer and when the store gave it.
        self._answers: dict[int, tuple[_Prompt, Optional[int], float]] = {}
        self._refresh_work: list[_Prompt] = []
        # By request id: a refresh's hit (None: the store call failed) and when the store gave it.
        self._refreshed: dict[int, tuple[Optional[int], float]] = {}
        self._stopped = False
        # Executor thread only.
        self._held: list[RequestQueueItem] = []
        self._waiting_hits: dict[int, _Hit] = {}
        self._next_scan = 0.0
        self._thread = threading.Thread(target=self._run, name="kv-sharing-probe", daemon=True)
        self._thread.start()

    def submit(self, item: RequestQueueItem) -> None:
        """Queues the request for a probe if it may fetch. Runs on the enqueuing thread."""
        prompt = self._prompt_to_probe(item)
        if prompt is None:
            return
        with self._cond:
            if self._stopped:
                return
            self._deadlines[item.id] = self._clock() + self._hold_s
            self._work.append(prompt)
            self._cond.notify()

    def hold_until_probed(self, fetched: list[RequestQueueItem]) -> list[RequestQueueItem]:
        """The items to broadcast now: held requests whose answer came or whose hold passed, then
        new ones likewise, then the other items. A cancel releases the held request it cancels and
        a shutdown releases them all, ahead of themselves. Runs on the executor thread."""
        shutdown = any(item.is_shutdown_request for item in fetched)
        cancelled = {item.id for item in fetched if item.is_canceled_request}
        now = self._clock()
        released, held = [], []
        with self._cond:
            for item in self._held + [item for item in fetched if item.is_normal_request]:
                release = shutdown or item.id in cancelled
                (released if self._leaves(item, now, release) else held).append(item)
        self._held = held
        return released + [item for item in fetched if not item.is_normal_request]

    def holds_requests(self) -> bool:
        """Whether requests wait for their answer outside the request queue."""
        return bool(self._held)

    def take_refreshed_hits(
        self, waiting: Iterable[RequestQueueItem]
    ) -> tuple[tuple[int, int], ...]:
        """The hits of waiting requests that a refresh changed, as ``(request id, hit)``. At most
        once per REFRESH_SCAN_S it forgets requests no longer ``waiting`` and queues one probe of
        the waiting hits last probed REFRESH_AFTER_S ago or earlier, in queue order, adding
        prompts until REFRESH_KEYS_PER_ROUND keys. Runs on the executor thread, every pass, before
        ``hold_until_probed``: ``waiting`` must hold every request an earlier pass released."""
        if not self._waiting_hits:
            return ()
        with self._cond:
            refreshed, self._refreshed = self._refreshed, {}
        changed = []
        for request_id, (hit, probed_at) in refreshed.items():
            kept = self._waiting_hits.get(request_id)
            if kept is None:
                continue
            kept.refreshing = False
            if hit is None:  # the store call failed: the old hit stands
                self.counters["refresh_errors"] += 1
                continue
            kept.probed_at = probed_at
            if hit != kept.store_hit_tokens:
                kept.store_hit_tokens = hit
                changed.append((request_id, hit))
            if hit < self._min_fetch_tokens:
                del self._waiting_hits[request_id]
        now = self._clock()
        if now >= self._next_scan:
            self._next_scan = now + REFRESH_SCAN_S
            self._queue_refreshes(waiting, now)
        return tuple(changed)

    def stop(self, wait_s: float = 0.0) -> None:
        """Probes nothing more; held requests leave at the next call, unanswered. Waits up to
        ``wait_s`` for the probe thread, which may be inside a store call, to end."""
        with self._cond:
            self._stopped = True
            self._deadlines.clear()
            self._work.clear()
            self._refresh_work.clear()
            self._cond.notify()
        if wait_s > 0:
            self._thread.join(wait_s)

    def _prompt_to_probe(self, item: RequestQueueItem) -> Optional[_Prompt]:
        """The prompt of a request that may fetch: one text prompt, no LoRA, no output computed
        from the prompt's own positions, longer than ``min_fetch_tokens``."""
        request = item.request
        if not item.is_normal_request or item.child_req_ids:
            return None
        output = request.output_config
        if (
            request.multimodal_input is not None
            or getattr(request, "py_multimodal_data", None) is not None
            or request.lora_config is not None
            or request.request_type == tllm_executor.RequestType.REQUEST_TYPE_GENERATION_ONLY
            or output.return_context_logits
            or output.additional_model_outputs
        ):
            return None
        tokens = request.input_token_ids
        if len(tokens) <= self._min_fetch_tokens:
            return None
        tpb = self._tokens_per_block
        num_blocks = len(tokens) // tpb
        # The answer is also where a publication starts. Without a window a fetch cuts it before
        # the last prompt token itself, so it may name every whole block (and a publication then
        # skips the block ending a whole-block prompt); with one a fetch takes the answer exactly,
        # so it ends at the whole blocks before the last prompt token.
        answer_limit = (len(tokens) - 1) // tpb * tpb if self._windowed else num_blocks * tpb
        prompt = np.asarray(tokens, dtype=np.int32)
        return _Prompt(item.id, prompt, request.cache_salt, num_blocks, answer_limit)

    def _leaves(self, item: RequestQueueItem, now: float, release: bool) -> bool:
        """Whether a request leaves the hold now, setting its answer if it came; a hit worth
        fetching is then kept for refreshes while the request waits."""
        if item.id in self._answers:
            prompt, hit, probed_at = self._answers.pop(item.id)
            item.store_hit_tokens = hit
            if hit is not None and hit >= self._min_fetch_tokens:
                self._waiting_hits[item.id] = _Hit(prompt, hit, probed_at)
            return True
        deadline = self._deadlines.get(item.id)
        if deadline is None:  # not probed
            return True
        if release or now >= deadline:
            del self._deadlines[item.id]
            if not release:
                self.counters["held_past_limit"] += 1
            return True
        return False

    def _queue_refreshes(self, waiting: Iterable[RequestQueueItem], now: float) -> None:
        """Forgets kept hits no longer waiting and hands the thread those due for a refresh."""
        waiting_ids, due, keys = set(), [], 0
        for item in waiting:
            waiting_ids.add(item.id)
            kept = self._waiting_hits.get(item.id)
            if (
                kept is None
                or kept.refreshing
                or now - kept.probed_at < REFRESH_AFTER_S
                or keys >= REFRESH_KEYS_PER_ROUND
            ):
                continue
            kept.refreshing = True
            due.append(kept.prompt)
            keys += kept.prompt.num_blocks * self._keys_per_block
        for request_id in self._waiting_hits.keys() - waiting_ids:  # admitted, cancelled, ended
            del self._waiting_hits[request_id]
        if due:
            with self._cond:
                self._refresh_work += due
                self._cond.notify()

    def _run(self) -> None:
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._work or self._refresh_work or self._stopped)
                if self._stopped:
                    return
                # A request released meanwhile needs no answer. Refreshes ride the same store
                # call, so steady arrivals never starve them.
                first = [p for p in self._work if p.request_id in self._deadlines]
                refresh = self._refresh_work
                self._work, self._refresh_work = [], []
            if not first and not refresh:
                continue
            probed_at = self._clock()  # the store's read leases run from the call
            answers = self._probe(first + refresh)
            with self._cond:
                for prompt, answer in zip(first, answers):
                    if prompt.request_id in self._deadlines:
                        del self._deadlines[prompt.request_id]
                        self._answers[prompt.request_id] = (prompt, answer, probed_at)
                for prompt, answer in zip(refresh, answers[len(first) :]):
                    self._refreshed[prompt.request_id] = (answer, probed_at)

    def _probe(self, batch: list[_Prompt]) -> list[Optional[int]]:
        """Each prompt's usable end; ``None`` for every one where naming, the store call or the
        lender's rule failed. One driver call for the batch; the driver asks PROBE_KEYS_PER_CALL
        keys per store call."""
        try:
            names = np.concatenate(
                [
                    self._lender.prompt_object_keys(
                        prompt.tokens,
                        cache_salt=prompt.cache_salt,
                        layer_group_index=group,
                        num_blocks=prompt.num_blocks,
                        shares=range(shares),
                    )
                    for prompt in batch
                    for group, shares in enumerate(self._shares_per_group)
                ]
            )
            hexed, width = names.tobytes().hex(), 2 * names.shape[1]  # one pass for the batch
            keys = [self._key_stem + hexed[i : i + width] for i in range(0, len(hexed), width)]
            statuses = self._driver.probe(keys, self._hold_s)
            held = np.fromiter((status is ProbeStatus.HIT for status in statuses), dtype=bool)
            answers = self._usable_ends(batch, held)
        except Exception as error:  # the requests compute their prompts
            self.counters["probe_errors"] += 1
            logger.warning_once(f"kv_sharing: a store probe failed: {error!r}", key="probe")
            return [None] * len(batch)
        self.counters["probes"] += 1
        return answers

    def _usable_ends(self, batch: list[_Prompt], held: np.ndarray) -> list[int]:
        """Each prompt's usable end by the lender's rule, given ``held``, whether the store holds
        each of the batch's keys."""
        rows = self._keys_per_block
        answers, offset = [], 0
        for prompt in batch:
            size = rows * prompt.num_blocks
            # Rows: one per (layer group, share), group-major; columns: blocks.
            present = held[offset : offset + size].reshape(rows, prompt.num_blocks)
            offset += size
            # One mask per layer group: a block counts where every share of it is held.
            by_group = np.logical_and.reduceat(present, self._group_rows, axis=0)
            answers.append(self._lender.usable_end(by_group, known=0, limit=prompt.answer_limit))
        return answers

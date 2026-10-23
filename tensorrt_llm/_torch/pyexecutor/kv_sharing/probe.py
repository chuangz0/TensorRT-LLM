# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rank 0 asks the store how much of each new prompt it holds before the request is scheduled."""

from __future__ import annotations

import collections
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from tensorrt_llm.bindings import executor as tllm_executor
from tensorrt_llm.logger import logger

from ...disaggregation.blob.store import ProbeStatus

if TYPE_CHECKING:
    from ...disaggregation.blob.store import StoreDriver
    from ..executor_request_queue import RequestQueueItem
    from ..kv_cache.sharing import Lender


@dataclass(frozen=True)
class _Prompt:
    """A request to probe: its tokens are copied at enqueue, so the probe thread never reads the
    request itself."""

    request_id: int
    tokens: list[int]
    cache_salt: Optional[str]
    num_blocks: int


class ProbeService:
    """Asks the store, on its own thread, how many leading whole blocks of each new prompt it
    holds, and keeps rank 0's requests out of the broadcast until the answer came or the hold limit
    passed. A request leaves with ``store_hit_tokens`` set to the answer, or ``None`` unanswered.

    Args:
        lender: Names the blocks as a publication names them.
        driver: The store driver; only this service's thread calls its ``probe``.
        key_stem: How every store key of this server's objects starts.
        tokens_per_block: The KV cache's block size.
        min_prompt_tokens: Shorter prompts are not probed: they cannot fetch enough to pay off.
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
        min_prompt_tokens: int,
        hold_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.counters: collections.Counter[str] = collections.Counter()
        self._lender = lender
        self._driver = driver
        self._key_stem = key_stem
        self._tokens_per_block = tokens_per_block
        self._min_prompt_tokens = min_prompt_tokens
        self._hold_s = hold_s
        self._clock = clock
        # Guards the four fields below and wakes the probe thread.
        self._cond = threading.Condition()
        self._work: list[_Prompt] = []
        self._deadlines: dict[int, float] = {}  # by request id, while held and unanswered
        self._answers: dict[int, Optional[int]] = {}
        self._stopped = False
        self._held: list[RequestQueueItem] = []  # executor thread only
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

    def stop(self, wait_s: float = 0.0) -> None:
        """Probes nothing more; held requests leave at the next call, unanswered. Waits up to
        ``wait_s`` for the probe thread, which may be inside a store call, to end."""
        with self._cond:
            self._stopped = True
            self._deadlines.clear()
            self._work.clear()
            self._cond.notify()
        if wait_s > 0:
            self._thread.join(wait_s)

    def _prompt_to_probe(self, item: RequestQueueItem) -> Optional[_Prompt]:
        """The prompt of a request that may fetch: one text prompt, no LoRA, no output computed
        from the prompt's own positions, at least ``min_prompt_tokens`` long."""
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
        if len(tokens) < self._min_prompt_tokens:
            return None
        # Every whole block: a fetch still stops before the last prompt token, and a publication
        # starts past every block the store holds, the one ending the prompt included.
        num_blocks = len(tokens) // self._tokens_per_block
        return _Prompt(item.id, tokens, request.cache_salt, num_blocks)

    def _leaves(self, item: RequestQueueItem, now: float, release: bool) -> bool:
        """Whether a request leaves the hold now, setting its answer if it came."""
        if item.id in self._answers:
            item.store_hit_tokens = self._answers.pop(item.id)
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

    def _run(self) -> None:
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._work or self._stopped)
                if self._stopped:
                    return
                # A request released meanwhile needs no answer.
                batch = [prompt for prompt in self._work if prompt.request_id in self._deadlines]
                self._work.clear()
            if not batch:
                continue
            answers = self._probe(batch)
            with self._cond:
                for prompt, answer in zip(batch, answers):
                    if prompt.request_id in self._deadlines:
                        del self._deadlines[prompt.request_id]
                        self._answers[prompt.request_id] = answer

    def _probe(self, batch: list[_Prompt]) -> list[Optional[int]]:
        """Each prompt's leading tokens the store holds, in whole blocks; ``None`` for every one
        where naming or the store call failed. One store call for the batch."""
        try:
            names = np.concatenate(
                [
                    self._lender.prompt_object_keys(
                        prompt.tokens,
                        cache_salt=prompt.cache_salt,
                        layer_group_index=0,
                        num_blocks=prompt.num_blocks,
                    )
                    for prompt in batch
                ]
            )
            hexed, width = names.tobytes().hex(), 2 * names.shape[1]  # one pass for the batch
            keys = [self._key_stem + hexed[i : i + width] for i in range(0, len(hexed), width)]
            statuses = self._driver.probe(keys, self._hold_s)
        except Exception as error:  # the requests compute their prompts
            self.counters["probe_errors"] += 1
            logger.warning_once(f"kv_sharing: a store probe failed: {error!r}", key="probe")
            return [None] * len(batch)
        answers, offset = [], 0
        for prompt in batch:
            present = statuses[offset : offset + prompt.num_blocks]
            offset += prompt.num_blocks
            leading = next(
                (i for i, status in enumerate(present) if status is not ProbeStatus.HIT),
                len(present),
            )
            answers.append(leading * self._tokens_per_block)
        self.counters["probes"] += 1
        return answers

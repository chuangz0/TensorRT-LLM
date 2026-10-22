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
"""Rank 0 keeps a new request out of the broadcast until the store said how much of its prompt it
holds, the hold limit passed, or a cancel or shutdown needs it broadcast."""

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.blob.store import ProbeStatus
from tensorrt_llm._torch.pyexecutor.executor_request_queue import (
    SHUTDOWN_REQUEST_ID,
    RequestQueueItem,
)
from tensorrt_llm._torch.pyexecutor.kv_sharing.probe import ProbeService
from tensorrt_llm.bindings import executor as tllm_executor

pytestmark = pytest.mark.cpu_only

TPB = 4
STEM = "t/v1/"
STORED_BLOCKS = 3
HOLD_S = 0.05
WAIT_S = 5.0


class Lender:
    """Names block ``i`` of any prompt ``bytes([i]) * 54``."""

    def prompt_object_keys(self, tokens, *, cache_salt, layer_group_index, num_blocks):
        assert layer_group_index == 0 and cache_salt == "tenant"
        return np.array([[i] * 54 for i in range(num_blocks)], dtype=np.uint8)


class Driver:
    """Holds each probe until ``answer`` is set, then finds the first ``STORED_BLOCKS`` blocks."""

    def __init__(self) -> None:
        self.answer = threading.Event()

    def probe(self, keys, budget_s, hint=None):
        assert self.answer.wait(WAIT_S)
        stored = {STEM + (bytes([i]) * 54).hex() for i in range(STORED_BLOCKS)}
        return tuple(ProbeStatus.HIT if key in stored else ProbeStatus.MISS for key in keys)


def request_item(request_id: int, prompt_tokens: int = 6 * TPB) -> RequestQueueItem:
    request = SimpleNamespace(
        input_token_ids=list(range(prompt_tokens)),
        cache_salt="tenant",
        multimodal_input=None,
        lora_config=None,
        request_type=tllm_executor.RequestType.REQUEST_TYPE_CONTEXT_AND_GENERATION,
        output_config=SimpleNamespace(return_context_logits=False, additional_model_outputs=None),
    )
    return RequestQueueItem(request_id, request)


def leave_with_answer(service: ProbeService, item: RequestQueueItem) -> None:
    """Waits until the held ``item`` leaves the hold."""
    deadline = time.monotonic() + WAIT_S
    while time.monotonic() < deadline:
        if service.hold_until_probed([]) == [item]:
            return
        time.sleep(0.001)
    raise AssertionError("the held request never left")


def test_held_requests_leave_with_their_answer_ahead_of_their_cancel_or_at_the_hold_limit():
    now = [0.0]
    driver = Driver()
    service = ProbeService(
        Lender(),
        driver,
        STEM,
        tokens_per_block=TPB,
        min_prompt_tokens=3 * TPB,
        hold_s=HOLD_S,
        clock=lambda: now[0],
    )
    try:
        # A prompt too short to fetch passes at once; an eligible one waits for the store.
        short, answered = request_item(1, 2 * TPB), request_item(2, 6 * TPB + 1)
        for item in (short, answered):
            service.submit(item)
        assert service.hold_until_probed([short, answered]) == [short]
        assert service.holds_requests() and short.store_hit_tokens is None
        driver.answer.set()
        leave_with_answer(service, answered)
        assert answered.store_hit_tokens == STORED_BLOCKS * TPB

        # A cancel releases the request it cancels ahead of itself, unanswered, and no other.
        driver.answer.clear()
        cancelled, waiting = request_item(3), request_item(4)
        for item in (cancelled, waiting):
            service.submit(item)
        assert service.hold_until_probed([cancelled, waiting]) == []
        cancel = RequestQueueItem(3, is_canceled_request=True)
        assert service.hold_until_probed([cancel]) == [cancelled, cancel]
        assert cancelled.store_hit_tokens is None and service.holds_requests()

        # Past the hold limit a request leaves unanswered.
        now[0] += HOLD_S
        assert service.hold_until_probed([]) == [waiting]
        assert waiting.store_hit_tokens is None and service.counters["held_past_limit"] == 1

        # A shutdown releases every held request ahead of itself.
        held = request_item(5)
        service.submit(held)
        assert service.hold_until_probed([held]) == []
        shutdown = RequestQueueItem(SHUTDOWN_REQUEST_ID)
        assert service.hold_until_probed([shutdown]) == [held, shutdown]
        assert not service.holds_requests()
    finally:
        driver.answer.set()
        service.stop(wait_s=WAIT_S)

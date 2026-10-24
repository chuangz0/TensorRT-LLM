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
holds, the hold limit passed, or a cancel or shutdown needs it broadcast. A block counts only where
the store holds every rank's share of it, the lender's rule turns each layer group's blocks into
the end a fetch may take, and the hits of requests that still wait are measured again."""

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
from tensorrt_llm._torch.pyexecutor.kv_sharing import probe as probe_module
from tensorrt_llm._torch.pyexecutor.kv_sharing.probe import ProbeService
from tensorrt_llm.bindings import executor as tllm_executor

pytestmark = pytest.mark.cpu_only

TPB = 4
STEM = "t/v1/"
STORED_BLOCKS = 3
HOLD_S = 0.05
WAIT_S = 5.0


def key(block: int, share: int = 0, group: int = 0) -> list[int]:
    """The 54 bytes naming share ``share`` of block ``block`` of layer group ``group`` of any
    prompt."""
    return [block] * 52 + [group, share]


def name(block: int, share: int = 0, group: int = 0) -> str:
    """The store key of ``key(block, share, group)``."""
    return STEM + bytes(key(block, share, group)).hex()


class Lender:
    """Names the blocks of any prompt by ``key``. A fetch may end where every layer group holds
    every block before it."""

    def __init__(self) -> None:
        self.asked: list[tuple] = []  # (present, known, limit) of each usable_end

    def prompt_object_keys(self, tokens, *, cache_salt, layer_group_index, num_blocks, shares):
        assert cache_salt == "tenant"
        return np.array(
            [key(i, s, layer_group_index) for s in shares for i in range(num_blocks)],
            dtype=np.uint8,
        )

    def usable_end(self, present, *, known, limit):
        self.asked.append((np.array(present), known, limit))
        held = np.logical_and.reduce(present, axis=0)[: limit // TPB]
        return (len(held) if held.all() else int(np.argmin(held))) * TPB


class Driver:
    """Holds each probe until ``answer`` is set, then finds the ``stored`` keys; raises while
    ``fails``."""

    def __init__(self, stored_blocks: int = STORED_BLOCKS) -> None:
        self.answer = threading.Event()
        self.stored = {name(i) for i in range(stored_blocks)}
        self.probed: list[tuple[str, ...]] = []
        self.fails = False

    def probe(self, keys, budget_s, hint=None):
        assert self.answer.wait(WAIT_S)
        self.probed.append(tuple(keys))
        if self.fails:
            raise RuntimeError("the store is gone")
        return tuple(ProbeStatus.HIT if key in self.stored else ProbeStatus.MISS for key in keys)


def probe_service(
    driver: Driver, shares: int = 1, clock=time.monotonic, *, groups: int = 1, lender=None
) -> ProbeService:
    """A probe over ``groups`` layer groups of ``shares`` shares each; with several, windowed."""
    return ProbeService(
        Lender() if lender is None else lender,
        driver,
        STEM,
        tokens_per_block=TPB,
        min_fetch_tokens=2 * TPB,
        shares_per_group=(shares,) * groups,
        windowed=groups > 1,
        hold_s=HOLD_S,
        clock=clock,
    )


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
    service = probe_service(driver, clock=lambda: now[0])
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


def probe_until_answered(service: ProbeService, item: RequestQueueItem) -> None:
    """Submits ``item`` and waits until it leaves the hold with the store's answer."""
    service.submit(item)
    if service.hold_until_probed([item]) != [item]:
        leave_with_answer(service, item)


def test_the_probe_answers_the_run_of_blocks_every_share_holds():
    driver = Driver()
    driver.answer.set()
    # Rank 1's share of block 3 is missing; rank 0's shares are all there.
    driver.stored |= {name(i, 1) for i in range(STORED_BLOCKS + 3) if i != 3}
    driver.stored |= {name(i) for i in range(STORED_BLOCKS + 3)}
    service = probe_service(driver, shares=2)
    try:
        item = request_item(1)
        probe_until_answered(service, item)
    finally:
        service.stop(wait_s=WAIT_S)
    assert item.store_hit_tokens == 3 * TPB
    (keys,) = driver.probed
    assert keys == tuple(name(i, share) for share in (0, 1) for i in range(6))


def test_the_probe_asks_the_lender_where_a_fetch_may_end_from_every_share_of_each_layer_group():
    """Two layer groups of two shares, a sliding window in one: the lender gets one mask per group,
    each block held where both its shares are, and may end a fetch no later than the whole blocks
    before the last prompt token."""
    driver = Driver()
    driver.answer.set()
    driver.stored = {name(i, s, g) for i in range(6) for s in (0, 1) for g in (0, 1)}
    driver.stored.discard(name(2, share=1, group=1))
    lender = Lender()
    service = probe_service(driver, shares=2, groups=2, lender=lender)
    try:
        item = request_item(1, 6 * TPB)  # its last token ends block 5
        probe_until_answered(service, item)
    finally:
        service.stop(wait_s=WAIT_S)
    ((present, known, limit),) = lender.asked
    assert present.tolist() == [[True] * 6, [True, True, False, True, True, True]]
    assert (known, limit) == (0, 5 * TPB)
    assert item.store_hit_tokens == 2 * TPB


def wait_for(condition) -> None:
    deadline = time.monotonic() + WAIT_S
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.001)


def test_waiting_hits_are_refreshed_and_a_failed_refresh_keeps_the_old_hit():
    now = [0.0]
    driver = Driver()
    driver.answer.set()
    service = probe_service(driver, clock=lambda: now[0])
    item = request_item(1)
    try:
        probe_until_answered(service, item)
        assert item.store_hit_tokens == STORED_BLOCKS * TPB
        # Not due before REFRESH_AFTER_S has passed since the probe.
        now[0] = probe_module.REFRESH_AFTER_S / 2
        assert service.take_refreshed_hits([item]) == () and len(driver.probed) == 1

        # Due: the store now holds two blocks more, and the change is reported once.
        driver.stored |= {name(i) for i in range(STORED_BLOCKS + 2)}
        now[0] = probe_module.REFRESH_AFTER_S
        changed = []

        def refreshed() -> bool:
            changed.extend(service.take_refreshed_hits([item]))
            return bool(changed)

        wait_for(refreshed)
        assert changed == [(1, (STORED_BLOCKS + 2) * TPB)]
        assert service.take_refreshed_hits([item]) == ()

        # A refresh whose store call fails changes nothing and is counted.
        driver.fails = True
        now[0] += probe_module.REFRESH_AFTER_S
        assert service.take_refreshed_hits([item]) == ()
        wait_for(
            lambda: service.take_refreshed_hits([item]) == () and service.counters["refresh_errors"]
        )

        # A request no longer waiting is forgotten: nothing probes it again.
        now[0] += probe_module.REFRESH_SCAN_S
        service.take_refreshed_hits([])
        probes = len(driver.probed)
        now[0] += 2 * probe_module.REFRESH_AFTER_S
        assert service.take_refreshed_hits([item]) == ()
        time.sleep(0.05)
        assert len(driver.probed) == probes
    finally:
        service.stop(wait_s=WAIT_S)

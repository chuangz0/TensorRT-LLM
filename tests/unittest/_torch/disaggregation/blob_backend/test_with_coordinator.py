# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``BlobStoreBackend`` under a real ``KVTransferCoordinator``: one coordinator publishes a
context request's blocks, a second one on another "rank" (its own memory, its own backend, the
same store) probes, plans, launches and lands the fetch. Engine effects, reader, queue and
collective are the ``kv_transfer`` suite's fakes; only the store backend is real.

The backend answers on its own threads, so each step that depends on it waits on an observable
(the fake client's objects, the backend's counters, an effect being recorded) before the next
``advance``.
"""

import time

__extra_import_path__ = ["~/tensorrt_llm/_torch", "../kv_transfer"]
from disaggregation.orchestration.kv_transfer_coordinator import KVTransferCoordinator  # noqa: E402
from disaggregation.orchestration.kv_transfer_interfaces import DEFER, FetchSource  # noqa: E402
from disaggregation.orchestration.remote_cache import FetchPlan, Planner  # noqa: E402
from fakes import (  # noqa: E402
    TPB,
    FakeDist,
    FakeEngineQueue,
    FakeReader,
    FakeRequest,
    RecordingEffects,
    full_attention,
)
from store_fakes import (  # noqa: E402
    FakeStoreClient,
    fill,
    make_rank,
    pattern,
    read,
    wait_until,
    write,
)

PROMPT = 29
END = (PROMPT - 1) // TPB * TPB  # 28: seven nameable blocks
BLOCKS = END // TPB
UNIT_BYTES = 64


class Side:
    """One rank: a store backend over the shared client, with a coordinator on top of it."""

    def __init__(self, client: FakeStoreClient, *, publishes: bool) -> None:
        self.rank = make_rank(client, arena_bytes=1 << 16)
        for o in range(BLOCKS + 2):
            self.rank.resolver.add(0, o, UNIT_BYTES // 2, UNIT_BYTES // 2)  # two segments each
        self.reader = FakeReader(groups=[full_attention(0)], tokens_per_block=TPB)
        self.effects = RecordingEffects()
        self.queue = FakeEngineQueue()
        self.source = FetchSource("store", self.rank.backend, None)
        self.planner = Planner([self.source], self.reader, TPB, probe_budget_rounds=2)
        self.coord = KVTransferCoordinator(
            [self.source],
            [self.rank.backend] if publishes else [],
            self.planner,
            self.reader,
            self.effects,
            self.queue,
            FakeDist(),
        )

    @property
    def backend(self):
        return self.rank.backend

    def block_bytes(self, ordinal: int) -> bytes:
        return read(self.rank.resolver.segments(0, ordinal))

    def write_block(self, ordinal: int, data: bytes) -> None:
        write(self.rank.resolver.segments(0, ordinal), data)

    def fill_all(self, byte: int) -> None:
        for o in range(BLOCKS + 2):
            fill(self.rank.resolver.segments(0, o), byte)

    def records(self):
        return self.coord.status_dump()["records"]

    def advance_until(self, effect: str, req=None, *, deadline_s: float = 5.0) -> None:
        """``advance`` (with ``req`` as the only candidate when given) until ``effect`` lands."""
        deadline = time.monotonic() + deadline_s
        t = 10.0
        while self.effects.count(effect) == 0:
            assert time.monotonic() < deadline, f"{effect} never happened: {self.effects.names()}"
            self.coord.advance([req] if req is not None else [], t)
            t += 1.0
            time.sleep(0.002)

    def close(self) -> None:
        self.rank.client.unblock()
        self.backend.close()


def _publish(ctx: Side, req: FakeRequest) -> None:
    """Context side: offer the request's blocks and drive the publish record to its end."""
    for o in range(BLOCKS):
        ctx.write_block(o, pattern(o + 1, UNIT_BYTES))
    ctx.coord.publish_committed_blocks([req], finished=[], now=0.0)
    assert ctx.records()[0]["state"] == "IN_FLIGHT" and ctx.records()[0]["direction"] == "publish"
    wait_until(lambda: len(ctx.rank.client.objects) == BLOCKS, what="publish to land in the store")
    ctx.coord.advance([], 1.0)
    assert ctx.records() == []  # LANDED, quiesced, released
    assert ctx.effects.calls == []  # a publish of a running request owes the engine nothing
    assert ctx.backend.counters.publish_stored == BLOCKS


def _probe_and_plan(gen: Side, req: FakeRequest) -> FetchPlan:
    """Generation side: the first ``advance`` queues the probe and defers; once the backend has
    answered, the next one plans from the store."""
    gen.coord.advance([req], 0.0)
    assert gen.coord.plan_fetch(req) is DEFER
    counters = gen.backend.counters
    wait_until(lambda: counters.probe_hits + counters.probe_misses == BLOCKS, what="probe answer")
    gen.coord.advance([req], 1.0)
    plan = gen.coord.plan_fetch(req)
    assert isinstance(plan, FetchPlan), plan
    return plan


def test_publish_on_one_rank_then_probe_plan_launch_and_land_on_another():
    client = FakeStoreClient()
    ctx, gen = Side(client, publishes=True), Side(client, publishes=False)
    try:
        req = FakeRequest(1, prompt_len=PROMPT)
        _publish(ctx, req)

        plan = _probe_and_plan(gen, req)
        assert plan.source == "store" and plan.token_end == END and plan.hint is None
        assert plan.units_by_group == {0: tuple(range(BLOCKS))}
        gen.fill_all(0xEE)
        gen.coord.launch_fetches([req], 1.0)
        assert gen.effects.names() == ["prepare_fetch_resources", "park_for_fetch"]
        assert gen.records()[0]["state"] == "IN_FLIGHT"

        gen.advance_until("unpark")
        assert gen.effects.only("unpark") == [(req, END, False, None)]
        assert gen.records()[0]["state"] == "LANDED"
        for o in range(BLOCKS):
            assert gen.block_bytes(o) == pattern(o + 1, UNIT_BYTES) == ctx.block_bytes(o)
        assert gen.block_bytes(BLOCKS) == bytes([0xEE]) * UNIT_BYTES  # not asked for, not touched
        assert gen.backend.counters.fetch_hits == BLOCKS and gen.backend.counters.fetch_misses == 0
        assert gen.backend.counters.failed_attempts == 0

        # The request's end is the release point: one quiesce (True), record gone, no hold.
        gen.coord.notify_request_finished(req)
        assert gen.records() == [] and gen.effects.count("hold_for_transfer") == 0
        assert gen.coord.status_dump() == {
            "records": [],
            "decided_plans": 0,
            "finished_pending": [],
        }
    finally:
        gen.close()
        ctx.close()


def test_content_gone_between_probe_and_fetch_is_a_short_serve_retried_once_then_local():
    client = FakeStoreClient()
    ctx, gen = Side(client, publishes=True), Side(client, publishes=False)
    try:
        req = FakeRequest(2, prompt_len=PROMPT)
        _publish(ctx, req)
        _probe_and_plan(gen, req)
        client.evict_all()  # the answer was advisory (SPEC §6.2 probe): stale before the fetch
        gen.fill_all(0xEE)
        gen.coord.launch_fetches([req], 1.0)

        gen.advance_until("give_back_fetch_pages")
        # Delivered(∅) is a miss, not a failure: quiesce, give the pages back, keep one retry.
        assert gen.effects.names() == [
            "prepare_fetch_resources",
            "park_for_fetch",
            "give_back_fetch_pages",
        ]
        rec = gen.records()[0]
        assert rec["state"] == "PLANNED" and rec["outcomes"] == ["Delivered"]
        assert (
            gen.backend.counters.fetch_misses == BLOCKS
            and gen.backend.counters.failed_attempts == 0
        )
        assert all(gen.block_bytes(o) == bytes([0xEE]) * UNIT_BYTES for o in range(BLOCKS))

        # The retry is bounded by the short serve (hint 0), so the request computes locally.
        gen.coord.advance([req], 5.0)
        assert gen.coord.plan_fetch(req) is None
        assert gen.records() == []
        assert gen.effects.count("fail_requests") == 0 and gen.effects.count("unpark") == 0
    finally:
        gen.close()
        ctx.close()


def test_store_outage_during_fetch_is_failed_gives_pages_back_and_the_retry_lands():
    client = FakeStoreClient()
    ctx, gen = Side(client, publishes=True), Side(client, publishes=False)
    try:
        req = FakeRequest(3, prompt_len=PROMPT)
        _publish(ctx, req)
        _probe_and_plan(gen, req)
        client.fail_next("batch_is_exist")  # the fetch's own lookup, not the probe's
        gen.coord.launch_fetches([req], 1.0)

        gen.advance_until("give_back_fetch_pages")
        rec = gen.records()[0]
        assert rec["state"] == "PLANNED" and rec["outcomes"] == ["Failed"]
        assert gen.backend.counters.failed_attempts == 1
        assert gen.effects.count("fail_requests") == 0  # local fallback exists

        # Re-planned from the cached probe answer, launched again (try 1), and this time lands.
        gen.coord.advance([req], 20.0)
        plan = gen.coord.plan_fetch(req)
        assert isinstance(plan, FetchPlan) and plan.token_end == END
        gen.coord.launch_fetches([req], 20.0)
        assert gen.records()[0]["try_index"] == 1 and gen.records()[0]["attempts"] == 2
        gen.advance_until("unpark")
        assert gen.effects.only("unpark") == [(req, END, False, None)]
        for o in range(BLOCKS):
            assert gen.block_bytes(o) == pattern(o + 1, UNIT_BYTES)
        assert (
            gen.backend.counters.fetch_hits == BLOCKS and gen.backend.counters.failed_attempts == 1
        )
    finally:
        gen.close()
        ctx.close()


def test_probe_outage_defers_within_budget_then_plans_local_without_a_fetch():
    client = FakeStoreClient()
    gen = Side(client, publishes=False)
    try:
        req = FakeRequest(4, prompt_len=PROMPT)
        asked = []

        def unreachable(keys):  # the store is down: every lookup answers an error status
            asked.append(tuple(keys))
            return [-1 for _ in keys]

        client.batch_is_exist = unreachable
        gen.coord.advance([req], 0.0)
        assert gen.coord.plan_fetch(req) is DEFER
        wait_until(lambda: len(asked) >= 1, what="the probe's lookup")
        # The lookup failed on the backend's thread. Whether the second probe finds the error
        # (and raises, which the coordinator logs and keeps pending) or is still waiting, the
        # answer is never an empty set, so the planner defers rather than reading a miss.
        gen.coord.advance([req], 1.0)
        assert gen.coord.plan_fetch(req) is DEFER
        gen.coord.advance([req], 2.0)  # probe budget spent: compute locally
        assert gen.coord.plan_fetch(req) is None
        assert gen.records() == [] and gen.effects.calls == []
        assert gen.backend.counters.probe_misses == 0 and gen.backend.counters.fetch_misses == 0
        assert gen.backend.counters.probe_hits == 0
    finally:
        gen.close()

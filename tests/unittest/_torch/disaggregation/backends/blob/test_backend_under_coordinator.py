# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``BlobStoreBackend`` under a real ``KVTransferCoordinator``: one coordinator publishes a
context request's blocks, a second one on another "rank" (its own memory, its own backend, the
same store) probes, plans, launches and lands the fetch. Engine effects, reader, queue and
collective are the ``kv_transfer`` suite's fakes; only the store backend is real.

The backend answers on its own threads, so each step that depends on it waits on an observable
(the fake store's objects, the backend's counters, an effect being recorded) before the next
``advance``.
"""

import time

import pytest

__extra_import_path__ = ["~/tensorrt_llm/_torch", "../../orchestration/kv_transfer"]
from disaggregation.backends.blob.store import BlobStoreError  # noqa: E402
from disaggregation.backends.config import (  # noqa: E402
    DEFAULT_FETCH_WAIT_TIMEOUT_S,
    DEFAULT_UNLAUNCHED_TIMEOUT_S,
)
from disaggregation.orchestration.kv_transfer.coordinator import KVTransferCoordinator  # noqa: E402
from disaggregation.remote_cache import DEFER, FetchPlan, FetchSource, Planner  # noqa: E402
from fakes import (  # noqa: E402
    EMPTY_DUMP,
    TPB,
    FakeEngineQueue,
    FakeReader,
    FakeRequest,
    RecordingEffects,
    SingleRankCollective,
    full_attention,
    ordinals_by_group,
)
from store_fakes import (  # noqa: E402
    FakeBlobStore,
    device_rank,
    fill,
    host_rank,
    pattern,
    read,
    wait_until,
    write,
)

pytestmark = pytest.mark.cpu_only

PROMPT = 29
END = (PROMPT - 1) // TPB * TPB  # 28: seven nameable blocks
BLOCKS = END // TPB
UNIT_BYTES = 64


class Side:
    """One rank: a store backend over the shared store, with a coordinator on top of it;
    ``config_overrides`` go to the backend's ``BlobBackendConfig``."""

    def __init__(self, store: FakeBlobStore, *, publishes: bool, **config_overrides) -> None:
        self.rank = device_rank(store, arena_bytes=1 << 16, **config_overrides)
        for o in range(BLOCKS + 2):
            self.rank.resolver.add(0, o, UNIT_BYTES // 2, UNIT_BYTES // 2)  # two segments each
        self.reader = FakeReader(groups=[full_attention(0)], tokens_per_block=TPB)
        self.effects = RecordingEffects()
        self.queue = FakeEngineQueue()
        self.source = FetchSource("store", self.rank.backend, None)
        # Measured on the ``now`` the tests pass to ``advance``: deferred at 0.0 and 1.0, local at 2.0.
        self.planner = Planner([self.source], self.reader, TPB, probe_timeout_s=2.0)
        self.coord = KVTransferCoordinator(
            [self.source],
            [self.rank.backend] if publishes else [],
            self.planner,
            self.reader,
            self.effects,
            self.queue,
            SingleRankCollective(),
            unlaunched_timeout_s=DEFAULT_UNLAUNCHED_TIMEOUT_S,
            fetch_wait_timeout_s=DEFAULT_FETCH_WAIT_TIMEOUT_S,
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
        self.rank.store.unblock()
        self.backend.close()


def _publish(ctx: Side, req: FakeRequest) -> None:
    """Context side: offer the request's blocks and drive the publish record to its end."""
    for o in range(BLOCKS):
        ctx.write_block(o, pattern(o + 1, UNIT_BYTES))
    ctx.coord.publish_committed_blocks([req], now=0.0)
    assert ctx.records()[0]["state"] == "IN_FLIGHT" and ctx.records()[0]["direction"] == "publish"
    wait_until(lambda: len(ctx.rank.store.objects) == BLOCKS, what="publish to land in the store")
    ctx.coord.advance([], 1.0)
    assert ctx.records() == []  # DELIVERED, quiesced, released
    assert ctx.effects.calls == []  # a publish of a running request owes the engine nothing
    assert ctx.backend.counters.publish_stored == BLOCKS


def _probe_and_plan(gen: Side, req: FakeRequest) -> FetchPlan:
    """Generation side: the first ``advance`` queues the probe and defers; once the backend has
    answered, the next one plans from the store."""
    gen.coord.advance([req], 0.0)
    assert gen.coord.fetch_answer(req) is DEFER
    counters = gen.backend.counters
    wait_until(lambda: counters.probe_hits + counters.probe_misses == BLOCKS, what="probe answer")
    gen.coord.advance([req], 1.0)
    plan = gen.coord.fetch_answer(req)
    assert isinstance(plan, FetchPlan), plan
    return plan


def test_publish_on_one_rank_then_probe_plan_launch_and_land_on_another():
    store = FakeBlobStore()
    ctx, gen = Side(store, publishes=True), Side(store, publishes=False)
    try:
        req = FakeRequest(1, prompt_len=PROMPT)
        _publish(ctx, req)

        plan = _probe_and_plan(gen, req)
        assert plan.source == "store" and plan.token_end == END and plan.hint is None
        assert ordinals_by_group(plan) == {0: tuple(range(BLOCKS))}
        gen.fill_all(0xEE)
        gen.coord.launch_reserved_fetches([req], 1.0)
        assert gen.effects.names() == ["prepare_fetch_resources", "park_for_fetch"]
        assert gen.records()[0]["state"] == "IN_FLIGHT"

        gen.advance_until("unpark")
        assert gen.effects.args_of("unpark") == [(req, END, False, None)]
        assert gen.records()[0]["state"] == "DELIVERED"
        for o in range(BLOCKS):
            assert gen.block_bytes(o) == pattern(o + 1, UNIT_BYTES) == ctx.block_bytes(o)
        assert gen.block_bytes(BLOCKS) == bytes([0xEE]) * UNIT_BYTES  # not asked for, not touched
        assert gen.backend.counters.fetch_hits == BLOCKS and gen.backend.counters.fetch_misses == 0
        assert gen.backend.counters.failed_attempts == 0

        # The request's end is the release point: one quiesce (True), record gone, no hold.
        gen.coord.holds_finished_request(req)
        assert gen.records() == [] and gen.effects.count("hold_for_transfer") == 0
        assert gen.coord.status_dump() == EMPTY_DUMP
    finally:
        gen.close()
        ctx.close()


def test_content_gone_between_probe_and_fetch_is_a_short_serve_retried_once_then_local():
    store = FakeBlobStore()
    ctx, gen = Side(store, publishes=True), Side(store, publishes=False)
    try:
        req = FakeRequest(2, prompt_len=PROMPT)
        _publish(ctx, req)
        _probe_and_plan(gen, req)
        store.evict_all()  # the answer was advisory (SPEC §6.2 probe): stale before the fetch
        gen.fill_all(0xEE)
        gen.coord.launch_reserved_fetches([req], 1.0)

        gen.advance_until("revert_fetch_pages")
        # Delivered(∅) is a miss, not a failure: quiesce, give the pages back, keep one retry.
        assert gen.effects.names() == [
            "prepare_fetch_resources",
            "park_for_fetch",
            "revert_fetch_pages",
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
        assert gen.coord.fetch_answer(req) is None
        assert gen.records() == []
        assert gen.effects.count("fail_requests") == 0 and gen.effects.count("unpark") == 0
    finally:
        gen.close()
        ctx.close()


def test_store_outage_during_fetch_is_failed_gives_pages_back_and_the_retry_lands():
    store = FakeBlobStore()
    ctx, gen = Side(store, publishes=True), Side(store, publishes=False)
    try:
        req = FakeRequest(3, prompt_len=PROMPT)
        _publish(ctx, req)
        _probe_and_plan(gen, req)
        store.fail_next("contains")  # the fetch's own lookup, not the probe's
        gen.coord.launch_reserved_fetches([req], 1.0)

        gen.advance_until("revert_fetch_pages")
        rec = gen.records()[0]
        assert rec["state"] == "PLANNED" and rec["outcomes"] == ["Failed"]
        assert gen.backend.counters.failed_attempts == 1
        assert gen.effects.count("fail_requests") == 0  # local fallback exists

        # Re-planned from the cached probe answer, launched again (try 1), and this time lands.
        gen.coord.advance([req], 20.0)
        plan = gen.coord.fetch_answer(req)
        assert isinstance(plan, FetchPlan) and plan.token_end == END
        gen.coord.launch_reserved_fetches([req], 20.0)
        assert gen.records()[0]["try_index"] == 1 and gen.records()[0]["attempts"] == 2
        gen.advance_until("unpark")
        assert gen.effects.args_of("unpark") == [(req, END, False, None)]
        for o in range(BLOCKS):
            assert gen.block_bytes(o) == pattern(o + 1, UNIT_BYTES)
        assert (
            gen.backend.counters.fetch_hits == BLOCKS and gen.backend.counters.failed_attempts == 1
        )
    finally:
        gen.close()
        ctx.close()


def test_probe_outage_defers_within_budget_then_plans_local_without_a_fetch():
    store = FakeBlobStore()
    gen = Side(store, publishes=False)
    try:
        req = FakeRequest(4, prompt_len=PROMPT)
        asked = []

        def unreachable(keys):  # the store is down: every lookup fails
            asked.append(tuple(keys))
            raise BlobStoreError("down")

        store.contains = unreachable
        gen.coord.advance([req], 0.0)
        assert gen.coord.fetch_answer(req) is DEFER
        wait_until(lambda: len(asked) >= 1, what="the probe's lookup")
        # The lookup failed on the backend's thread. Whether the second probe finds the error
        # (and raises, which the coordinator logs and keeps pending) or is still waiting, the
        # answer is never an empty set, so the planner defers rather than reading a miss.
        gen.coord.advance([req], 1.0)
        assert gen.coord.fetch_answer(req) is DEFER
        gen.coord.advance([req], 2.0)  # probe budget spent: compute locally
        assert gen.coord.fetch_answer(req) is None
        assert gen.records() == [] and gen.effects.calls == []
        assert gen.backend.counters.probe_misses == 0 and gen.backend.counters.fetch_misses == 0
        assert gen.backend.counters.probe_hits == 0
    finally:
        gen.close()


PROBE_TTL_S = 0.5
"""Short enough to expire inside a test, long enough that a loaded machine cannot expire an
answer before the first ``wait_until`` has seen it."""


def test_probe_answer_expiring_before_the_planner_reads_it_is_asked_again_not_read_as_a_miss():
    """The backend keeps an unclaimed probe answer for ``probe_ttl_s``; a planner that comes
    back later finds it gone. The lookup is asked again and the request stays deferred: an
    expired answer is never an empty one, so no miss is read into it, and the fetch is planned
    from the store once the fresh answer is in."""
    store = FakeBlobStore()
    ctx, gen = Side(store, publishes=True), Side(store, publishes=False, probe_ttl_s=PROBE_TTL_S)
    try:
        req = FakeRequest(5, prompt_len=PROMPT)
        _publish(ctx, req)
        gen.coord.advance([req], 0.0)
        assert gen.coord.fetch_answer(req) is DEFER
        counters = gen.backend.counters
        wait_until(lambda: counters.probe_hits == BLOCKS, what="the first lookup")
        lookups_before = store.count("contains")
        time.sleep(PROBE_TTL_S + 0.1)  # past the TTL: the answer nobody read is dropped
        gen.coord.advance([req], 1.0)
        assert gen.coord.fetch_answer(req) is DEFER  # asked again, not read as a miss
        wait_until(lambda: counters.probe_hits == 2 * BLOCKS, what="the second lookup")
        assert store.count("contains") == lookups_before + 1  # one fresh lookup, nothing else
        gen.coord.advance([req], 1.5)
        plan = gen.coord.fetch_answer(req)
        assert isinstance(plan, FetchPlan) and plan.token_end == END
        assert counters.probe_misses == 0 and gen.records()[0]["state"] == "PLANNED"
    finally:
        gen.close()
        ctx.close()


# ---------------------------------------------------------------------------------------------
# landing: host
# ---------------------------------------------------------------------------------------------


class HostSide(Side):
    """A rank whose backend is the ``landing: host`` shape: fetches land in the backend's host
    memory and are placed into pages once the scheduler has reserved them."""

    def __init__(self, store: FakeBlobStore, *, publishes: bool) -> None:
        self.rank = host_rank(
            store,
            arena_bytes=1 << 16,
            landing_slots=BLOCKS + 2,
            unit_bytes_of=lambda name: UNIT_BYTES,  # every unit is one block of one group
        )
        for o in range(BLOCKS + 2):
            self.rank.resolver.add(0, o, UNIT_BYTES // 2, UNIT_BYTES // 2)
        self.reader = FakeReader(groups=[full_attention(0)], tokens_per_block=TPB)
        self.effects = RecordingEffects()
        self.queue = FakeEngineQueue()
        self.source = FetchSource("store", self.rank.backend, None)
        self.planner = Planner([self.source], self.reader, TPB, probe_timeout_s=2.0)
        self.coord = KVTransferCoordinator(
            [self.source],
            [self.rank.inner] if publishes else [],
            self.planner,
            self.reader,
            self.effects,
            self.queue,
            SingleRankCollective(),
            unlaunched_timeout_s=DEFAULT_UNLAUNCHED_TIMEOUT_S,
            fetch_wait_timeout_s=DEFAULT_FETCH_WAIT_TIMEOUT_S,
        )


def test_host_landing_rank_lands_first_then_places_after_the_scheduler_reserves():
    """The host-first flow end to end over a real backend: the plan starts the landing at once
    (``LANDING``, no pages), ``fetch_answer`` defers until the content is on the host, then answers
    the plan so the scheduler reserves pages, and ``launch_reserved_fetches`` places instead of fetching.
    The landing is released in the same round the request is unparked."""
    store = FakeBlobStore()
    ctx, gen = Side(store, publishes=True), HostSide(store, publishes=False)
    try:
        req = FakeRequest(1, prompt_len=PROMPT)
        _publish(ctx, req)

        # Probe, then plan: the plan is decided and the landing started in the same advance.
        gen.coord.advance([req], 0.0)
        assert gen.coord.fetch_answer(req) is DEFER
        counters = gen.backend.counters
        wait_until(lambda: counters.probe_hits + counters.probe_misses == BLOCKS, what="probe")
        gen.coord.advance([req], 1.0)
        assert gen.coord.fetch_answer(req) is DEFER  # LANDING: no pages yet
        assert gen.records()[0]["state"] == "LANDING" and gen.records()[0]["has_landing"]
        assert gen.effects.names() == []  # nothing parked, nothing prepared
        gen.fill_all(0xEE)

        # Landed on the host: the record is LANDED and the plan is offered to the scheduler.
        wait_until(lambda: counters.fetch_hits == BLOCKS, what="landing")
        gen.coord.advance([req], 2.0)
        plan = gen.coord.fetch_answer(req)
        assert isinstance(plan, FetchPlan) and plan.token_end == END
        assert gen.records()[0]["state"] == "LANDED"
        assert gen.block_bytes(0) == bytes([0xEE]) * UNIT_BYTES  # pages untouched so far
        assert gen.rank.backend.landings_held() == 1

        # The scheduler reserved: launch places, parks, and the next round unparks + releases.
        gen.coord.launch_reserved_fetches([req], 2.0)
        assert gen.effects.names() == ["prepare_fetch_resources", "park_for_fetch"]
        assert gen.records()[0]["state"] == "IN_FLIGHT"
        gen.advance_until("unpark")
        assert gen.effects.args_of("unpark") == [(req, END, False, None)]
        assert gen.records()[0]["state"] == "DELIVERED"
        assert not gen.records()[0]["has_landing"]
        assert gen.rank.backend.landings_held() == 0
        for o in range(BLOCKS):
            assert gen.block_bytes(o) == pattern(o + 1, UNIT_BYTES) == ctx.block_bytes(o)
        assert gen.block_bytes(BLOCKS) == bytes([0xEE]) * UNIT_BYTES
        assert counters.fetch_hits == BLOCKS and counters.fetch_misses == 0
        assert counters.failed_attempts == 0

        gen.coord.holds_finished_request(req)
        assert gen.records() == [] and gen.effects.count("hold_for_transfer") == 0
    finally:
        gen.close()
        ctx.close()

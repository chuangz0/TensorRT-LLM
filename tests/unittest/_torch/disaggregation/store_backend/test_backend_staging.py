# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The host-staging path: ``HostStagingPool`` with a ``FakeCopier``, and the backend driving it.

The property that distinguishes staging is the design's §10.1 split: a staged publish is *quiet*
(the caller's memory is no longer read) as soon as its units are gathered into host slots, before
the store has taken them -- so ``quiesce`` answers ``True`` while ``poll`` still answers ``None``.
"""

import threading
import time

import pytest

__extra_import_path__ = ["~/tensorrt_llm/_torch"]
from disaggregation.backends.store.staging import HostStagingPool, plan_slot_geometry  # noqa: E402
from disaggregation.base.cache_backend import Delivered, Failed, SubmissionRejected  # noqa: E402
from store_fakes import (  # noqa: E402
    FakeCopier,
    FakeStoreClient,
    MemoryArena,
    extent,
    make_rank,
    make_staging,
    pattern,
    read,
    wait_until,
)

SLOT = 128


def _staged(client=None, *, slots: int = 4, slot_bytes: int = SLOT, **overrides):
    """A staging backend whose pinned buffer is a registered host arena; the caller's pool is
    deliberately *not* registered, as it would not be on a machine without GPUDirect."""
    client = client if client is not None else FakeStoreClient()
    pool, copier, host, trace = make_staging(client, slots=slots, slot_bytes=slot_bytes)
    rank = make_rank(client, stage_through_host=True, staging=pool, **overrides)
    rank.pool, rank.copier, rank.host, rank.trace = pool, copier, host, trace
    return rank


def _slot_of(rank, address: int) -> int:
    return (address - rank.pool.slot_address(0)) // rank.pool.slot_bytes


# ---- plan_slot_geometry ----


@pytest.mark.parametrize(
    "unit, batch, budget, expected",
    [
        (100, 8, 1000, (100, 8)),  # budget covers the batch
        (100, 8, 350, (100, 3)),  # budget bounds the batch
        (100, 8, 50, (100, 1)),  # budget below one unit still yields one slot
        (100, 8, 0, (100, 1)),
        (100, 1, 10**9, (100, 1)),
        (7, 3, 21, (7, 3)),
    ],
)
def test_plan_slot_geometry(unit, batch, budget, expected):
    assert plan_slot_geometry(unit, batch, budget) == expected


@pytest.mark.parametrize("unit, batch", [(0, 8), (-1, 8), (100, 0), (100, -3)])
def test_plan_slot_geometry_rejects_non_positive_inputs(unit, batch):
    with pytest.raises(ValueError):
        plan_slot_geometry(unit, batch, 1000)


# ---- HostStagingPool alone ----


def test_pool_geometry_and_slot_addresses():
    host = MemoryArena(3 * 16)
    pool = HostStagingPool(host.address, 16, 3, FakeCopier())
    assert (pool.slot_bytes, pool.num_slots) == (16, 3)
    assert [pool.slot_address(i) for i in range(3)] == [host.address + 16 * i for i in range(3)]
    for bad in (-1, 3):
        with pytest.raises(IndexError):
            pool.slot_address(bad)
    assert pool.fits(16) and pool.fits(1) and not pool.fits(17) and not pool.fits(0)
    with pytest.raises(ValueError):
        HostStagingPool(host.address, 0, 3, FakeCopier())
    with pytest.raises(ValueError):
        HostStagingPool(host.address, 16, 0, FakeCopier())


def test_pool_acquire_is_all_or_nothing_and_release_wakes_waiters():
    host = MemoryArena(3 * 16)
    pool = HostStagingPool(host.address, 16, 3, FakeCopier())
    with pytest.raises(ValueError):
        pool.acquire(0)
    with pytest.raises(ValueError):
        pool.acquire(4)
    taken = pool.acquire(2)
    assert sorted(taken) == [0, 1]
    got = []
    t = threading.Thread(target=lambda: got.append(pool.acquire(2)))
    t.start()
    time.sleep(0.05)
    assert t.is_alive() and got == []  # one free slot is not two
    pool.release(taken[:1])
    t.join(5)
    assert not t.is_alive()
    assert sorted(got[0]) == [0, 2]


def test_gather_and_scatter_concatenate_in_resolver_order():
    host = MemoryArena(2 * 32)
    copier = FakeCopier()
    pool = HostStagingPool(host.address, 32, 2, copier)
    src = MemoryArena(64)
    s1, s2 = src.carve(12), src.carve(20)
    import ctypes

    ctypes.memmove(s1[0], pattern(1, 12), 12)
    ctypes.memmove(s2[0], pattern(2, 20), 20)
    assert pool.gather(1, [s1, s2]) == 32
    assert read([(pool.slot_address(1), 32)]) == pattern(1, 12) + pattern(2, 20)
    assert copier.kinds() == ["d2h", "d2h"]
    dst = MemoryArena(64)
    d1, d2 = dst.carve(12), dst.carve(20)
    pool.scatter(1, [d1, d2])
    assert read([d1]) == pattern(1, 12) and read([d2]) == pattern(2, 20)
    assert copier.kinds() == ["d2h", "d2h", "h2d", "h2d"]
    pool.sync()
    assert copier.syncs == 1
    with pytest.raises(ValueError, match="exceeds"):
        pool.gather(0, [s1, s2, (s1[0], 1)])


# ---- backend over the pool ----


def test_staged_publish_is_quiet_before_the_store_takes_it_then_delivered():
    with _staged() as rank:
        a, b = rank.unit(0, 0, 64), rank.unit(0, 1, 40, 24)
        rank.write(a, pattern(1, 64))
        rank.write(b, pattern(2, 64))
        rank.client.block("batch_put_from_multi_buffers")
        attempt = rank.backend.publish(extent([a, b]))
        rank.client.wait_entered(3)  # register (host), is_exist, put
        # §10.1: gathered into slots and synced, so the caller's memory is out of the picture.
        assert rank.backend.quiesce([attempt]) is True
        assert attempt.poll() is None
        assert rank.copier.kinds() == ["d2h", "d2h", "d2h"] and rank.copier.syncs == 1
        # Overwriting the source now must not change what the store receives.
        rank.fill(a, 0x00)
        rank.fill(b, 0x00)
        rank.client.unblock()
        outcome = rank.finish(attempt)
        assert outcome == Delivered(frozenset({a.name, b.name}))
        assert rank.client.objects[rank.key(a)] == pattern(1, 64)
        assert rank.client.objects[rank.key(b)] == pattern(2, 64)
        # The store was handed host memory (the fake refuses unregistered buffers, and the
        # caller's pool is not registered), and every slot is free again afterwards.
        assert rank.client.count("batch_put_from_multi_buffers") == 1
        assert sorted(rank.pool.acquire(rank.pool.num_slots)) == list(range(rank.pool.num_slots))


def test_staged_fetch_round_trips_bytes_through_slots():
    client = FakeStoreClient()
    with make_rank(client) as direct, _staged(client) as staged:
        # Publish directly from registered memory; fetch through host staging.
        ua = direct.unit(0, 0, 40, 24)
        direct.write(ua, pattern(7, 64))
        assert isinstance(direct.finish(direct.backend.publish(extent([ua]))), Delivered)
        ub, missing = staged.unit(0, 0, 40, 24), staged.unit(0, 1, 16)
        staged.fill(ub, 0xEE)
        staged.fill(missing, 0xEE)
        outcome = staged.finish(staged.backend.fetch(extent([ub, missing])))
        assert outcome == Delivered(frozenset({ub.name}))
        assert staged.read(ub) == pattern(7, 64)
        assert staged.read(missing) == bytes([0xEE]) * 16
        assert staged.copier.kinds() == ["h2d", "h2d"] and staged.copier.syncs == 1
        # And a staged publish is readable by a direct fetch: same byte string either way.
        uc = staged.unit(1, 0, 8, 8)
        staged.write(uc, pattern(9, 16))
        assert isinstance(staged.finish(staged.backend.publish(extent([uc]))), Delivered)
        ud = direct.unit(1, 0, 16)
        assert isinstance(direct.finish(direct.backend.fetch(extent([ud]))), Delivered)
        assert direct.read(ud) == pattern(9, 16)


def test_slot_exhaustion_waits_then_proceeds():
    with _staged(slots=1) as rank:
        a, b = rank.unit(0, 0, 32), rank.unit(0, 1, 32)
        rank.write(a, pattern(1, 32))
        rank.write(b, pattern(2, 32))
        held = rank.pool.acquire(1)  # someone else has the only slot
        attempt = rank.backend.publish(extent([a, b]))
        # One slot bounds the batch to one unit, so the lookup is two calls; then the worker
        # parks in ``acquire`` until the slot comes back. (``close`` would wake it with a failed
        # delivery instead; see ``test_close_wakes_a_worker_parked_for_a_slot``.)
        wait_until(lambda: rank.client.count("batch_is_exist") == 2)
        time.sleep(0.05)
        assert attempt.poll() is None and rank.copier.copies == []
        rank.pool.release(held)
        outcome = rank.finish(attempt)
        assert outcome == Delivered(frozenset({a.name, b.name}))
        # Two rounds of one slot: gather, put, gather, put.
        assert rank.copier.kinds() == ["d2h", "d2h"]
        puts = [args[0] for m, args in rank.client.calls if m == "batch_put_from_multi_buffers"]
        assert [len(k) for k in puts] == [1, 1]
        assert rank.client.objects[rank.key(b)] == pattern(2, 32)


def test_staged_publish_larger_than_the_pool_is_quiet_only_after_its_last_round():
    with _staged(slots=2) as rank:
        units = [rank.unit(0, i, 16) for i in range(3)]
        rank.client.block("batch_put_from_multi_buffers")
        attempt = rank.backend.publish(extent(units))
        rank.client.wait_entered(3)  # first round's put is blocked; a third unit is still unread
        quiet = []
        t = threading.Thread(target=lambda: quiet.append(rank.backend.quiesce([attempt])))
        t.start()
        time.sleep(0.05)
        assert t.is_alive()
        rank.client.unblock()
        t.join(5)
        assert quiet == [True]
        assert isinstance(rank.finish(attempt), Delivered)


def test_copier_exception_fails_the_delivery_and_frees_the_slots():
    with _staged(slots=2) as rank:
        a = rank.unit(0, 0, 32)
        rank.copier.fail_next("copy", RuntimeError("cudaMemcpyAsync failed"))
        outcome = rank.finish(rank.backend.publish(extent([a])))
        assert isinstance(outcome, Failed) and "cudaMemcpyAsync" in outcome.reason
        assert rank.client.count("batch_put_from_multi_buffers") == 0
        assert rank.backend.counters.failed_attempts == 1
        assert sorted(rank.pool.acquire(2)) == [0, 1]  # nothing leaked
        rank.pool.release([0, 1])
        # On the fetch side too: a scatter that fails is Failed, not a short serve.
        b = rank.unit(0, 1, 32)
        rank.client.objects[rank.key(b)] = pattern(3, 32)
        rank.copier.fail_next("copy")
        outcome = rank.finish(rank.backend.fetch(extent([b])))
        assert isinstance(outcome, Failed)
        assert sorted(rank.pool.acquire(2)) == [0, 1]


def test_unit_larger_than_a_slot_fails_before_anything_moves():
    with _staged(slot_bytes=32) as rank:
        big = rank.unit(0, 0, 33)
        outcome = rank.backend.publish(extent([big])).poll()
        assert isinstance(outcome, Failed) and "exceeds the staging slot" in outcome.reason
        assert rank.client.count("batch_is_exist") == 0 and rank.copier.copies == []


def test_staging_does_not_require_the_callers_pool_to_be_registered():
    with _staged() as rank:
        assert rank.registration is None
        assert rank.client.count("register_buffer") == 1  # the host buffer only
        u = rank.unit(0, 0, 8)
        rank.write(u, pattern(1, 8))
        assert isinstance(rank.finish(rank.backend.publish(extent([u]))), Delivered)
        assert rank.client.objects[rank.key(u)] == pattern(1, 8)


# ---- copier failure part-way through a unit ----


def _sync_precedes_release(rank, thread: int) -> None:
    """On ``thread``, the last ``sync`` happened before the last ``release`` (copies drained
    before the slots go back, so nobody reuses a slot a copy may still be landing in)."""
    events = [e for e, _ in rank.trace.by_thread(thread)]
    assert "sync" in events and "release" in events, events
    last_sync = max(i for i, e in enumerate(events) if e == "sync")
    last_release = max(i for i, e in enumerate(events) if e == "release")
    assert last_sync < last_release, events


@pytest.mark.parametrize("direction", ["publish", "fetch"])
def test_copier_failing_on_the_second_segment_syncs_before_releasing_the_slot(direction):
    with _staged(slots=2) as rank:
        two_seg = rank.unit(0, 0, 32, 32)
        rank.write(two_seg, pattern(1, 64))
        if direction == "fetch":
            rank.client.objects[rank.key(two_seg)] = pattern(1, 64)
            rank.fill(two_seg, 0xEE)
        rank.copier.fail_at("copy", 2, RuntimeError("cudaMemcpyAsync failed on segment 2"))
        attempt = getattr(rank.backend, direction)(extent([two_seg]))
        quiet = []
        t = threading.Thread(target=lambda: quiet.append(rank.backend.quiesce([attempt])))
        t.start()
        outcome = rank.finish(attempt)
        t.join(5)
        assert isinstance(outcome, Failed) and "segment 2" in outcome.reason
        assert quiet == [True]
        # The failing worker synced, then released; the first segment's copy was issued, the
        # second never was.
        copies = [(e, a) for e, _, a in rank.trace.events if e == "copy"]
        assert len(copies) == 2  # the second call is recorded before it raises
        worker = next(t_ for e, t_, _ in rank.trace.events if e == "copy")
        _sync_precedes_release(rank, worker)
        assert rank.copier.syncs >= 1
        if direction == "publish":
            assert rank.client.count("batch_put_from_multi_buffers") == 0
        else:
            # SPEC §5.2 inv. 3: after Failed the destination is undefined. Here the first segment
            # landed and the second never did, which is exactly what the caller must not trust.
            assert rank.read(two_seg) == pattern(1, 64)[:32] + bytes([0xEE]) * 32
        # Another delivery through the same slots afterwards sees nothing of the failed one.
        fresh = rank.unit(0, 1, 64)
        rank.write(fresh, pattern(7, 64))
        assert isinstance(rank.finish(rank.backend.publish(extent([fresh]))), Delivered)
        assert rank.client.objects[rank.key(fresh)] == pattern(7, 64)
        rank.trace.check_slot_exclusivity(rank.pool)


# ---- interleaved staged traffic ----


def test_interleaved_staged_publish_and_fetch_keep_slots_exclusive_and_bytes_exact():
    client = FakeStoreClient()
    with (
        make_rank(client) as direct,
        _staged(client, slots=2, transfer_batch_size=8, num_workers=2) as staged,
    ):
        # Six units to publish through staging (three rounds of two) ...
        outgoing = [staged.unit(0, i, 40, 24) for i in range(6)]
        for i, u in enumerate(outgoing):
            staged.write(u, pattern(10 + i, 64))
        # ... and six others, published directly, to fetch through staging at the same time.
        sources = [direct.unit(1, i, 64) for i in range(6)]
        for i, u in enumerate(sources):
            direct.write(u, pattern(20 + i, 64))
        assert isinstance(direct.finish(direct.backend.publish(extent(sources))), Delivered)
        incoming = [staged.unit(1, i, 32, 32) for i in range(6)]
        for u in incoming:
            staged.fill(u, 0xEE)

        # Force the overlap: the publish parks at its first put holding both slots, the fetch
        # is submitted meanwhile and parks in ``acquire``; from the unblock on they interleave.
        client.block("batch_put_from_multi_buffers")
        pub = staged.backend.publish(extent(outgoing, name=b"out"))
        wait_until(lambda: client.count("batch_put_from_multi_buffers") == 2, what="first put")
        lookups = client.count("batch_is_exist")
        fetch = staged.backend.fetch(extent(incoming, name=b"in"))
        wait_until(lambda: client.count("batch_is_exist") > lookups, what="fetch lookup")
        time.sleep(0.02)
        assert fetch.poll() is None and pub.poll() is None
        client.unblock()
        assert staged.finish(pub) == Delivered(frozenset(u.name for u in outgoing))
        assert staged.finish(fetch) == Delivered(frozenset(u.name for u in incoming))
        for i, u in enumerate(outgoing):
            assert client.objects[staged.key(u)] == pattern(10 + i, 64)
        for i, u in enumerate(incoming):
            assert staged.read(u) == pattern(20 + i, 64) == direct.read(sources[i])
        staged.trace.check_slot_exclusivity(staged.pool)
        threads = {t for e, t, _ in staged.trace.events if e == "copy"}
        assert len(threads) == 2  # both workers took part
        acquires = [a[0] for e, _, a in staged.trace.events if e == "acquire"]
        assert all(len(s) <= 2 for s in acquires) and len(acquires) >= 6


# ---- close while parked or racing submissions ----


def test_close_wakes_a_worker_parked_for_a_slot_and_fails_its_delivery():
    rank = _staged(slots=1)
    held = rank.pool.acquire(1)
    u = rank.unit(0, 0, 32)
    attempt = rank.backend.publish(extent([u]))
    wait_until(lambda: rank.client.count("batch_is_exist") == 1)
    time.sleep(0.02)  # let the worker reach acquire
    start = time.monotonic()
    rank.backend.close()
    assert time.monotonic() - start < 5
    outcome = attempt.poll()
    assert isinstance(outcome, Failed) and "shut down" in outcome.reason
    assert rank.backend.quiesce([attempt]) is True
    assert rank.client.count("batch_put_from_multi_buffers") == 0
    rank.pool.release(held)
    with pytest.raises(RuntimeError):
        rank.pool.acquire(1)  # the pool stays shut


def test_submissions_racing_close_are_rejected_or_reach_an_outcome():
    rank = _staged(slots=2, max_inflight_ops=64, num_workers=2)
    units = [rank.unit(0, i, 16) for i in range(40)]
    attempts, rejected = [], []
    go = threading.Event()

    def submitter(mine):
        go.wait()
        for u in mine:
            try:
                attempts.append(rank.backend.publish(extent([u])))
            except SubmissionRejected:
                rejected.append(u)

    threads = [threading.Thread(target=submitter, args=(units[i::4],)) for i in range(4)]
    for t in threads:
        t.start()
    go.set()
    time.sleep(0.005)
    rank.backend.close()
    for t in threads:
        t.join(5)
        assert not t.is_alive()
    assert len(attempts) + len(rejected) == len(units)
    deadline = time.monotonic() + 10
    for attempt in attempts:
        rank.backend.settle([attempt])
        assert attempt.poll() is not None
        assert time.monotonic() < deadline
    assert rank.backend.quiesce(attempts) is True
    # Every attempt that was Delivered really stored its unit; nothing is half-way.
    for attempt in attempts:
        outcome = attempt.poll()
        assert isinstance(outcome, (Delivered, Failed))
    rank.trace.check_slot_exclusivity(rank.pool)


# ---- the store declining or failing a put once the units are already in host slots ----


def test_staged_declined_put_for_a_unit_another_publisher_made_present_counts_as_taken():
    """Staged variant of the publish race: between our lookup and our put another publisher
    stores ``first``; the store declines our write of it. The unit is held under its name, so it
    is served, the slots go back, and the attempt is quiet."""
    client = FakeStoreClient()
    with make_rank(client) as other, _staged(client, slots=4) as staged:
        first, second = staged.unit(0, 0, 32), staged.unit(0, 1, 32)
        staged.write(first, pattern(1, 32))
        staged.write(second, pattern(2, 32))
        theirs = other.unit(0, 0, 32)  # the same name from another rank's memory
        other.write(theirs, pattern(1, 32))
        orig = client.batch_put_from_multi_buffers

        def raced_put(keys, ptrs, sizes):
            client.batch_put_from_multi_buffers = orig  # one-shot: ``other`` needs the real one
            assert isinstance(other.finish(other.backend.publish(extent([theirs]))), Delivered)
            results = list(orig(keys, ptrs, sizes))
            results[keys.index(staged.key(first))] = -1
            return results

        client.batch_put_from_multi_buffers = raced_put
        attempt = staged.backend.publish(extent([first, second]))
        outcome = staged.finish(attempt)
        assert outcome == Delivered(frozenset({first.name, second.name}))
        assert staged.backend.counters.publish_stored == 1
        assert staged.backend.counters.publish_raced == 1
        assert staged.backend.counters.failed_attempts == 0
        assert client.objects[staged.key(first)] == pattern(1, 32)
        assert client.objects[staged.key(second)] == pattern(2, 32)
        assert staged.backend.quiesce([attempt]) is True
        every_slot = list(range(staged.pool.num_slots))
        assert sorted(staged.pool.acquire(staged.pool.num_slots)) == every_slot  # all came back
        staged.pool.release(every_slot)
        staged.trace.check_slot_exclusivity(staged.pool)


def test_staged_put_raising_after_the_gather_fails_frees_the_slots_and_is_quiet():
    """The units were copied into host slots (the caller's memory is done with) when the store
    call raises: the delivery fails, nothing is stored, the slots are released, and ``quiesce``
    answers True because the gather is what read the caller's memory."""
    with _staged(slots=2) as rank:
        a, b = rank.unit(0, 0, 32), rank.unit(0, 1, 32)
        rank.write(a, pattern(1, 32))
        rank.write(b, pattern(2, 32))
        rank.client.fail_next("batch_put_from_multi_buffers", RuntimeError("store unreachable"))
        attempt = rank.backend.publish(extent([a, b]))
        outcome = rank.finish(attempt)
        assert isinstance(outcome, Failed) and "store unreachable" in outcome.reason
        assert rank.client.objects == {}
        assert rank.copier.kinds() == ["d2h", "d2h"]  # gathered before the put
        assert rank.backend.counters.failed_attempts == 1
        assert rank.backend.counters.publish_stored == 0
        assert rank.backend.quiesce([attempt]) is True
        assert sorted(rank.pool.acquire(2)) == [0, 1]
        rank.pool.release([0, 1])
        rank.trace.check_slot_exclusivity(rank.pool)

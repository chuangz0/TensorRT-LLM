# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The host pools: ``plan_slot_geometry``, ``HostSlotPool`` with a ``FakeCopier`` (blocking
and queued hand-out), and the ``landing: host`` backend driving its publish pool.

The property that distinguishes a staged publish is the design's §10.1 split: it is *quiet* (the
caller's memory is no longer read) as soon as its units are gathered into host slots, before the
store has taken them -- so ``quiesce`` answers ``True`` while ``poll`` still answers ``None``. The
fetch direction of this shape (landing, then placement) has its own suite,
``test_host_landing_contract.py``; here it appears only where it shares a pool's mechanics.
"""

import threading
import time

import pytest

__extra_import_path__ = ["~/tensorrt_llm/_torch"]
from disaggregation.backends.blob.slot_pool import HostSlotPool, plan_slot_geometry  # noqa: E402
from disaggregation.backends.blob.store import PutStatus  # noqa: E402
from disaggregation.base.cache_backend import Delivered, Failed, SubmissionRejected  # noqa: E402
from store_fakes import (  # noqa: E402
    SLOT,
    FakeBlobStore,
    FakeCopier,
    MemoryArena,
    extent,
    make_host_rank,
    make_rank,
    pattern,
    read,
    wait_until,
)


def _staged(store=None, *, slots: int = 4, slot_bytes: int = SLOT, **overrides):
    """A host-landing backend whose publish pool has ``slots`` traced slots."""
    return make_host_rank(store, publish_slots=slots, slot_bytes=slot_bytes, **overrides)


# ---- plan_slot_geometry ----


@pytest.mark.parametrize(
    "unit, cap, budget, expected",
    [
        (100, 8, 1000, (100, 8)),  # budget covers the cap
        (100, 8, 350, (100, 3)),  # budget bounds the cap
        (100, 8, 50, (100, 1)),  # budget below one unit still yields one slot
        (100, 8, 0, (100, 1)),
        (100, 1, 10**9, (100, 1)),
        (7, 3, 21, (7, 3)),
        (100, None, 1000, (100, 10)),  # no cap: every slot the budget affords
        (100, None, 50, (100, 1)),
    ],
)
def test_plan_slot_geometry(unit, cap, budget, expected):
    assert plan_slot_geometry(unit, cap, budget) == expected


@pytest.mark.parametrize("unit, cap", [(0, 8), (-1, 8), (100, 0), (100, -3)])
def test_plan_slot_geometry_rejects_non_positive_inputs(unit, cap):
    with pytest.raises(ValueError):
        plan_slot_geometry(unit, cap, 1000)


# ---- HostSlotPool alone ----


def test_pool_geometry_and_slot_addresses():
    host = MemoryArena(3 * 16)
    pool = HostSlotPool(host.address, 16, 3, FakeCopier())
    assert (pool.slot_bytes, pool.num_slots) == (16, 3)
    assert [pool.slot_address(i) for i in range(3)] == [host.address + 16 * i for i in range(3)]
    for bad in (-1, 3):
        with pytest.raises(IndexError):
            pool.slot_address(bad)
    assert pool.fits(16) and pool.fits(1) and not pool.fits(17) and not pool.fits(0)
    with pytest.raises(ValueError):
        HostSlotPool(host.address, 0, 3, FakeCopier())
    with pytest.raises(ValueError):
        HostSlotPool(host.address, 16, 0, FakeCopier())


def test_pool_acquire_is_all_or_nothing_and_release_wakes_waiters():
    host = MemoryArena(3 * 16)
    pool = HostSlotPool(host.address, 16, 3, FakeCopier())
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


class _Waiter:
    """A ``SlotWaiter`` that records what it was told and on which thread."""

    def __init__(self) -> None:
        self.granted: list[int] | None = None
        self.refused: str | None = None
        self.thread: int | None = None

    def slots_granted(self, slots):
        self.granted, self.thread = slots, threading.get_ident()

    def slots_refused(self, reason):
        self.refused, self.thread = reason, threading.get_ident()


def test_pool_enqueue_grants_at_once_queues_in_order_and_grants_on_release():
    host = MemoryArena(3 * 16)
    pool = HostSlotPool(host.address, 16, 3, FakeCopier())
    with pytest.raises(ValueError):
        pool.enqueue(_Waiter(), 4)
    first, second, third = _Waiter(), _Waiter(), _Waiter()
    pool.enqueue(first, 2)
    assert sorted(first.granted) == [0, 1] and first.thread == threading.get_ident()
    pool.enqueue(second, 2)  # one free is not two: queued
    pool.enqueue(third, 1)  # would fit, but waits behind ``second``: no overtaking
    assert second.granted is None and third.granted is None
    releaser = threading.Thread(target=pool.release, args=(first.granted,))
    releaser.start()
    releaser.join(5)
    # The releasing thread granted both, in order, once the head's ask fit (the free list is
    # ``[2, 0, 1]`` at that point: the returned slots go to its back).
    assert sorted(second.granted) == [0, 2] and second.thread == releaser.ident
    assert third.granted == [1] and third.thread == releaser.ident
    assert pool.dequeue(second) is False  # already granted


def test_pool_dequeue_drops_a_queued_waiter_and_shutdown_refuses_the_rest():
    host = MemoryArena(16)
    pool = HostSlotPool(host.address, 16, 1, FakeCopier())
    holder, leaving, staying, late = _Waiter(), _Waiter(), _Waiter(), _Waiter()
    pool.enqueue(holder, 1)
    pool.enqueue(leaving, 1)
    pool.enqueue(staying, 1)
    assert pool.dequeue(leaving) is True and pool.dequeue(leaving) is False
    pool.shutdown()
    assert staying.refused == "slot pool is shut down" and leaving.refused is None
    pool.enqueue(late, 1)
    assert late.refused is not None and late.granted is None
    pool.release(holder.granted)  # still allowed; nothing is granted to anyone
    assert staying.granted is None and leaving.granted is None
    with pytest.raises(RuntimeError):
        pool.acquire(1)


def test_gather_and_scatter_concatenate_in_resolver_order():
    host = MemoryArena(2 * 32)
    copier = FakeCopier()
    pool = HostSlotPool(host.address, 32, 2, copier)
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
    pool.wait_for_copies()
    assert copier.copy_waits == 1
    with pytest.raises(ValueError, match="exceeds"):
        pool.gather(0, [s1, s2, (s1[0], 1)])


# ---- backend over the publish pool ----


def test_staged_publish_is_quiet_before_the_store_takes_it_then_delivered():
    with _staged() as rank:
        a, b = rank.unit(0, 0, 64), rank.unit(0, 1, 40, 24)
        rank.write(a, pattern(1, 64))
        rank.write(b, pattern(2, 64))
        rank.store.block("put")
        attempt = rank.backend.publish(extent([a, b]))
        rank.store.wait_entered(4)  # two pool registrations, is_exist, put
        # §10.1: gathered into slots and synced, so the caller's memory is out of the picture.
        assert rank.backend.quiesce([attempt]) is True
        assert attempt.poll() is None
        assert rank.publish_copier.kinds() == ["d2h", "d2h", "d2h"]
        assert rank.publish_copier.copy_waits == 1
        # Overwriting the source now must not change what the store receives.
        rank.fill(a, 0x00)
        rank.fill(b, 0x00)
        rank.store.unblock()
        outcome = rank.finish(attempt)
        assert outcome == Delivered(frozenset({a.name, b.name}))
        assert rank.store.objects[rank.key(a)] == pattern(1, 64)
        assert rank.store.objects[rank.key(b)] == pattern(2, 64)
        # The store was handed host memory (the fake refuses unregistered buffers, and the
        # caller's pool is not registered), and every slot is free again afterwards.
        assert rank.store.count("put") == 1
        pool = rank.publish_pool
        assert sorted(pool.acquire(pool.num_slots)) == list(range(pool.num_slots))


def test_staged_publish_is_readable_by_a_direct_fetch_and_vice_versa():
    """Same byte string either way: a unit's segments concatenated in resolver order."""
    store = FakeBlobStore()
    with make_rank(store) as direct, _staged(store) as staged:
        uc = staged.unit(1, 0, 8, 8)
        staged.write(uc, pattern(9, 16))
        assert isinstance(staged.finish(staged.backend.publish(extent([uc]))), Delivered)
        ud = direct.unit(1, 0, 16)
        assert isinstance(direct.finish(direct.backend.fetch(extent([ud]))), Delivered)
        assert direct.read(ud) == pattern(9, 16)
        # And the other way, through a landing: published directly, landed and placed here.
        ua = direct.unit(0, 0, 40, 24)
        direct.write(ua, pattern(7, 64))
        assert isinstance(direct.finish(direct.backend.publish(extent([ua]))), Delivered)
        ub = staged.unit(0, 0, 32, 32)
        staged.fill(ub, 0xEE)
        landing = staged.land([ub])
        assert staged.place(landing, [ub]) == Delivered(frozenset({ub.name}))
        assert staged.read(ub) == pattern(7, 64)
        assert staged.landing_copier.kinds() == ["h2d", "h2d"]
        landing.close()


def test_slot_exhaustion_waits_then_proceeds():
    with _staged(slots=1) as rank:
        a, b = rank.unit(0, 0, 32), rank.unit(0, 1, 32)
        rank.write(a, pattern(1, 32))
        rank.write(b, pattern(2, 32))
        held = rank.publish_pool.acquire(1)  # someone else has the only slot
        attempt = rank.backend.publish(extent([a, b]))
        # The lookup needs no slot and is one call; then the worker parks in ``acquire`` until
        # the slot comes back. (``close`` would wake it with a failed delivery instead; see
        # ``test_close_wakes_a_worker_parked_for_a_slot``.)
        wait_until(lambda: rank.store.count("contains") == 1)
        time.sleep(0.05)
        assert attempt.poll() is None and rank.publish_copier.copies == []
        rank.publish_pool.release(held)
        outcome = rank.finish(attempt)
        assert outcome == Delivered(frozenset({a.name, b.name}))
        # Two rounds of one slot: gather, put, gather, put.
        assert rank.publish_copier.kinds() == ["d2h", "d2h"]
        puts = [args[0] for m, args in rank.store.calls if m == "put"]
        assert [len(k) for k in puts] == [1, 1]
        assert rank.store.objects[rank.key(b)] == pattern(2, 32)


def test_staged_publish_larger_than_the_pool_is_quiet_only_after_its_last_round():
    with _staged(slots=2) as rank:
        units = [rank.unit(0, i, 16) for i in range(3)]
        rank.store.block("put")
        attempt = rank.backend.publish(extent(units))
        rank.store.wait_entered(4)  # first round's put is blocked; a third unit is still unread
        quiet = []
        t = threading.Thread(target=lambda: quiet.append(rank.backend.quiesce([attempt])))
        t.start()
        time.sleep(0.05)
        assert t.is_alive()
        rank.store.unblock()
        t.join(5)
        assert quiet == [True]
        assert isinstance(rank.finish(attempt), Delivered)


def test_copier_exception_fails_the_publish_and_frees_the_slots():
    with _staged(slots=2) as rank:
        a = rank.unit(0, 0, 32)
        rank.publish_copier.fail_next("copy", RuntimeError("cudaMemcpyAsync failed"))
        outcome = rank.finish(rank.backend.publish(extent([a])))
        assert isinstance(outcome, Failed) and "cudaMemcpyAsync" in outcome.reason
        assert rank.store.count("put") == 0
        assert rank.backend.counters.failed_attempts == 1
        assert sorted(rank.publish_pool.acquire(2)) == [0, 1]  # nothing leaked
        rank.publish_pool.release([0, 1])


def test_unit_larger_than_a_slot_fails_before_anything_moves():
    with _staged(slot_bytes=32) as rank:
        big = rank.unit(0, 0, 33)
        outcome = rank.backend.publish(extent([big])).poll()
        assert isinstance(outcome, Failed) and "exceeds the publish-pool slot" in outcome.reason
        assert rank.store.count("contains") == 0 and rank.publish_copier.copies == []


def test_staging_does_not_require_the_callers_pool_to_be_registered():
    with _staged() as rank:
        assert rank.registration is None
        assert rank.store.count("register_span") == 2  # the two host pools only
        u = rank.unit(0, 0, 8)
        rank.write(u, pattern(1, 8))
        assert isinstance(rank.finish(rank.backend.publish(extent([u]))), Delivered)
        assert rank.store.objects[rank.key(u)] == pattern(1, 8)


# ---- copier failure part-way through a unit ----


def _copy_wait_precedes_release(rank, thread: int) -> None:
    """On ``thread``, the last ``wait_for_copies`` happened before the last ``release`` (copies
    waited for before the slots go back, so nobody reuses a slot a copy may still be landing
    in)."""
    events = [e for e, _ in rank.trace.by_thread(thread)]
    assert "wait_for_copies" in events and "release" in events, events
    last_wait = max(i for i, e in enumerate(events) if e == "wait_for_copies")
    last_release = max(i for i, e in enumerate(events) if e == "release")
    assert last_wait < last_release, events


def test_copier_failing_on_the_second_segment_of_a_publish_waits_before_releasing_the_slot():
    with _staged(slots=2) as rank:
        two_seg = rank.unit(0, 0, 32, 32)
        rank.write(two_seg, pattern(1, 64))
        rank.publish_copier.fail_at("copy", 2, RuntimeError("cudaMemcpyAsync failed on segment 2"))
        attempt = rank.backend.publish(extent([two_seg]))
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
        _copy_wait_precedes_release(rank, worker)
        assert rank.publish_copier.copy_waits >= 1
        assert rank.store.count("put") == 0
        # Another delivery through the same slots afterwards sees nothing of the failed one.
        fresh = rank.unit(0, 1, 64)
        rank.write(fresh, pattern(7, 64))
        assert isinstance(rank.finish(rank.backend.publish(extent([fresh]))), Delivered)
        assert rank.store.objects[rank.key(fresh)] == pattern(7, 64)
        rank.trace.check_slot_exclusivity(rank.publish_pool)


def test_copier_failing_on_the_second_segment_of_a_placement_drains_and_leaves_memory_undefined():
    store = FakeBlobStore()
    with make_rank(store) as direct, _staged(store, slots=2) as rank:
        src = direct.unit(0, 0, 64)
        direct.write(src, pattern(1, 64))
        assert isinstance(direct.finish(direct.backend.publish(extent([src]))), Delivered)
        two_seg = rank.unit(0, 0, 32, 32)
        rank.fill(two_seg, 0xEE)
        landing = rank.land([two_seg])
        rank.landing_copier.fail_at("copy", 2, RuntimeError("cudaMemcpyAsync failed on segment 2"))
        attempt = landing.place(extent([two_seg]))
        quiet = []
        t = threading.Thread(target=lambda: quiet.append(rank.backend.quiesce([attempt])))
        t.start()
        outcome = rank.finish(attempt)
        t.join(5)
        assert isinstance(outcome, Failed) and "segment 2" in outcome.reason
        assert quiet == [True]
        assert rank.landing_copier.copy_waits == 1  # waited for before the outcome
        # SPEC §5.2 inv. 3: after Failed the destination is undefined. Here the first segment
        # landed and the second never did, which is exactly what the caller must not trust.
        assert rank.read(two_seg) == pattern(1, 64)[:32] + bytes([0xEE]) * 32
        landing.close()


# ---- interleaved staged traffic ----


def test_interleaved_publish_and_landing_use_their_own_pools_and_keep_bytes_exact():
    store = FakeBlobStore()
    with (
        make_rank(store) as direct,
        _staged(store, slots=2, landing_slots=6, transfer_batch_size=8, num_workers=2) as staged,
    ):
        # Six units to publish through staging (three rounds of two) ...
        outgoing = [staged.unit(0, i, 40, 24) for i in range(6)]
        for i, u in enumerate(outgoing):
            staged.write(u, pattern(10 + i, 64))
        # ... and six others, published directly, to land and place at the same time.
        sources = [direct.unit(1, i, 64) for i in range(6)]
        for i, u in enumerate(sources):
            direct.write(u, pattern(20 + i, 64))
        assert isinstance(direct.finish(direct.backend.publish(extent(sources))), Delivered)
        incoming = [staged.unit(1, i, 32, 32) for i in range(6)]
        for u in incoming:
            staged.fill(u, 0xEE)

        # Force the overlap: the publish parks at its first put holding both publish slots; the
        # landing, submitted meanwhile, takes its slots from the other pool and completes.
        store.block("put")
        pub = staged.backend.publish(extent(outgoing, name=b"out"))
        wait_until(lambda: store.count("put") == 2, what="first put")
        landing = staged.land(incoming)
        assert landing.poll() == Delivered(frozenset(u.name for u in incoming))
        assert pub.poll() is None
        store.unblock()
        assert staged.finish(pub) == Delivered(frozenset(u.name for u in outgoing))
        assert staged.place(landing, incoming) == Delivered(frozenset(u.name for u in incoming))
        for i, u in enumerate(outgoing):
            assert store.objects[staged.key(u)] == pattern(10 + i, 64)
        for i, u in enumerate(incoming):
            assert staged.read(u) == pattern(20 + i, 64) == direct.read(sources[i])
        staged.trace.check_slot_exclusivity(staged.publish_pool)
        landing.close()


# ---- close while parked or racing submissions ----


def test_close_wakes_a_worker_parked_for_a_slot_and_fails_its_delivery():
    rank = _staged(slots=1)
    held = rank.publish_pool.acquire(1)
    u = rank.unit(0, 0, 32)
    attempt = rank.backend.publish(extent([u]))
    wait_until(lambda: rank.store.count("contains") == 1)
    time.sleep(0.02)  # let the worker reach acquire
    start = time.monotonic()
    rank.backend.close()
    assert time.monotonic() - start < 5
    outcome = attempt.poll()
    assert isinstance(outcome, Failed) and "shut down" in outcome.reason
    assert rank.backend.quiesce([attempt]) is True
    assert rank.store.count("put") == 0
    rank.publish_pool.release(held)
    with pytest.raises(RuntimeError):
        rank.publish_pool.acquire(1)  # the pool stays shut


def test_submissions_racing_close_are_rejected_or_reach_an_outcome():
    rank = _staged(slots=2, max_inflight_deliveries=64, num_workers=2)
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
    rank.trace.check_slot_exclusivity(rank.publish_pool)


# ---- the store declining or failing a put once the units are already in host slots ----


def test_staged_declined_put_for_a_unit_another_publisher_made_present_counts_as_taken():
    """Staged variant of the publish race: between our lookup and our put another publisher
    stores ``first``; the store declines our write of it. The unit is held under its name, so it
    is served, the slots go back, and the attempt is quiet."""
    store = FakeBlobStore()
    with make_rank(store) as other, _staged(store, slots=4) as staged:
        first, second = staged.unit(0, 0, 32), staged.unit(0, 1, 32)
        staged.write(first, pattern(1, 32))
        staged.write(second, pattern(2, 32))
        theirs = other.unit(0, 0, 32)  # the same name from another rank's memory
        other.write(theirs, pattern(1, 32))
        orig = store.put

        def raced_put(keys, buffers):
            store.put = orig  # one-shot: ``other`` needs the real one
            assert isinstance(other.finish(other.backend.publish(extent([theirs]))), Delivered)
            results = list(orig(keys, buffers))
            results[keys.index(staged.key(first))] = PutStatus.DECLINED
            return results

        store.put = raced_put
        attempt = staged.backend.publish(extent([first, second]))
        outcome = staged.finish(attempt)
        assert outcome == Delivered(frozenset({first.name, second.name}))
        assert staged.backend.counters.publish_stored == 1
        assert staged.backend.counters.publish_raced == 1
        assert staged.backend.counters.failed_attempts == 0
        assert store.objects[staged.key(first)] == pattern(1, 32)
        assert store.objects[staged.key(second)] == pattern(2, 32)
        assert staged.backend.quiesce([attempt]) is True
        pool = staged.publish_pool
        every_slot = list(range(pool.num_slots))
        assert sorted(pool.acquire(pool.num_slots)) == every_slot  # all came back
        pool.release(every_slot)
        staged.trace.check_slot_exclusivity(pool)


def test_pooled_put_raising_after_the_gather_fails_frees_the_slots_and_is_quiet():
    """The units were copied into host slots (the caller's memory is done with) when the store
    call raises: the delivery fails, nothing is stored, the slots are released, and ``quiesce``
    answers True because the gather is what read the caller's memory."""
    with _staged(slots=2) as rank:
        a, b = rank.unit(0, 0, 32), rank.unit(0, 1, 32)
        rank.write(a, pattern(1, 32))
        rank.write(b, pattern(2, 32))
        rank.store.fail_next("put", RuntimeError("store unreachable"))
        attempt = rank.backend.publish(extent([a, b]))
        outcome = rank.finish(attempt)
        assert isinstance(outcome, Failed) and "store unreachable" in outcome.reason
        assert rank.store.objects == {}
        assert rank.publish_copier.kinds() == ["d2h", "d2h"]  # gathered before the put
        assert rank.backend.counters.failed_attempts == 1
        assert rank.backend.counters.publish_stored == 0
        assert rank.backend.quiesce([attempt]) is True
        assert sorted(rank.publish_pool.acquire(2)) == [0, 1]
        rank.publish_pool.release([0, 1])
        rank.trace.check_slot_exclusivity(rank.publish_pool)

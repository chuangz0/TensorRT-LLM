# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``HostLandingBlobBackend`` (``landing: host``) against the ``LandsOnHost`` / ``Landing``
contract, over the in-process store: a fetch lands in a landing-pool slot per unit, is placed into
the caller's pages later, and the landing holds its slots across rounds until ``release``.

What distinguishes this shape from the device one, and what the tests pin down:

* A landing short of slots waits in the pool's queue, never on a worker; the thread that returns
  slots hands it to a worker. Landings do not take the in-flight semaphore.
* The publish pool and the landing pool are separate, so a staged publish completes while every
  landing slot is held by a landing waiting for pages.
* ``release`` is non-blocking on the caller's thread in every state: queued, get in flight, landed,
  and after the backend has closed.
* ``poll`` bounds the queue wait; ``place`` writes only the units of its extent.

Every backend is closed inside the test body (``with make_host_rank() as rank``): the repository's
thread-leak check runs before fixture teardown.
"""

import threading
import time

import pytest

__extra_import_path__ = ["~/tensorrt_llm/_torch"]
from disaggregation.backends.blob.backend import (  # noqa: E402
    BlobStoreBackend,
    HostLandingBlobBackend,
)
from disaggregation.backends.blob.store import BlobStoreError  # noqa: E402
from disaggregation.base.cache_backend import (  # noqa: E402
    Attempt,
    Delivered,
    Failed,
    Fetches,
    Publishes,
    SubmissionRejected,
    Unit,
)
from disaggregation.orchestration.kv_transfer.interfaces import Landing, LandsOnHost  # noqa: E402
from store_fakes import (  # noqa: E402
    FakeBlobStore,
    extent,
    make_host_rank,
    make_rank,
    pattern,
    wait_until,
)

UNIT = 64


def _published(store: FakeBlobStore, count: int, seed: int = 1):
    """Publish ``count`` units of ``UNIT`` bytes from a device-landing rank, keeping it open so
    the test can compare bytes. Returns ``(rank, units)``; the caller closes the rank."""
    direct = make_rank(store)
    units = [direct.unit(0, i, UNIT) for i in range(count)]
    for i, u in enumerate(units):
        direct.write(u, pattern(seed + i, UNIT))
    assert isinstance(direct.finish(direct.backend.publish(extent(units))), Delivered)
    return direct, units


def _mirror(rank, units, fill: int = 0xEE):
    """The same names on the host rank's memory, filled so an untouched unit is recognisable."""
    mine = [rank.unit(u.local_group, u.local, UNIT) for u in units]
    for u in mine:
        rank.fill(u, fill)
    return mine


# ---- shape ----


def test_host_shape_is_lands_on_host_and_not_fetches():
    with make_host_rank() as rank:
        assert isinstance(rank.backend, HostLandingBlobBackend)
        assert isinstance(rank.backend, LandsOnHost)
        assert not isinstance(rank.backend, Fetches)
        assert isinstance(rank.inner, BlobStoreBackend) and isinstance(rank.inner, Publishes)
        landing = rank.backend.fetch_to_host(b"n", [])
        assert isinstance(landing, Landing)
        assert landing.poll() == Delivered(frozenset())
        assert isinstance(landing.place(extent([])), Attempt)
        landing.release()
        landing.release()  # idempotent
        # The inner backend has a publish pool and no fetch path of its own: a wiring error.
        with pytest.raises(RuntimeError, match="fetch_to_host"):
            rank.inner.fetch(extent([rank.unit(0, 0, 8)]))


def test_host_shape_registers_only_its_pools_and_publishes_through_the_publish_pool():
    with make_host_rank() as rank:
        assert rank.registration is None
        assert rank.store.count("register_span") == 2  # publish pool, landing pool
        u = rank.unit(0, 0, UNIT)
        rank.write(u, pattern(1, UNIT))
        assert isinstance(rank.finish(rank.backend.publish(extent([u]))), Delivered)
        assert rank.store.objects[rank.key(u)] == pattern(1, UNIT)
        assert rank.publish_copier.kinds() == ["d2h"] and rank.landing_copier.copies == []
        assert rank.backend.counters.publish_stored == 1


# ---- land, then place ----


def test_landing_then_place_round_trips_bytes_and_writes_only_the_extent():
    store = FakeBlobStore()
    direct, theirs = _published(store, 3)
    with direct, make_host_rank(store) as rank:
        a, b, c = _mirror(rank, theirs)
        missing = rank.unit(0, 9, UNIT)
        rank.fill(missing, 0xEE)
        landing = rank.land([a, b, c, missing])
        assert landing.poll() == Delivered(frozenset({a.name, b.name, c.name}))
        # Landed in host slots only: the caller's memory is untouched until placed.
        assert rank.read(a) == bytes([0xEE]) * UNIT and rank.landing_copier.copies == []
        assert rank.backend.counters.fetch_hits == 3 and rank.backend.counters.fetch_misses == 1
        assert rank.free_landing_slots() == 0  # four asked, four held (the miss's slot too)

        # Place two of the three: only those are written.
        assert rank.place(landing, [a, c]) == Delivered(frozenset({a.name, c.name}))
        assert rank.read(a) == pattern(1, UNIT) == direct.read(theirs[0])
        assert rank.read(c) == pattern(3, UNIT) == direct.read(theirs[2])
        assert rank.read(b) == bytes([0xEE]) * UNIT
        assert rank.landing_copier.kinds() == ["h2d", "h2d"] and rank.landing_copier.syncs == 1
        # The rest can still be placed: the slots are held until release.
        assert rank.place(landing, [b]) == Delivered(frozenset({b.name}))
        assert rank.read(b) == pattern(2, UNIT)
        assert rank.backend.landings_held() == 1
        landing.release()
        assert rank.backend.landings_held() == 0 and rank.free_landing_slots() == 4
        assert rank.backend.counters.failed_attempts == 0


def test_place_of_a_unit_the_landing_did_not_serve_fails_without_touching_memory():
    store = FakeBlobStore()
    direct, theirs = _published(store, 1)
    with direct, make_host_rank(store) as rank:
        (a,) = _mirror(rank, theirs)
        never = rank.unit(0, 5, UNIT)
        rank.fill(never, 0xEE)
        landing = rank.land([a, never])  # ``never`` is a miss
        attempt = landing.place(extent([a, never]))
        outcome = attempt.poll()  # decided at submission
        assert isinstance(outcome, Failed) and "not landed" in outcome.reason
        assert rank.landing_copier.copies == [] and rank.read(never) == bytes([0xEE]) * UNIT
        assert rank.backend.quiesce([attempt]) is True
        assert rank.backend.counters.failed_attempts == 1
        landing.release()


def test_place_before_the_landing_has_content_is_rejected():
    with make_host_rank() as rank:
        a = rank.unit(0, 0, UNIT)
        rank.store.block("holds")
        landing = rank.backend.fetch_to_host(b"n", [a.name])
        rank.store.wait_entered(3)  # two registrations, then the lookup parked
        with pytest.raises(SubmissionRejected, match="no content"):
            landing.place(extent([a]))
        rank.store.unblock()
        wait_until(lambda: landing.poll() is not None)
        assert landing.poll() == Delivered(frozenset())  # a miss, honestly reported
        # Landed with nothing: a placement is accepted and fails through its attempt, since a
        # unit the landing did not serve is a wiring error, not back-pressure.
        outcome = landing.place(extent([a])).poll()
        assert isinstance(outcome, Failed) and "not landed" in outcome.reason
        landing.release()


def test_place_copier_failure_is_failed_quiet_and_keeps_the_landing_placeable():
    store = FakeBlobStore()
    direct, theirs = _published(store, 1)
    with direct, make_host_rank(store) as rank:
        (a,) = _mirror(rank, theirs)
        landing = rank.land([a])
        rank.landing_copier.fail_next("copy", RuntimeError("cudaMemcpyAsync failed"))
        attempt = landing.place(extent([a]))
        outcome = rank.finish(attempt)
        assert isinstance(outcome, Failed) and "cudaMemcpyAsync" in outcome.reason
        assert rank.backend.quiesce([attempt]) is True
        assert rank.landing_copier.syncs == 1  # drained before the outcome
        assert rank.backend.landings_held() == 1  # the failed placement released nothing
        assert rank.place(landing, [a]) == Delivered(frozenset({a.name}))
        assert rank.read(a) == pattern(1, UNIT)
        landing.release()


# ---- the queue: slots without threads ----


def test_slot_shortage_queues_the_landing_and_release_hands_it_to_a_worker():
    store = FakeBlobStore()
    direct, theirs = _published(store, 2)
    with direct, make_host_rank(store, landing_slots=1) as rank:
        a, b = _mirror(rank, theirs)
        first = rank.land([a])
        lookups = rank.store.count("holds")
        second = rank.backend.fetch_to_host(b"second", [b.name])
        time.sleep(0.05)
        for _ in range(20):
            assert second.poll() is None  # queued, non-blocking
        assert rank.store.count("holds") == lookups  # the worker was not asked yet
        assert rank.inflight_slots_free() == 256
        first.release()  # the engine thread returns the slot and dispatches the queue head
        wait_until(lambda: second.poll() is not None)
        assert second.poll() == Delivered(frozenset({b.name}))
        assert rank.place(second, [b]) == Delivered(frozenset({b.name}))
        assert rank.read(b) == pattern(2, UNIT)
        second.release()


def test_one_worker_one_slot_three_landings_all_complete_in_order():
    store = FakeBlobStore()
    direct, theirs = _published(store, 3)
    with direct, make_host_rank(store, landing_slots=1, num_workers=1) as rank:
        mine = _mirror(rank, theirs)
        landings = [
            rank.backend.fetch_to_host(f"l{i}".encode(), [u.name]) for i, u in enumerate(mine)
        ]
        for i, landing in enumerate(landings):
            wait_until(lambda: landing.poll() is not None, what=f"landing {i}")
            assert landing.poll() == Delivered(frozenset({mine[i].name}))
            assert rank.place(landing, [mine[i]]) == Delivered(frozenset({mine[i].name}))
            later = [other.poll() for other in landings[i + 1 :]]
            assert later == [None] * len(later)  # the rest still wait for the one slot
            landing.release()
        for i, u in enumerate(mine):
            assert rank.read(u) == pattern(1 + i, UNIT)


def test_publish_completes_while_a_landing_holds_every_landing_slot_on_the_only_worker():
    """The two-pool property: a staged publish only ever waits for other publishes."""
    store = FakeBlobStore()
    direct, theirs = _published(store, 1)
    with direct, make_host_rank(store, landing_slots=1, publish_slots=1, num_workers=1) as rank:
        (a,) = _mirror(rank, theirs)
        held = rank.land([a])  # holds the single landing slot, waiting for pages
        out = [rank.unit(1, i, UNIT) for i in range(2)]
        for i, u in enumerate(out):
            rank.write(u, pattern(10 + i, UNIT))
        assert rank.finish(rank.backend.publish(extent(out))) == Delivered(
            frozenset(u.name for u in out)
        )
        for i, u in enumerate(out):
            assert store.objects[rank.key(u)] == pattern(10 + i, UNIT)
        assert held.poll() == Delivered(frozenset({a.name}))  # still landed, still held
        held.release()


def test_landings_take_no_inflight_slot_but_placements_do():
    store = FakeBlobStore()
    direct, theirs = _published(store, 2)
    with direct, make_host_rank(store, max_inflight_ops=1) as rank:
        a, b = _mirror(rank, theirs)
        rank.store.block("get")
        first = rank.backend.fetch_to_host(b"a", [a.name])
        second = rank.backend.fetch_to_host(b"b", [b.name])
        rank.store.wait_entered(6)  # 2 registrations, 2 lookups, 2 gets parked
        assert rank.inflight_slots_free() == 1  # two gets in flight, no inflight slot taken
        rank.store.unblock()
        wait_until(lambda: first.poll() is not None and second.poll() is not None)
        rank.landing_copier.block("copy")
        placing = first.place(extent([a]))
        rank.landing_copier.wait_entered(1)
        assert rank.inflight_slots_free() == 0
        with pytest.raises(SubmissionRejected, match="1 deliveries already in flight"):
            second.place(extent([b]))
        rank.landing_copier.unblock()
        assert isinstance(rank.finish(placing), Delivered)
        assert rank.inflight_slots_free() == 1
        first.release()
        second.release()


def test_holds_and_get_run_back_to_back_once_the_slots_are_granted():
    store = FakeBlobStore()
    direct, theirs = _published(store, 2)
    with direct, make_host_rank(store, landing_slots=1) as rank:
        a, b = _mirror(rank, theirs)
        first = rank.land([a])
        calls_before = len(rank.store.calls)
        second = rank.backend.fetch_to_host(b"b", [b.name])
        time.sleep(0.05)
        assert rank.store.calls[calls_before:] == []  # nothing asked while queued
        first.release()
        wait_until(lambda: second.poll() is not None)
        assert [m for m, _ in rank.store.calls[calls_before:]] == ["holds", "get"]
        second.release()


# ---- release in every state ----


def test_release_while_queued_dequeues_and_takes_no_slot():
    store = FakeBlobStore()
    direct, theirs = _published(store, 2)
    with direct, make_host_rank(store, landing_slots=1) as rank:
        a, b = _mirror(rank, theirs)
        first = rank.land([a])
        lookups = rank.store.count("holds")
        second = rank.backend.fetch_to_host(b"b", [b.name])
        second.release()  # gives up its place in the queue
        assert rank.backend.landings_held() == 1
        first.release()  # the slot comes back and nobody is dispatched
        time.sleep(0.05)
        assert rank.store.count("holds") == lookups
        assert rank.free_landing_slots() == 1 and rank.backend.landings_held() == 0
        assert rank.inflight_slots_free() == 256
        assert rank.backend.counters.failed_attempts == 0  # a release is not a failure


def test_release_while_the_get_is_in_flight_returns_the_slots_after_it():
    store = FakeBlobStore()
    direct, theirs = _published(store, 1)
    with direct, make_host_rank(store, landing_slots=1) as rank:
        (a,) = _mirror(rank, theirs)
        rank.store.block("get")
        landing = rank.backend.fetch_to_host(b"a", [a.name])
        rank.store.wait_entered(4)  # 2 registrations, lookup, get parked
        start = time.monotonic()
        landing.release()
        assert time.monotonic() - start < 0.5  # did not wait for the get
        assert rank.free_landing_slots() == 0  # the store may still write the slot
        rank.store.unblock()
        wait_until(lambda: rank.free_landing_slots() == 1, what="slot returned after the get")
        assert rank.backend.landings_held() == 0
        assert rank.read(a) == bytes([0xEE]) * UNIT


def test_queue_wait_past_the_bound_fails_the_landing_and_dequeues_it():
    store = FakeBlobStore()
    direct, theirs = _published(store, 2)
    with direct, make_host_rank(store, landing_slots=1, landing_wait_timeout_s=0.05) as rank:
        a, b = _mirror(rank, theirs)
        first = rank.land([a])
        second = rank.backend.fetch_to_host(b"b", [b.name])
        assert second.poll() is None
        time.sleep(0.1)
        outcome = second.poll()  # the bound is checked here, on the caller's clock
        assert isinstance(outcome, Failed) and "waited" in outcome.reason
        assert rank.backend.counters.failed_attempts == 0  # a wait, not a store failure
        lookups = rank.store.count("holds")
        first.release()
        time.sleep(0.05)
        assert rank.store.count("holds") == lookups  # the failed landing was not dispatched
        assert rank.free_landing_slots() == 1
        second.release()  # nothing to give back; no error


def test_landing_failure_gives_the_slots_back_at_once():
    store = FakeBlobStore()
    direct, theirs = _published(store, 1)
    with direct, make_host_rank(store) as rank:
        (a,) = _mirror(rank, theirs)
        rank.store.fail_next("get", RuntimeError("store unreachable"))
        landing = rank.land([a])
        outcome = landing.poll()
        assert isinstance(outcome, Failed) and "store unreachable" in outcome.reason
        assert rank.free_landing_slots() == 4 and rank.backend.landings_held() == 0
        assert rank.backend.counters.failed_attempts == 1
        landing.release()
        # A lookup failure is a failure too, not a miss.
        rank.store.fail_next("holds", BlobStoreError("master unreachable"))
        outcome = rank.land([a]).poll()
        assert isinstance(outcome, Failed) and "lookup failed" in outcome.reason
        assert rank.backend.counters.fetch_misses == 0


def test_more_units_than_slots_or_an_unknown_unit_fails_before_the_store_is_asked():
    with make_host_rank(landing_slots=2) as rank:
        units = [rank.unit(0, i, UNIT) for i in range(3)]
        calls = len(rank.store.calls)
        outcome = rank.backend.fetch_to_host(b"big", [u.name for u in units]).poll()
        assert isinstance(outcome, Failed) and "exceed the 2 landing slots" in outcome.reason
        outcome = rank.backend.fetch_to_host(b"?", [b"nobody"]).poll()
        assert isinstance(outcome, Failed) and "no known size" in outcome.reason
        assert len(rank.store.calls) == calls and rank.backend.counters.failed_attempts == 2
        assert rank.free_landing_slots() == 2


def test_unit_larger_than_a_slot_fails_the_landing_at_once():
    with make_host_rank(slot_bytes=32) as rank:
        big = rank.unit(0, 0, 33)
        outcome = rank.backend.fetch_to_host(b"n", [big.name]).poll()
        assert isinstance(outcome, Failed) and "exceeds the landing slot" in outcome.reason


# ---- close ----


def test_close_refuses_queued_landings_frees_held_slots_and_release_afterwards_is_inert():
    store = FakeBlobStore()
    direct, theirs = _published(store, 2)
    with direct:
        rank = make_host_rank(store, landing_slots=1)
        a, b = _mirror(rank, theirs)
        landed = rank.land([a])
        queued = rank.backend.fetch_to_host(b"b", [b.name])
        rank.backend.close()
        outcome = queued.poll()
        assert isinstance(outcome, Failed) and "shut down" in outcome.reason
        assert rank.backend.landings_held() == 0 and rank.free_landing_slots() == 1
        assert store.closed == 1
        # Late releases from the coordinator: nothing to do, nothing to submit, no error.
        landed.release()
        queued.release()
        with pytest.raises(SubmissionRejected):
            rank.backend.fetch_to_host(b"c", [a.name])
        with pytest.raises(SubmissionRejected):
            landed.place(extent([a]))
        rank.backend.close()  # idempotent
        assert store.closed == 1


def test_close_waits_for_a_get_in_flight_then_returns_its_slots():
    store = FakeBlobStore()
    direct, theirs = _published(store, 1)
    with direct:
        rank = make_host_rank(store, landing_slots=1)
        (a,) = _mirror(rank, theirs)
        store.block("get")
        landing = rank.backend.fetch_to_host(b"a", [a.name])
        store.wait_entered(4)
        closer = threading.Thread(target=rank.backend.close)
        closer.start()
        time.sleep(0.05)
        assert closer.is_alive()  # the worker is inside the store call
        store.unblock()
        closer.join(5)
        assert not closer.is_alive()
        assert landing.poll() == Delivered(frozenset({a.name}))
        assert rank.free_landing_slots() == 1  # close asked for the slots back
        landing.release()


# ---- misc contract points ----


def test_empty_landing_and_empty_placement_complete_at_once_without_the_store():
    with make_host_rank() as rank:
        calls = len(rank.store.calls)
        landing = rank.backend.fetch_to_host(b"n", [])
        assert landing.poll() == Delivered(frozenset())
        attempt = landing.place(extent([]))
        assert attempt.poll() == Delivered(frozenset())
        assert rank.backend.quiesce([attempt]) is True
        rank.backend.settle([attempt])
        landing.release()
        assert len(rank.store.calls) == calls
        assert rank.free_landing_slots() == 4


def test_quiesce_and_settle_only_know_placements():
    with make_host_rank() as rank:
        with pytest.raises(TypeError):
            rank.backend.quiesce([object()])
        u = Unit(name=b"x", local_group=0, local=0)
        rank.resolver._table[(0, 0)] = ((rank.arena.address, 8),)
        attempt = rank.backend.fetch_to_host(b"n", []).place(extent([u]))
        outcome = attempt.poll()
        assert isinstance(outcome, Failed) and "not landed" in outcome.reason
        assert rank.backend.quiesce([attempt]) is True


def test_no_wait_bound_keeps_a_queued_landing_waiting():
    store = FakeBlobStore()
    direct, theirs = _published(store, 2)
    with direct, make_host_rank(store, landing_slots=1, landing_wait_timeout_s=None) as rank:
        a, b = _mirror(rank, theirs)
        first = rank.land([a])
        second = rank.backend.fetch_to_host(b"b", [b.name])
        second.enqueued_at -= 3600.0  # queued "an hour ago": still no bound to hit
        assert second.poll() is None
        first.release()
        wait_until(lambda: second.poll() is not None)
        assert second.poll() == Delivered(frozenset({b.name}))
        second.release()


def test_close_fails_a_landing_still_queued_so_poll_is_never_none_forever():
    store = FakeBlobStore()
    direct, theirs = _published(store, 2)
    with direct:
        rank = make_host_rank(store, landing_slots=1)
        a, b = _mirror(rank, theirs)
        held = rank.land([a])
        queued = rank.backend.fetch_to_host(b"b", [b.name])
        rank.backend.close()
        assert isinstance(queued.poll(), Failed) and isinstance(held.poll(), Delivered)
        with pytest.raises(SubmissionRejected):  # its slots went back at close
            held.place(extent([a]))

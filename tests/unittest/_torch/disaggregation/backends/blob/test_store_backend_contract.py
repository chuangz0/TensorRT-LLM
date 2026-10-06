# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``BlobStoreBackend`` against the cache backend contract's shape: protocols satisfied,
registration rules (SPEC §6.4), submission rules (§6.2/§6.3), ``poll``/``settle``/``quiesce``
(§6.1), back-pressure and lifetime. Content semantics are in ``test_backend_semantics.py``.

Every backend is closed inside the test body (``with make_rank() as rank``): the repository's
thread-leak check runs before fixture teardown, and ``close`` joining its workers is itself part
of the behaviour under test.
"""

import threading
import time

import pytest

__extra_import_path__ = ["~/tensorrt_llm/_torch"]
from disaggregation.backends.blob.backend import BlobStoreBackend, StoreCounters  # noqa: E402
from disaggregation.backends.blob.store import BlobStoreError  # noqa: E402
from disaggregation.base.cache_backend import (  # noqa: E402
    Attempt,
    Delivered,
    Failed,
    Fetches,
    Publishes,
    RegistersPools,
    SubmissionRejected,
    Unit,
)
from store_fakes import (  # noqa: E402
    FakeBlobStore,
    MemoryArena,
    config,
    extent,
    make_host_rank,
    make_rank,
    pattern,
    wait_until,
)

pytestmark = pytest.mark.cpu_only


# ---- protocols ----


def test_backend_satisfies_the_three_protocols():
    with make_rank() as rank:
        assert isinstance(rank.backend, Fetches)
        assert isinstance(rank.backend, Publishes)
        assert isinstance(rank.backend, RegistersPools)
        assert isinstance(rank.backend.counters, StoreCounters)
        assert isinstance(rank.backend.publish(extent([])), Attempt)


def test_host_landing_requires_a_publish_pool_when_configured():
    with pytest.raises(ValueError, match="HostStagingPool"):
        BlobStoreBackend(FakeBlobStore(), config(landing="host"), lambda group, local: (), b"\x01")


# ---- RegistersPools (§6.4) ----


def test_register_pool_forwards_to_store_and_refuses_overlap():
    with make_rank() as rank:
        a = rank.arena.address
        assert rank.store.calls[0] == ("register_span", (a, rank.arena.size))
        other = MemoryArena(64)
        reg = rank.backend.register_pool(other.address, other.size)
        for start, size in [(a, 1), (a + 10, 5), (a - 1, 2), (a + rank.arena.size - 1, 100)]:
            with pytest.raises(ValueError, match="overlaps"):
                rank.backend.register_pool(start, size)
        reg.close()
        with pytest.raises(ValueError, match="size must be > 0"):
            rank.backend.register_pool(other.address, 0)
        assert rank.store.count("register_span") == 2  # a refused registration reaches nothing


def test_concurrent_duplicate_register_pool_is_refused_before_it_reaches_the_store():
    # The winner is parked inside the store's register call; the loser, asking for the same span
    # meanwhile, must be refused without a call of its own (a store that keys registrations by
    # address would otherwise have the loser's cleanup tear the winner down).
    with make_rank() as rank:
        other = MemoryArena(64)
        rank.store.block("register_span")
        winner = []
        t = threading.Thread(
            target=lambda: winner.append(rank.backend.register_pool(other.address, other.size))
        )
        t.start()
        rank.store.wait_entered(2)  # the pool's own registration, then the winner's, parked
        with pytest.raises(ValueError, match="overlaps"):
            rank.backend.register_pool(other.address, other.size)
        with pytest.raises(ValueError, match="overlaps"):
            rank.backend.register_pool(other.address + 8, 8)  # partial overlap, same answer
        assert rank.store.count("register_span") == 2  # nothing from the losers
        rank.store.unblock()
        t.join(5)
        assert not t.is_alive() and len(winner) == 1
        # The winner's registration is live and untouched by the refusals.
        rank.resolver._table[(5, 5)] = ((other.address, 8),)
        u = Unit(name=b"x", local_group=5, local=5)
        assert isinstance(rank.finish(rank.backend.publish(extent([u]))), Delivered)
        winner[0].close()
        assert rank.store.count("unregister_span") == 1


def test_register_pool_raises_and_registers_nothing_when_store_refuses():
    with make_rank() as rank:
        other = MemoryArena(64)
        # The store refuses ...
        rank.store.fail_next("register_span", BlobStoreError("registration refused with status -5"))
        with pytest.raises(RuntimeError, match="status -5"):
            rank.backend.register_pool(other.address, other.size)
        # ... or its transport raises outright.
        rank.store.fail_next("register_span", OSError("rdma device gone"))
        with pytest.raises(OSError):
            rank.backend.register_pool(other.address, other.size)
        # Nothing was kept from either failure: the span is free, and a unit there is refused.
        rank.resolver._table[(5, 5)] = ((other.address, 8),)
        outcome = rank.backend.publish(extent([Unit(name=b"x", local_group=5, local=5)])).poll()
        assert isinstance(outcome, Failed) and "not registered" in outcome.reason
        rank.backend.register_pool(other.address, other.size).close()


def test_unregister_rpc_raising_keeps_the_handle_live_and_retry_works():
    with make_rank() as rank:
        other = MemoryArena(64)
        reg = rank.backend.register_pool(other.address, other.size)
        rank.store.fail_next("unregister_span", OSError("transport hiccup"))
        with pytest.raises(OSError):
            reg.close()
        # Still registered: overlap is refused and deliveries into the span still run.
        with pytest.raises(ValueError, match="overlaps"):
            rank.backend.register_pool(other.address, other.size)
        rank.resolver._table[(5, 5)] = ((other.address, 8),)
        u = Unit(name=b"x", local_group=5, local=5)
        assert isinstance(rank.finish(rank.backend.publish(extent([u]))), Delivered)
        reg.close()  # the retry succeeds
        assert rank.store.count("unregister_span") == 2
        assert isinstance(rank.backend.publish(extent([u])).poll(), Failed)
        reg.close()  # idempotent once it has succeeded
        assert rank.store.count("unregister_span") == 2


def test_registration_close_is_idempotent_and_by_handle():
    with make_rank() as rank:
        other = MemoryArena(64)
        stale = rank.backend.register_pool(other.address, other.size)
        stale.close()
        stale.close()
        assert rank.store.count("unregister_span") == 1
        # Pool rebuild: register the same span again, then close the stale handle once more
        # (§6.4 3c): the live registration must not be taken down.
        live = rank.backend.register_pool(other.address, other.size)
        stale.close()
        assert rank.store.count("unregister_span") == 1
        with pytest.raises(ValueError, match="overlaps"):
            rank.backend.register_pool(other.address, other.size)
        live.close()
        assert rank.store.count("unregister_span") == 2


def test_register_pool_during_close_is_refused():
    """``close`` is under way, waiting for a delivery in flight; a registration asked for
    meanwhile is refused as on a closed backend and reaches neither the store nor the table,
    so ``close`` leaves nothing registered behind it."""
    with make_rank() as rank:
        u = rank.unit(0, 0, 8)
        rank.store.block("put")
        attempt = rank.backend.publish(extent([u]))
        rank.store.wait_entered(3)  # register, holds, put parked
        closer = threading.Thread(target=rank.backend.close)
        closer.start()
        time.sleep(0.05)
        assert closer.is_alive()  # waiting for the worker inside the store call
        other = MemoryArena(64)
        with pytest.raises(RuntimeError, match="closed"):
            rank.backend.register_pool(other.address, other.size)
        rank.store.unblock()
        closer.join(5)
        assert not closer.is_alive()
        assert isinstance(attempt.poll(), Delivered)
        assert rank.store.count("register_span") == 1  # the pool's own, at construction
        assert rank.store.count("unregister_span") == 1  # undone by close


def test_registration_in_flight_when_close_runs_is_undone_or_refused():
    """A registration whose store call is still running when ``close`` runs: once the store
    answers, the registration is either refused or undone (whether ``close`` returned at once
    or waited for it). A closed backend leaves nothing registered, or the transport keeps memory
    the caller believes it owns again."""
    with make_rank() as rank:
        other = MemoryArena(64)
        rank.store.block("register_span")
        answers = []

        def register():
            try:
                answers.append(rank.backend.register_pool(other.address, other.size))
            except RuntimeError as exc:
                answers.append(exc)

        registrar = threading.Thread(target=register)
        registrar.start()
        rank.store.wait_entered(2)  # the pool's own registration, then this one, parked
        closer = threading.Thread(target=rank.backend.close)
        closer.start()
        time.sleep(0.05)  # close either returns at once or waits for the parked registration
        rank.store.unblock()
        registrar.join(5)
        closer.join(5)
        assert not registrar.is_alive() and not closer.is_alive() and len(answers) == 1
        assert rank.store.count("register_span") == 2
        refused = isinstance(answers[0], RuntimeError)
        assert refused or rank.store.count("unregister_span") == 2, (
            "a registration completed on a closed backend and was never undone"
        )


def test_registration_close_that_raises_has_not_closed():
    with make_rank() as rank:
        other = MemoryArena(64)
        reg = rank.backend.register_pool(other.address, other.size)
        rank.store.fail_next("unregister_span")
        with pytest.raises(RuntimeError):
            reg.close()
        with pytest.raises(ValueError, match="overlaps"):  # still registered
            rank.backend.register_pool(other.address, other.size)
        reg.close()
        rank.backend.register_pool(other.address, other.size).close()


@pytest.mark.parametrize("direction", ["fetch", "publish"])
def test_delivery_into_unregistered_memory_fails_and_is_not_a_miss(direction):
    with make_rank(register=False) as rank:
        u = rank.unit(0, 0, 32)
        attempt = getattr(rank.backend, direction)(extent([u]))
        outcome = attempt.poll()  # decided at submission: nothing was queued
        assert isinstance(outcome, Failed) and "not registered" in outcome.reason
        assert rank.store.count("holds") == 0
        assert rank.backend.counters.failed_attempts == 1
        assert rank.backend.counters.fetch_misses == 0
        assert rank.backend.quiesce([attempt]) is True
        rank.backend.settle([attempt])


def test_delivery_into_closed_registration_fails():
    with make_rank() as rank:
        u = rank.unit(0, 0, 32)
        rank.registration.close()
        outcome = rank.backend.fetch(extent([u])).poll()
        assert isinstance(outcome, Failed) and "not registered" in outcome.reason


def test_segment_running_past_its_registration_is_refused_before_the_store():
    # Coverage is per segment and per registration: a segment that starts inside the pool but
    # runs past its end is not registered memory, whatever else is registered.
    with make_rank() as rank:
        straddle = (rank.arena.address + rank.arena.size - 8, 16)
        rank.resolver._table[(0, 9)] = (straddle,)
        bad = Unit(name=b"straddle", local_group=0, local=9)
        outcome = rank.backend.publish(extent([bad])).poll()
        assert isinstance(outcome, Failed) and "not registered" in outcome.reason
        assert rank.store.count("put") == 0


def test_one_bad_unit_fails_the_whole_delivery_and_nothing_is_moved():
    with make_rank() as rank:
        good = rank.unit(0, 0, 8)
        unknown = Unit(name=b"?", local_group=7, local=7)
        outcome = rank.backend.publish(extent([good, unknown])).poll()
        assert isinstance(outcome, Failed) and "does not resolve" in outcome.reason
        assert rank.store.count("holds") == 0
        assert rank.key(good) not in rank.store.objects


def test_unit_resolving_to_no_memory_fails():
    with make_rank() as rank:
        rank.resolver._table[(0, 0)] = ()
        empty = Unit(name=b"e", local_group=0, local=0)
        outcome = rank.backend.fetch(extent([empty])).poll()
        assert isinstance(outcome, Failed) and "no memory" in outcome.reason


# ---- submission (§6.2 / §6.3) ----


def test_empty_extent_is_delivered_empty_immediately():
    with make_rank() as rank:
        calls = len(rank.store.calls)
        for submit in (rank.backend.fetch, rank.backend.publish):
            attempt = submit(extent([]))
            assert attempt.poll() == Delivered(frozenset())
            assert rank.backend.quiesce([attempt]) is True
            rank.backend.settle([attempt])
        assert len(rank.store.calls) == calls


def test_fetch_with_any_route_is_rejected():
    class SomeRoute:
        def close(self):
            pass

    with make_rank() as rank:
        with pytest.raises(SubmissionRejected):
            rank.backend.fetch(extent([rank.unit(0, 0, 8)]), route=SomeRoute())
        assert rank.store.count("holds") == 0


def test_open_route_is_not_implemented():
    with make_rank() as rank:
        with pytest.raises(NotImplementedError):
            rank.backend.open_route({"peer": "x"})


def test_submit_after_close_is_rejected_and_close_is_idempotent():
    with make_rank() as rank:
        u = rank.unit(0, 0, 8)
        rank.backend.close()
        rank.backend.close()
        assert rank.store.closed == 1
        with pytest.raises(SubmissionRejected):
            rank.backend.fetch(extent([u]))
        with pytest.raises(SubmissionRejected):
            rank.backend.publish(extent([u]))
        with pytest.raises(RuntimeError):
            rank.backend.probe(b"n", [u.name])
        # An empty extent needs no worker and is still answered.
        assert rank.backend.publish(extent([])).poll() == Delivered(frozenset())
    assert rank.store.closed == 1


def test_close_finishes_work_in_flight_before_closing_the_store():
    with make_rank() as rank:
        u = rank.unit(0, 0, 8)
        rank.store.block("put")
        attempt = rank.backend.publish(extent([u]))
        rank.store.wait_entered(3)  # register, is_exist, put
        closer = threading.Thread(target=rank.backend.close)
        closer.start()
        time.sleep(0.05)
        assert closer.is_alive() and rank.store.closed == 0
        rank.store.unblock()
        closer.join(5)
        assert not closer.is_alive() and rank.store.closed == 1
        assert isinstance(attempt.poll(), Delivered)


# ---- Attempt / settle / quiesce (§6.1) ----


def test_poll_is_non_blocking_while_workers_are_blocked():
    with make_rank() as rank:
        u = rank.unit(0, 0, 8)
        rank.store.block("holds")
        attempt = rank.backend.publish(extent([u]))
        rank.store.wait_entered(2)
        start = time.monotonic()
        for _ in range(50):
            assert attempt.poll() is None
        assert time.monotonic() - start < 0.5
        rank.store.unblock()
        assert isinstance(rank.finish(attempt), Delivered)


def test_settle_blocks_until_an_outcome_then_poll_is_set():
    with make_rank() as rank:
        u = rank.unit(0, 0, 8)
        rank.store.block("holds")
        attempt = rank.backend.fetch(extent([u]))
        rank.store.wait_entered(2)
        settled = threading.Event()

        def settle():
            rank.backend.settle([attempt])
            settled.set()

        t = threading.Thread(target=settle)
        t.start()
        assert not settled.wait(0.1)
        assert attempt.poll() is None
        rank.store.unblock()
        assert settled.wait(5)
        t.join(5)
        assert attempt.poll() is not None


def test_outcome_is_immutable_once_set():
    with make_rank() as rank:
        u = rank.unit(0, 0, 8)
        attempt = rank.backend.publish(extent([u]))
        first = rank.finish(attempt)
        assert isinstance(first, Delivered)
        for _ in range(3):
            assert attempt.poll() is first
        rank.backend.settle([attempt, attempt])
        assert attempt.poll() is first


def test_quiesce_is_true_once_done_and_blocks_while_running():
    with make_rank() as rank:
        u = rank.unit(0, 0, 8)
        rank.store.block("put")
        attempt = rank.backend.publish(extent([u]))
        rank.store.wait_entered(3)
        answered = []
        t = threading.Thread(target=lambda: answered.append(rank.backend.quiesce([attempt])))
        t.start()
        time.sleep(0.05)
        assert t.is_alive()  # direct path: the put reads the caller's memory, so not quiet yet
        rank.store.unblock()
        t.join(5)
        assert answered == [True]
        assert rank.backend.quiesce([attempt, attempt]) is True


# ---- back-pressure ----


def test_max_inflight_ops_rejects_the_excess_and_nothing_escapes():
    with make_rank(max_inflight_ops=1, num_workers=2) as rank:
        a, b = rank.unit(0, 0, 8), rank.unit(0, 1, 8)
        rank.write(b, pattern(2, 8))
        rank.store.block("holds")
        first = rank.backend.publish(extent([a], name=b"a"))
        rank.store.wait_entered(2)
        with pytest.raises(SubmissionRejected, match="1 deliveries already in flight"):
            rank.backend.publish(extent([b], name=b"b"))
        assert rank.store.count("holds") == 1  # only the first delivery reached it
        rank.store.unblock()
        assert isinstance(rank.finish(first), Delivered)
        assert rank.key(b) not in rank.store.objects  # nothing of the rejected one escaped
        assert rank.backend.counters.failed_attempts == 0
        # The slot is free again once the first is done.
        second = rank.backend.publish(extent([b], name=b"b"))
        assert isinstance(rank.finish(second), Delivered)
        assert rank.store.objects[rank.key(b)] == pattern(2, 8)


def test_rejected_submission_succeeds_once_an_in_flight_delivery_finishes():
    # The semaphore is restored by the finishing delivery, whatever its outcome.
    with make_rank(max_inflight_ops=2, num_workers=2) as rank:
        units = [rank.unit(0, i, 8) for i in range(3)]
        rank.store.block("holds")
        first = rank.backend.publish(extent([units[0]]))
        second = rank.backend.publish(extent([units[1]]))
        rank.store.wait_entered(3)  # register, then both lookups parked at the gate
        with pytest.raises(SubmissionRejected):
            rank.backend.publish(extent([units[2]]))
        rank.store.fail_next("put")  # one of the two fails, not both
        rank.store.unblock()
        outcomes = [rank.finish(first), rank.finish(second)]
        assert sorted(type(o).__name__ for o in outcomes) == ["Delivered", "Failed"]
        third = rank.backend.publish(extent([units[2]]))
        assert isinstance(rank.finish(third), Delivered)
        assert rank.key(units[2]) in rank.store.objects


def test_failed_delivery_releases_its_inflight_slot():
    with make_rank(max_inflight_ops=1) as rank:
        u = rank.unit(0, 0, 8)
        rank.store.fail_next("holds")
        assert isinstance(rank.finish(rank.backend.fetch(extent([u]))), Failed)
        assert isinstance(rank.finish(rank.backend.fetch(extent([u]))), Delivered)


def test_pre_submission_failure_does_not_consume_an_inflight_slot():
    with make_rank(max_inflight_ops=1, register=False) as rank:
        u = rank.unit(0, 0, 8)
        for _ in range(3):
            assert isinstance(rank.backend.fetch(extent([u])).poll(), Failed)
        rank.backend.register_pool(rank.arena.address, rank.arena.size)
        assert isinstance(rank.finish(rank.backend.fetch(extent([u]))), Delivered)


# ---- counters and batching ----


def test_counters_count_hits_misses_stored_and_present():
    with make_rank() as rank:
        a, b, c = rank.unit(0, 0, 8), rank.unit(0, 1, 8), rank.unit(0, 2, 8)
        assert isinstance(rank.finish(rank.backend.publish(extent([a, b]))), Delivered)
        assert rank.backend.counters.publish_stored == 2
        assert rank.backend.counters.publish_present == 0
        assert isinstance(rank.finish(rank.backend.fetch(extent([a, b, c]))), Delivered)
        assert rank.backend.counters.fetch_hits == 2
        assert rank.backend.counters.fetch_misses == 1
        assert isinstance(rank.finish(rank.backend.publish(extent([a, b, c]))), Delivered)
        assert rank.backend.counters.publish_present == 2
        assert rank.backend.counters.publish_stored == 3
        assert rank.backend.counters.failed_attempts == 0
        assert rank.backend.probe(b"n", [a.name, c.name]) is None
        counters = rank.backend.counters
        wait_until(lambda: counters.probe_hits + counters.probe_misses == 2, what="probe")
        assert (counters.probe_hits, counters.probe_misses) == (2, 0)


def test_transfer_batch_size_bounds_one_store_call_not_one_delivery():
    with make_rank(transfer_batch_size=2) as rank:
        units = [rank.unit(0, i, 4) for i in range(5)]
        outcome = rank.finish(rank.backend.publish(extent(units)))
        assert outcome == Delivered(frozenset(u.name for u in units))
        puts = [args[0] for m, args in rank.store.calls if m == "put"]
        assert [len(keys) for keys in puts] == [2, 2, 1]
        exists = [args[0] for m, args in rank.store.calls if m == "holds"]
        assert [len(keys) for keys in exists] == [2, 2, 1]


def test_staged_put_batch_is_bounded_by_the_publish_pool_slot_count():
    """A staged put holds a publish-pool slot per unit for the length of the call, so with
    fewer slots than ``transfer_batch_size`` the store is asked in rounds of the slot count."""
    with make_host_rank(publish_slots=3, transfer_batch_size=64) as rank:
        units = [rank.unit(0, i, 8) for i in range(5)]
        outcome = rank.finish(rank.backend.publish(extent(units)))
        assert outcome == Delivered(frozenset(u.name for u in units))
        puts = [args[0] for m, args in rank.store.calls if m == "put"]
        assert [len(keys) for keys in puts] == [3, 2]
        lookups = [args[0] for m, args in rank.store.calls if m == "holds"]
        assert [len(keys) for keys in lookups] == [5]  # lookups hold no slot

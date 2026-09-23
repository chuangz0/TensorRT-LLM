# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What a fetch serves and what a publish stores, across two backends sharing one store: the
byte contract (SPEC §5.1), miss versus failure (§5.2), merge-not-replace (§6.3 req. 2), the
probe protocol (§6.2 ``probe``), and that a store never reads ``is_last`` (§4.2 inv. 10)."""

import time

import pytest

__extra_import_path__ = ["~/tensorrt_llm/_torch"]
from disaggregation.backends.blob.store import BlobStoreError, GetStatus, PutStatus  # noqa: E402
from disaggregation.base.cache_backend import Delivered, Failed, Unit  # noqa: E402
from store_fakes import FakeBlobStore, extent, make_rank, pattern, wait_until  # noqa: E402


def _pair(**overrides):
    """A publisher and a fetcher over one store, each with its own memory and matching units:
    (0,0) one segment, (0,1) two segments, (1,0) one segment in another layer group."""
    store = FakeBlobStore()
    a = make_rank(store, **overrides)
    b = make_rank(store, **overrides)
    for rank in (a, b):
        rank.unit(0, 0, 64)
        rank.unit(0, 1, 40, 24)
        rank.unit(1, 0, 16)
    return a, b


def _seed(rank, seeds: dict) -> None:
    for coords, seed in seeds.items():
        unit = rank.units[coords]
        rank.write(unit, pattern(seed, sum(s for _, s in rank.segments(unit))))


# ---- publish then fetch ----


def test_fetch_serves_exactly_what_was_published_with_equal_bytes():
    a, b = _pair()
    with a, b:
        _seed(a, {(0, 0): 1, (0, 1): 2, (1, 0): 3})
        published = a.finish(a.backend.publish(extent(a.units.values(), name=b"pub")))
        assert published == Delivered(frozenset(u.name for u in a.units.values()))
        assert a.store.objects[a.key(a.units[(0, 1)])] == pattern(2, 64)  # 40 + 24 concatenated
        for unit in b.units.values():
            b.fill(unit, 0xEE)
        fetched = b.finish(b.backend.fetch(extent(b.units.values(), name=b"other-extent")))
        assert fetched == Delivered(frozenset(u.name for u in b.units.values()))
        for coords in a.units:
            assert b.read(b.units[coords]) == a.read(a.units[coords])
        two_seg = b.segments(b.units[(0, 1)])
        assert [s for _, s in two_seg] == [40, 24]
        assert b.read(b.units[(0, 1)]) == pattern(2, 64)


def test_missing_unit_is_not_served_and_its_destination_is_untouched():
    a, b = _pair()
    with a, b:
        _seed(a, {(0, 0): 1, (1, 0): 3})
        held = [a.units[(0, 0)], a.units[(1, 0)]]
        assert isinstance(a.finish(a.backend.publish(extent(held))), Delivered)
        for unit in b.units.values():
            b.fill(unit, 0xEE)
        fetched = b.finish(b.backend.fetch(extent(b.units.values())))
        assert fetched == Delivered(frozenset(u.name for u in held))
        assert b.read(b.units[(0, 1)]) == bytes([0xEE]) * 64  # SPEC §5.1 inv. 5
        assert b.read(b.units[(0, 0)]) == pattern(1, 64)
        gets = [args[0] for m, args in b.store.calls if m == "get"]
        assert gets == [(b.key(held[0]), b.key(held[1]))]  # the miss is never asked for
        assert b.backend.counters.failed_attempts == 0


def test_fetch_of_nothing_held_is_delivered_empty_not_failed():
    a, b = _pair()
    with a, b:
        fetched = b.finish(b.backend.fetch(extent(b.units.values())))
        assert fetched == Delivered(frozenset())
        assert b.backend.counters.fetch_misses == 3 and b.backend.counters.failed_attempts == 0


def test_republish_merges_present_units_without_rewriting():
    a, b = _pair()
    with a, b:
        _seed(a, {(0, 0): 1})
        first = a.units[(0, 0)]
        assert isinstance(a.finish(a.backend.publish(extent([first], name=b"p1"))), Delivered)
        stored = dict(a.store.objects)
        # The same content under the same name, plus a new unit, from a rewritten source.
        a.fill(first, 0x00)
        _seed(a, {(0, 1): 2})
        second = a.finish(a.backend.publish(extent([first, a.units[(0, 1)]], name=b"p1")))
        assert second == Delivered(frozenset({first.name, a.units[(0, 1)].name}))
        assert a.store.objects[a.key(first)] == stored[a.key(first)] == pattern(1, 64)
        assert a.store.objects[a.key(a.units[(0, 1)])] == pattern(2, 64)
        puts = [args[0] for m, args in a.store.calls if m == "put"]
        assert puts[-1] == (a.key(a.units[(0, 1)]),)  # the present unit was not put again
        assert a.backend.counters.publish_present == 1
        assert a.backend.counters.publish_stored == 2


def test_publish_where_the_store_declines_a_unit_is_not_served_not_failed():
    # A declined put with a clean lookup afterwards is "not taken" (SPEC §5.1 inv. 2, publish
    # direction): the store simply does not hold it, and the caller learns so from ``served``.
    with make_rank() as rank:
        a, b = rank.unit(0, 0, 8), rank.unit(0, 1, 8)
        rank.store.put = lambda keys, buffers: [PutStatus.DECLINED for _ in keys]
        outcome = rank.finish(rank.backend.publish(extent([a, b])))
        assert outcome == Delivered(frozenset())
        assert rank.backend.counters.publish_stored == 0
        assert rank.backend.counters.failed_attempts == 0
        assert rank.store.objects == {}


def test_publish_where_a_unit_fails_to_write_is_failed():
    # FAILED is not DECLINED: a unit the store could not write fails the attempt, and the store
    # is not asked whether it holds it. Units stored in the same call still count as stored.
    with make_rank() as rank:
        a, b = rank.unit(0, 0, 8), rank.unit(0, 1, 8)
        rank.store.put = lambda keys, buffers: [PutStatus.STORED, PutStatus.FAILED]
        outcome = rank.finish(rank.backend.publish(extent([a, b])))
        assert isinstance(outcome, Failed) and "1 of 2 units could not be written" in outcome.reason
        assert rank.store.count("holds") == 1  # the lookup before the put; none after
        assert rank.backend.counters.publish_stored == 1
        assert rank.backend.counters.failed_attempts == 1


def test_publish_whose_lookup_fails_is_failed():
    """A lookup the store cannot answer is a failed lookup, not "absent": the publish is
    Failed and nothing is put (the contract forbids reading a failure as a miss, §5.2)."""
    with make_rank() as rank:
        u = rank.unit(0, 0, 8)
        rank.write(u, pattern(1, 8))
        rank.store.fail_next("holds", BlobStoreError("down"))
        outcome = rank.finish(rank.backend.publish(extent([u])))
        assert isinstance(outcome, Failed) and "lookup failed" in outcome.reason
        assert rank.store.count("put") == 0
        assert rank.store.objects == {}
        assert rank.backend.counters.publish_stored == 0
        assert rank.backend.counters.failed_attempts == 1


def test_declined_put_for_a_unit_another_publisher_made_present_counts_as_taken():
    # A race: the store refuses our write because a concurrent publisher got there first. The
    # unit is held under its name, so it is served; the bytes are the same by construction.
    a, b = _pair()
    with a, b:
        _seed(a, {(0, 0): 1, (0, 1): 2})
        _seed(b, {(0, 0): 1, (0, 1): 2})
        first, second = a.units[(0, 0)], a.units[(0, 1)]
        orig = a.store.put

        def raced_put(keys, buffers):
            # ``b`` publishes ``first`` in between our lookup and our put; the store then
            # declines our write of it and takes the other. One-shot: the store is shared, so
            # ``b``'s own put must see the real method again.
            a.store.put = orig
            assert isinstance(b.finish(b.backend.publish(extent([b.units[(0, 0)]]))), Delivered)
            results = list(orig(keys, buffers))
            results[keys.index(a.key(first))] = PutStatus.DECLINED
            return results

        a.store.put = raced_put
        outcome = a.finish(a.backend.publish(extent([first, second])))
        assert outcome == Delivered(frozenset({first.name, second.name}))
        assert a.backend.counters.publish_stored == 1  # ``second``, written by us
        assert a.backend.counters.publish_raced == 1  # ``first``, held thanks to ``b``
        assert a.backend.counters.failed_attempts == 0
        assert a.store.objects[a.key(first)] == pattern(1, 64)


def test_declined_put_then_lookup_trouble_is_failed():
    with make_rank() as rank:
        u = rank.unit(0, 0, 8)
        rank.store.put = lambda keys, buffers: [PutStatus.DECLINED for _ in keys]
        # The lookup after a declined put raises something unforeseen ...
        rank.store.fail_at("holds", 2)
        assert isinstance(rank.finish(rank.backend.publish(extent([u]))), Failed)
        # ... or fails as a store lookup does (the second ``holds``: the one after the put) ...
        rank.store.fail_at("holds", 2, BlobStoreError("down"))
        outcome = rank.finish(rank.backend.publish(extent([u])))
        assert (
            isinstance(outcome, Failed) and "lookup failed after a declined put" in outcome.reason
        )
        # ... or answers the wrong number of keys.
        orig = rank.store.holds
        calls = []

        def short_after_decline(keys):
            calls.append(keys)
            return orig(keys) if len(calls) == 1 else []

        rank.store.holds = short_after_decline
        assert isinstance(rank.finish(rank.backend.publish(extent([u]))), Failed)
        assert rank.backend.counters.publish_stored == 0
        assert rank.backend.counters.failed_attempts == 3


def test_put_answering_the_wrong_count_is_failed():
    with make_rank() as rank:
        u = rank.unit(0, 0, 8)
        rank.store.put = lambda keys, buffers: []
        outcome = rank.finish(rank.backend.publish(extent([u])))
        assert isinstance(outcome, Failed) and "answered 0 of 1" in outcome.reason


# ---- miss versus failure (§5.2) ----


def test_lookup_outage_is_failed_not_a_miss():
    a, b = _pair()
    with a, b:
        b.store.fail_next("holds")
        outcome = b.finish(b.backend.fetch(extent(b.units.values())))
        assert isinstance(outcome, Failed) and "RuntimeError" in outcome.reason
        assert b.backend.counters.fetch_misses == 0
        assert b.backend.counters.failed_attempts == 1
        # A lookup the store reports as failed is the same outage, named as such.
        b.store.fail_next("holds", BlobStoreError("down"))
        outcome = b.finish(b.backend.fetch(extent(b.units.values())))
        assert isinstance(outcome, Failed) and "lookup failed" in outcome.reason


def test_lookup_answering_the_wrong_count_is_failed():
    with make_rank() as rank:
        u = rank.unit(0, 0, 8)
        rank.store.holds = lambda keys: []
        assert isinstance(rank.finish(rank.backend.fetch(extent([u]))), Failed)
        assert isinstance(rank.finish(rank.backend.publish(extent([u]))), Failed)


def test_present_at_lookup_but_unreadable_is_failed():
    a, b = _pair()
    with a, b:
        _seed(a, {(0, 0): 1, (1, 0): 3})
        held = [a.units[(0, 0)], a.units[(1, 0)]]
        assert isinstance(a.finish(a.backend.publish(extent(held))), Delivered)
        b.store.fail_next("get")
        outcome = b.finish(b.backend.fetch(extent(b.units.values())))
        assert isinstance(outcome, Failed)
        # And when the get answers FAILED for one present unit (unreadable), the whole attempt
        # fails rather than serving fewer.
        orig = b.store.get

        def one_bad(keys, buffers):
            results = list(orig(keys, buffers))
            results[0] = GetStatus.FAILED
            return results

        b.store.get = one_bad
        outcome = b.finish(b.backend.fetch(extent(b.units.values())))
        assert isinstance(outcome, Failed) and "1 of 2 present units" in outcome.reason
        assert b.backend.counters.failed_attempts == 2


def test_unit_gone_between_lookup_and_get_is_a_miss_with_destination_untouched():
    # ``holds`` said yes, the get says MISS. The store wrote nothing, so this is a miss (SPEC
    # §5.1 inv. 2), counted as one, and the other present unit is still served.
    a, b = _pair()
    with a, b:
        _seed(a, {(0, 0): 1, (1, 0): 3})
        held = [a.units[(0, 0)], a.units[(1, 0)]]
        assert isinstance(a.finish(a.backend.publish(extent(held))), Delivered)
        gone, kept = b.units[(0, 0)], b.units[(1, 0)]
        for unit in b.units.values():
            b.fill(unit, 0xEE)
        orig = b.store.get

        def evict_then_get(keys, buffers):
            b.store.evict(b.key(gone))
            return orig(keys, buffers)

        b.store.get = evict_then_get
        outcome = b.finish(b.backend.fetch(extent(b.units.values())))
        assert outcome == Delivered(frozenset({kept.name}))
        assert b.read(gone) == bytes([0xEE]) * 64
        assert b.read(kept) == pattern(3, 16)
        assert b.backend.counters.fetch_misses == 2  # never published + gone
        assert b.backend.counters.fetch_hits == 1
        assert b.backend.counters.failed_attempts == 0


def test_failed_read_of_a_present_unit_is_failed():
    a, b = _pair()
    with a, b:
        _seed(a, {(0, 0): 1})
        assert isinstance(a.finish(a.backend.publish(extent([a.units[(0, 0)]]))), Delivered)
        b.store.get = lambda keys, buffers: [GetStatus.FAILED for _ in keys]
        outcome = b.finish(b.backend.fetch(extent([b.units[(0, 0)]])))
        assert isinstance(outcome, Failed) and "could not be read" in outcome.reason
        assert b.backend.counters.fetch_misses == 0


def test_get_answering_the_wrong_count_is_failed():
    a, b = _pair()
    with a, b:
        _seed(a, {(0, 0): 1})
        assert isinstance(a.finish(a.backend.publish(extent([a.units[(0, 0)]]))), Delivered)
        b.store.get = lambda keys, buffers: []
        assert isinstance(b.finish(b.backend.fetch(extent([b.units[(0, 0)]]))), Failed)


def test_short_read_is_failed_not_served():
    # A name that matches but a destination another size: SPEC §5.2 inv. 1b.
    store = FakeBlobStore()
    a = make_rank(store)
    b = make_rank(store)
    with a, b:
        ua = a.unit(0, 0, 32)
        ub = b.unit(0, 0, 64)  # same name, larger unit on the fetching side
        a.write(ua, pattern(1, 32))
        assert isinstance(a.finish(a.backend.publish(extent([ua]))), Delivered)
        outcome = b.finish(b.backend.fetch(extent([ub])))
        assert isinstance(outcome, Failed)
    store2 = FakeBlobStore()
    a = make_rank(store2)
    b = make_rank(store2)
    with a, b:
        ua = a.unit(0, 0, 64)
        ub = b.unit(0, 0, 32)  # smaller destination: the object does not fit
        a.write(ua, pattern(1, 64))
        assert isinstance(a.finish(a.backend.publish(extent([ua]))), Delivered)
        b.fill(ub, 0xEE)
        outcome = b.finish(b.backend.fetch(extent([ub])))
        assert isinstance(outcome, Failed)
        assert b.read(ub) == bytes([0xEE]) * 32


def test_failure_in_a_later_batch_fails_the_whole_attempt():
    # Earlier batches may already have written; SPEC §5.2 inv. 3 says the attempt is then Failed,
    # not Delivered with a subset.
    a, b = _pair(transfer_batch_size=1)
    with a, b:
        _seed(a, {(0, 0): 1, (0, 1): 2, (1, 0): 3})
        assert isinstance(a.finish(a.backend.publish(extent(a.units.values()))), Delivered)
        orig = b.store.holds
        seen = []

        def second_lookup_fails(keys):
            seen.append(keys)
            if len(seen) == 2:
                raise RuntimeError("store went away")
            return orig(keys)

        b.store.holds = second_lookup_fails
        outcome = b.finish(b.backend.fetch(extent(b.units.values())))
        assert isinstance(outcome, Failed)
        assert b.store.count("get") == 1  # first batch was written


# ---- probe ----


def test_probe_answers_none_then_the_set_once_then_asks_again():
    a, b = _pair()
    with a, b:
        _seed(a, {(0, 0): 1, (1, 0): 3})
        held = [a.units[(0, 0)], a.units[(1, 0)]]
        assert isinstance(a.finish(a.backend.publish(extent(held))), Delivered)
        names = [u.name for u in b.units.values()]
        before = b.store.count("holds")  # the publisher's own lookup is on it too
        assert b.backend.probe(b"n", names) is None
        wait_until(lambda: b.backend.counters.probe_hits + b.backend.counters.probe_misses == 3)
        lookups = b.store.count("holds")
        assert lookups == before + 1
        answer = b.backend.probe(b"n", names)
        assert answer == frozenset(u.name for u in held)
        assert isinstance(answer, frozenset)
        # Consumed: the next call asks the store again.
        assert b.backend.probe(b"n", names) is None
        wait_until(lambda: b.store.count("holds") == lookups + 1)
        # A different unit list under the same name is a different question.
        assert b.backend.probe(b"n", names[:1]) is None
        assert b.backend.probe(b"", []) == frozenset()


def test_probe_repeated_while_pending_does_not_requeue():
    with make_rank() as rank:
        u = rank.unit(0, 0, 8)
        rank.store.block("holds")
        assert rank.backend.probe(b"n", [u.name]) is None
        rank.store.wait_entered(2)
        for _ in range(5):
            assert rank.backend.probe(b"n", [u.name]) is None
        assert rank.store.count("holds") == 1
        rank.store.unblock()
        wait_until(lambda: rank.backend.counters.probe_misses == 1, what="lookup")
        assert rank.backend.probe(b"n", [u.name]) == frozenset()


def test_probe_answer_expires_after_ttl():
    with make_rank(probe_ttl_s=0.05) as rank:
        u = rank.unit(0, 0, 8)
        assert rank.backend.probe(b"n", [u.name]) is None
        wait_until(lambda: rank.backend.counters.probe_misses == 1)
        time.sleep(0.1)
        # Expired: the stale answer is dropped and a fresh lookup is queued.
        assert rank.backend.probe(b"n", [u.name]) is None
        wait_until(lambda: rank.store.count("holds") == 2, what="second lookup")
        wait_until(lambda: rank.backend.counters.probe_misses == 2)
        assert rank.backend.probe(b"n", [u.name]) == frozenset()


def test_probe_lookup_failure_raises_once_and_never_reads_as_empty():
    with make_rank() as rank:
        u = rank.unit(0, 0, 8)
        rank.store.fail_next("holds")
        assert rank.backend.probe(b"n", [u.name]) is None
        wait_until(lambda: rank.store.count("holds") == 1)
        # The worker has to have recorded the error; poll until the probe is decided.
        deadline = time.monotonic() + 5
        while True:
            try:
                answer = rank.backend.probe(b"n", [u.name])
            except RuntimeError as exc:
                assert "lookup failed" in str(exc) and exc.__cause__ is not None
                break
            assert answer is None, f"an outage answered {answer!r}"
            assert time.monotonic() < deadline
            time.sleep(0.002)
        # Forgotten: the next call asks again and, the store being back, answers normally.
        assert rank.backend.probe(b"n", [u.name]) is None
        wait_until(lambda: rank.backend.counters.probe_misses == 1)
        assert rank.backend.probe(b"n", [u.name]) == frozenset()


def test_probe_lookup_the_store_reports_failed_is_an_outage_not_an_empty_answer():
    with make_rank() as rank:
        u = rank.unit(0, 0, 8)

        def down(keys):
            raise BlobStoreError("down")

        rank.store.holds = down
        assert rank.backend.probe(b"n", [u.name]) is None
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                answer = rank.backend.probe(b"n", [u.name])
            except RuntimeError:
                break
            assert answer is None
            time.sleep(0.002)
        else:
            pytest.fail("outage never raised")
        assert rank.backend.counters.probe_misses == 0


def test_probe_is_answered_while_every_delivery_worker_is_blocked():
    # Lookups run on their own thread, so a probe is not queued behind deliveries in flight.
    with make_rank(num_workers=2) as rank:
        a, b, c = rank.unit(0, 0, 8), rank.unit(0, 1, 8), rank.unit(0, 2, 8)
        assert isinstance(rank.finish(rank.backend.publish(extent([a]))), Delivered)
        rank.store.block("put")
        busy = [rank.backend.publish(extent([b])), rank.backend.publish(extent([c]))]
        # One put already happened for ``a``; both workers are parked once two more have entered.
        wait_until(lambda: rank.store.count("put") == 3, what="workers")
        assert rank.backend.probe(b"n", [a.name, b.name]) is None
        deadline = time.monotonic() + 5
        answer = None
        while answer is None and time.monotonic() < deadline:
            answer = rank.backend.probe(b"n", [a.name, b.name])
            time.sleep(0.002)
        assert answer == frozenset({a.name})  # answered with both workers still blocked
        assert all(attempt.poll() is None for attempt in busy)
        rank.store.unblock()
        for attempt in busy:
            assert isinstance(rank.finish(attempt), Delivered)


def test_probe_answer_never_names_units_it_was_not_asked_about():
    with make_rank() as rank:
        a, b = rank.unit(0, 0, 8), rank.unit(0, 1, 8)
        assert isinstance(rank.finish(rank.backend.publish(extent([a, b]))), Delivered)
        assert rank.backend.probe(b"n", [a.name]) is None
        wait_until(lambda: rank.backend.counters.probe_hits == 1)
        assert rank.backend.probe(b"n", [a.name]) == frozenset({a.name})
        assert rank.backend.probe(b"n", [b"never-published"]) is None
        wait_until(lambda: rank.backend.counters.probe_misses == 1)
        assert rank.backend.probe(b"n", [b"never-published"]) == frozenset()


def test_probe_keys_units_like_fetch_does():
    with make_rank() as rank:
        u = rank.unit(0, 0, 8)
        assert rank.backend.probe(b"n", [u.name]) is None
        wait_until(lambda: rank.store.count("holds") == 1)
        (asked,) = [args[0] for m, args in rank.store.calls if m == "holds"]
        assert asked == (rank.key(u),)


# ---- is_last is never read ----


class _ExtentThatForbidsIsLast:
    """Looks like a ``CacheExtent`` to a store; reading ``is_last`` is a contract breach."""

    def __init__(self, units):
        self.name = b"n"
        self.units = tuple(units)

    @property
    def is_last(self):
        raise AssertionError("a store read CacheExtent.is_last")


def test_store_never_reads_is_last_on_either_path():
    a, b = _pair()
    with a, b:
        _seed(a, {(0, 0): 1})
        published = a.finish(a.backend.publish(_ExtentThatForbidsIsLast([a.units[(0, 0)]])))
        assert published == Delivered(frozenset({a.units[(0, 0)].name}))
        fetched = b.finish(b.backend.fetch(_ExtentThatForbidsIsLast(b.units.values())))
        assert fetched == Delivered(frozenset({b.units[(0, 0)].name}))
        assert b.read(b.units[(0, 0)]) == pattern(1, 64)
        # Failure paths do not read it either.
        assert isinstance(
            a.backend.publish(
                _ExtentThatForbidsIsLast([Unit(name=b"?", local_group=9, local=9)])
            ).poll(),
            Failed,
        )


# ---- the probe table is bounded and forgets ----


def test_probe_table_is_bounded_and_a_new_lookup_past_the_bound_raises():
    from disaggregation.backends.blob.backend import MAX_PROBES

    with make_rank() as rank:
        rank.store.block("holds")  # every lookup stays pending
        for i in range(MAX_PROBES):
            assert rank.backend.probe(f"n{i}".encode(), [b"u"]) is None
        with pytest.raises(RuntimeError, match=f"{MAX_PROBES} store lookups already remembered"):
            rank.backend.probe(b"one-too-many", [b"u"])
        rank.store.unblock()


def test_pending_probe_past_the_ttl_is_dropped_and_asked_again():
    with make_rank(probe_ttl_s=0.05) as rank:
        rank.store.block("holds")
        assert rank.backend.probe(b"n", [b"u"]) is None
        rank.store.wait_entered(2)  # register_span, then the lookup parked at the gate
        time.sleep(0.1)  # past the TTL while still pending
        # The stale pending entry is dropped and the same question is asked afresh.
        assert rank.backend.probe(b"n", [b"u"]) is None
        rank.store.unblock()
        wait_until(lambda: rank.store.count("holds") == 2, what="second lookup")
        # The fresh lookup answers; an answer is consumed once.
        wait_until(lambda: rank.backend.probe(b"n", [b"u"]) is not None, what="answer")
        assert rank.backend.probe(b"n", [b"u"]) is None  # consumed: asked again


def test_answered_probe_past_the_ttl_is_forgotten():
    with make_rank(probe_ttl_s=0.05) as rank:
        u = rank.unit(0, 0, 8)
        rank.write(u, pattern(1, 8))
        assert isinstance(rank.finish(rank.backend.publish(extent([u]))), Delivered)
        assert rank.backend.probe(b"n", [u.name]) is None
        wait_until(lambda: rank.backend.counters.probe_hits == 1, what="lookup")
        time.sleep(0.1)  # nobody collected the answer
        assert rank.backend.probe(b"n", [u.name]) is None  # forgotten: a new lookup starts
        wait_until(lambda: rank.backend.counters.probe_hits == 2, what="second lookup")
        assert rank.backend.probe(b"n", [u.name]) == frozenset({u.name})

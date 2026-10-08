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
"""Staging sizing and the slot queue alone, over plain layouts, the usable end readiness finds and
how it and fetch ends keep out of runs of multimodal tokens, with the runtime's own life cycles
behind a stand-in manager, and the bytes the test kit stages: no real manager, no memory, no GPU."""

import hashlib
import random
import weakref
from types import SimpleNamespace

import numpy as np
import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import StagingOptions, _lender
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing._slots import (
    Runs,
    Slots,
    fetch_rows,
    slot_counts,
)
from tensorrt_llm.runtime.kv_cache_manager_v2 import AttnLifeCycle

pytestmark = pytest.mark.cpu_only

MiB = 1 << 20
GiB = 1 << 30


def layout(tokens_per_block, groups):
    """What the sizing reads of a manager's layout, for layer groups ``(window, sink blocks, pool
    group, page bytes)``."""
    return SimpleNamespace(
        tokens_per_block=tokens_per_block,
        windows=tuple(window for window, _, _, _ in groups),
        sink_blocks=tuple(sinks for _, sinks, _, _ in groups),
        pool_group_of=tuple(g for _, _, g, _ in groups),
        page_bytes={g: page for _, _, g, page in groups},
    )


def one_fetch(lay, fetch_tokens):
    return sum(rows * lay.page_bytes[g] for g, rows in fetch_rows(lay, fetch_tokens).items())


# TinyLlama-1.1B, TP=1, 32 tokens per block: 22 layers of K and V, 4 heads of 64 bf16 each.
TINYLLAMA_PAGE = 22 * 2 * 4 * 64 * 2 * 32
TINYLLAMA = layout(32, [(None, 0, 0, TINYLLAMA_PAGE)])

# DeepSeek-V4-Pro as its cache manager lays it out on every rank, 128 tokens per block: a
# 128-token window, compressed full attention, and the compressors' state with an 8-token window.
DSV4_PAGES = {1: 20250624, 0: 572672, 2: 39321600}
DSV4_PRO = layout(
    128, [(128, 0, 1, DSV4_PAGES[1]), (None, 0, 0, DSV4_PAGES[0]), (8, 0, 2, DSV4_PAGES[2])]
)


# -- sizing -----------------------------------------------------------------------------------


def test_tinyllama_stages_eight_whole_prompts():
    assert fetch_rows(TINYLLAMA, 2048) == {0: 64}
    assert one_fetch(TINYLLAMA, 2048) == 44 * MiB
    assert slot_counts(TINYLLAMA, StagingOptions(2048, max_fetches=8)) == {0: 8 * 64}


def test_deepseek_v4_windows_take_their_window_and_the_full_group_the_range():
    assert fetch_rows(DSV4_PRO, 4096) == {1: 1, 0: 32, 2: 1}
    assert 74 * MiB < one_fetch(DSV4_PRO, 4096) < 75 * MiB
    assert slot_counts(DSV4_PRO, StagingOptions(4096, max_fetches=8)) == {1: 8, 0: 256, 2: 8}


def test_deepseek_v4_long_ranges_capped_at_one_fetch_keep_one_fetch():
    one = one_fetch(DSV4_PRO, 1 << 20)
    assert fetch_rows(DSV4_PRO, 1 << 20) == {1: 1, 0: 8192, 2: 1}
    assert 4 * GiB < one < 5 * GiB
    options = StagingOptions(1 << 20, max_fetches=8, max_bytes=one)
    assert slot_counts(DSV4_PRO, options) == {1: 1, 0: 8192, 2: 1}
    # A cap between one and eight fetches: every group keeps its rows of one fetch and more.
    cap = 960 * GiB // 8 // 4
    counts = slot_counts(DSV4_PRO, StagingOptions(1 << 20, max_fetches=8, max_bytes=cap))
    assert counts[0] >= 8192 and counts[1] >= 1 and counts[2] >= 1
    assert sum(counts[g] * DSV4_PAGES[g] for g in counts) <= cap


def test_short_ranges_take_no_more_than_their_blocks_even_in_a_window():
    assert fetch_rows(DSV4_PRO, 100) == {1: 1, 0: 1, 2: 1}
    assert slot_counts(DSV4_PRO, StagingOptions(100)) == {1: 1, 0: 1, 2: 1}


def test_a_window_wider_than_the_range_takes_the_range_blocks():
    lay = layout(32, [(1024, 0, 0, 100)])  # a window of 32 blocks over a range of 8
    assert fetch_rows(lay, 256) == {0: 8}
    assert slot_counts(lay, StagingOptions(256, max_fetches=2)) == {0: 16}


def test_sink_blocks_count_with_the_window():
    lay = layout(32, [(64, 1, 0, 100), (None, 0, 1, 10)])
    assert fetch_rows(lay, 320) == {0: 1 + 2, 1: 10}
    assert one_fetch(lay, 320) == 3 * 100 + 10 * 10
    assert slot_counts(lay, StagingOptions(320, max_fetches=3)) == {0: 9, 1: 30}


def test_layer_groups_sharing_a_pool_group_add_up():
    lay = layout(16, [(None, 0, 0, 8), (32, 0, 0, 8)])
    assert fetch_rows(lay, 160) == {0: 10 + 2}


def test_a_partial_block_counts_as_a_whole_one():
    assert fetch_rows(TINYLLAMA, 33) == {0: 2}
    assert fetch_rows(TINYLLAMA, 1) == {0: 1}


@pytest.mark.parametrize("fetch_tokens", [0, -32])
def test_fetch_tokens_must_be_positive(fetch_tokens):
    with pytest.raises(ValueError):
        fetch_rows(TINYLLAMA, fetch_tokens)


def test_max_bytes_below_one_fetch_is_rejected():
    one = one_fetch(DSV4_PRO, 4096)
    with pytest.raises(ValueError, match="one fetch"):
        slot_counts(DSV4_PRO, StagingOptions(4096, max_bytes=one - 1))
    assert slot_counts(DSV4_PRO, StagingOptions(4096, max_bytes=one)) == fetch_rows(DSV4_PRO, 4096)


def test_max_bytes_above_max_fetches_changes_nothing():
    options = StagingOptions(4096, max_fetches=2, max_bytes=100 * one_fetch(DSV4_PRO, 4096))
    assert slot_counts(DSV4_PRO, options) == {1: 2, 0: 64, 2: 2}


def random_layout(rng):
    """A layout with up to four layer groups over up to three pool groups, and its life cycles."""
    tpb = rng.choice([4, 8, 16, 32])
    pages = {g: rng.randint(1, 1000) for g in range(rng.randint(1, 3))}
    groups, life_cycles = [], []
    for _ in range(rng.randint(1, 4)):
        window = rng.choice([None, rng.randint(1, 6 * tpb)])
        sink_tokens = 0 if window is None else rng.choice([0, 0, rng.randint(1, 3 * tpb)])
        life_cycle = AttnLifeCycle.make(window, sink_tokens, tpb)
        g = rng.choice(sorted(pages))
        groups.append((window, int(life_cycle.num_sink_blocks), g, pages[g]))
        life_cycles.append(life_cycle)
    return layout(tpb, groups), life_cycles


@pytest.mark.parametrize("seed", range(40))
def test_any_whole_block_range_of_at_most_fetch_tokens_fits_one_fetch(seed):
    # Rows a range [start, end) needs: its blocks that a history of end still reads, as the
    # runtime's own life cycle says; the longest range ending at end needs the most.
    rng = random.Random(seed)
    lay, life_cycles = random_layout(rng)
    tpb = lay.tokens_per_block
    fetch_tokens = rng.randint(1, 12 * tpb)
    bound = fetch_rows(lay, fetch_tokens)
    span = fetch_tokens // tpb
    for end_block in range(1, 40):
        end = end_block * tpb
        need = dict.fromkeys(bound, 0)
        for lg, life_cycle in enumerate(life_cycles):
            stale = life_cycle.get_stale_range(end, tpb)
            blocks = range(max(0, end_block - span), end_block)
            need[lay.pool_group_of[lg]] += sum(not stale.beg <= b < stale.end for b in blocks)
        assert all(need[g] <= bound[g] for g in bound), (end, need, bound)


@pytest.mark.parametrize("seed", range(40))
def test_counts_are_max_fetches_fetches_or_the_cap_rounded_down(seed):
    rng = random.Random(seed)
    lay, _ = random_layout(rng)
    fetch_tokens = rng.randint(1, 12 * lay.tokens_per_block)
    rows = fetch_rows(lay, fetch_tokens)
    one = one_fetch(lay, fetch_tokens)
    max_fetches = rng.randint(1, 9)
    assert slot_counts(lay, StagingOptions(fetch_tokens, max_fetches)) == {
        g: max_fetches * rows[g] for g in rows
    }
    cap = rng.randint(one, max_fetches * one)
    counts = slot_counts(lay, StagingOptions(fetch_tokens, max_fetches, max_bytes=cap))
    assert sum(counts[g] * lay.page_bytes[g] for g in counts) <= cap
    # As many whole fetches as the cap holds hold slots at once.
    whole = cap // one
    assert all(whole * rows[g] <= counts[g] <= max_fetches * rows[g] for g in rows)


@pytest.mark.parametrize("max_bytes_in_fetches", [None, 1, 2.5])
def test_max_fetches_ranges_hold_slots_at_once_and_one_more_waits(max_bytes_in_fetches):
    rows = fetch_rows(DSV4_PRO, 4096)
    max_bytes = None
    at_once = 3
    if max_bytes_in_fetches is not None:
        max_bytes = int(max_bytes_in_fetches * one_fetch(DSV4_PRO, 4096))
        at_once = int(max_bytes_in_fetches)
    slots = Slots(slot_counts(DSV4_PRO, StagingOptions(4096, max_fetches=3, max_bytes=max_bytes)))
    for _ in range(at_once):
        assert slots.take(slots.ask(rows)) is not None
    assert slots.take(slots.ask(rows)) is None
    assert slots.num_waiting == 1


# -- the slot queue ---------------------------------------------------------------------------


def take_now(slots, counts):
    ticket = slots.ask(counts)
    runs = slots.take(ticket)
    assert runs is not None, f"{counts} did not fit"
    return runs


def test_slots_of_one_lease_are_contiguous_per_group():
    slots = Slots({0: 8, 1: 8})
    assert (slots.num_slots(0), slots.free_slots(1)) == (8, 8)
    runs = take_now(slots, {0: 3, 1: 5})
    assert runs.slots(0).tolist() == [0, 1, 2]
    assert runs.slots(1).tolist() == [0, 1, 2, 3, 4]
    assert runs.slots(7).tolist() == []
    assert runs.slots(0).dtype.name == "int64"


def test_a_lease_is_all_or_nothing_across_groups():
    slots = Slots({0: 8, 1: 4})
    held = take_now(slots, {1: 3})
    ticket = slots.ask({0: 2, 1: 2})

    assert slots.take(ticket) is None
    # Group 0 had room, but nothing was taken from it.
    assert slots.free_slots(0) == 8
    assert slots.free_slots(1) == 1

    slots.give(held)
    assert slots.take(ticket) is not None
    assert (slots.free_slots(0), slots.free_slots(1)) == (6, 2)


def test_contiguity_waits_for_a_run_even_when_enough_slots_are_free():
    slots = Slots({0: 6})
    a, b, c = (take_now(slots, {0: 2}) for _ in range(3))
    slots.give(a)
    slots.give(c)
    assert slots.free_slots(0) == 4

    ticket = slots.ask({0: 3})
    assert slots.take(ticket) is None, "two free runs of 2 are not a run of 3"

    slots.give(b)
    assert slots.take(ticket).slots(0).tolist() == [0, 1, 2]


def test_waiting_leases_are_served_strictly_in_order():
    slots = Slots({0: 4})
    held = take_now(slots, {0: 3})
    first = slots.ask({0: 2})
    second = slots.ask({0: 1})

    # One slot is free and ``second`` would fit, but ``first`` asked earlier.
    assert slots.take(second) is None
    assert slots.take(first) is None
    assert slots.num_waiting == 2

    slots.give(held)
    assert slots.take(second) is None, "still behind the first"
    assert slots.take(first).slots(0).tolist() == [0, 1]
    assert slots.take(second).slots(0).tolist() == [2]
    assert slots.num_waiting == 0


def test_cancelling_the_head_lets_the_next_lease_through():
    slots = Slots({0: 4})
    held = take_now(slots, {0: 3})
    first = slots.ask({0: 4})
    second = slots.ask({0: 1})
    assert slots.take(second) is None

    slots.cancel(first)
    assert slots.take(second).slots(0).tolist() == [3]
    slots.give(held)
    # Cancelling an unknown or granted ticket does nothing.
    slots.cancel(first)
    slots.cancel(12345)
    assert (slots.free_slots(0), slots.num_waiting) == (3, 0)


def test_a_ticket_that_is_not_waiting_is_a_key_error():
    slots = Slots({0: 8})
    with pytest.raises(KeyError):
        slots.take(12345)
    ticket = slots.ask({0: 1})
    assert slots.take(ticket) is not None
    with pytest.raises(KeyError):
        slots.take(ticket)


def test_frees_coalesce_into_one_run():
    slots = Slots({0: 8})
    held = [take_now(slots, {0: 2}) for _ in range(4)]
    for runs in (held[1], held[3], held[0], held[2]):
        slots.give(runs)
    assert slots.free_slots(0) == 8
    # Only a single coalesced run can serve the whole group at once.
    assert take_now(slots, {0: 8}).slots(0).tolist() == list(range(8))


def test_a_double_free_is_rejected_and_changes_nothing():
    slots = Slots({0: 4, 1: 4})
    runs = take_now(slots, {0: 2, 1: 2})
    slots.give(Runs({1: (0, 2)}))
    assert (slots.free_slots(0), slots.free_slots(1)) == (2, 4)

    with pytest.raises(ValueError, match="freed twice"):
        slots.give(runs)
    # Group 0's run was not returned on the way to finding group 1's double free.
    assert (slots.free_slots(0), slots.free_slots(1)) == (2, 4)


@pytest.mark.parametrize("run", [(3, 2), (-1, 1)], ids=["past_the_end", "negative"])
def test_freeing_slots_outside_the_group_is_rejected(run):
    slots = Slots({0: 4})
    take_now(slots, {0: 4})
    with pytest.raises(ValueError, match="outside"):
        slots.give(Runs({0: run}))
    assert slots.free_slots(0) == 0


def test_freeing_slots_of_an_unknown_group_is_rejected():
    slots = Slots({0: 4})
    take_now(slots, {0: 4})
    with pytest.raises(ValueError, match="pool group 5"):
        slots.give(Runs({0: (0, 4), 5: (0, 1)}))
    assert slots.free_slots(0) == 0


@pytest.mark.parametrize(
    "counts", [{0: 5}, {3: 1}, {0: -1}], ids=["too_many", "unknown", "negative"]
)
def test_a_request_that_can_never_be_granted_is_rejected_at_once(counts):
    slots = Slots({0: 4})
    with pytest.raises(ValueError):
        slots.check(counts)
    with pytest.raises(ValueError):
        slots.ask(counts)
    assert slots.num_waiting == 0
    assert slots.free_slots(0) == 4


def test_check_queues_nothing():
    slots = Slots({0: 4})
    slots.check({0: 4})
    assert slots.num_waiting == 0
    # A later ask is first in line.
    assert take_now(slots, {0: 4}).slots(0).tolist() == [0, 1, 2, 3]


def test_zero_counts_need_no_slots():
    slots = Slots({0: 1})
    take_now(slots, {0: 1})
    assert take_now(slots, {0: 0, 9: 0}).runs == {}


class _Owner:
    """What a stand-in lender's weak references point at."""


_OWNER = _Owner()


def staging_over(windows, sinks, tpb):
    """A staging lender and a manager stand-in answering what readiness asks: the tokens per block
    and each layer group's stale range, from the runtime's own life cycle."""
    cycles = [AttnLifeCycle.make(w, s * tpb, tpb) for w, s in zip(windows, sinks)]

    def stale_block_range(lg, history):
        stale = cycles[lg].get_stale_range(history, tpb)
        return stale.beg, stale.end

    manager = SimpleNamespace(tokens_per_block=tpb, _stale_block_range=stale_block_range)
    lay = SimpleNamespace(
        tokens_per_block=tpb, windows=tuple(windows), pool_groups=(), num_layer_groups=len(windows)
    )
    ref = weakref.ref(_OWNER)
    return _lender.Staging(ref, lay, None, (), None, ref), manager


def reads_all(manager, windows, blocks, first, end_block, tpb):
    """Every layer group has, among the blocks from ``first`` and the delivered ones, every block a
    start at ``end_block * tpb`` reads: all of them for full attention, else all but the stale."""
    for lg, delivered in enumerate(blocks):
        beg, end = (0, 0)
        if windows[lg] is not None:
            beg, end = manager._stale_block_range(lg, end_block * tpb)
        for ordinal in range(first, end_block):
            if not beg <= ordinal < end and not (ordinal < len(delivered) and delivered[ordinal]):
                return False
    return True


def test_usable_until_is_the_largest_start_every_layer_group_reads():
    """Random layouts, delivered rows and computed prefixes: the usable end is the largest
    whole-block start past the computed prefix where every layer group has what it reads, whatever
    the lowest fetch start, else the computed prefix."""
    rng = random.Random(0)
    for _ in range(3000):
        tpb = rng.choice([16, 32])
        windows = [None if rng.random() < 0.25 else rng.randint(1, 200) for _ in range(3)]
        windows = windows[: rng.randint(1, 3)]
        sinks = [0 if w is None else rng.choice([0, 0, 1, 2]) for w in windows]
        known = rng.choice([0, rng.randint(0, 12) * tpb, rng.randint(0, 12 * tpb)])
        origin = rng.choice([known // tpb * tpb, rng.randint(0, 14) * tpb])
        length = rng.randint(0, 20)
        blocks = []
        for _ in windows:
            row = np.zeros(max(0, length + rng.choice([0, -1, 1])), dtype=bool)
            row[rng.choice([0, origin // tpb, rng.randint(0, 20)]) :] = True
            for _ in range(rng.choice([0, 0, 1, 2])):
                if len(row):
                    row[rng.randrange(len(row))] = False
            row[: origin // tpb] = False  # fetches start at origin or later
            blocks.append(row)
        lender, manager = staging_over(windows, sinks, tpb)
        first = known // tpb
        last = max((len(b) for b in blocks), default=0)
        ends = range(first + 1, last + 1)
        starts = [e for e in ends if reads_all(manager, windows, blocks, first, e, tpb)]
        want = starts[-1] * tpb if starts else known
        delivered = _lender._Delivered(origin, [b.copy() for b in blocks])
        got = lender._usable_until(manager, delivered, known)
        assert got == want, (
            f"windows {windows}, sinks {sinks}, tpb {tpb}, origin {origin}, known {known}, "
            f"delivered {[np.nonzero(b)[0].tolist() for b in blocks]}: usable until {got}, "
            f"not {want}"
        )


def test_usable_until_after_a_whole_fetch_checks_its_end_alone(monkeypatch):
    """Two windowed layer groups, nothing computed and 4096 blocks delivered in each: the end of the
    delivered rows is usable, found with one stale-range lookup per layer group, not one per block
    below it, so readiness after each lease of a split fetch costs no more as the fetch grows."""
    lender, manager = staging_over([4096, 8192], [0, 0], 32)
    blocks = [np.ones(4096, dtype=bool), np.ones(4096, dtype=bool)]
    real_stale, calls = _lender._stale, []

    def counted(*args):
        calls.append(args)
        return real_stale(*args)

    monkeypatch.setattr(_lender, "_stale", counted)
    got = lender._usable_until(manager, _lender._Delivered(0, blocks), 0)
    assert got == 4096 * 32, f"usable until {got}"
    assert len(calls) == 2, f"{len(calls)} stale-range lookups for one usable end"


def test_usable_until_checks_no_start_past_a_missing_full_attention_block(monkeypatch):
    """One full-attention layer group, nothing computed and 8192 blocks delivered but one: no start
    past the missing block has every block it reads, so none is checked, and readiness after each
    later lease of a split fetch that lost an early block checks no more starts as it grows."""
    lender, manager = staging_over([None], [0], 32)
    real_stale, calls = _lender._stale, []

    def counted(*args):
        calls.append(args)
        return real_stale(*args)

    monkeypatch.setattr(_lender, "_stale", counted)
    for hole, usable, checked in ((0, 0, 0), (4000, 4000 * 32, 1)):
        blocks = np.ones(8192, dtype=bool)
        blocks[hole] = False
        calls.clear()
        got = lender._usable_until(manager, _lender._Delivered(0, [blocks]), 0)
        assert got == usable, f"usable until {got} with block {hole} missing"
        assert len(calls) == checked, f"{len(calls)} starts checked with block {hole} missing"


# -- runs of multimodal tokens ----------------------------------------------------------------


def multimodal(runs, length, flagged=True):
    """A request stand-in whose multimodal data marks ``runs`` as tokens that attend both ways."""
    mask = np.zeros(length, dtype=np.int64)
    for beg, end in runs:
        mask[beg:end] = 1
    data = {"mm_bidirectional_blocks": flagged, "multimodal_embed_mask_cumsum": np.cumsum(mask)}
    return SimpleNamespace(py_multimodal_data=data)


def strictly_inside(runs, position):
    return any(beg < position < end for beg, end in runs)


def test_readiness_and_fetch_ends_keep_out_of_bidirectional_runs():
    """Random runs of multimodal tokens attending both ways: a fetch end strictly inside a run is
    refused and any other is not; readiness lowers its end to the largest position at or below it
    that no run holds strictly inside, and needs a floor at the end of the last run at or below
    that, so no position from that floor to that end lies strictly inside a run."""
    rng = random.Random(0)
    # With a window too, which the refusal must not read: the model's window can be narrower.
    lender, windowed = (staging_over(w, [0] * len(w), 32)[0] for w in ([None], [64, None]))
    for _ in range(3000):
        length = rng.randint(1, 400)
        runs, pos = [], rng.randint(0, 20)
        while pos < length - 1 and rng.random() < 0.7:
            stop = min(length, pos + rng.randint(1, 120))
            runs.append((pos, stop))
            pos = stop + rng.randint(1, 40)
        request = multimodal(runs, length)
        usable = rng.randint(0, length)
        got, floor = lender._outside_bidirectional_spans(request, usable)
        want = max(p for p in range(usable + 1) if not strictly_inside(runs, p))
        case = f"runs {runs}, usable {usable}"
        assert got == want, f"{case}: the end lowered to {got}, not {want}"
        assert floor == max((stop for _, stop in runs if stop <= want), default=0), (
            f"{case}: a floor of {floor}"
        )
        assert not any(strictly_inside(runs, p) for p in range(floor, got + 1)), (
            f"{case}: [{floor}, {got}] holds a position strictly inside a run"
        )
        end = rng.randint(0, length)
        for each in (lender, windowed):
            refused = each._splits_bidirectional_span(request, end) is not None
            assert refused == strictly_inside(runs, end), (
                f"{case}: an end at {end} refused: {refused}"
            )


def test_a_request_without_bidirectional_runs_keeps_its_interval_and_ends():
    """No flag, no multimodal data, or no mask: nothing lowers, rises or fails."""
    lender, _ = staging_over([64, None], [0, 0], 32)
    for request in (
        multimodal([(1, 300)], 400, flagged=False),
        SimpleNamespace(py_multimodal_data=None),
        SimpleNamespace(py_multimodal_data={"mm_bidirectional_blocks": True}),
    ):
        assert lender._outside_bidirectional_spans(request, 200) == (200, 0)
        assert lender._splits_bidirectional_span(request, 200) is None


# -- the bytes the test kit stages ------------------------------------------------------------


def test_the_pieces_of_a_staged_row_differ(kit):
    """The 32-byte pieces of a staged row are pairwise distinct, so two of them trading places
    shows in the row's bytes."""
    row = kit.staged_row(0x5A, 1, 3, 4096)
    pieces = [row[i : i + 32] for i in range(0, len(row), 32)]
    assert len(set(pieces)) == len(pieces), "a staged row repeats a piece, so a swap would not show"


def periodic_row(byte, layer_group, ordinal, nbytes):  # one digest over and over, as before
    seed = hashlib.sha256(bytes([byte, layer_group]) + int(ordinal).to_bytes(8, "little")).digest()
    return (seed * (nbytes // len(seed) + 1))[:nbytes]


def test_the_check_catches_a_staged_row_repeating_one_digest(kit, monkeypatch):
    monkeypatch.setattr(kit, "staged_row", periodic_row)
    with pytest.raises(AssertionError, match="a staged row repeats a piece"):
        test_the_pieces_of_a_staged_row_differ(kit)

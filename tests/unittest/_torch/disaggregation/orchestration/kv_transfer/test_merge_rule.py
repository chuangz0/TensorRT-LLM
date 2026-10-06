# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure-function tests of the merge rule (design §6.3), ``servable_blocks`` (the store decision
over every paged group) and their shared input ``required_ordinals``.

Synthetic model: tpb = 4, a windowed group of W = 3 blocks (12 tokens) with 1 sink block, and
7 full blocks (``token_end = 28``). Expected stale ranges are computed by hand from
``AttnLifeCycle.get_stale_range``::

    num_blocks = ceil(history / tpb); start = min(num_blocks, sink_blocks)
    end = max(start, (history + 1 - window_size) // tpb)          # windowed only

    history  num_blocks  stale     live blocks (< full = history // tpb)
      28         7       [1, 4)    {0, 4, 5, 6}
      24         6       [1, 3)    {0, 3, 4, 5}
      20         5       [1, 2)    {0, 2, 3, 4}
      16         4       [1, 1)    {0, 1, 2, 3}
      12         3       [1, 1)    {0, 1, 2}
       8         2       [1, 1)    {0, 1}
       4         1       [1, 1)    {0}
"""

import pytest

__extra_import_path__ = ["~/tensorrt_llm/_torch"]
from disaggregation.base.backend import CacheKind  # noqa: E402
from disaggregation.base.views import GroupSpec  # noqa: E402
from disaggregation.remote_cache import (  # noqa: E402
    FetchSource,
    Planner,
    _stale_range,
    merge,
    required_ordinals,
    servable_blocks,
)
from fakes import (  # noqa: E402
    TPB,
    FakeFetches,
    FakeReader,
    FakeRequest,
    full_attention,
    keys_for,
    make_plan,
    names,
    ordinals_by_group,
    plan_unit_names,
    state_group,
    windowed,
)

KEYS = keys_for("prompt", 7)
FULL = full_attention(0)
WINDOW = windowed(1, window_blocks=3, sink_blocks=1)
STATE = state_group(2)
END = 7 * TPB  # 28
NAMEABLE = 7


def held(*per_group: tuple) -> frozenset[bytes]:
    """A probe answer: ``(spec, ordinals)`` pairs, each naming the blocks the store holds."""
    return frozenset().union(*(names(spec, KEYS, ordinals) for spec, ordinals in per_group))


def store_planner(reader: FakeReader) -> Planner:
    return Planner([FetchSource("store", FakeFetches(name="store"), None)], reader, TPB)


# ---- required_ordinals mirrors get_stale_range ----


@pytest.mark.parametrize(
    "history, expected",
    [
        (28, {0, 4, 5, 6}),
        (24, {0, 3, 4, 5}),
        (20, {0, 2, 3, 4}),
        (16, {0, 1, 2, 3}),
        (12, {0, 1, 2}),
        (8, {0, 1}),
        (4, {0}),
        (0, set()),
    ],
)
def test_window_required_ordinals_match_stale_range_table(history, expected):
    assert required_ordinals(WINDOW, history, 0, TPB) == frozenset(expected)


def test_window_required_ordinals_drop_local_prefix_including_sink():
    # Design §6.2 example: local hit on blocks 0..1, so the sink block is already local.
    assert required_ordinals(WINDOW, END, 2, TPB) == frozenset({4, 5, 6})


def test_full_attention_required_ordinals_are_the_range_above_reuse():
    assert required_ordinals(FULL, END, 0, TPB) == frozenset(range(7))
    assert required_ordinals(FULL, END, 2, TPB) == frozenset({2, 3, 4, 5, 6})
    # A gen-init target that is not a block boundary still counts full blocks only.
    assert required_ordinals(FULL, 30, 0, TPB) == frozenset(range(7))


def test_state_required_ordinals_is_exact_snapshot_or_nothing():
    assert required_ordinals(STATE, END, 0, TPB) == frozenset({6})
    assert required_ordinals(STATE, END, 2, TPB) == frozenset({6})
    # Not a block boundary: the live state has no name.
    assert required_ordinals(STATE, 30, 0, TPB) == frozenset()
    # Snapshot already inside the local prefix.
    assert required_ordinals(STATE, 8, 2, TPB) == frozenset()


def test_plan_units_by_group_match_design_example():
    plan = make_plan([FULL, WINDOW, STATE], token_end=END, keys=KEYS, reuse_end=2)
    assert ordinals_by_group(plan) == {0: (2, 3, 4, 5, 6), 1: (4, 5, 6), 2: (6,)}
    assert plan_unit_names(plan) == (
        names(FULL, KEYS, [2, 3, 4, 5, 6])
        | names(WINDOW, KEYS, [4, 5, 6])
        | names(STATE, KEYS, [6])
    )
    assert WINDOW.kind is CacheKind.PAGED and STATE.kind is CacheKind.STATE


# ---- full attention only ----


def test_full_attention_all_served_lands_at_token_end():
    plan = make_plan([FULL], token_end=END, keys=KEYS, reuse_end=2)
    assert merge(plan, plan_unit_names(plan)) == END


def test_full_attention_missing_last_block_steps_down_one_boundary():
    plan = make_plan([FULL], token_end=END, keys=KEYS, reuse_end=2)
    served = plan_unit_names(plan) - names(FULL, KEYS, [6])
    assert merge(plan, served) == 24


def test_full_attention_gap_in_the_middle_stops_below_the_gap():
    plan = make_plan([FULL], token_end=END, keys=KEYS, reuse_end=2)
    served = plan_unit_names(plan) - names(FULL, KEYS, [3])
    assert merge(plan, served) == 12  # blocks 2 present, 3 missing -> B = 3 * tpb


def test_extra_served_names_are_ignored():
    plan = make_plan([FULL], token_end=END, keys=KEYS, reuse_end=2)
    served = plan_unit_names(plan) | {b"unrelated"}
    assert merge(plan, served) == END


# ---- empty and partial ----


def test_empty_served_is_the_local_prefix_end():
    plan = make_plan([FULL, WINDOW, STATE], token_end=END, keys=KEYS, reuse_end=2)
    assert merge(plan, frozenset()) == 2 * TPB


def test_empty_served_with_no_local_prefix_is_zero():
    plan = make_plan([FULL], token_end=END, keys=KEYS, reuse_end=0)
    assert merge(plan, frozenset()) == 0


# ---- window with sink ----


def test_window_all_served_lands():
    plan = make_plan([WINDOW], token_end=END, keys=KEYS, reuse_end=0)
    assert ordinals_by_group(plan) == {1: (0, 4, 5, 6)}
    assert merge(plan, plan_unit_names(plan)) == END


def test_window_missing_a_window_block_falls_to_where_only_the_sink_is_needed():
    # Missing u5: at 24 the window needs {3,4,5}, at 20 {2,3,4} ... none of which were asked for
    # (blocks 1..3 are stale at token_end), down to history 4 where only the sink block is live.
    plan = make_plan([WINDOW], token_end=END, keys=KEYS, reuse_end=0)
    served = plan_unit_names(plan) - names(WINDOW, KEYS, [5])
    assert merge(plan, served) == 4


def test_window_missing_sink_block_lands_nowhere_above_zero():
    plan = make_plan([WINDOW], token_end=END, keys=KEYS, reuse_end=0)
    served = plan_unit_names(plan) - names(WINDOW, KEYS, [0])
    assert merge(plan, served) == 0


def test_window_with_local_sink_short_window_falls_to_local_prefix():
    plan = make_plan([FULL, WINDOW], token_end=END, keys=KEYS, reuse_end=2)
    served = plan_unit_names(plan) - names(WINDOW, KEYS, [4])
    # Full attention alone would allow 28; the window group drags B to the local prefix.
    assert merge(plan, served) == 2 * TPB


def test_combined_model_missing_full_block_drops_to_the_local_prefix():
    plan = make_plan([FULL, WINDOW, STATE], token_end=END, keys=KEYS, reuse_end=2)
    served = plan_unit_names(plan) - names(FULL, KEYS, [3])
    # Full attention alone would allow 12 (block 2 present, 3 missing); at 12 the window needs
    # block 2, which was never asked for, so B drops to the local prefix.
    assert merge(plan, served) == 2 * TPB


# ---- state group ----


def test_state_requires_exact_snapshot():
    plan = make_plan([FULL, STATE], token_end=END, keys=KEYS, reuse_end=2)
    assert merge(plan, plan_unit_names(plan)) == END
    served = plan_unit_names(plan) - names(STATE, KEYS, [6])
    # Every lower boundary needs a different snapshot that was never asked for.
    assert merge(plan, served) == 2 * TPB


def test_state_snapshot_alone_does_not_rescue_a_short_full_group():
    plan = make_plan([FULL, STATE], token_end=END, keys=KEYS, reuse_end=2)
    served = plan_unit_names(plan) - names(FULL, KEYS, [6])
    assert merge(plan, served) == 2 * TPB


def test_state_only_model_lands_only_on_its_exact_snapshot():
    plan = make_plan([STATE], token_end=END, keys=KEYS, reuse_end=0)
    assert ordinals_by_group(plan) == {2: (6,)}
    assert merge(plan, plan_unit_names(plan)) == END
    assert merge(plan, frozenset()) == 0


def test_merge_never_exceeds_token_end():
    plan = make_plan([FULL], token_end=24, keys=KEYS, reuse_end=0)
    served = names(FULL, KEYS, range(7))
    assert merge(plan, served) == 24


# ---- servable_blocks: the store decision over every paged group ----

ALL = range(7)


def test_servable_blocks_lands_at_nameable_when_every_group_is_whole():
    assert (
        servable_blocks(held((FULL, ALL), (WINDOW, ALL)), KEYS, [FULL, WINDOW], NAMEABLE, TPB) == 7
    )


def test_servable_blocks_steps_down_to_where_the_missing_window_block_is_stale():
    # Window block 5 missing: e=7 needs {0,4,5,6}, e=6 needs {0,3,4,5}, e=5 needs {0,2,3,4}.
    answer = held((FULL, ALL), (WINDOW, [0, 1, 2, 3, 4, 6]))
    assert servable_blocks(answer, KEYS, [FULL, WINDOW], NAMEABLE, TPB) == 5
    # The same missing block after a fetch to 28: blocks 2 and 3 were stale at 28 and never
    # asked for, so merge cannot stop at 20 and falls to where only the sink block is live.
    plan = make_plan([FULL, WINDOW], token_end=END, keys=KEYS, reuse_end=0)
    assert merge(plan, plan_unit_names(plan) - names(WINDOW, KEYS, [5])) == 4


def test_servable_blocks_counts_a_block_below_the_local_prefix_as_missing():
    # Full block 0 is not in the store (the local tree has it): the decision is computed with
    # reuse_end = 0 for rank agreement, so no target is servable ...
    answer = held((FULL, range(1, 7)), (WINDOW, ALL))
    assert servable_blocks(answer, KEYS, [FULL, WINDOW], NAMEABLE, TPB) == 0
    # ... while a plan already built above that prefix never asked for block 0 and merges whole.
    plan = make_plan([FULL, WINDOW], token_end=END, keys=KEYS, reuse_end=2)
    assert merge(plan, plan_unit_names(plan)) == END


@pytest.mark.parametrize(
    "specs, answer, expected",
    [
        ([FULL, WINDOW], held((FULL, ALL), (WINDOW, range(1, 7))), 0),  # sink block missing
        ([FULL, WINDOW], held((FULL, range(6)), (WINDOW, ALL)), 6),  # full block 6 missing
        ([WINDOW], held((WINDOW, range(3))), 3),  # no full-attention group: the window decides
        ([FULL, WINDOW, STATE], held((FULL, ALL), (WINDOW, ALL)), 7),  # state is not consulted
        ([STATE], held((FULL, ALL)), 0),  # no paged group
    ],
    ids=["sink_missing", "full_tail_missing", "window_only", "state_ignored", "no_paged_group"],
)
def test_servable_blocks_table(specs, answer, expected):
    assert servable_blocks(answer, KEYS, specs, NAMEABLE, TPB) == expected


def test_servable_blocks_with_nothing_nameable_is_zero():
    assert servable_blocks(held((FULL, ALL)), KEYS, [FULL], 0, TPB) == 0


def test_publisher_window_one_block_ahead_of_the_fetch_target_serves_nothing():
    # F6: a publisher with prompt_len L = 28 (L % tpb == 0) named the window from
    # stale_end(28) = 4 on, a fetcher's largest target is B = 24 and needs from stale_end(24) = 3.
    # Every smaller target needs an even earlier block, so without sink blocks nothing serves.
    window = windowed(1, window_blocks=3, sink_blocks=0)
    keys = KEYS[:6]  # nameable = (28 - 1) // 4 = 6
    answer = held((FULL, range(6)), (window, [4, 5]))
    assert servable_blocks(answer, keys, [FULL, window], 6, TPB) == 0
    assert servable_blocks(answer, keys, [FULL], 6, TPB) == 6  # the full group alone would allow B


# ---- retry: the hint is merge's B, and the store answer is judged again below it ----


def test_retry_at_merged_b_asks_only_for_units_that_arrived():
    reader = FakeReader(groups=[FULL, WINDOW])
    req = FakeRequest(1, prompt_len=29)
    keys = reader.block_keys(req)
    plan = make_plan([FULL, WINDOW], token_end=END, keys=keys, reuse_end=0)
    served = plan_unit_names(plan) - names(WINDOW, keys, [5])
    b = merge(plan, served)
    assert b == 4
    answer = plan_unit_names(plan) - names(WINDOW, keys, [5])  # what the store told us before
    retry = store_planner(reader).decide(req, {"store": answer}, now=0.0, retry_hint=b)
    assert retry.token_end == b
    assert plan_unit_names(retry) <= served
    assert ordinals_by_group(retry) == {0: (0,), 1: (0,)}


def test_retry_can_fall_below_the_merged_b_to_a_plan_with_empty_asks():
    # Local prefix of 2 blocks; the store holds every full block and the window blocks live at 28.
    reader = FakeReader(groups=[FULL, WINDOW])
    req = FakeRequest(1, prompt_len=29)
    reader.reuse_tokens[1] = 2 * TPB
    keys = reader.block_keys(req)
    answer = names(FULL, keys, ALL) | names(WINDOW, keys, [0, 4, 5, 6])
    plan = store_planner(reader).decide(req, {"store": answer}, now=0.0)
    assert plan.token_end == END and plan.reuse_end == 2
    # Full block 6 never arrives. merge trims below reuse_end 2 and walks down from 28: at 24
    # the window needs block 3, never asked for; ... down to the reuse floor 8.
    served = plan_unit_names(plan) - names(FULL, keys, [6])
    b = merge(plan, served)
    assert b == 8
    # The retry judges the cached answer with reuse_end = 0 up to cap 8 // 4 = 2: e=2 needs window
    # block 1, which the store never held, so e=1 wins; reuse then trims every ask to nothing.
    retry = store_planner(reader).decide(req, {"store": answer}, now=0.0, retry_hint=b)
    assert retry.token_end == TPB < b
    assert retry.reuse_end == 1
    assert plan_unit_names(retry) == frozenset()


# ---- property check against a literal mirror of AttnLifeCycle.get_stale_range ----


def _div_up(a, b):
    return (a + b - 1) // b


def _mirror_stale_range(window_size, num_sink_blocks, history_length, tokens_per_block):
    """``AttnLifeCycle.get_stale_range`` from ``kv_cache_manager_v2/_life_cycle_registry.py``,
    transcribed line for line."""
    num_blocks = _div_up(history_length, tokens_per_block)
    start = min(num_blocks, num_sink_blocks)
    if window_size is None:
        return start, start
    return start, max(start, (history_length + 1 - window_size) // tokens_per_block)


@pytest.mark.parametrize("tpb", [1, 2, 4, 8])
@pytest.mark.parametrize("sink_blocks", [0, 1, 2])
@pytest.mark.parametrize("window_blocks", [None, 1, 2, 3, 5])
def test_required_ordinals_match_mirrored_stale_range_over_a_grid(tpb, sink_blocks, window_blocks):
    window = None if window_blocks is None else window_blocks * tpb
    spec = GroupSpec(0, CacheKind.PAGED, b"\0" * 8, window_size=window, sink_blocks=sink_blocks)
    for history in range(0, 12 * tpb + 1):
        beg, end = _mirror_stale_range(window, sink_blocks, history, tpb)
        assert _stale_range(spec, history, tpb) == (beg, end), (history,)
        full = history // tpb
        expected = set(range(min(beg, full))) | set(range(end, full))
        assert required_ordinals(spec, history, 0, tpb) == frozenset(expected), (history,)
        # The local prefix only removes ordinals; it never adds any.
        for reuse_end in range(0, full + 1):
            got = required_ordinals(spec, history, reuse_end, tpb)
            assert got == frozenset(o for o in expected if o >= reuse_end), (history, reuse_end)


@pytest.mark.parametrize("tpb", [1, 4])
def test_merge_lands_at_token_end_when_everything_asked_is_served(tpb):
    for window_blocks in (None, 2, 3):
        for token_blocks in range(1, 9):
            spec = (
                windowed(1, window_blocks=window_blocks, sink_blocks=1, tpb=tpb)
                if window_blocks
                else FULL
            )
            keys = keys_for("grid", token_blocks)
            plan = make_plan([spec, STATE], token_end=token_blocks * tpb, keys=keys, tpb=tpb)
            assert merge(plan, plan_unit_names(plan)) == token_blocks * tpb
            assert merge(plan, frozenset()) <= plan.token_end

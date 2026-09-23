# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The planner's decision table (design §7.2), one case per row, with tpb = 4."""

import pytest

__extra_import_path__ = ["~/tensorrt_llm/_torch"]
from disaggregation.orchestration.kv_transfer_interfaces import (  # noqa: E402
    DEFER,
    Defer,
    FetchSource,
    GroupKind,
)
from disaggregation.orchestration.remote_cache import FetchPlan, Planner  # noqa: E402
from fakes import (  # noqa: E402
    TPB,
    FakeFetches,
    FakeReader,
    FakeRequest,
    full_attention,
    state_group,
    windowed,
)

HINT = {"peer": "ctx-0"}


def make_planner(reader, *, sources=("worker", "store"), **kw):
    worker = FakeFetches(name="worker")
    store = FakeFetches(name="store", single_destination=True)
    table = {
        "worker": FetchSource("worker", worker, "ctx"),
        "store": FetchSource("store", store, None),
    }
    return Planner([table[s] for s in sources], reader, TPB, **kw)


@pytest.fixture
def reader():
    return FakeReader(groups=[full_attention(0), windowed(1), state_group(2)])


# ---- step 1: gen-init short-circuit ----


def test_gen_init_short_circuit_uses_prompt_len_and_routed_worker(reader):
    planner = make_planner(reader)
    req = FakeRequest(1, prompt_len=30, is_gen_init=True, route_hints={"ctx": HINT})
    plan = planner.decide(req, {})
    assert isinstance(plan, FetchPlan)
    assert plan.token_end == 30  # prompt_len, not a block boundary
    assert plan.no_local_fallback is True
    assert plan.source == "worker" and plan.hint == HINT
    # The live recurrent state is not part of the named ask for gen-init.
    assert {g.spec.kind for g in plan.group_plans} == {GroupKind.PAGED}
    assert 2 not in plan.units_by_group
    assert plan.units_by_group[0] == tuple(range(7))


def test_gen_init_without_matching_hint_is_none(reader):
    planner = make_planner(reader)
    assert planner.decide(FakeRequest(1, prompt_len=30, is_gen_init=True), {}) is None
    other = FakeRequest(1, prompt_len=30, is_gen_init=True, route_hints={"other": HINT})
    assert planner.decide(other, {}) is None


def test_gen_init_without_any_worker_source_is_none(reader):
    planner = make_planner(reader, sources=("store",))
    req = FakeRequest(1, prompt_len=30, is_gen_init=True, route_hints={"ctx": HINT})
    assert planner.decide(req, {}) is None


def test_gen_init_ignores_probe_answers_and_local_reuse(reader):
    planner = make_planner(reader)
    reader.reuse_tokens[1] = 8
    req = FakeRequest(1, prompt_len=30, is_gen_init=True, route_hints={"ctx": HINT})
    plan = planner.decide(req, {"store": frozenset()})
    assert plan.token_end == 30 and plan.source == "worker"
    assert plan.reuse_end == 2 and plan.units_by_group[0] == (2, 3, 4, 5, 6)


# ---- step 2: gen-first context waits for the generation side ----


def test_gen_first_not_ready_defers(reader):
    planner = make_planner(reader)
    req = FakeRequest(1, prompt_len=29, is_gen_first_context=True, route_hints={"ctx": HINT})
    reader.ready[1] = False
    answer = planner.decide(req, {})
    assert answer is DEFER and isinstance(answer, Defer)
    reader.ready[1] = True
    assert isinstance(planner.decide(req, {}), FetchPlan)


# ---- step 3: nothing nameable ----


@pytest.mark.parametrize("prompt_len", [1, 2, 4])
def test_nothing_nameable_is_none(reader, prompt_len):
    # (prompt_len - 1) // tpb == 0 for these: the last prompt token never counts, so a prompt of
    # exactly one block has no nameable block (prompt_len 5 would have one).
    planner = make_planner(reader)
    req = FakeRequest(1, prompt_len=prompt_len, route_hints={"ctx": HINT})
    assert planner.decide(req, {}) is None
    assert planner.probe_query(req) is None


def test_local_reuse_covering_every_nameable_block_still_plans_with_empty_asks(reader):
    # Rank-independence (§7.2 step 6): the local hit depth only trims the ask, never the decision.
    planner = make_planner(reader)
    req = FakeRequest(1, prompt_len=29, route_hints={"ctx": HINT})  # nameable = 7
    reader.reuse_tokens[1] = 28
    plan = planner.decide(req, {})
    assert plan.token_end == 28 and plan.source == "worker"
    assert plan.units_by_group == {0: (), 1: (), 2: ()} and plan.unit_names == frozenset()
    assert plan.reuse_end == 7


# ---- step 4/5: which source, how far ----


def test_worker_with_hint_takes_the_whole_nameable_prefix(reader):
    planner = make_planner(reader)
    req = FakeRequest(1, prompt_len=29, route_hints={"ctx": HINT})
    plan = planner.decide(req, {})  # store never answered; the worker wins first
    assert plan.source == "worker" and plan.hint == HINT
    assert plan.token_end == 28 and plan.no_local_fallback is False
    assert plan.units_by_group == {0: tuple(range(7)), 1: (0, 4, 5, 6), 2: (6,)}
    assert plan.mode == "PREFETCH"


def test_worker_source_is_skipped_without_its_hint(reader):
    planner = make_planner(reader)
    req = FakeRequest(1, prompt_len=29, route_hints={"other": HINT})
    held = reader.unit_names(req, range(7))
    plan = planner.decide(req, {"store": held})
    assert plan.source == "store" and plan.hint is None and plan.token_end == 28


def test_store_probe_unanswered_defers_within_budget_then_none(reader):
    planner = make_planner(reader, sources=("store",), probe_budget_rounds=2)
    req = FakeRequest(1, prompt_len=29)
    assert planner.decide(req, {}) is DEFER
    assert planner.decide(req, {"store": None}) is DEFER
    assert planner.decide(req, {}) is None
    # Budget is per request and reset once decided.
    assert planner.decide(req, {}) is DEFER


def test_store_probe_empty_is_an_answer_not_a_deferral(reader):
    planner = make_planner(reader, sources=("store",))
    req = FakeRequest(1, prompt_len=29)
    assert planner.decide(req, {"store": frozenset()}) is None


def test_store_uses_contiguous_prefix_of_full_attention_units(reader):
    planner = make_planner(reader, sources=("store",))
    req = FakeRequest(1, prompt_len=29)
    full, window = reader.groups[0], reader.groups[1]
    keys = reader.block_keys(req)
    # Full-attention blocks 0,1,3..6 held (gap at 2); window blocks all held.
    held = frozenset(full.tag + keys[o] for o in (0, 1, 3, 4, 5, 6)) | frozenset(
        window.tag + keys[o] for o in range(7)
    )
    plan = planner.decide(req, {"store": held})
    assert plan.source == "store" and plan.token_end == 8


def test_store_prefix_ignores_window_group_gaps(reader):
    planner = make_planner(reader, sources=("store",))
    req = FakeRequest(1, prompt_len=29)
    full = reader.groups[0]
    keys = reader.block_keys(req)
    held = frozenset(full.tag + keys[o] for o in range(7))  # no window units at all
    plan = planner.decide(req, {"store": held})
    assert plan.token_end == 28


def test_store_prefix_uses_every_paged_group_when_no_full_attention():
    reader = FakeReader(groups=[windowed(0), state_group(1)])
    planner = make_planner(reader, sources=("store",))
    req = FakeRequest(1, prompt_len=29)
    window = reader.groups[0]
    keys = reader.block_keys(req)
    held = frozenset(window.tag + keys[o] for o in range(3))
    plan = planner.decide(req, {"store": held})
    assert plan.token_end == 12


def test_store_prefix_inside_local_reuse_plans_with_reuse_capped_at_target(reader):
    planner = make_planner(reader, sources=("store",))
    req = FakeRequest(1, prompt_len=29)
    reader.reuse_tokens[1] = 20  # reuse_end = 5, past what the store offers
    held = reader.unit_names(req, range(3))
    plan = planner.decide(req, {"store": held})
    assert plan.source == "store" and plan.token_end == 12
    # A plan never asks below its own target: reuse_end is capped at token_end // tpb.
    assert plan.reuse_end == 3
    assert plan.units_by_group == {0: (), 1: (), 2: ()}


def test_sources_are_tried_in_table_order(reader):
    planner = make_planner(reader, sources=("store", "worker"))
    req = FakeRequest(1, prompt_len=29, route_hints={"ctx": HINT})
    held = reader.unit_names(req, range(2))
    plan = planner.decide(req, {"store": held})
    assert plan.source == "store" and plan.token_end == 8
    # Store unanswered: the worker further down the table is still consulted, no DEFER.
    plan = planner.decide(req, {})
    assert plan.source == "worker" and plan.token_end == 28


# ---- token_end cap ----


@pytest.mark.parametrize(
    "prompt_len, expected_end",
    [(29, 28), (32, 28), (33, 32), (9, 8), (8, 4)],
)
def test_token_end_cap_is_floor_of_prompt_len_minus_one(reader, prompt_len, expected_end):
    planner = make_planner(reader)
    req = FakeRequest(1, prompt_len=prompt_len, route_hints={"ctx": HINT})
    plan = planner.decide(req, {})
    assert plan.token_end == expected_end == (prompt_len - 1) // TPB * TPB
    assert plan.token_end % TPB == 0


# ---- retry hint ----


def test_retry_hint_caps_token_end(reader):
    planner = make_planner(reader)
    req = FakeRequest(1, prompt_len=29, route_hints={"ctx": HINT})
    plan = planner.decide(req, {}, retry_hint=16)
    assert plan.token_end == 16
    assert plan.units_by_group == {0: (0, 1, 2, 3), 1: (0, 1, 2, 3), 2: (3,)}


def test_retry_hint_inside_local_prefix_plans_empty_and_zero_is_none(reader):
    planner = make_planner(reader)
    req = FakeRequest(1, prompt_len=29, route_hints={"ctx": HINT})
    reader.reuse_tokens[1] = 8
    plan = planner.decide(req, {}, retry_hint=8)
    assert plan.token_end == 8 and plan.reuse_end == 2
    assert plan.units_by_group == {0: (), 1: (), 2: ()}
    assert planner.decide(req, {}, retry_hint=0) is None


# ---- step 6: rank-independence ----


def test_units_by_group_pruned_by_local_reuse_but_decision_is_not():
    req = FakeRequest(1, prompt_len=29, route_hints={"ctx": HINT})
    plans = []
    for reuse in (0, 8, 24):
        reader = FakeReader(groups=[full_attention(0), windowed(1), state_group(2)])
        reader.reuse_tokens[1] = reuse
        plans.append(make_planner(reader).decide(req, {}))
    assert [p.token_end for p in plans] == [28, 28, 28]
    assert [p.source for p in plans] == ["worker"] * 3
    assert plans[0].units_by_group == {0: tuple(range(7)), 1: (0, 4, 5, 6), 2: (6,)}
    assert plans[1].units_by_group == {0: (2, 3, 4, 5, 6), 1: (4, 5, 6), 2: (6,)}
    # Pruned to almost nothing is still a legal plan.
    assert plans[2].units_by_group == {0: (6,), 1: (6,), 2: (6,)}
    assert plans[2].unit_names == reader.unit_names(
        req, [6], kinds=(GroupKind.PAGED, GroupKind.STATE)
    )
    assert plans[2].reuse_end == 6


def test_probe_query_is_rank_independent(reader):
    planner = make_planner(reader)
    req = FakeRequest(1, prompt_len=29)
    reader.reuse_tokens[1] = 16  # would prune the ask; the probe still starts at block 0
    name, units = planner.probe_query(req)
    keys = reader.block_keys(req)
    assert name == keys[6]
    assert frozenset(units) == reader.unit_names(
        req, range(7), kinds=(GroupKind.PAGED, GroupKind.STATE)
    )
    assert len(units) == 21  # three groups x 7 blocks: state snapshots are named too


def test_probe_query_none_for_gen_init(reader):
    planner = make_planner(reader)
    assert planner.probe_query(FakeRequest(1, prompt_len=29, is_gen_init=True)) is None


def test_aligned_prompt_names_one_block_fewer_than_it_fills(reader):
    # prompt_len 28 fills 7 blocks, but the 28th token is the last prompt token and never takes
    # part in reuse, so the reader hands out 6 keys and the ask stops at 24.
    planner = make_planner(reader)
    req = FakeRequest(1, prompt_len=28, route_hints={"ctx": HINT})
    assert len(reader.block_keys(req)) == 6
    plan = planner.decide(req, {})
    assert plan.token_end == 24 and plan.units_by_group[0] == tuple(range(6))


def test_hybrid_layout_state_group_named_for_content_fetch_not_for_gen_init(reader):
    planner = make_planner(reader)
    state = reader.groups[2]
    keys = reader.block_keys(FakeRequest(1, prompt_len=29))

    content = planner.decide(FakeRequest(1, prompt_len=29, route_hints={"ctx": HINT}), {})
    assert [g.spec.kind for g in content.group_plans] == [
        GroupKind.PAGED,
        GroupKind.PAGED,
        GroupKind.STATE,
    ]
    assert content.units_by_group[2] == (6,)
    assert state.tag + keys[6] in content.unit_names

    gen_init = planner.decide(
        FakeRequest(2, prompt_len=29, is_gen_init=True, route_hints={"ctx": HINT}), {}
    )
    assert all(g.spec.kind is GroupKind.PAGED for g in gen_init.group_plans)
    assert 2 not in gen_init.units_by_group
    assert not any(name.startswith(state.tag) for name in gen_init.unit_names)

    _, units = planner.probe_query(FakeRequest(3, prompt_len=29))
    assert sum(1 for u in units if u.startswith(state.tag)) == 7


def test_forget_drops_defer_budget(reader):
    planner = make_planner(reader, sources=("store",), probe_budget_rounds=1)
    req = FakeRequest(1, prompt_len=29)
    assert planner.decide(req, {}) is DEFER
    planner.forget(1)
    assert planner.decide(req, {}) is DEFER
    assert planner.decide(req, {}) is None

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The KV v2 wrapper changes and ``resource/kv_v2_*`` against a real ``KVCacheManagerV2``
(2 layers, 4 KV heads, tpb 32, 2048 tokens of pool).

``context_block_keys`` count, determinism and divergence; ``reserve_transfer_pages(token_end)``
declaring history to the fetch target; the fetch extent naming exactly the nameable blocks the
local radix tree does not serve; two identical committed requests publishing under identical
names; ``layout_fingerprint`` stable per layout and different across layouts; the region
resolver's spans; and the idempotent ``release_index_slot``. Allocates device pools.

``TestVariableSlidingWindow`` repeats the fetch and publish walk on a two-group manager (window
64 on layers 0 and 2, full attention on 1 and 3): the planner's stale-range mirror against the
cache's own, the reserved pages of the windowed group, the published ordinals, and a second
request served by local reuse alone.
"""

import gc

import pytest
import torch
from engine_fakes import FakeFetches

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.disaggregation.base.backend import CacheKind
from tensorrt_llm._torch.disaggregation.remote_cache import (
    FetchPlan,
    FetchSource,
    Planner,
    _stale_range,
    required_ordinals,
    servable_block_end,
    served_token_end,
)
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import build_page_table_from_manager
from tensorrt_llm._torch.disaggregation.resource.kv_v2_view import KVv2ResourceView
from tensorrt_llm._torch.disaggregation.resource.region import (
    KVv2RegionResolver,
    layout_fingerprint,
    parallel_shard_tag,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_transfer.assembly import _check_layer_groups_are_paged
from tensorrt_llm._torch.pyexecutor.kv_transfer.effects import EngineRequestView
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX

DataType = tensorrt_llm.bindings.DataType
CacheType = tensorrt_llm.bindings.internal.batch_manager.CacheType

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="allocates real KV cache pools"
)

TPB = 32
PROMPT_LEN = 230  # 7 full blocks + 6 tokens; (230 - 1) // 32 == 7 nameable blocks
NAMEABLE = (PROMPT_LEN - 1) // TPB


def make_manager(**overrides) -> KVCacheManagerV2:
    kwargs = dict(
        kv_cache_config=KvCacheConfig(max_tokens=2048, enable_block_reuse=True),
        kv_cache_type=CacheType.SELF,
        num_layers=2,
        num_kv_heads=4,
        head_dim=64,
        tokens_per_block=TPB,
        max_seq_len=512,
        max_batch_size=4,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=DataType.HALF,
        vocab_size=32000,
    )
    kwargs.update(overrides)
    return KVCacheManagerV2(**kwargs)


def prompt_tokens(seed: int, prompt_len: int = PROMPT_LEN) -> list[int]:
    return [(seed * 7919 + i * 13 + 1) % 31000 + 100 for i in range(prompt_len)]


def make_request(request_id: int, tokens: list[int]) -> LlmRequest:
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=4,
        input_tokens=list(tokens),
        sampling_config=SamplingConfig(1),
        is_streaming=False,
    )


def compute_and_commit(manager: KVCacheManagerV2, request: LlmRequest) -> None:
    """One full context step: schedule, run, commit; as the executor loop does."""
    assert manager.prepare_context(request)
    assert manager.resize_context(request, request.context_remaining_length)
    batch = ScheduledRequests()
    batch.append_context_request(request)
    manager.prepare_resources(batch)
    request.move_to_next_context_chunk()  # the forward pass advanced the cursor
    assert request.context_remaining_length == 0
    manager.update_context_resources(batch)


@pytest.fixture
def manager():
    torch.cuda.init()
    gc.collect()
    torch.cuda.empty_cache()
    mgr = make_manager()
    yield mgr
    mgr.shutdown()
    del mgr
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture
def reader(manager):
    return KVv2ResourceView(manager, build_page_table_from_manager(manager))


# ---------------------------------------------------------------------------------------------
# context_block_keys
# ---------------------------------------------------------------------------------------------


class TestContextBlockKeys:
    def test_count_is_full_blocks_of_the_prompt_minus_its_last_token(self, manager):
        keys = manager.context_block_keys(make_request(1, prompt_tokens(1)))
        assert len(keys) == NAMEABLE == 7
        assert all(isinstance(k, bytes) and len(k) > 0 for k in keys)
        assert len(set(keys)) == len(keys)

    @pytest.mark.parametrize("prompt_len, expected", [(224, 6), (225, 7), (33, 1), (32, 0), (5, 0)])
    def test_aligned_prompt_names_one_block_fewer(self, manager, prompt_len, expected):
        keys = manager.context_block_keys(make_request(1, prompt_tokens(1, prompt_len)))
        assert len(keys) == expected == (prompt_len - 1) // TPB

    def test_same_prompt_same_keys_across_requests(self, manager):
        a = manager.context_block_keys(make_request(1, prompt_tokens(1)))
        b = manager.context_block_keys(make_request(2, prompt_tokens(1)))
        assert a == b

    def test_prompts_diverge_from_the_changed_block_on(self, manager):
        base = prompt_tokens(1)
        changed = list(base)
        changed[2 * TPB + 5] += 1  # inside block 2
        a = manager.context_block_keys(make_request(1, base))
        b = manager.context_block_keys(make_request(2, changed))
        assert a[:2] == b[:2]
        assert all(x != y for x, y in zip(a[2:], b[2:]))  # a chain: every later key changes

    def test_keys_are_the_ones_the_radix_tree_commits_under(self, manager, reader):
        """A request computed and committed makes its blocks reusable by a second request under
        the same keys: the reuse probe finds every full block the keys name."""
        first = make_request(1, prompt_tokens(1))
        compute_and_commit(manager, first)
        second = make_request(2, prompt_tokens(1))
        reusable = manager.probe_context_reuse(second)
        # Every full block plus, with partial reuse, the committed tail short of the last token.
        assert reusable >= NAMEABLE * TPB
        assert reusable == PROMPT_LEN - 1
        assert reader.local_reuse_tokens(EngineRequestView(second)) == reusable

    def test_reader_caches_keys_per_request_until_the_prompt_changes(self, manager, reader):
        request = make_request(1, prompt_tokens(1))
        view = EngineRequestView(request)
        first = reader.block_keys(view)
        assert reader.block_keys(view) is first  # remembered
        reader.forget_request(1)
        assert reader.block_keys(view) == first and reader.block_keys(view) is not first


# ---------------------------------------------------------------------------------------------
# reserve_transfer_pages(token_end)
# ---------------------------------------------------------------------------------------------


class TestPrepareDisaggGenInitTokenEnd:
    def test_history_is_declared_to_token_end_only(self, manager):
        request = make_request(1, prompt_tokens(1))
        assert manager.reserve_transfer_pages(request, 128)
        assert manager.get_history_length(request) == 128
        kv_cache = manager.kv_cache_map[1]
        assert kv_cache.capacity >= 128
        assert kv_cache.enable_swa_scratch_reuse is False
        assert request.py_ctx_pre_resize_cap == 0  # the capacity grew from nothing
        # The cursor is not touched: the request is still a first chunk at position 0.
        assert request.context_current_position == 0 and request.is_first_context_chunk
        page_indices = list(kv_cache.get_aggregated_page_indices(0, valid_only=False))
        assert len(page_indices) >= 128 // TPB
        assert all(p != BAD_PAGE_INDEX for p in page_indices[: 128 // TPB])

    def test_without_token_end_history_is_the_whole_prompt(self, manager):
        request = make_request(1, prompt_tokens(1))
        assert manager.prepare_disagg_gen_init(request)
        assert manager.get_history_length(request) == PROMPT_LEN
        assert manager.kv_cache_map[1].capacity >= PROMPT_LEN

    def test_reservation_below_a_local_partial_reuse_match_keeps_the_reused_history(self, manager):
        """The same prompt was computed here before: local reuse serves 229 tokens, more than the
        planner's block-aligned ``token_end`` of 224 (a plan with an empty ask is legal, design
        §7.2 step 6). Declaring history at ``token_end`` would lower it below the reuse match,
        which the cache refuses ("History length cannot be decreased"); the reservation must keep
        what reuse gave and succeed."""
        earlier = make_request(1, prompt_tokens(1))
        compute_and_commit(manager, earlier)
        request = make_request(2, prompt_tokens(1))
        assert manager.probe_context_reuse(request) == PROMPT_LEN - 1 > NAMEABLE * TPB

        assert manager.reserve_transfer_pages(request, NAMEABLE * TPB)

        assert 2 in manager.kv_cache_map
        assert manager.get_history_length(request) >= NAMEABLE * TPB

    def test_fetch_reservation_does_not_fill_the_reserved_pages(self, manager):
        """The diagnostic fresh-page fill is an asynchronous write on the engine stream; a store
        backend writes the fetched pages from its own thread with no ordering against it. The
        ``token_end`` path therefore skips the fill (the fetch fills every reserved block); the
        gen-init path keeps it."""
        manager._fresh_page_fill = 0.0  # what TRTLLM_KV_FRESH_PAGE_FILL=zero parses to
        fetching = make_request(1, prompt_tokens(1))
        assert manager.reserve_transfer_pages(fetching, 128)
        assert not manager._fresh_pages_filled.get(1)
        gen_init = make_request(2, prompt_tokens(2))
        assert manager.prepare_disagg_gen_init(gen_init)
        assert manager._fresh_pages_filled.get(2)

    def test_reserving_again_keeps_the_cache_and_never_lowers_its_history(self, manager):
        """The scheduler may ask again for a request that already has a cache (its launch did
        not go through and the pages were kept, or the plan was retried at another target):
        every reservation finds the same cache and leaves the request a first chunk at position
        0. The same target changes nothing; a lower one keeps the history (it never decreases);
        a higher one raises it and grows the pages."""
        request = make_request(1, prompt_tokens(1))
        assert manager.reserve_transfer_pages(request, 128)
        kv_cache = manager.kv_cache_map[1]
        pages = list(kv_cache.get_aggregated_page_indices(0, valid_only=False))

        assert manager.reserve_transfer_pages(request, 128)  # the same target
        assert manager.kv_cache_map[1] is kv_cache
        assert manager.get_history_length(request) == 128
        assert list(kv_cache.get_aggregated_page_indices(0, valid_only=False)) == pages

        assert manager.reserve_transfer_pages(request, 64)  # a lower target
        assert manager.kv_cache_map[1] is kv_cache
        assert manager.get_history_length(request) == 128
        assert list(kv_cache.get_aggregated_page_indices(0, valid_only=False)) == pages

        assert manager.reserve_transfer_pages(request, 192)  # a higher target
        assert manager.kv_cache_map[1] is kv_cache
        assert manager.get_history_length(request) == 192
        grown = list(kv_cache.get_aggregated_page_indices(0, valid_only=False))
        assert grown[: len(pages)] == pages and len(grown) >= 192 // TPB
        assert all(p != BAD_PAGE_INDEX for p in grown[: 192 // TPB])
        assert request.context_current_position == 0 and request.is_first_context_chunk

    def test_revert_after_a_fetch_reservation_drops_the_cache(self, manager):
        """The reservation grew the cache from nothing to ``token_end`` with history declared
        there; reverting cannot shrink below the history, so the cache is dropped and the
        request re-enters as a fresh first chunk."""
        request = make_request(1, prompt_tokens(1))
        assert manager.reserve_transfer_pages(request, 128)
        assert manager.revert_allocate_context(request) is False
        assert 1 not in manager.kv_cache_map
        assert request.context_current_position == 0
        assert request.context_chunk_size == PROMPT_LEN
        assert request.py_ctx_pre_resize_cap is None


# ---------------------------------------------------------------------------------------------
# fetch_extent_and_committed: nameable blocks minus the local reuse prefix
# ---------------------------------------------------------------------------------------------


class TestFetchExtent:
    def test_units_cover_nameable_minus_reuse_in_the_reserved_pages(self, manager, reader):
        # Blocks 0-1 of the prompt are computed locally by an earlier request with the same first
        # 64 tokens; blocks 2-5 come from the store (its answer covers keys 0-5 only).
        local = prompt_tokens(1)[: 2 * TPB] + prompt_tokens(99)[2 * TPB :]
        earlier = make_request(1, local)
        compute_and_commit(manager, earlier)

        request = make_request(2, prompt_tokens(1))
        view = EngineRequestView(request)
        keys = reader.block_keys(view)
        assert reader.local_reuse_tokens(view) == 2 * TPB
        store = FakeFetches(name="store", probe_answer="all")
        planner = Planner([FetchSource("store", store, None)], reader, TPB)
        (spec,) = reader.group_specs()
        answer = frozenset(spec.tag + keys[o] for o in range(6))
        plan = planner.decide(view, {"store": answer}, now=0.0)
        assert isinstance(plan, FetchPlan)
        assert plan.token_end == 6 * TPB and plan.reuse_end_blocks == 2
        assert [g.ordinals for g in plan.group_plans] == [(2, 3, 4, 5)]

        # The scheduler's reservation, then the extent the coordinator launches.
        assert manager.reserve_transfer_pages(request, plan.token_end)
        assert manager.get_history_length(request) == plan.token_end
        extent, committed = reader.fetch_extent_and_committed(view, plan)

        assert extent.name == b"fetch:2" and extent.is_last is True
        assert len(extent.units) == 6 - 2 == 4
        assert [u.name for u in extent.units] == [spec.tag + keys[o] for o in (2, 3, 4, 5)]
        assert all(u.local_group == 0 for u in extent.units)
        kv_cache = manager.kv_cache_map[2]
        pages = list(kv_cache.get_aggregated_page_indices(0, valid_only=False))
        assert [u.local for u in extent.units] == [pages[o] for o in (2, 3, 4, 5)]
        assert all(u.local != BAD_PAGE_INDEX for u in extent.units)
        # The reused blocks are the earlier request's very pages.
        earlier_pages = list(
            manager.kv_cache_map[1].get_aggregated_page_indices(0, valid_only=False)
        )
        assert pages[:2] == earlier_pages[:2]

    def test_without_local_reuse_every_nameable_block_is_fetched(self, manager, reader):
        request = make_request(1, prompt_tokens(1))
        view = EngineRequestView(request)
        store = FakeFetches(name="store", probe_answer="all")
        planner = Planner([FetchSource("store", store, None)], reader, TPB)
        name, units = planner.probe_query(view)
        assert len(units) == NAMEABLE  # one group
        plan = planner.decide(view, {"store": frozenset(units)}, now=0.0)
        assert plan.token_end == NAMEABLE * TPB == 224
        assert manager.reserve_transfer_pages(request, plan.token_end)
        extent, committed = reader.fetch_extent_and_committed(view, plan)
        assert len(extent.units) == NAMEABLE
        assert frozenset(u.name for u in extent.units) == frozenset(units)

    def test_extent_names_only_the_ordinals_the_reservation_has_pages_for(self, manager, reader):
        request = make_request(1, prompt_tokens(1))
        view = EngineRequestView(request)
        store = FakeFetches(name="store", probe_answer="all")
        planner = Planner([FetchSource("store", store, None)], reader, TPB)
        _, units = planner.probe_query(view)
        plan = planner.decide(view, {"store": frozenset(units)}, now=0.0)
        assert manager.reserve_transfer_pages(request, 3 * TPB)  # fewer pages than planned
        extent, committed = reader.fetch_extent_and_committed(view, plan)
        assert [u.name for u in extent.units] == [
            reader.group_specs()[0].tag + k for k in reader.block_keys(view)[:3]
        ]
        assert committed == frozenset()  # a shortfall is not "already here"

    def test_blocks_committed_since_the_plan_are_returned_by_name_not_fetched_over(
        self, manager, reader
    ):
        """The plan was made when nothing was local; by the time the scheduler reserves, another
        request has committed the first two blocks. The reservation shares their pages, so the
        extent leaves them out and names them as committed; merged together they still reach
        the plan's target."""
        request = make_request(2, prompt_tokens(1))
        view = EngineRequestView(request)
        keys = reader.block_keys(view)
        (spec,) = reader.group_specs()
        store = FakeFetches(name="store", probe_answer="all")
        planner = Planner([FetchSource("store", store, None)], reader, TPB)
        _, units = planner.probe_query(view)
        plan = planner.decide(view, {"store": frozenset(units)}, now=0.0)
        assert plan.reuse_end_blocks == 0 and [g.ordinals for g in plan.group_plans] == [
            tuple(range(NAMEABLE))
        ]

        local = prompt_tokens(1)[: 2 * TPB] + prompt_tokens(99)[2 * TPB :]
        compute_and_commit(manager, make_request(1, local))  # meanwhile, elsewhere
        assert manager.reserve_transfer_pages(request, plan.token_end)
        assert manager.kv_cache_map[2].num_committed_tokens == 2 * TPB

        extent, committed = reader.fetch_extent_and_committed(view, plan)
        assert committed == frozenset(spec.tag + keys[o] for o in range(2))
        assert [u.name for u in extent.units] == [spec.tag + keys[o] for o in range(2, NAMEABLE)]
        assert (
            served_token_end(plan, frozenset(u.name for u in extent.units) | committed)
            == plan.token_end
        )
        assert (
            served_token_end(plan, frozenset(u.name for u in extent.units)) == 0
        )  # without the names


# ---------------------------------------------------------------------------------------------
# publish_extent_and_chunk
# ---------------------------------------------------------------------------------------------


class TestPublishDescription:
    def test_two_identical_committed_requests_publish_identical_names(self, manager, reader):
        a = make_request(1, prompt_tokens(1))
        compute_and_commit(manager, a)
        extent_a, chunk_a = reader.publish_extent_and_chunk(EngineRequestView(a))
        b = make_request(2, prompt_tokens(1))
        compute_and_commit(manager, b)
        extent_b, chunk_b = reader.publish_extent_and_chunk(EngineRequestView(b))

        assert chunk_a is None and chunk_b is None
        assert extent_a.is_last and extent_b.is_last
        assert extent_a.name == b"publish:1" and extent_b.name == b"publish:2"
        assert len(extent_a.units) == NAMEABLE
        assert [u.name for u in extent_a.units] == [u.name for u in extent_b.units]
        (spec,) = reader.group_specs()
        assert [u.name for u in extent_a.units] == [
            spec.tag + k for k in manager.context_block_keys(a)
        ]
        # Same content, same blocks: the second request reused the first one's pages.
        assert [u.local for u in extent_a.units] == [u.local for u in extent_b.units]
        assert all(u.local != BAD_PAGE_INDEX for u in extent_a.units)

    def test_different_prompts_publish_different_names(self, manager, reader):
        a = make_request(1, prompt_tokens(1))
        b = make_request(2, prompt_tokens(2))
        compute_and_commit(manager, a)
        compute_and_commit(manager, b)
        names_a = {u.name for u in reader.publish_extent_and_chunk(EngineRequestView(a))[0].units}
        names_b = {u.name for u in reader.publish_extent_and_chunk(EngineRequestView(b))[0].units}
        assert not names_a & names_b

    def test_publish_offers_only_committed_blocks(self, manager, reader):
        """A request whose cache exists but committed nothing yet offers nothing."""
        request = make_request(1, prompt_tokens(1))
        assert manager.reserve_transfer_pages(request, 128)
        extent, _ = reader.publish_extent_and_chunk(EngineRequestView(request))
        assert extent.units == ()
        assert extent.is_last is False  # prefill has not ended

    def test_group_specs_describe_one_full_attention_group(self, reader):
        specs = reader.group_specs()
        assert len(specs) == 1
        (spec,) = specs
        assert spec.kind is CacheKind.PAGED and spec.window_size is None
        assert spec.local_group == 0 and spec.sink_blocks == 0
        assert len(spec.tag) == 8
        assert reader.tokens_per_block == TPB
        assert reader.generation_first_ready(None) is True


# ---------------------------------------------------------------------------------------------
# layout_fingerprint and KVv2RegionResolver
# ---------------------------------------------------------------------------------------------


class TestLayout:
    def test_fingerprint_is_stable_for_one_layout(self, manager):
        page_table = build_page_table_from_manager(manager)
        first = layout_fingerprint(manager, page_table)
        assert isinstance(first, bytes) and len(first) == 16
        assert layout_fingerprint(manager, page_table) == first
        assert layout_fingerprint(manager) == first  # page table rebuilt from the manager
        # A second manager with the same configuration (different pool addresses) agrees.
        other = make_manager()
        try:
            other_table = build_page_table_from_manager(other)
            assert KVv2RegionResolver(other_table).pool_memory_spans() != (
                KVv2RegionResolver(page_table).pool_memory_spans()
            )
            assert layout_fingerprint(other, other_table) == first
        finally:
            other.shutdown()

    @pytest.mark.parametrize(
        "override",
        [
            dict(tokens_per_block=64),
            dict(num_kv_heads=2),
            dict(head_dim=128),
            dict(dtype=DataType.BF16),
        ],
        ids=["tpb", "kv_heads", "head_dim", "dtype"],
    )
    def test_fingerprint_changes_with_the_layout(self, manager, override):
        other = make_manager(**override)
        try:
            assert layout_fingerprint(other) != layout_fingerprint(manager)
        finally:
            other.shutdown()

    def test_parallel_shard_tag_names_the_tp_slice_and_is_shared_by_full_head_holders(self):
        # A TP=1 worker and an attention-DP replica both hold every head: one tag, shared entries.
        assert parallel_shard_tag(Mapping(world_size=1, tp_size=1, rank=0)) == "heads=all"
        tp_ranks = [Mapping(world_size=2, tp_size=2, rank=r) for r in range(2)]
        assert [parallel_shard_tag(m) for m in tp_ranks] == ["heads=0/2", "heads=1/2"]
        adp_ranks = [
            Mapping(world_size=2, tp_size=2, rank=r, enable_attention_dp=True) for r in range(2)
        ]
        assert [parallel_shard_tag(m) for m in adp_ranks] == ["heads=all", "heads=all"]
        # Context parallelism splits the sequence: each rank's slice is named too.
        cp_ranks = [Mapping(world_size=2, cp_size=2, rank=r) for r in range(2)]
        assert [parallel_shard_tag(m) for m in cp_ranks] == ["heads=all;cp=0/2", "heads=all;cp=1/2"]
        tp_cp = Mapping(world_size=4, tp_size=2, cp_size=2, rank=3)
        assert parallel_shard_tag(tp_cp) == f"heads={tp_cp.tp_rank}/2;cp={tp_cp.cp_rank}/2"

    def test_fingerprint_separates_tp_shards_and_shares_attention_dp_replicas(self, manager):
        """Two TP ranks hold byte-identical layouts of different heads: their fingerprints must
        differ. Two attention-DP replicas hold the same heads: theirs must agree. The default
        (no shard tag) is stable."""
        page_table = build_page_table_from_manager(manager)
        by_tag = {
            tag: layout_fingerprint(manager, page_table, parallel_shard=tag)
            for tag in ("heads=0/2", "heads=1/2", "heads=all", "")
        }
        assert by_tag["heads=0/2"] != by_tag["heads=1/2"]
        assert by_tag["heads=all"] == layout_fingerprint(
            manager, page_table, parallel_shard="heads=all"
        )
        assert by_tag[""] == layout_fingerprint(manager, page_table) == layout_fingerprint(manager)
        assert len(set(by_tag.values())) == 4

    def test_fingerprint_separates_models_of_one_geometry_by_identity(self, manager):
        """Two models with identical KV geometry must not read each other's pages: the model
        identity the assembly passes is part of the digest. Unknown (empty) is the default and
        is what every call without the argument gets."""
        page_table = build_page_table_from_manager(manager)
        unknown = layout_fingerprint(manager, page_table)
        assert unknown == layout_fingerprint(manager, page_table, model_identity="")
        by_model = {
            name: layout_fingerprint(manager, page_table, model_identity=name)
            for name in ("meta-llama/Llama-3.1-8B@main", "mistralai/Mistral-7B-v0.3@main")
        }
        assert len({unknown, *by_model.values()}) == 3
        assert by_model["meta-llama/Llama-3.1-8B@main"] == layout_fingerprint(
            manager, page_table, model_identity="meta-llama/Llama-3.1-8B@main"
        )
        # Identity and shard tag are independent dimensions of the key.
        assert layout_fingerprint(
            manager, page_table, parallel_shard="heads=0/2", model_identity="m"
        ) not in {
            by_model["meta-llama/Llama-3.1-8B@main"],
            layout_fingerprint(manager, page_table, parallel_shard="heads=0/2"),
        }

    def test_resolver_spans_and_segments(self, manager):
        page_table = build_page_table_from_manager(manager)
        resolver = KVv2RegionResolver(page_table)
        pools = [pool for group in page_table.pool_groups for pool in group.pools]
        spans = resolver.pool_memory_spans()
        assert len(spans) == len({int(p.base_address) for p in pools})
        assert spans == sorted(spans)
        for (addr_a, size_a), (addr_b, _) in zip(spans, spans[1:]):
            assert addr_a + size_a <= addr_b  # no overlap
        for address, size in spans:
            assert address > 0 and size > 0
        assert sum(size for _, size in spans) == sum(
            int(p.slot_bytes) * int(p.num_slots) for p in pools
        )

        segments = resolver(0, 0)
        group_pools = page_table.pool_groups[page_table.layer_groups[0].pool_group_idx].pools
        assert len(segments) == len(group_pools)
        assert [size for _, size in segments] == [int(p.slot_bytes) for p in group_pools]
        # Consecutive pages are consecutive slots.
        for (a0, _), (a1, _), pool in zip(segments, resolver(0, 1), group_pools):
            assert a1 - a0 == int(pool.slot_bytes)
        assert resolver.max_unit_bytes() == max(
            sum(int(p.slot_bytes) for p in g.pools) for g in page_table.pool_groups
        )
        # One page of tpb tokens, 2 layers x (K+V) x 4 heads x 64 dims x 2 bytes.
        assert resolver.max_unit_bytes() == TPB * 2 * 2 * 4 * 64 * 2

    def test_resolver_rejects_out_of_range_coordinates(self, manager):
        resolver = KVv2RegionResolver(build_page_table_from_manager(manager))
        with pytest.raises(KeyError):
            resolver(1, 0)
        with pytest.raises(KeyError):
            resolver(0, 10**9)
        with pytest.raises(KeyError):
            resolver(0, -1)

    def test_unit_addresses_lie_inside_a_registered_span(self, manager, reader):
        request = make_request(1, prompt_tokens(1))
        compute_and_commit(manager, request)
        extent, _ = reader.publish_extent_and_chunk(EngineRequestView(request))
        page_table = build_page_table_from_manager(manager)
        resolver = KVv2RegionResolver(page_table)
        spans = resolver.pool_memory_spans()
        for unit in extent.units:
            for address, size in resolver(unit.local_group, unit.local):
                assert any(a <= address and address + size <= a + s for a, s in spans)


# ---------------------------------------------------------------------------------------------
# release_index_slot idempotence
# ---------------------------------------------------------------------------------------------


class _CountingIndexMapper:
    """Delegates to the real (C++, read-only) ``IndexMapper`` and counts ``remove_sequence``."""

    def __init__(self, real) -> None:
        self._real = real
        self.removed: list[int] = []

    def remove_sequence(self, request_id: int) -> None:
        self.removed.append(request_id)
        return self._real.remove_sequence(request_id)

    def __getattr__(self, name):
        return getattr(self._real, name)


class TestReleaseIndexSlot:
    def test_second_release_is_a_no_op_and_free_still_works(self, manager, monkeypatch):
        request = make_request(1, prompt_tokens(1))
        compute_and_commit(manager, request)
        mapper = _CountingIndexMapper(manager.index_mapper)
        monkeypatch.setattr(manager, "index_mapper", mapper)
        manager.release_index_slot(1)
        manager.release_index_slot(1)  # the second transfer owner
        assert mapper.removed == [1]
        assert 1 in manager._early_freed_index_requests
        assert 1 in manager.kv_cache_map  # the pages stay allocated
        manager.free_resources(request)
        assert mapper.removed == [1]  # free knows the slot went early
        assert 1 not in manager.kv_cache_map
        assert 1 not in manager._early_freed_index_requests

    def test_release_then_disagg_style_free_leaves_no_early_freed_entry(self, manager):
        """The disagg send path releases the slot, then ``free_resources`` ends the request; a
        store publish releasing the same slot in between must not disturb either."""
        request = make_request(1, prompt_tokens(1))
        compute_and_commit(manager, request)
        manager.release_index_slot(1)  # disagg send
        manager.release_index_slot(1)  # store publish hold
        manager.free_resources(request)
        assert manager._early_freed_index_requests == set()
        # A new request takes a slot without complaint.
        again = make_request(2, prompt_tokens(2))
        compute_and_commit(manager, again)
        assert 2 in manager.kv_cache_map


# ---------------------------------------------------------------------------------------------
# Variable sliding window: a windowed group next to a full-attention group
# ---------------------------------------------------------------------------------------------

WINDOW = 64
"""Layers 0 and 2 read the last 64 tokens; ``max_attention_window=[64, 512]`` with
``max_seq_len=512`` makes the 512 entry the full-attention default (``None``)."""
B = NAMEABLE * TPB  # 224, the largest fetch target for PROMPT_LEN 230


def stale_end(history: int) -> int:
    return (history + 1 - WINDOW) // TPB


@pytest.fixture
def vswa_manager():
    torch.cuda.init()
    gc.collect()
    torch.cuda.empty_cache()
    mgr = make_manager(
        num_layers=4,
        kv_cache_config=KvCacheConfig(
            max_tokens=2048, enable_block_reuse=True, max_attention_window=[WINDOW, 512]
        ),
    )
    yield mgr
    mgr.shutdown()
    del mgr
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture
def vswa_reader(vswa_manager):
    return KVv2ResourceView(vswa_manager, build_page_table_from_manager(vswa_manager))


def groups_of(reader):
    """``(windowed, full)`` specs of the two-group reader."""
    (windowed,) = [s for s in reader.group_specs() if s.window_size is not None]
    (full,) = [s for s in reader.group_specs() if s.window_size is None]
    return windowed, full


def store_planner(reader) -> Planner:
    store = FakeFetches(name="store", probe_answer="all")
    return Planner([FetchSource("store", store, None)], reader, TPB)


class TestVariableSlidingWindow:
    def test_group_specs_describe_a_windowed_and_a_full_attention_group(self, vswa_reader):
        windowed, full = groups_of(vswa_reader)
        assert {s.kind for s in vswa_reader.group_specs()} == {CacheKind.PAGED}
        assert windowed.window_size == WINDOW and windowed.sink_blocks == 0
        assert full.window_size is None and full.sink_blocks == 0
        assert windowed.tag != full.tag
        _check_layer_groups_are_paged(vswa_reader)  # the assembly guard admits this manager

    @pytest.mark.parametrize("history", [B, PROMPT_LEN, 256, 300])
    def test_planner_stale_range_mirrors_the_cache_life_cycle(
        self, vswa_manager, vswa_reader, history
    ):
        for spec in vswa_reader.group_specs():
            assert _stale_range(spec, history, TPB) == tuple(
                vswa_manager.stale_block_range(spec.local_group, history)
            )
        windowed, _ = groups_of(vswa_reader)
        assert _stale_range(windowed, history, TPB) == (0, stale_end(history))

    def test_fetch_extent_names_the_reserved_pages_of_every_group(self, vswa_manager, vswa_reader):
        request = make_request(1, prompt_tokens(1))
        view = EngineRequestView(request)
        windowed, full = groups_of(vswa_reader)
        keys = vswa_reader.block_keys(view)
        planner = store_planner(vswa_reader)
        _, units = planner.probe_query(view)
        assert len(units) == 2 * NAMEABLE  # both groups, every nameable block
        plan = planner.decide(view, {"store": frozenset(units)}, now=0.0)
        assert isinstance(plan, FetchPlan) and plan.token_end == B and plan.reuse_end_blocks == 0
        expected = {
            windowed.local_group: tuple(sorted(required_ordinals(windowed, B, 0, TPB))),
            full.local_group: tuple(range(NAMEABLE)),
        }
        assert {g.spec.local_group: g.ordinals for g in plan.group_plans} == expected
        assert expected[windowed.local_group] == (5, 6) == tuple(range(stale_end(B), NAMEABLE))

        assert vswa_manager.reserve_transfer_pages(request, plan.token_end)
        assert vswa_manager.get_history_length(request) == B
        extent, committed = vswa_reader.fetch_extent_and_committed(view, plan)
        assert committed == frozenset()
        # Every required ordinal has a page: the reservation skipped exactly the stale range.
        assert len(extent.units) == sum(len(g.ordinals) for g in plan.group_plans) == 9
        kv_cache = vswa_manager.kv_cache_map[1]
        for group_plan in plan.group_plans:
            pages = list(
                kv_cache.get_aggregated_page_indices(group_plan.spec.local_group, valid_only=False)
            )
            for ordinal in group_plan.ordinals:
                assert pages[ordinal] != BAD_PAGE_INDEX
        window_pages = list(
            kv_cache.get_aggregated_page_indices(windowed.local_group, valid_only=False)
        )
        assert all(p == BAD_PAGE_INDEX for p in window_pages[: stale_end(B)])
        by_group = {}
        for unit in extent.units:
            by_group.setdefault(unit.local_group, []).append(unit)
        assert [u.name for u in by_group[windowed.local_group]] == [
            windowed.tag + keys[o] for o in (5, 6)
        ]
        assert [u.local for u in by_group[windowed.local_group]] == window_pages[5:7]
        assert [u.name for u in by_group[full.local_group]] == [
            full.tag + keys[o] for o in range(NAMEABLE)
        ]
        full_pages = list(kv_cache.get_aggregated_page_indices(full.local_group, valid_only=False))
        assert [u.local for u in by_group[full.local_group]] == full_pages[:NAMEABLE]
        resolver = KVv2RegionResolver(build_page_table_from_manager(vswa_manager))
        spans = resolver.pool_memory_spans()
        for unit in extent.units:
            for address, size in resolver(unit.local_group, unit.local):
                assert any(a <= address and address + size <= a + s for a, s in spans)

    def test_publish_names_the_sink_and_live_window_of_the_windowed_group(
        self, vswa_manager, vswa_reader
    ):
        request = make_request(1, prompt_tokens(1))
        compute_and_commit(vswa_manager, request)
        windowed, full = groups_of(vswa_reader)
        keys = vswa_reader.block_keys(EngineRequestView(request))
        extent, chunk = vswa_reader.publish_extent_and_chunk(EngineRequestView(request))
        assert chunk is None and extent.is_last
        names = {u.name for u in extent.units}
        # History is the whole prompt: (230 + 1 - 64) // 32 == 5, so window blocks 5 and 6 live.
        assert stale_end(PROMPT_LEN) == 5
        assert names == {windowed.tag + keys[o] for o in (5, 6)} | {
            full.tag + keys[o] for o in range(NAMEABLE)
        }
        assert len(extent.units) == 9
        assert all(u.local != BAD_PAGE_INDEX for u in extent.units)

    def test_second_request_is_served_by_local_reuse_and_asks_for_nothing(
        self, vswa_manager, vswa_reader
    ):
        earlier = make_request(1, prompt_tokens(1))
        compute_and_commit(vswa_manager, earlier)
        request = make_request(2, prompt_tokens(1))
        view = EngineRequestView(request)
        # Partial reuse matches the prompt short of its last token; the windowed group has pages
        # for the blocks live at 229 (5 and 6), so the match is not cut short.
        assert vswa_reader.local_reuse_tokens(view) == PROMPT_LEN - 1 == 229
        planner = store_planner(vswa_reader)
        _, units = planner.probe_query(view)
        keys = vswa_reader.block_keys(view)
        assert (
            servable_block_end(frozenset(units), keys, vswa_reader.group_specs(), NAMEABLE, TPB)
            == 7
        )
        plan = planner.decide(view, {"store": frozenset(units)}, now=0.0)
        assert plan.token_end == B and plan.reuse_end_blocks == 7
        assert all(g.ordinals == () for g in plan.group_plans)

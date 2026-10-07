# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Several context instances sharing one store, several generation instances taking KV from
them.

The deployment this models: a pool of context instances with the same parallelism publish to
and fetch from one Mooncake store, so a prompt any of them has computed is a store hit for all
the others; generation instances receive their KV from whichever context instance served the
request, over the paired transceiver path, and never touch the store. Context and generation
parallelism may differ in such a deployment; here every engine is TP=1 and all four live in one
process on one GPU (``ctx_a``, ``ctx_b``, ``gen_a``, ``gen_b``).

What the test proves, over two identical rounds:

- ``ctx_a`` serves prompt P1 as ``context_only`` and publishes its blocks; ``gen_a`` continues it
  as ``generation_only`` and generates what a plain single engine does.
- ``ctx_b`` serves the same P1: it has never seen the prompt, so it probes the store, fetches
  every nameable block and computes only the tail; ``gen_b`` continues it to the same output.
- P2 goes the other way (``ctx_b`` publishes, ``ctx_a`` fetches), so reuse is symmetric.
- Each context engine fetched exactly the blocks of the prompt it did not compute, published
  exactly the blocks of the one it did, and failed nothing; each generation engine neither
  published nor fetched. No engine leaves coordinator records behind: nothing but the
  ``context_only`` requests is sent to a context engine, so each reaped its disagg sends on its
  own and released every request through the gate.
"""

import os

import pytest
from status_dumps import assert_no_leftover_records, counters, read_single_rank_dump
from store_engine import (
    KV_TRANSFER_CONFIG_ENV,
    NAMEABLE_BLOCKS,
    generate_ids,
    prompt_token_ids,
    run_engine,
    sampling_params,
    start_engine,
    timeout_mark,
    write_kv_transfer_yaml,
)

pytestmark = [pytest.mark.threadleak(enabled=False), pytest.mark.private_mpi_session]

CTX_ENGINES = ("ctx_a", "ctx_b")
GEN_ENGINES = ("gen_a", "gen_b")
FREE_GPU_MEMORY_FRACTION = 0.08
"""Of the memory free at each engine's start: four TinyLlama engines must fit on one GPU."""
ROUNDS = 2


def context_then_generate(llm_ctx, llm_gen, prompt, expected) -> None:
    """``context_only`` on ``llm_ctx``, then ``generation_only`` on ``llm_gen`` with the params
    the context engine handed back; the output must equal the single-engine reference."""
    from tensorrt_llm.disaggregated_params import DisaggregatedParams

    (ctx_output,) = llm_ctx.generate(
        [prompt],
        sampling_params=sampling_params(1),
        disaggregated_params=DisaggregatedParams(request_type="context_only"),
    )
    assert list(ctx_output.outputs[0].token_ids) == expected[:1]
    disaggregated_params = ctx_output.disaggregated_params
    disaggregated_params.request_type = "generation_only"
    tokens = generate_ids(llm_gen, prompt, disaggregated_params=disaggregated_params)
    assert tokens == expected, (tokens, expected)


@timeout_mark(900)
def test_context_instances_share_one_store_and_generation_instances_never_touch_it(
    mooncake_cluster, tinyllama_path, tmp_path, monkeypatch
):
    prompts = {"P1": prompt_token_ids(seed=1), "P2": prompt_token_ids(seed=2)}

    # One plain engine, no store, no transceiver; created and gone before the four.
    monkeypatch.delenv(KV_TRANSFER_CONFIG_ENV, raising=False)
    plain = run_engine(tmp_path, monkeypatch, None, tinyllama_path, list(prompts.values()))
    expected = dict(zip(prompts, plain.tokens))
    assert all(len(tokens) > 0 for tokens in expected.values())
    assert expected["P1"] != expected["P2"]

    namespace = f"multi-instance-{os.getpid()}"
    config_path = write_kv_transfer_yaml(tmp_path, mooncake_cluster.master_address, namespace)
    monkeypatch.setenv(KV_TRANSFER_CONFIG_ENV, config_path)

    engines = {}
    dumps = {}
    try:
        for tag in CTX_ENGINES + GEN_ENGINES:
            engines[tag] = start_engine(
                tmp_path,
                monkeypatch,
                tag,
                tinyllama_path,
                cache_transceiver=True,
                free_gpu_memory_fraction=FREE_GPU_MEMORY_FRACTION,
            )
        ctx_a, ctx_b = (engines[tag] for tag in CTX_ENGINES)
        gen_a, gen_b = (engines[tag] for tag in GEN_ENGINES)

        # (context engine, generation engine, prompt): P1 is computed by ctx_a and fetched by
        # ctx_b, P2 the other way around. The same order every round.
        schedule = (
            (ctx_a, gen_a, "P1"),
            (ctx_b, gen_b, "P1"),
            (ctx_b, gen_b, "P2"),
            (ctx_a, gen_a, "P2"),
        )
        for _ in range(ROUNDS):
            for llm_ctx, llm_gen, name in schedule:
                context_then_generate(llm_ctx, llm_gen, prompts[name], expected[name])
    finally:
        for tag, llm in engines.items():
            llm.shutdown()
            dumps[tag] = read_single_rank_dump(tmp_path, tag)

    assert set(dumps) == set(CTX_ENGINES + GEN_ENGINES)
    assert len({dump["pid"] for dump in dumps.values()}) == len(dumps)

    for tag in CTX_ENGINES:
        ctx_counters = counters(dumps[tag])
        # One prompt computed and published, the other found in the store and fetched whole;
        # the second round is served by the local radix tree.
        assert ctx_counters["publish_stored"] == NAMEABLE_BLOCKS, (tag, ctx_counters)
        assert ctx_counters["fetch_hits"] == NAMEABLE_BLOCKS, (tag, ctx_counters)
        assert ctx_counters["fetch_misses"] == 0, (tag, ctx_counters)
        # The first sight of the prompt this engine computed is the only probe that misses.
        assert ctx_counters["probe_misses"] == NAMEABLE_BLOCKS, (tag, ctx_counters)
        assert ctx_counters["probe_hits"] >= NAMEABLE_BLOCKS, (tag, ctx_counters)
        assert ctx_counters["failed_attempts"] == 0, (tag, ctx_counters)
        assert_no_leftover_records(dumps[tag])

    for tag in GEN_ENGINES:
        gen_counters = counters(dumps[tag])
        # Gen-init requests belong to the transceiver: the store is never asked.
        assert gen_counters["publish_stored"] == 0, (tag, gen_counters)
        assert gen_counters["fetch_hits"] == 0, (tag, gen_counters)
        assert gen_counters["probe_hits"] == 0, (tag, gen_counters)
        assert gen_counters["failed_attempts"] == 0, (tag, gen_counters)
        assert_no_leftover_records(dumps[tag])

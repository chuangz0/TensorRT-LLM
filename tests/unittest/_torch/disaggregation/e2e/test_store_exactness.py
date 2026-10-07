# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A fetched prefix is bit-exact with a locally reused one.

Comparing engine B (prefix fetched from the store, tail computed) against engine A (whole prompt
computed) only holds within numerical noise: "prefix from cache, tail computed" and "one pass"
run different kernels, and on a real model the greedy tokens can flip on that noise alone. The
exact reference for B is the same engine shape reusing the prefix from its own radix tree: same
pages, same tail computation. So:

* engine C0 (no store) computes the prompt from scratch;
* engine C1 (no store) first runs a sibling prompt that shares exactly the prompt's nameable full
  blocks, then the prompt: it reuses those blocks -- what B fetches -- and computes the same tail
  B computes (a plain second run of the prompt could also reuse the partial last block);
* engine A publishes, engine B fetches the full blocks from the store and computes the tail;
* B's tokens and last-prompt-position logits must equal C1's exactly, and A's equal C0's.
"""

import os
import time

import pytest
from mooncake_cluster import (
    KV_TRANSFER_CONFIG_ENV,
    KV_TRANSFER_STATUS_DUMP_ENV,
    LAYOUTS,
    MAX_TOKENS,
    MODELS,
    SELECTED_EXACT_LAYOUTS,
    SELECTED_MODELS,
    SETTLE_S,
    TOKENS_PER_BLOCK,
    TRANSPORTS,
    counters,
    dump_template,
    kv_cache_config,
    model_path_for,
    prompt_token_ids,
    read_rank_dumps,
    timeout_mark,
    world_size_of,
    write_kv_transfer_yaml,
)

pytestmark = [pytest.mark.threadleak(enabled=False), pytest.mark.private_mpi_session]
EXACT_LAYOUTS = {"tp1": {}, **LAYOUTS}


def _engine(model_path, layout, max_seq_len):
    from tensorrt_llm import LLM

    extra = {} if max_seq_len is None else {"max_seq_len": max_seq_len}
    return LLM(
        model=model_path,
        kv_cache_config=kv_cache_config(),
        disable_overlap_scheduler=True,
        **EXACT_LAYOUTS[layout],
        **extra,
    )


def _generate(llm, prompt):
    from tensorrt_llm.sampling_params import SamplingParams

    sp = SamplingParams(max_tokens=MAX_TOKENS, return_context_logits=True)
    (out,) = llm.generate([prompt], sp)
    return list(out.outputs[0].token_ids), out.context_logits[-1].float().cpu()


def _diff(a, b):
    return (a - b).abs().max().item()


@timeout_mark(1800)
@pytest.mark.parametrize("transport_store", TRANSPORTS, indirect=True)
@pytest.mark.parametrize("layout", SELECTED_EXACT_LAYOUTS)
@pytest.mark.parametrize("model", SELECTED_MODELS)
def test_fetched_prefix_is_exact(transport_store, model, layout, request, tmp_path, monkeypatch):
    import torch

    world_size = world_size_of(EXACT_LAYOUTS[layout])
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    _, prompt_len, max_seq_len, _ = MODELS[model]
    model_path = model_path_for(request, model)
    protocol, landing, master_address = transport_store
    prompt = prompt_token_ids(prompt_len=prompt_len)

    # C: the local-reuse reference, no store attached. Run 0 computes the prompt from scratch.
    # A sibling prompt sharing exactly the nameable full blocks then commits those blocks and a
    # different tail, so run 1 of the prompt reuses exactly what B fetches -- the full blocks --
    # and computes the same tail B computes (a plain second run could also reuse the partial
    # last block and compute fewer tokens than B). Under tp2_adp the reference holds only if the
    # ADP router sends the prompt to the rank that ran the sibling: an engine without a store
    # cannot show which rank reused, so a routing change would fail B == C1 for the wrong reason.
    nameable_tokens = (prompt_len - 1) // TOKENS_PER_BLOCK * TOKENS_PER_BLOCK
    sibling = prompt[:nameable_tokens] + [t + 1 for t in prompt[nameable_tokens:]]
    monkeypatch.delenv(KV_TRANSFER_CONFIG_ENV, raising=False)
    llm = _engine(model_path, layout, max_seq_len)
    try:
        c0 = _generate(llm, prompt)
    finally:
        llm.shutdown()
    llm = _engine(model_path, layout, max_seq_len)
    try:
        _generate(llm, sibling)
        c1 = _generate(llm, prompt)
    finally:
        llm.shutdown()

    namespace = f"e5-{os.getpid()}-{model}-{layout}-{protocol}-{landing}"
    monkeypatch.setenv(
        KV_TRANSFER_CONFIG_ENV,
        write_kv_transfer_yaml(
            tmp_path, master_address, namespace, protocol=protocol, landing=landing
        ),
    )
    if protocol == "rdma":
        monkeypatch.setenv("WITH_NVIDIA_PEERMEM", os.environ.get("WITH_NVIDIA_PEERMEM", "0"))
    results = {}
    for tag in ("a", "b"):
        monkeypatch.setenv(KV_TRANSFER_STATUS_DUMP_ENV, dump_template(tmp_path, tag))
        llm = _engine(model_path, layout, max_seq_len)
        try:
            results[tag] = _generate(llm, prompt)
            if tag == "a" and world_size > 1:
                time.sleep(SETTLE_S)
        finally:
            llm.shutdown()
    fetched = sum(counters(d)["fetch_hits"] for d in read_rank_dumps(tmp_path, "b", world_size))
    assert fetched > 0

    tokens_a, logits_a = results["a"]
    tokens_b, logits_b = results["b"]
    tokens_c0, logits_c0 = c0
    tokens_c1, logits_c1 = c1
    assert tokens_a == tokens_c0, (tokens_a, tokens_c0)
    assert torch.equal(logits_a, logits_c0), f"A vs C0 max_abs={_diff(logits_a, logits_c0)}"
    assert tokens_b == tokens_c1, (tokens_b, tokens_c1)
    assert torch.equal(logits_b, logits_c1), f"B vs C1 max_abs={_diff(logits_b, logits_c1)}"

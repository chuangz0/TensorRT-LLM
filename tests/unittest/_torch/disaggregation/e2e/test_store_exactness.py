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

import pytest
from mooncake_cluster import TRANSPORTS
from status_dumps import counters
from store_engine import (
    KV_TRANSFER_CONFIG_ENV,
    LAYOUTS,
    MODELS,
    SELECTED_EXACT_LAYOUTS,
    SELECTED_MODELS,
    TOKENS_PER_BLOCK,
    model_path_for,
    prompt_token_ids,
    run_engine,
    timeout_mark,
    world_size_of,
    write_kv_transfer_yaml,
)

pytestmark = [pytest.mark.threadleak(enabled=False), pytest.mark.private_mpi_session]
EXACT_LAYOUTS = {"tp1": {}, **LAYOUTS}


def _max_abs_diff(a, b) -> float:
    return (a - b).abs().max().item()


@timeout_mark(1800)
@pytest.mark.parametrize("transport_store", TRANSPORTS, indirect=True)
@pytest.mark.parametrize("layout", SELECTED_EXACT_LAYOUTS)
@pytest.mark.parametrize("model", SELECTED_MODELS)
def test_a_fetched_prefix_is_bit_exact_with_a_locally_reused_one(
    transport_store, model, layout, request, tmp_path, monkeypatch
):
    import torch

    params = EXACT_LAYOUTS[layout]
    world_size = world_size_of(params)
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    case = MODELS[model]
    model_path = model_path_for(request, model)
    protocol, landing, master_address = transport_store
    prompt = prompt_token_ids(prompt_len=case.prompt_len)

    def run(tag, prompts):
        return run_engine(
            tmp_path,
            monkeypatch,
            tag,
            model_path,
            prompts,
            layout=params,
            max_seq_len=case.max_seq_len,
            return_logits=True,
        )

    # C: the local-reuse reference, no store attached. Run 0 computes the prompt from scratch.
    # A sibling prompt sharing exactly the nameable full blocks then commits those blocks and a
    # different tail, so run 1 of the prompt reuses exactly what B fetches -- the full blocks --
    # and computes the same tail B computes (a plain second run could also reuse the partial
    # last block and compute fewer tokens than B). Under tp2_adp the reference holds only if the
    # ADP router sends the prompt to the rank that ran the sibling: an engine without a store
    # cannot show which rank reused, so a routing change would fail B == C1 for the wrong reason.
    nameable_tokens = (case.prompt_len - 1) // TOKENS_PER_BLOCK * TOKENS_PER_BLOCK
    sibling = prompt[:nameable_tokens] + [t + 1 for t in prompt[nameable_tokens:]]
    monkeypatch.delenv(KV_TRANSFER_CONFIG_ENV, raising=False)
    c0 = run(None, [prompt])
    c1 = run(None, [sibling, prompt])

    namespace = f"exactness-{os.getpid()}-{model}-{layout}-{protocol}-{landing}"
    monkeypatch.setenv(
        KV_TRANSFER_CONFIG_ENV,
        write_kv_transfer_yaml(
            tmp_path, master_address, namespace, protocol=protocol, landing=landing
        ),
    )
    if protocol == "rdma":
        monkeypatch.setenv("WITH_NVIDIA_PEERMEM", os.environ.get("WITH_NVIDIA_PEERMEM", "0"))
    a = run("a", [prompt])
    b = run("b", [prompt])
    fetched = sum(counters(d)["fetch_hits"] for d in b.dumps)
    assert fetched > 0

    tokens_a, logits_a = a.tokens[0], a.logits[0]
    tokens_b, logits_b = b.tokens[0], b.logits[0]
    tokens_c0, logits_c0 = c0.tokens[0], c0.logits[0]
    tokens_c1, logits_c1 = c1.tokens[1], c1.logits[1]  # the prompt's run, after the sibling's
    assert tokens_a == tokens_c0, (tokens_a, tokens_c0)
    assert torch.equal(logits_a, logits_c0), f"A vs C0 max_abs={_max_abs_diff(logits_a, logits_c0)}"
    assert tokens_b == tokens_c1, (tokens_b, tokens_c1)
    assert torch.equal(logits_b, logits_c1), f"B vs C1 max_abs={_max_abs_diff(logits_b, logits_c1)}"

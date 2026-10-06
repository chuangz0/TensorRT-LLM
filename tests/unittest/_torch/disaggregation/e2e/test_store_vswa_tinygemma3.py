# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The store path on a variable-sliding-window model.

Two engines, one prompt, one store, as ``test_store_e2e_tinyllama``, on a Gemma3 whose layers
alternate sliding and full attention, so the KV cache has a windowed group and a full-attention
group. The main path builds a random-weight four-layer Gemma3 with ``transformers`` in a temporary
directory (no model files needed); the real Gemma-3-1b-it runs when its weights are present.

Per-engine counters, with W the window, tpb = 32, L the prompt length, nameable = (L - 1) // tpb,
B = nameable * tpb the largest fetch target and stale_end(h) = (h + 1 - W) // tpb:

* Engine A computes and publishes every nameable full-attention block plus the windowed blocks
  live at history L, ``[stale_end(L), nameable)``.
* Engine B probes 2 * nameable names and fetches whenever the windowed blocks it needs at B,
  ``[stale_end(B), nameable)``, were all published, that is when stale_end(L) == stale_end(B).
  With W a multiple of tpb that fails exactly when L % tpb is 0 or tpb - 1: A dropped the window's
  first block at B before publishing, no smaller target does without it either, so B fetches
  nothing, computes locally and republishes what the store already holds.

  W = 128 (tiny): L = 330 gives nameable 10, stale_end(320) = stale_end(330) = 6, so A publishes
  10 + 4 = 14 and B fetches 14. L = 352 gives stale_end(352) = 7: A publishes 13, B fetches 0.
  W = 512 (real): L = 1000 gives 31 + 16 = 47 published and fetched; L = 1024 gives 46 and 0.

Counters are asserted exactly, and so are the generated tokens. The proof that the fetched pages
carry the right values is the first-step logits: B's last prompt token attends to the fetched
windowed and full-attention blocks, and its context logits at that position must match A's, which
computed everything locally, within bfloat16 tolerance.
"""

import os

import pytest
import torch
from mooncake_cluster import (
    KV_TRANSFER_CONFIG_ENV,
    KV_TRANSFER_STATUS_DUMP_ENV,
    MAX_TOKENS,
    TOKENS_PER_BLOCK,
    assert_no_leftover_records,
    counters,
    dump_template,
    kv_cache_config,
    prompt_token_ids,
    read_status_dump,
    timeout_mark,
    write_kv_transfer_yaml,
)

pytestmark = [pytest.mark.threadleak(enabled=False), pytest.mark.private_mpi_session]

TINY_WINDOW = 128
REAL_WINDOW = 512
MAX_SEQ_LEN = {"tiny": 1024, "real": 2048}
"""Bounds the full-attention window the engine derives for the full layers; comfortably above
every prompt plus its generation."""


@pytest.fixture(scope="module")
def tiny_gemma3_path(tmp_path_factory) -> str:
    """A four-layer Gemma3 with random weights: ``layer_types`` alternates sliding and full
    attention, which the engine derives ``max_attention_window=[128, None, 128, None]`` from."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config = transformers.Gemma3TextConfig(
        num_hidden_layers=4,
        sliding_window=TINY_WINDOW,
        sliding_window_pattern=2,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        vocab_size=32000,  # prompt ids stay below 31100
        max_position_embeddings=MAX_SEQ_LEN["tiny"],
    )
    assert list(config.layer_types) == ["sliding_attention", "full_attention"] * 2
    torch.manual_seed(0)
    model = transformers.Gemma3ForCausalLM(config).to(torch.bfloat16)
    path = str(tmp_path_factory.mktemp("tiny-gemma3"))
    model.save_pretrained(path)
    return path


def real_gemma3_path() -> str | None:
    """``$GEMMA3_MODEL_PATH``, else ``gemma/gemma-3-1b-it`` under ``$LLM_MODELS_ROOT``."""
    explicit = os.environ.get("GEMMA3_MODEL_PATH")
    if explicit:
        return explicit
    root = os.environ.get("LLM_MODELS_ROOT")
    if root:
        candidate = os.path.join(root, "gemma", "gemma-3-1b-it")
        if os.path.isdir(candidate):
            return candidate
    return None


def run_engine(
    tmp_path, monkeypatch, tag: str, model_path: str, prompt, *, max_seq_len: int, **llm_kwargs
):
    """One ``LLM()`` on integer prompts that generates once and shuts down; returns
    (tokens, first-step logits, status dump)."""
    from tensorrt_llm import LLM
    from tensorrt_llm.sampling_params import SamplingParams

    monkeypatch.setenv(KV_TRANSFER_STATUS_DUMP_ENV, dump_template(tmp_path, tag))
    llm = LLM(
        model=model_path,
        skip_tokenizer_init=True,
        kv_cache_config=kv_cache_config(),
        max_seq_len=max_seq_len,
        disable_overlap_scheduler=True,
        **llm_kwargs,
    )
    # No tokenizer, so the end id is given by hand (Gemma's <eos>); ignored anyway, so that every
    # run generates the same number of tokens whatever the random weights produce.
    sampling = SamplingParams(
        max_tokens=MAX_TOKENS, end_id=1, ignore_eos=True, return_context_logits=True
    )
    try:
        (output,) = llm.generate([prompt], sampling)
        tokens = list(output.outputs[0].token_ids)
        # The last prompt position: the only one whose logits both engines compute themselves.
        first_step_logits = output.context_logits[-1].detach().float().cpu()
    finally:
        llm.shutdown()
    return tokens, first_step_logits, read_status_dump(tmp_path, tag)


def run_two_engines_and_check(
    mooncake_cluster,
    tmp_path,
    monkeypatch,
    model_path: str,
    namespace: str,
    prompt_len: int,
    published_by_a: int,
    fetched_by_b: int,
    **llm_kwargs,
) -> None:
    """A then B on one prompt, both with ``llm_kwargs``; the exact counters, equal tokens and
    first-step logits within bfloat16 tolerance."""
    nameable = (prompt_len - 1) // TOKENS_PER_BLOCK
    config_path = write_kv_transfer_yaml(tmp_path, mooncake_cluster.master_address, namespace)
    monkeypatch.setenv(KV_TRANSFER_CONFIG_ENV, config_path)
    prompt = prompt_token_ids(prompt_len=prompt_len)

    tokens_a, logits_a, dump_a = run_engine(
        tmp_path, monkeypatch, "a", model_path, prompt, **llm_kwargs
    )
    tokens_b, logits_b, dump_b = run_engine(
        tmp_path, monkeypatch, "b", model_path, prompt, **llm_kwargs
    )

    assert len(tokens_a) == MAX_TOKENS
    assert tokens_a == tokens_b
    assert logits_a.shape == logits_b.shape
    torch.testing.assert_close(logits_b, logits_a, atol=1e-2, rtol=1e-2)

    counters_a = counters(dump_a)
    assert counters_a["publish_stored"] == published_by_a, counters_a
    assert counters_a["fetch_hits"] == 0 and counters_a["fetch_misses"] == 0, counters_a
    assert counters_a["failed_attempts"] == 0, counters_a
    assert_no_leftover_records(dump_a)

    counters_b = counters(dump_b)
    # B asks about every nameable block of both groups; the store holds what A published.
    assert counters_b["probe_hits"] == published_by_a, counters_b
    assert counters_b["probe_misses"] == 2 * nameable - published_by_a, counters_b
    assert counters_b["fetch_hits"] == fetched_by_b, counters_b
    assert counters_b["fetch_misses"] == 0, counters_b
    assert counters_b["failed_attempts"] == 0, counters_b
    # Whether fetched or recomputed, what B offers at the end is already in the store.
    assert counters_b["publish_stored"] == 0, counters_b
    assert_no_leftover_records(dump_b)


CASES = [
    pytest.param("tiny", 330, 14, 14, id="tiny-L330-window-aligned"),
    pytest.param("tiny", 352, 13, 0, id="tiny-L352-window-one-block-ahead"),
    pytest.param("real", 1000, 47, 47, id="gemma3_1b-L1000-window-aligned"),
    pytest.param("real", 1024, 46, 0, id="gemma3_1b-L1024-window-one-block-ahead"),
]


@timeout_mark(900)
@pytest.mark.parametrize("model, prompt_len, published_by_a, fetched_by_b", CASES)
def test_store_fetch_on_a_vswa_model(
    mooncake_cluster,
    tmp_path,
    monkeypatch,
    request,
    model,
    prompt_len,
    published_by_a,
    fetched_by_b,
):
    if model == "tiny":
        model_path = request.getfixturevalue("tiny_gemma3_path")
    else:
        model_path = real_gemma3_path()
        if model_path is None:
            pytest.skip("Gemma-3-1b-it weights not found (GEMMA3_MODEL_PATH or LLM_MODELS_ROOT)")
    run_two_engines_and_check(
        mooncake_cluster,
        tmp_path,
        monkeypatch,
        model_path,
        f"e3-{os.getpid()}-{model}-{prompt_len}",
        prompt_len,
        published_by_a,
        fetched_by_b,
        max_seq_len=MAX_SEQ_LEN[model],
    )


CHUNK_TOKENS = 64
"""Two blocks per context chunk: the 330-token prompt prefills in six chunks on A and, after
the fetch lands at 320, in one on B."""


@timeout_mark(900)
def test_store_fetch_on_a_vswa_model_with_chunked_prefill(
    mooncake_cluster, tmp_path, monkeypatch, tiny_gemma3_path
):
    """Chunked prefill on both engines, on the tiny model's window-aligned case: A publishes
    once its last chunk has committed, B's first chunk is the fetch and the tail runs in chunks
    afterwards; counters, tokens and first-step logits are those of the unchunked run."""
    run_two_engines_and_check(
        mooncake_cluster,
        tmp_path,
        monkeypatch,
        tiny_gemma3_path,
        f"e3-chunked-{os.getpid()}",
        330,
        14,
        14,
        max_seq_len=MAX_SEQ_LEN["tiny"],
        enable_chunked_prefill=True,
        max_num_tokens=CHUNK_TOKENS,
        max_batch_size=4,
    )

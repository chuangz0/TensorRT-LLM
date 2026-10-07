# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The store path with more than one rank per engine, over loopback TCP and over the NIC.

Engine A (a TP / PP / attention-DP layout) computes a prompt of seven full blocks and publishes
them; engine B, same layout, started after A shut down, probes the store, fetches the blocks and
generates the same tokens. Every rank writes its own status dump.

- TP and PP: every rank holds a shard of every block (TP: its heads, PP: its layers), its layout
  fingerprint differs from its peers', so every rank publishes and fetches all seven blocks of
  its own shard.
- attention-DP: the request runs on one replica (heads=all, so replicas share names); the
  counters are summed over ranks.

Transports: ``tcp`` (loopback, host landing); ``rdma_device`` (store reads and writes the
KV pools directly through the NIC) and ``rdma_host`` (pinned host landing over the NIC), enabled by
``KV_TRANSFER_E2E_RDMA=1``; ``KV_TRANSFER_E2E_RDMA_DEVICES`` lists the HCAs (empty: Mooncake
discovers them). GPU memory is registered through dma-buf: the Mooncake wheel registers it with
``ibv_reg_mr`` (the nvidia-peermem path) unless ``WITH_NVIDIA_PEERMEM=0``. The model is
``KV_TRANSFER_E2E_MODELS`` (``MODELS`` in ``mooncake_cluster.py``, TinyLlama by default).
"""

import os

import pytest
from mooncake_cluster import (
    KV_TRANSFER_CONFIG_ENV,
    KV_TRANSFER_STATUS_DUMP_ENV,
    LAYOUTS,
    MODELS,
    SELECTED_MODELS,
    TOKENS_PER_BLOCK,
    TRANSPORTS,
    assert_no_leftover_records,
    counters,
    dump_template,
    generate_ids,
    kv_cache_config,
    landing_of,
    model_path_for,
    prompt_token_ids,
    read_rank_dumps,
    timeout_mark,
    world_size_of,
    write_kv_transfer_yaml,
)

pytestmark = [pytest.mark.threadleak(enabled=False), pytest.mark.private_mpi_session]


def torch_device_count() -> int:
    import torch

    return torch.cuda.device_count()


def run_engine(
    tmp_path,
    monkeypatch,
    tag,
    model_path,
    prompt,
    layout,
    *,
    disable_overlap,
    max_seq_len=None,
):
    from tensorrt_llm import LLM

    monkeypatch.setenv(KV_TRANSFER_STATUS_DUMP_ENV, dump_template(tmp_path, tag))
    extra = {} if max_seq_len is None else {"max_seq_len": max_seq_len}
    llm = LLM(
        model=model_path,
        kv_cache_config=kv_cache_config(),
        disable_overlap_scheduler=disable_overlap,
        **LAYOUTS[layout],
        **extra,
    )
    try:
        tokens = generate_ids(llm, prompt)
    finally:
        llm.shutdown()
    return tokens


@timeout_mark(1500)
@pytest.mark.parametrize("transport_store", TRANSPORTS, indirect=True)
@pytest.mark.parametrize("layout", list(LAYOUTS))
@pytest.mark.parametrize("model", SELECTED_MODELS)
def test_store_fetch_multi_rank(transport_store, layout, model, request, tmp_path, monkeypatch):
    params = LAYOUTS[layout]
    world_size = world_size_of(params)
    if torch_device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    _, prompt_len, max_seq_len, vswa = MODELS[model]
    model_path = model_path_for(request, model)
    protocol, landing, master_address = transport_store
    namespace = f"e4-{os.getpid()}-{model}-{layout}-{protocol}-{landing}"
    monkeypatch.setenv(
        KV_TRANSFER_CONFIG_ENV,
        write_kv_transfer_yaml(
            tmp_path, master_address, namespace, protocol=protocol, landing=landing
        ),
    )
    if protocol == "rdma":
        monkeypatch.setenv("WITH_NVIDIA_PEERMEM", os.environ.get("WITH_NVIDIA_PEERMEM", "0"))
    prompt = prompt_token_ids(prompt_len=prompt_len)
    nameable = (prompt_len - 1) // TOKENS_PER_BLOCK

    tokens_a = run_engine(
        tmp_path,
        monkeypatch,
        "a",
        model_path,
        prompt,
        layout,
        disable_overlap=True,
        max_seq_len=max_seq_len,
    )
    tokens_b = run_engine(
        tmp_path,
        monkeypatch,
        "b",
        model_path,
        prompt,
        layout,
        disable_overlap=False,
        max_seq_len=max_seq_len,
    )
    assert len(tokens_a) > 0
    assert tokens_a == tokens_b, (tokens_a, tokens_b)

    dumps_a = read_rank_dumps(tmp_path, "a", world_size)
    dumps_b = read_rank_dumps(tmp_path, "b", world_size)
    expected_landing = landing or "host"
    for dump in dumps_a + dumps_b:
        assert landing_of(dump) == expected_landing
    counters_a = [counters(d) for d in dumps_a]
    counters_b = [counters(d) for d in dumps_b]

    for per_rank in counters_a + counters_b:
        assert per_rank["failed_attempts"] == 0, per_rank
        assert per_rank["fetch_misses"] == 0, per_rank
    # Full attention: every nameable block. Sliding windows add the live windowed blocks, so
    # there the check is that B fetched exactly what A published, and more than the full prefix.
    if params.get("enable_attention_dp"):
        published = sum(c["publish_stored"] for c in counters_a)
        fetched = sum(c["fetch_hits"] for c in counters_b)
        assert published == fetched > 0, (counters_a, counters_b)
        if not vswa:
            assert fetched == nameable, counters_b
    else:
        for ca, cb in zip(counters_a, counters_b):
            assert ca["publish_stored"] == cb["fetch_hits"] > 0, (counters_a, counters_b)
            assert ca["fetch_hits"] == 0 and cb["publish_stored"] == 0, (counters_a, counters_b)
            if vswa:
                assert cb["fetch_hits"] > nameable, counters_b
            else:
                assert cb["fetch_hits"] == nameable, counters_b
    for dump in dumps_a + dumps_b:
        assert_no_leftover_records(dump)

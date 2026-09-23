# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E1 (integration plan §11): two engines, one prompt, one store.

Engine A computes a prompt of seven full blocks and publishes them to a Mooncake store over
loopback TCP; engine B, started after A is gone, probes the store, fetches the seven
blocks into its own host memory first, places them into pages once the scheduler reserved them,
computes only the tail, and generates the same tokens (the store runs over TCP, so the YAML's
unset ``landing`` resolves to ``host``). Counters come from the status dump each engine writes at
``close``. B runs with and without the overlap scheduler, which is what proves the overlap-loop
publish point offers pages whose forward has completed.
"""

import os

import pytest
from mooncake_cluster import (
    KV_TRANSFER_CONFIG_ENV,
    KV_TRANSFER_STATUS_DUMP_ENV,
    NAMEABLE_BLOCKS,
    assert_no_leftover_records,
    counters,
    dump_template,
    generate_ids,
    kv_cache_config,
    landing_of,
    prompt_token_ids,
    read_status_dump,
    timeout_mark,
    write_kv_transfer_yaml,
)

pytestmark = [pytest.mark.threadleak(enabled=False), pytest.mark.private_mpi_session]


def run_engine(
    tmp_path, monkeypatch, tag: str, model_path: str, prompt, *, disable_overlap_scheduler: bool
):
    """One ``LLM()`` that generates once and shuts down; returns (tokens, status dump)."""
    from tensorrt_llm import LLM

    monkeypatch.setenv(KV_TRANSFER_STATUS_DUMP_ENV, dump_template(tmp_path, tag))
    llm = LLM(
        model=model_path,
        kv_cache_config=kv_cache_config(),
        disable_overlap_scheduler=disable_overlap_scheduler,
    )
    try:
        tokens = generate_ids(llm, prompt)
    finally:
        llm.shutdown()
    return tokens, read_status_dump(tmp_path, tag)


RDMA_ENV = "KV_TRANSFER_E2E_RDMA"
"""Set to ``1`` on a machine whose transport writes GPU memory directly: the explicit
``landing: device`` variant registers the KV pools with Mooncake, which over loopback TCP is the
unverified path (integration plan §11 #12), so it is skipped by default."""

LANDINGS = [
    pytest.param(None, id="landing_default_host"),
    pytest.param(
        "device",
        id="landing_device",
        marks=pytest.mark.skipif(
            os.environ.get(RDMA_ENV) != "1", reason=f"{RDMA_ENV}=1 only: GPU-direct landing"
        ),
    ),
]


@timeout_mark(600)
@pytest.mark.parametrize("landing", LANDINGS)
@pytest.mark.parametrize(
    "disable_overlap_scheduler_b", [True, False], ids=["no_overlap", "overlap"]
)
def test_store_fetch_two_instances(
    mooncake_cluster, tinyllama_path, tmp_path, monkeypatch, disable_overlap_scheduler_b, landing
):
    namespace = f"e1-{os.getpid()}-{int(disable_overlap_scheduler_b)}-{landing or 'default'}"
    config_path = write_kv_transfer_yaml(
        tmp_path, mooncake_cluster.master_address, namespace, landing=landing
    )
    monkeypatch.setenv(KV_TRANSFER_CONFIG_ENV, config_path)
    prompt = prompt_token_ids()

    tokens_a, dump_a = run_engine(
        tmp_path, monkeypatch, "a", tinyllama_path, prompt, disable_overlap_scheduler=True
    )
    tokens_b, dump_b = run_engine(
        tmp_path,
        monkeypatch,
        "b",
        tinyllama_path,
        prompt,
        disable_overlap_scheduler=disable_overlap_scheduler_b,
    )

    assert len(tokens_a) > 0
    assert tokens_a == tokens_b

    assert dump_a["started_at"] < dump_b["started_at"]
    assert dump_a["pid"] != dump_b["pid"]
    # With no landing in the YAML, the factory lands on host memory first over TCP.
    expected_landing = landing or "host"
    assert landing_of(dump_a) == expected_landing and landing_of(dump_b) == expected_landing

    counters_a = counters(dump_a)
    assert counters_a["publish_stored"] == NAMEABLE_BLOCKS, counters_a
    assert counters_a["fetch_hits"] == 0 and counters_a["fetch_misses"] == 0, counters_a
    assert counters_a["failed_attempts"] == 0, counters_a
    assert_no_leftover_records(dump_a)

    counters_b = counters(dump_b)
    assert counters_b["fetch_hits"] == NAMEABLE_BLOCKS, counters_b
    assert counters_b["fetch_misses"] == 0, counters_b
    assert counters_b["failed_attempts"] == 0, counters_b
    assert counters_b["probe_hits"] == NAMEABLE_BLOCKS, counters_b
    # B recomputed nothing the store had: what it offers is already present.
    assert counters_b["publish_stored"] == 0, counters_b
    assert_no_leftover_records(dump_b)

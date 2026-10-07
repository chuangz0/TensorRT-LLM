# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two engines, one prompt, one store.

Engine A computes a prompt of seven full blocks and publishes them to a Mooncake store over
loopback TCP; engine B, started after A is gone, probes the store, fetches the seven
blocks into its own host memory first, places them into pages once the scheduler reserved them,
computes only the tail, and generates the same tokens (the store runs over TCP, so the YAML's
unset ``landing`` resolves to ``host``). Counters come from the status dump each engine writes at
``close``. B runs with and without the overlap scheduler, which is what proves the overlap-loop
publish point offers pages whose forward has completed; a further case runs A with it, which
proves the same point for the publisher. The config without its coordinator timeouts and a batch
with the same prompt twice are the two remaining variations of the same round trip.
"""

import os

import pytest
from mooncake_cluster import (
    KV_TRANSFER_CONFIG_ENV,
    KV_TRANSFER_STATUS_DUMP_ENV,
    NAMEABLE_BLOCKS,
    RDMA_ENV,
    assert_no_leftover_records,
    counters,
    dump_template,
    generate_ids,
    kv_cache_config,
    landing_of,
    prompt_token_ids,
    read_status_dump,
    sampling_params,
    timeout_mark,
    write_kv_transfer_yaml,
)

pytestmark = [pytest.mark.threadleak(enabled=False), pytest.mark.private_mpi_session]


def run_engine(
    tmp_path,
    monkeypatch,
    tag: str,
    model_path: str,
    prompts,
    *,
    disable_overlap_scheduler: bool,
    batch: bool = False,
):
    """One ``LLM()`` that generates each prompt once and shuts down: one ``generate`` call per
    prompt, or all of them in one call with ``batch``. Returns (tokens per prompt, status dump)."""
    from tensorrt_llm import LLM

    monkeypatch.setenv(KV_TRANSFER_STATUS_DUMP_ENV, dump_template(tmp_path, tag))
    llm = LLM(
        model=model_path,
        kv_cache_config=kv_cache_config(),
        disable_overlap_scheduler=disable_overlap_scheduler,
    )
    try:
        if batch:
            outputs = llm.generate(prompts, sampling_params())
            tokens = [list(output.outputs[0].token_ids) for output in outputs]
        else:
            tokens = [generate_ids(llm, prompt) for prompt in prompts]
    finally:
        llm.shutdown()
    return tokens, read_status_dump(tmp_path, tag)


def assert_a_published_and_b_fetched_everything(
    dump_a, dump_b, *, probe_hits_by_b: int = NAMEABLE_BLOCKS
):
    """A published the one prompt's blocks and fetched nothing; B's probes found exactly
    ``probe_hits_by_b`` (one lookup of the prompt's blocks per request B saw), it fetched the
    prompt's blocks once, missed nothing, failed nothing and offered nothing the store lacked."""
    counters_a = counters(dump_a)
    assert counters_a["publish_stored"] == NAMEABLE_BLOCKS, counters_a
    assert counters_a["fetch_hits"] == 0 and counters_a["fetch_misses"] == 0, counters_a
    assert counters_a["failed_attempts"] == 0, counters_a
    assert_no_leftover_records(dump_a)

    counters_b = counters(dump_b)
    assert counters_b["fetch_hits"] == NAMEABLE_BLOCKS, counters_b
    assert counters_b["fetch_misses"] == 0, counters_b
    assert counters_b["failed_attempts"] == 0, counters_b
    assert counters_b["probe_hits"] == probe_hits_by_b, counters_b
    # B recomputed nothing the store had: what it offers is already present.
    assert counters_b["publish_stored"] == 0, counters_b
    assert_no_leftover_records(dump_b)


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
"""The explicit ``landing: device`` variant registers the KV pools with Mooncake, which over
loopback TCP is the unverified path: it runs only with ``RDMA_ENV`` set, on a machine whose
transport writes GPU memory directly."""


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

    (tokens_a,), dump_a = run_engine(
        tmp_path, monkeypatch, "a", tinyllama_path, [prompt], disable_overlap_scheduler=True
    )
    (tokens_b,), dump_b = run_engine(
        tmp_path,
        monkeypatch,
        "b",
        tinyllama_path,
        [prompt],
        disable_overlap_scheduler=disable_overlap_scheduler_b,
    )

    assert len(tokens_a) > 0
    assert tokens_a == tokens_b

    assert dump_a["started_at"] < dump_b["started_at"]
    assert dump_a["pid"] != dump_b["pid"]
    # With no landing in the YAML, the factory lands on host memory first over TCP.
    expected_landing = landing or "host"
    assert landing_of(dump_a) == expected_landing and landing_of(dump_b) == expected_landing
    assert_a_published_and_b_fetched_everything(dump_a, dump_b)


@timeout_mark(600)
def test_engine_a_with_the_overlap_scheduler_publishes_every_block(
    mooncake_cluster, tinyllama_path, tmp_path, monkeypatch
):
    """The overlap loop's publish point offers the previous batch's pages, whose forward has
    completed: a publisher running it stores every block, and the fetcher reads them whole."""
    namespace = f"e1-overlap-a-{os.getpid()}"
    config_path = write_kv_transfer_yaml(tmp_path, mooncake_cluster.master_address, namespace)
    monkeypatch.setenv(KV_TRANSFER_CONFIG_ENV, config_path)
    prompt = prompt_token_ids()

    (tokens_a,), dump_a = run_engine(
        tmp_path, monkeypatch, "a", tinyllama_path, [prompt], disable_overlap_scheduler=False
    )
    (tokens_b,), dump_b = run_engine(
        tmp_path, monkeypatch, "b", tinyllama_path, [prompt], disable_overlap_scheduler=True
    )

    assert len(tokens_a) > 0 and tokens_a == tokens_b
    assert_a_published_and_b_fetched_everything(dump_a, dump_b)


@timeout_mark(600)
def test_config_without_coordinator_timeouts_runs_on_the_defaults(
    mooncake_cluster, tinyllama_path, tmp_path, monkeypatch
):
    """A config file naming only the backend: the coordinator's timeouts take their defaults
    and the round trip is the same as with them written out."""
    namespace = f"e1-defaults-{os.getpid()}"
    config_path = write_kv_transfer_yaml(
        tmp_path, mooncake_cluster.master_address, namespace, omit_coordinator_timeouts=True
    )
    monkeypatch.setenv(KV_TRANSFER_CONFIG_ENV, config_path)
    prompt = prompt_token_ids()

    (tokens_a,), dump_a = run_engine(
        tmp_path, monkeypatch, "a", tinyllama_path, [prompt], disable_overlap_scheduler=True
    )
    (tokens_b,), dump_b = run_engine(
        tmp_path, monkeypatch, "b", tinyllama_path, [prompt], disable_overlap_scheduler=True
    )

    assert len(tokens_a) > 0 and tokens_a == tokens_b
    assert_a_published_and_b_fetched_everything(dump_a, dump_b)


@timeout_mark(600)
def test_the_same_prompt_twice_in_one_batch_generates_alike_and_leaves_nothing_behind(
    mooncake_cluster, tinyllama_path, tmp_path, monkeypatch
):
    """Engine A publishes two prompts; engine B is given the first twice and the second once in
    one batch. Every output matches A's. The duplicates either both fetch (planned in the same
    round) or the second is served by local reuse once the first has landed, so B fetches the
    second prompt's seven blocks plus seven or fourteen for the first; nothing fails and no
    record is left."""
    namespace = f"e1-duplicates-{os.getpid()}"
    config_path = write_kv_transfer_yaml(tmp_path, mooncake_cluster.master_address, namespace)
    monkeypatch.setenv(KV_TRANSFER_CONFIG_ENV, config_path)
    first, second = prompt_token_ids(seed=1), prompt_token_ids(seed=2)

    (tokens_first, tokens_second), dump_a = run_engine(
        tmp_path, monkeypatch, "a", tinyllama_path, [first, second], disable_overlap_scheduler=True
    )
    tokens_b, dump_b = run_engine(
        tmp_path,
        monkeypatch,
        "b",
        tinyllama_path,
        [first, first, second],
        disable_overlap_scheduler=True,
        batch=True,
    )

    assert tokens_first != tokens_second
    assert tokens_b == [tokens_first, tokens_first, tokens_second]
    counters_a = counters(dump_a)
    assert counters_a["publish_stored"] == 2 * NAMEABLE_BLOCKS, counters_a
    assert counters_a["failed_attempts"] == 0, counters_a
    assert_no_leftover_records(dump_a)
    counters_b = counters(dump_b)
    # One lookup per request: the duplicates share a pending lookup for one round, then the
    # second asks on its own, so three lookups of seven blocks answer.
    assert counters_b["probe_hits"] == 3 * NAMEABLE_BLOCKS, counters_b
    assert counters_b["fetch_hits"] in (2 * NAMEABLE_BLOCKS, 3 * NAMEABLE_BLOCKS), counters_b
    assert counters_b["fetch_misses"] == 0 and counters_b["failed_attempts"] == 0, counters_b
    assert counters_b["publish_stored"] == 0, counters_b
    assert_no_leftover_records(dump_b)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The store path when something is short or goes away: the engine must finish every request
with the right tokens, fail nothing it could compute itself, hang nowhere, and leave no record.

Engine A publishes, engine B fetches, as in ``test_store_e2e_tinyllama``; what differs is what
B meets: a store too small for the prompt (its publish partly declined, the oldest objects
evicted), a master that is gone before the fetch, a landing pool too small for the plan, a
request aborted while its fetch may be in flight, and a shutdown with a fetch possibly in flight.
Counters come from the status dumps; the fixture's teardown proves no Mooncake process survived.
"""

import os

import pytest
from mooncake_cluster import (
    KV_TRANSFER_CONFIG_ENV,
    KV_TRANSFER_STATUS_DUMP_ENV,
    MAX_TOKENS,
    NAMEABLE_BLOCKS,
    TOKENS_PER_BLOCK,
    assert_no_leftover_records,
    counters,
    dump_template,
    generate_ids,
    kv_cache_config,
    prompt_token_ids,
    read_status_dump,
    sampling_params,
    timeout_mark,
    write_kv_transfer_yaml,
)

pytestmark = [pytest.mark.threadleak(enabled=False), pytest.mark.private_mpi_session]

SMALL_SEGMENT_BYTES = 16 << 20
"""The smallest segment the Mooncake client accepts (8 MiB is refused). A TinyLlama block is
22 layers x 2 x 4 KV heads x 64 x 32 tokens x 2 B = 704 KiB, so the segment holds about 22
blocks and the master evicts the oldest objects once it is 90% full."""
OVERFLOW_PROMPT_LEN = 24 * TOKENS_PER_BLOCK + 2
"""24 nameable blocks: more than the small segment holds."""


def start_engine(tmp_path, monkeypatch, tag: str, model_path: str, **llm_kwargs):
    from tensorrt_llm import LLM

    monkeypatch.setenv(KV_TRANSFER_STATUS_DUMP_ENV, dump_template(tmp_path, tag))
    return LLM(
        model=model_path,
        kv_cache_config=kv_cache_config(),
        disable_overlap_scheduler=True,
        **llm_kwargs,
    )


def run_engine(tmp_path, monkeypatch, tag: str, model_path: str, prompt):
    """One ``LLM()`` that generates once and shuts down; returns (tokens, status dump)."""
    llm = start_engine(tmp_path, monkeypatch, tag, model_path)
    try:
        tokens = generate_ids(llm, prompt)
    finally:
        llm.shutdown()
    return tokens, read_status_dump(tmp_path, tag)


def write_config(tmp_path, monkeypatch, cluster, namespace: str, **yaml_kwargs) -> None:
    config_path = write_kv_transfer_yaml(tmp_path, cluster.master_address, namespace, **yaml_kwargs)
    monkeypatch.setenv(KV_TRANSFER_CONFIG_ENV, config_path)


@timeout_mark(600)
@pytest.mark.parametrize("mooncake_cluster", [SMALL_SEGMENT_BYTES], indirect=True)
def test_store_too_small_for_the_prompt_declines_the_overflow_and_the_rest_is_computed(
    mooncake_cluster, tinyllama_path, tmp_path, monkeypatch
):
    """A's publish of 24 blocks does not fit: the store declines the overflow (not a failure)
    and evicts its oldest objects. B fetches whatever prefix is still there, computes the rest
    and offers it; neither engine fails anything, and both generate the same tokens."""
    write_config(tmp_path, monkeypatch, mooncake_cluster, f"fault-small-{os.getpid()}")
    prompt = prompt_token_ids(prompt_len=OVERFLOW_PROMPT_LEN)
    blocks = (OVERFLOW_PROMPT_LEN - 1) // TOKENS_PER_BLOCK

    tokens_a, dump_a = run_engine(tmp_path, monkeypatch, "a", tinyllama_path, prompt)
    tokens_b, dump_b = run_engine(tmp_path, monkeypatch, "b", tinyllama_path, prompt)

    assert len(tokens_a) == MAX_TOKENS and tokens_a == tokens_b
    counters_a = counters(dump_a)
    assert 0 < counters_a["publish_stored"] < blocks, counters_a
    assert counters_a["publish_declined"] >= 1, counters_a
    assert counters_a["publish_stored"] + counters_a["publish_declined"] == blocks, counters_a
    assert counters_a["failed_attempts"] == 0, counters_a
    assert_no_leftover_records(dump_a)
    counters_b = counters(dump_b)
    assert counters_b["fetch_hits"] <= counters_b["probe_hits"] < blocks, counters_b
    assert counters_b["failed_attempts"] == 0, counters_b
    assert_no_leftover_records(dump_b)


def assert_settled_or_one_publish_still_in_flight(dump: dict) -> None:
    """The two states a dump may show for a request whose publish met a dead master: the held
    request terminated and forgotten, or still held with its one publish record in flight."""
    coordinator = dump["coordinator"]
    records = coordinator["records"]
    if not records:
        assert coordinator["finished_pending"] == [], coordinator
        return
    (record,) = records
    assert record["direction"] == "publish" and record["state"] == "IN_FLIGHT", coordinator
    assert coordinator["finished_pending"] == [record["request_id"]], coordinator


@timeout_mark(600)
def test_master_gone_before_the_fetch_computes_locally_and_does_not_hang(
    mooncake_cluster, tinyllama_path, tmp_path, monkeypatch
):
    """B is up and connected when the master dies. Its probe cannot be answered, so the planner
    runs out its budget and the request computes locally; its publish fails against the dead
    master, which is a warning, not a request failure. B generates A's tokens and shuts down.

    What the dump shows is fixed by ``close``, which waits for the backend's threads: the
    publish's one store call and the lookup's retried calls each fail after the Mooncake
    client's RPC timeout (about 4 s here), well inside ``close_timeout_s`` (30 s), so both
    failures are counted. Whether the loop terminated the held request before ``shutdown`` or
    ``close`` freed it depends on when the publish's call returns, which the LLM API cannot
    observe, so the dump shows either the settled table or exactly that one publish record.
    """
    write_config(tmp_path, monkeypatch, mooncake_cluster, f"fault-master-{os.getpid()}")
    prompt = prompt_token_ids()

    tokens_a, dump_a = run_engine(tmp_path, monkeypatch, "a", tinyllama_path, prompt)
    assert counters(dump_a)["publish_stored"] == NAMEABLE_BLOCKS

    llm_b = start_engine(tmp_path, monkeypatch, "b", tinyllama_path)
    try:
        mooncake_cluster.kill_master()
        tokens_b = generate_ids(llm_b, prompt)
    finally:
        llm_b.shutdown()
    dump_b = read_status_dump(tmp_path, "b")

    assert tokens_b == tokens_a
    counters_b = counters(dump_b)
    assert counters_b["fetch_hits"] == 0 and counters_b["fetch_misses"] == 0, counters_b
    assert counters_b["probe_hits"] == 0, counters_b
    # The lookup and the publish both met the dead master; each is reported as what it is. A
    # lookup that fails fast is asked once more within the probe budget, hence at least one.
    assert counters_b["probe_failed"] >= 1, counters_b
    assert counters_b["failed_attempts"] == 1, counters_b  # the one publish
    assert_settled_or_one_publish_still_in_flight(dump_b)


@timeout_mark(600)
def test_landing_pool_too_small_for_the_plan_gives_the_fetch_up_and_computes_locally(
    mooncake_cluster, tinyllama_path, tmp_path, monkeypatch
):
    """``max_landed_units: 3`` leaves the landing pool three slots; a plan of seven units can
    never land, so the landing fails at once, the one retry fails the same way, and the request
    computes locally: two failed attempts, no fetch, the right tokens."""
    write_config(
        tmp_path,
        monkeypatch,
        mooncake_cluster,
        f"fault-landing-{os.getpid()}",
        backend_overrides={"max_landed_units": 3},
    )
    prompt = prompt_token_ids()

    tokens_a, dump_a = run_engine(tmp_path, monkeypatch, "a", tinyllama_path, prompt)
    tokens_b, dump_b = run_engine(tmp_path, monkeypatch, "b", tinyllama_path, prompt)

    assert tokens_a == tokens_b
    assert counters(dump_a)["publish_stored"] == NAMEABLE_BLOCKS
    counters_b = counters(dump_b)
    assert counters_b["probe_hits"] == NAMEABLE_BLOCKS, counters_b
    assert counters_b["fetch_hits"] == 0 and counters_b["fetch_misses"] == 0, counters_b
    assert counters_b["failed_attempts"] == 2, counters_b  # the plan and its one retry
    assert counters_b["landings_held"] == 0, counters_b
    assert counters_b["publish_stored"] == 0, counters_b  # the store already holds them
    assert_no_leftover_records(dump_b)


@timeout_mark(600)
def test_abort_right_after_a_concurrent_submit_terminates_once_and_the_next_request_is_served(
    mooncake_cluster, tinyllama_path, tmp_path, monkeypatch
):
    """Two requests for the published prompt submitted together, the first aborted at once:
    whether the abort finds its fetch in flight (the cancel waits for the landing) or already
    done, the request ends exactly once, the other request generates A's tokens, and a request
    afterwards is served; no record is left behind."""
    write_config(tmp_path, monkeypatch, mooncake_cluster, f"fault-abort-{os.getpid()}")
    prompt = prompt_token_ids()

    tokens_a, dump_a = run_engine(tmp_path, monkeypatch, "a", tinyllama_path, prompt)
    assert counters(dump_a)["publish_stored"] == NAMEABLE_BLOCKS

    llm_b = start_engine(tmp_path, monkeypatch, "b", tinyllama_path)
    try:
        aborted = llm_b.generate_async(prompt, sampling_params())
        kept = llm_b.generate_async(prompt, sampling_params())
        aborted.abort()
        aborted_output = aborted.result(timeout=120)
        kept_output = kept.result(timeout=120)
        later = generate_ids(llm_b, prompt)
    finally:
        llm_b.shutdown()
    dump_b = read_status_dump(tmp_path, "b")

    assert list(kept_output.outputs[0].token_ids) == tokens_a
    assert later == tokens_a
    assert aborted_output.outputs[0].finish_reason in ("cancelled", "length")
    assert (
        list(aborted_output.outputs[0].token_ids)
        == tokens_a[: len(aborted_output.outputs[0].token_ids)]
    )
    counters_b = counters(dump_b)
    assert counters_b["failed_attempts"] == 0 and counters_b["fetch_misses"] == 0, counters_b
    assert counters_b["fetch_hits"] >= NAMEABLE_BLOCKS, counters_b
    assert_no_leftover_records(dump_b)


@timeout_mark(600)
def test_shutdown_with_a_fetch_possibly_in_flight_returns_and_writes_the_dump(
    mooncake_cluster, tinyllama_path, tmp_path, monkeypatch
):
    """``shutdown`` right after a request for the published prompt was submitted: the fetch may
    be probing, landing or placing. The layer's ``close`` stops the backends, frees what it owns
    and writes its dump; the process exits with no Mooncake helper left behind."""
    write_config(tmp_path, monkeypatch, mooncake_cluster, f"fault-shutdown-{os.getpid()}")
    prompt = prompt_token_ids()

    tokens_a, dump_a = run_engine(tmp_path, monkeypatch, "a", tinyllama_path, prompt)
    assert counters(dump_a)["publish_stored"] == NAMEABLE_BLOCKS

    llm_b = start_engine(tmp_path, monkeypatch, "b", tinyllama_path)
    try:
        llm_b.generate_async(prompt, sampling_params())
    finally:
        llm_b.shutdown()
    dump_b = read_status_dump(tmp_path, "b")

    counters_b = counters(dump_b)
    assert counters_b["failed_attempts"] == 0, counters_b
    assert counters_b["landings_held"] == 0, counters_b  # close gave every landing back

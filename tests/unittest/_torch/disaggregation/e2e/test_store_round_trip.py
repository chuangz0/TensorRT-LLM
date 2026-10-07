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
from mooncake_cluster import RDMA_ENV
from status_dumps import (
    assert_no_leftover_records,
    assert_round_trip_counters,
    counters,
    landing_of,
)
from store_engine import (
    KV_TRANSFER_CONFIG_ENV,
    NAMEABLE_BLOCKS,
    prompt_token_ids,
    run_engine,
    timeout_mark,
    write_kv_transfer_yaml,
)

pytestmark = [pytest.mark.threadleak(enabled=False), pytest.mark.private_mpi_session]

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
def test_engine_b_fetches_what_engine_a_published_and_generates_the_same_tokens(
    mooncake_cluster, tinyllama_path, tmp_path, monkeypatch, disable_overlap_scheduler_b, landing
):
    namespace = (
        f"round-trip-{os.getpid()}-{int(disable_overlap_scheduler_b)}-{landing or 'default'}"
    )
    config_path = write_kv_transfer_yaml(
        tmp_path, mooncake_cluster.master_address, namespace, landing=landing
    )
    monkeypatch.setenv(KV_TRANSFER_CONFIG_ENV, config_path)
    prompt = prompt_token_ids()

    a = run_engine(tmp_path, monkeypatch, "a", tinyllama_path, [prompt])
    b = run_engine(
        tmp_path,
        monkeypatch,
        "b",
        tinyllama_path,
        [prompt],
        disable_overlap_scheduler=disable_overlap_scheduler_b,
    )

    (tokens_a,), (tokens_b,) = a.tokens, b.tokens
    assert len(tokens_a) > 0
    assert tokens_a == tokens_b

    assert a.dump["started_at"] < b.dump["started_at"]
    assert a.dump["pid"] != b.dump["pid"]
    # With no landing in the YAML, the factory lands on host memory first over TCP.
    expected_landing = landing or "host"
    assert landing_of(a.dump) == expected_landing and landing_of(b.dump) == expected_landing
    assert_round_trip_counters(a.dump, b.dump, blocks=NAMEABLE_BLOCKS)


@timeout_mark(600)
def test_engine_a_with_the_overlap_scheduler_publishes_every_block(
    mooncake_cluster, tinyllama_path, tmp_path, monkeypatch
):
    """The overlap loop's publish point offers the previous batch's pages, whose forward has
    completed: a publisher running it stores every block, and the fetcher reads them whole."""
    namespace = f"round-trip-overlap-a-{os.getpid()}"
    config_path = write_kv_transfer_yaml(tmp_path, mooncake_cluster.master_address, namespace)
    monkeypatch.setenv(KV_TRANSFER_CONFIG_ENV, config_path)
    prompt = prompt_token_ids()

    a = run_engine(
        tmp_path, monkeypatch, "a", tinyllama_path, [prompt], disable_overlap_scheduler=False
    )
    b = run_engine(tmp_path, monkeypatch, "b", tinyllama_path, [prompt])

    assert len(a.tokens[0]) > 0 and a.tokens == b.tokens
    assert_round_trip_counters(a.dump, b.dump, blocks=NAMEABLE_BLOCKS)


@timeout_mark(600)
def test_config_without_coordinator_timeouts_runs_on_the_defaults(
    mooncake_cluster, tinyllama_path, tmp_path, monkeypatch
):
    """A config file naming only the backend: the coordinator's timeouts take their defaults
    and the round trip is the same as with them written out."""
    namespace = f"round-trip-defaults-{os.getpid()}"
    config_path = write_kv_transfer_yaml(
        tmp_path, mooncake_cluster.master_address, namespace, omit_coordinator_timeouts=True
    )
    monkeypatch.setenv(KV_TRANSFER_CONFIG_ENV, config_path)
    prompt = prompt_token_ids()

    a = run_engine(tmp_path, monkeypatch, "a", tinyllama_path, [prompt])
    b = run_engine(tmp_path, monkeypatch, "b", tinyllama_path, [prompt])

    assert len(a.tokens[0]) > 0 and a.tokens == b.tokens
    assert_round_trip_counters(a.dump, b.dump, blocks=NAMEABLE_BLOCKS)


@timeout_mark(600)
def test_the_same_prompt_twice_in_one_batch_generates_alike_and_leaves_nothing_behind(
    mooncake_cluster, tinyllama_path, tmp_path, monkeypatch
):
    """Engine A publishes two prompts; engine B is given the first twice and the second once in
    one batch. Every output matches A's. The duplicates either both fetch (planned in the same
    round) or the second is served by local reuse once the first has landed, so B fetches the
    second prompt's seven blocks plus seven or fourteen for the first; nothing fails and no
    record is left."""
    namespace = f"round-trip-duplicates-{os.getpid()}"
    config_path = write_kv_transfer_yaml(tmp_path, mooncake_cluster.master_address, namespace)
    monkeypatch.setenv(KV_TRANSFER_CONFIG_ENV, config_path)
    first, second = prompt_token_ids(seed=1), prompt_token_ids(seed=2)

    a = run_engine(tmp_path, monkeypatch, "a", tinyllama_path, [first, second])
    b = run_engine(tmp_path, monkeypatch, "b", tinyllama_path, [first, first, second], batch=True)

    tokens_first, tokens_second = a.tokens
    assert tokens_first != tokens_second
    assert b.tokens == [tokens_first, tokens_first, tokens_second]
    counters_a = counters(a.dump)
    assert counters_a["publish_stored"] == 2 * NAMEABLE_BLOCKS, counters_a
    assert counters_a["failed_attempts"] == 0, counters_a
    assert_no_leftover_records(a.dump)
    counters_b = counters(b.dump)
    # One lookup per request: the duplicates share a pending lookup for one round, then the
    # second asks on its own, so three lookups of seven blocks answer.
    assert counters_b["probe_hits"] == 3 * NAMEABLE_BLOCKS, counters_b
    assert counters_b["fetch_hits"] in (2 * NAMEABLE_BLOCKS, 3 * NAMEABLE_BLOCKS), counters_b
    assert counters_b["fetch_misses"] == 0 and counters_b["failed_attempts"] == 0, counters_b
    assert counters_b["publish_stored"] == 0, counters_b
    assert_no_leftover_records(b.dump)

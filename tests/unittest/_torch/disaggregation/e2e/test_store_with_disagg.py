# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The store layer coexisting with disaggregated serving.

A context engine and a generation engine live in one process, as in
``test_llm_pytorch.py::test_llm_disagg_gen_cancelled``, both with the same KV transfer config.
A ``context_only`` request is sent to the gen engine by the transceiver *and* published to the
store; the same prompt again probes the store on the context side. The generation output equals
a plain single-engine run. Nothing but the ``context_only`` requests is sent to the context
engine: it reaps each disagg send on its own once the generation engine has pulled, so the
request is released through the gate and no coordinator record is left behind at shutdown.
The generation engine publishes nothing: gen-init requests belong to the transceiver.
"""

import os

import pytest
from status_dumps import assert_no_leftover_records, counters, read_single_rank_dump
from store_engine import (
    KV_TRANSFER_CONFIG_ENV,
    generate_ids,
    prompt_token_ids,
    run_engine,
    sampling_params,
    start_engine,
    timeout_mark,
    write_kv_transfer_yaml,
)

pytestmark = [pytest.mark.threadleak(enabled=False), pytest.mark.private_mpi_session]

ROUNDS = 2


@timeout_mark(600)
def test_context_engine_publishes_to_the_store_while_sending_kv_to_the_generation_engine(
    mooncake_cluster, tinyllama_path, tmp_path, monkeypatch
):
    from tensorrt_llm.disaggregated_params import DisaggregatedParams

    prompt = prompt_token_ids()

    # One plain engine, no store, no transceiver; created and gone before the pair.
    monkeypatch.delenv(KV_TRANSFER_CONFIG_ENV, raising=False)
    (expected,) = run_engine(tmp_path, monkeypatch, None, tinyllama_path, [prompt]).tokens
    assert len(expected) > 0

    namespace = f"with-disagg-{os.getpid()}"
    # The planner waits longer than the standard 1 s for the store's answer: a cold engine's
    # first lookup can outlast that, and the backend then hands the unclaimed answer to the next
    # ask for the same prompt instead of looking the blocks up again, which would leave the
    # second request's probe uncounted (``probe_hits`` below).
    config_path = write_kv_transfer_yaml(
        tmp_path, mooncake_cluster.master_address, namespace, probe_timeout_s=10.0
    )
    monkeypatch.setenv(KV_TRANSFER_CONFIG_ENV, config_path)

    llm_ctx = start_engine(tmp_path, monkeypatch, "ctx", tinyllama_path, cache_transceiver=True)
    llm_gen = None
    dump_ctx = dump_gen = None
    try:
        llm_gen = start_engine(tmp_path, monkeypatch, "gen", tinyllama_path, cache_transceiver=True)

        outputs = []
        for _ in range(ROUNDS):
            (ctx_output,) = llm_ctx.generate(
                [prompt],
                sampling_params=sampling_params(1),
                disaggregated_params=DisaggregatedParams(request_type="context_only"),
            )
            assert len(ctx_output.outputs[0].token_ids) == 1
            assert ctx_output.outputs[0].token_ids[0] == expected[0]
            disaggregated_params = ctx_output.disaggregated_params
            disaggregated_params.request_type = "generation_only"
            outputs.append(generate_ids(llm_gen, prompt, disaggregated_params=disaggregated_params))

        assert outputs[0] == expected, (outputs[0], expected)
        assert outputs[1] == expected, (outputs[1], expected)
    finally:
        llm_ctx.shutdown()
        dump_ctx = read_single_rank_dump(tmp_path, "ctx")
        if llm_gen is not None:
            llm_gen.shutdown()
            dump_gen = read_single_rank_dump(tmp_path, "gen")

    counters_ctx = counters(dump_ctx)
    assert counters_ctx["publish_stored"] > 0, counters_ctx
    # The second identical request probes the store and finds every block (probe_hits), but on
    # the same engine the local radix tree already serves the whole prompt, so the plan's ask is
    # empty and no unit is fetched: a store *fetch* is only observable across engines, which is
    # what ``test_store_round_trip`` shows.
    assert counters_ctx["probe_hits"] > 0, counters_ctx
    assert counters_ctx["fetch_misses"] == 0, counters_ctx
    assert counters_ctx["failed_attempts"] == 0, counters_ctx
    assert_no_leftover_records(dump_ctx)

    counters_gen = counters(dump_gen)
    assert counters_gen["publish_stored"] == 0, counters_gen
    assert counters_gen["fetch_hits"] == 0, counters_gen
    assert counters_gen["failed_attempts"] == 0, counters_gen
    assert_no_leftover_records(dump_gen)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""KV sharing through a real Mooncake store on several GPUs of one node: Qwen3-8B at TP2 (BF16 KV
cache), and one case per other layout: DeepSeek-V3-Lite (MLA) at TP2 and under attention DP, also
with an FP8 KV cache, Qwen3-8B with an FP8 KV cache and an NVFP4 one at TP2, and the
sliding-window models gemma-3-1b at TP1 and gpt-oss-20b at TP2, whose readers extend the writer's
prompt. On request, DeepSeek-V3.2 (DeepSeek sparse attention, FP8 KV cache) under attention DP on
the node's four GPUs.

Each stored object is put once: each rank puts its own share of every prompt block, and ranks
holding the same bytes (MLA's latent cache) put one stripe each, or under host staging the first
of them all. A fresh reader of the same layout fetches the blocks and serves the tokens of a server
that reused them locally, on either data path and also when part of the prefix is in its own cache
already. A fetch running late drops on every rank and the request computes its prompt; a cancel
and a shutdown with fetches in flight end cleanly. Two instances of one service share system
prompts and follow-up turns under concurrent load. Every rank writes a trace of its rounds
(``TRTLLM_KV_SHARING_TRACE_DIR``): under tensor parallelism the ranks of a server must write the
same one, which proves they entered the same store collectives with the same tables and page
counts; under attention DP each rank closes its own store work before it exits.

Runs only with ``TRTLLM_MOONCAKE_E2E=1`` (it skips otherwise), from ``tests/unittest``:
``TRTLLM_MOONCAKE_E2E=1 pytest _torch/executor/kv_sharing/test_mooncake_store_multi_rank.py``;
the DeepSeek-V3.2 case also needs ``TRTLLM_MOONCAKE_E2E_DSA=1``.
Then it needs four GPUs, ``mooncake_master`` on the path, the ``mooncake.store`` SDK and, for the
native driver, the built ``tensorrt_llm.mooncake_store_worker``; fails without them. The store and
its drivers are the module's (``conftest.py``). Servers run their ranks in worker processes.
"""

from __future__ import annotations

import gc
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from conftest import (
    TOKENS_PER_BLOCK,
    Store,
    decisions,
    puts_since,
    rank_traces,
    run_two_turn_workload,
    skip_unless_opted_in,
)
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.pyexecutor.kv_sharing.startup import TRACE_DIR_ENV
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.scheduling_params import SchedulingParams

pytestmark = skip_unless_opted_in

NEW_TOKENS = 32
TP = 2
_tokens = random.Random(20261023)
P = [1000] + [_tokens.randrange(1000, 30000) for _ in range(999)]  # 31 whole blocks
P_BLOCKS = len(P) // TOKENS_PER_BLOCK
# Siblings of P: its first 16 blocks, then 200 tokens of their own; one per reader.
FETCHED = 512
FETCHED_BLOCKS = FETCHED // TOKENS_PER_BLOCK
T_BLOCKS = (FETCHED + 200) // TOKENS_PER_BLOCK  # 22
T = {
    name: P[:FETCHED] + [_tokens.randrange(1000, 30000) for _ in range(200)]
    for name in ("exact", "late")
}
# A follow-up turn: P's first 24 blocks (a system prompt of 16 blocks and a first turn of 8, which
# another server computed), then a question of its own.
SYSTEM, FIRST_TURN = 512, 256
FOLLOW_UP = P[: SYSTEM + FIRST_TURN] + [_tokens.randrange(1000, 30000) for _ in range(200)]
# The sibling every other model class's reader fetches.
SIBLING = P[:FETCHED] + [_tokens.randrange(1000, 30000) for _ in range(200)]
# A follow-up of P, the prompt windows serve: P's whole blocks, then 200 tokens of its own.
P_END = P_BLOCKS * TOKENS_PER_BLOCK
EXTENDS_P = P[:P_END] + [_tokens.randrange(1000, 30000) for _ in range(200)]
# DeepSeek-V3.2's prompts pass its indexer's top-k (2048 tokens), so the sparse path runs: its
# reader's queries pick their 2048 keys among fetched indexer keys. LONG_P is 96 blocks, its sibling
# shares the first 80.
LONG_P = [1000] + [_tokens.randrange(1000, 30000) for _ in range(96 * TOKENS_PER_BLOCK - 1)]
LONG_FETCHED = 80 * TOKENS_PER_BLOCK
LONG_SIBLING = LONG_P[:LONG_FETCHED] + [_tokens.randrange(1000, 30000) for _ in range(512)]


DEEPSEEK = "DeepSeek-V3-Lite/bf16"
# NVFP4 MoE weights, BF16 attention: the checkpoint TensorRT-LLM runs with an FP8 KV cache and reuse.
DEEPSEEK_NVFP4_MOE = "DeepSeek-V3-Lite/nvfp4_moe_only"
# DeepSeek sparse attention; about 390 GiB, so one server takes the whole node.
DEEPSEEK_V32 = "DeepSeek-V3.2-NVFP4"
QWEN3 = "Qwen3/Qwen3-8B"
# NVFP4 weights; the checkpoint carries the KV cache scales an NVFP4 KV cache uses.
QWEN3_NVFP4 = "Qwen3/nvidia-Qwen3-8B-NVFP4"
GEMMA3 = "gemma/gemma-3-1b-it"
GPT_OSS = "gpt_oss/gpt-oss-20b"


def model_path(model: str = QWEN3) -> str:
    return str(llm_models_root(check=True) / model)


def server(
    monkeypatch: pytest.MonkeyPatch,
    tp: int,
    trace_dir: Path,
    *,
    model: str = QWEN3,
    max_batch_size: int = 8,
    max_seq_len: int = 2048,
    kv_dtype: str = "auto",
    **kwargs,
) -> LLM:
    """A server of ``model`` (Qwen3-8B by default) over ``tp`` GPUs whose ranks trace their
    rounds into ``trace_dir``."""
    monkeypatch.delenv("TLLM_WORKER_USE_SINGLE_PROCESS", raising=False)
    monkeypatch.setenv(TRACE_DIR_ENV, str(trace_dir))
    return LLM(
        model=model_path(model),
        tensor_parallel_size=tp,
        kv_cache_config=KvCacheConfig(
            tokens_per_block=TOKENS_PER_BLOCK, free_gpu_memory_fraction=0.1, dtype=kv_dtype
        ),
        cuda_graph_config=None,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        **kwargs,
    )


def sharing(store: Store, key_prefix: str, **knobs) -> dict:
    """``kv_sharing`` sized for these prompts, of 4096 tokens at most; readers fetch prefixes of
    two blocks or more."""
    return {
        "backends": [{"type": "mooncake", "config": store.config(key_prefix)}],
        "max_fetch_tokens": 4096,
        "min_fetch_tokens": 2 * TOKENS_PER_BLOCK,
        **knobs,
    }


def greedy() -> SamplingParams:
    """Fresh per call: the LLM fills in its own model's end id and generation defaults in place, so
    parameters shared across servers of different models would carry the first one's."""
    return SamplingParams(max_tokens=NEW_TOKENS, temperature=0.0)


def complete(llm: LLM, prompt: list[int], **kwargs):
    return llm.generate([prompt], greedy(), **kwargs)[0]


def tokens(output) -> list[int]:
    return list(output.outputs[0].token_ids)


def shut_down(llm: LLM) -> None:
    llm.shutdown()
    gc.collect()
    torch.cuda.empty_cache()


def gpus_by_process() -> dict[int, set[str]]:
    """The GPUs each process holds a CUDA context on."""
    listing = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    print(f"nvidia-smi compute apps:\n{listing}")
    held: dict[int, set[str]] = {}
    for line in listing.splitlines():
        pid, gpu, _ = (field.strip() for field in line.split(","))
        held.setdefault(int(pid), set()).add(gpu)
    return held


@dataclass
class Written:
    """P published by a writer, and that writer's tokens for the readers to match."""

    key_prefix: str
    references: dict[str, list[int]]


@pytest.fixture(scope="module")
def writer(store: Store, tmp_path_factory: pytest.TempPathFactory) -> Written:
    """A TP2 writer (GPU Direct, which 'auto' picks here) publishes P; each rank puts its share of
    each block once. Then, under salts nobody reads, it computes the references: the siblings with
    P's blocks reused locally, and computed whole."""
    key_prefix, trace_dir = "tp2", tmp_path_factory.mktemp("writer")
    with pytest.MonkeyPatch.context() as monkeypatch:
        llm = server(monkeypatch, TP, trace_dir, kv_sharing=sharing(store, key_prefix))
    stored = TP * P_BLOCKS
    references = {}
    try:
        before = store.master_metrics()
        references["P"] = tokens(complete(llm, P))
        store.assert_readable(key_prefix, stored)
        assert puts_since(store, before) == (stored, 0)
        complete(llm, P, cache_salt="reused")
        reused = complete(llm, T["exact"], cache_salt="reused")
        assert reused.cached_tokens == FETCHED
        references["exact"] = tokens(reused)
        references["late"] = tokens(complete(llm, T["late"], cache_salt="whole-late"))
        follow_up = complete(llm, FOLLOW_UP, cache_salt="reused")
        assert follow_up.cached_tokens == SYSTEM + FIRST_TURN
        references["follow-up"] = tokens(follow_up)
    finally:
        shut_down(llm)
    rank_traces(trace_dir, TP)
    return Written(key_prefix, references)


def test_a_host_staging_reader_fetches_a_gpu_direct_writers_blocks_exactly_and_puts_each_once(
    store: Store, writer: Written, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reader over host staging fetches what the GPU Direct writer put."""
    staged = sharing(store, writer.key_prefix, data_path="host_staging")
    llm = server(monkeypatch, TP, tmp_path, kv_sharing=staged)
    # Each rank puts its share of the blocks past the fetched ones.
    new_objects = TP * (T_BLOCKS - FETCHED_BLOCKS)
    try:
        held, _ = store.readable_objects(writer.key_prefix)
        before = store.master_metrics()
        output = complete(llm, T["exact"])
        store.assert_readable(writer.key_prefix, held + new_objects)
        puts = puts_since(store, before)
    finally:
        shut_down(llm)
    assert tokens(output) == writer.references["exact"]
    assert output.cached_tokens == FETCHED
    assert puts == (new_objects, 0)
    rank_traces(tmp_path, TP)


def test_a_fetch_past_its_deadline_drops_on_every_rank_and_serving_goes_on(
    store: Store, writer: Written, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Past by the first round that covers the fetch, on every rank's clock.
    timeout = sharing(store, writer.key_prefix, fetch_timeout_s=1e-6)
    llm = server(monkeypatch, TP, tmp_path, kv_sharing=timeout)
    try:
        late = complete(llm, T["late"])
        again = complete(llm, P)
    finally:
        shut_down(llm)
    assert tokens(late) == writer.references["late"] and late.cached_tokens == 0
    # P fetches past its first 16 blocks, which the late request left in the cache.
    assert tokens(again) == writer.references["P"] and again.cached_tokens == FETCHED
    assert len(decisions(rank_traces(tmp_path, TP), "drop")) == 2


def test_a_prefix_partly_in_the_local_cache_reads_the_rest_from_the_store_exactly(
    store: Store, writer: Written, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A follow-up turn over GPU Direct: the system prompt is in the reader's own cache, the first
    turn only in the store. The request reuses both, and the store gains only the new question's
    blocks. Every rank's store threads use its own GPU only, and a cancel and a shutdown with
    fetches in flight end cleanly."""
    new_blocks = len(FOLLOW_UP) // TOKENS_PER_BLOCK - (SYSTEM + FIRST_TURN) // TOKENS_PER_BLOCK
    llm = server(monkeypatch, TP, tmp_path, kv_sharing=sharing(store, writer.key_prefix))
    try:
        # The system prompt and one token: all 16 of its blocks are read from the store, so the
        # reader's own cache holds them as the other server computed them in one pass (a block
        # computed here over a cached prefix can differ from that in the last bits).
        complete(llm, P[: SYSTEM + 1])
        held, _ = store.readable_objects(writer.key_prefix)
        before = store.master_metrics()
        output = complete(llm, FOLLOW_UP)
        store.assert_readable(writer.key_prefix, held + TP * new_blocks)
        puts = puts_since(store, before)
        # Each rank's process and the GPUs its threads use, the store worker's included.
        ranks = {pid: gpus for pid, gpus in gpus_by_process().items() if pid != os.getpid()}
        # The reader now holds P's first 24 blocks and the store all 31: both of these fetch.
        cancelled = llm.generate_async(EXTENDS_P, greedy())
        cancelled.abort()
        llm.generate_async(P, greedy())  # the shutdown finds its fetch admitted or on its way
    finally:
        shut_down(llm)
    assert tokens(output) == writer.references["follow-up"]
    assert output.cached_tokens == SYSTEM + FIRST_TURN
    assert puts == (TP * new_blocks, 0)
    # Every rank's store threads use its own GPU only.
    assert len(ranks) >= TP and all(len(gpus) == 1 for gpus in ranks.values()), ranks
    # The cancel and the shutdown ended cleanly: every rank wrote the same rounds and its exit.
    rank_traces(tmp_path, TP)


def test_two_instances_of_one_service_share_system_prompts_and_follow_up_turns(
    store: Store, tmp_path: Path
) -> None:
    """Two TP2 instances, with CUDA graphs as served by default, on the node's four GPUs."""
    run_two_turn_workload(store, model_path(), tp=TP, gpu_groups=("0,1", "2,3"), run_dir=tmp_path)


@dataclass(frozen=True)
class Cell:
    """A model class on a layout of ``tp`` GPUs: a writer publishes a prompt over
    ``writer_path``, then a fresh reader of the same layout fetches a sibling's shared prefix over
    'auto' (GPU Direct here).

    Attributes:
        objects_per_block: Objects the store holds per prompt block: one per share of the KV
            heads, one for MLA's latent cache, which every rank holds alike.
    """

    model: str
    objects_per_block: int
    writer_path: str = "auto"
    attention_dp: bool = False
    kv_dtype: str = "auto"
    tp: int = TP


CELLS = {
    "deepseek-tp2-host-staging-writer": Cell(DEEPSEEK, 1, writer_path="host_staging"),
    "qwen3-fp8-tp2": Cell(QWEN3, 2, kv_dtype="fp8"),
    "deepseek-attention-dp2": Cell(DEEPSEEK, 1, attention_dp=True),
    "deepseek-fp8-kv-tp2": Cell(DEEPSEEK_NVFP4_MOE, 1, kv_dtype="fp8"),
    # NVFP4 block scales are a second pool, so each object is two spans; the writer stages them
    # through host memory, the reader writes them in place.
    "qwen3-nvfp4-kv-tp2": Cell(QWEN3_NVFP4, 2, writer_path="host_staging", kv_dtype="nvfp4"),
}
# Under attention DP every request goes to rank 1, never to rank 0 (which probes): rank 0 keeps
# stepping only because rank 1 reports open store work, and rank 1, whose one request is parked,
# still gets a padding request.
ON_RANK_1 = SchedulingParams(attention_dp_rank=1, attention_dp_relax=False)


def cell_server(
    monkeypatch: pytest.MonkeyPatch, cell: Cell, trace_dir: Path, kv_sharing: dict, **llm_args
) -> LLM:
    return server(
        monkeypatch,
        cell.tp,
        trace_dir,
        model=cell.model,
        kv_dtype=cell.kv_dtype,
        enable_attention_dp=cell.attention_dp,
        # The kernels the autotuner picks differ from one server process to the next, and with
        # them the last bits of MoE outputs: fixed kernels let a reader match its writer exactly.
        enable_autotuner=False,
        kv_sharing=kv_sharing,
        **llm_args,
    )


def cell_trace(trace_dir: Path, cell: Cell) -> str:
    """The trace of the rank that served the requests: under tensor parallelism every rank's,
    which must agree; under attention DP rank 1's, after every rank ended with nothing open."""
    if not cell.attention_dp:
        return rank_traces(trace_dir, cell.tp)
    traces = [(trace_dir / f"rank{rank}.txt").read_text() for rank in range(cell.tp)]
    for rank, trace in enumerate(traces):
        assert trace.splitlines()[-1].endswith(" open=0"), f"rank {rank}: {trace[-500:]}"
    return traces[1]


def check_a_reader_fetches_exactly_and_each_object_is_put_once(
    store: Store,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    cell: Cell,
    prompt: list[int],
    sibling: list[int],
    fetched: int,
    **llm_args,
) -> None:
    """A writer publishes ``prompt``, every object once, then serves ``sibling`` reusing their
    first ``fetched`` tokens locally; a fresh reader fetches those from the store, serves the same
    tokens and puts only its own new blocks."""
    pinned = {"scheduling_params": ON_RANK_1} if cell.attention_dp else {}
    with pytest.MonkeyPatch.context() as writer_env:
        writer = cell_server(
            writer_env,
            cell,
            tmp_path / "writer",
            sharing(store, name, data_path=cell.writer_path),
            **llm_args,
        )
    stored = cell.objects_per_block * (len(prompt) // TOKENS_PER_BLOCK)
    try:
        before = store.master_metrics()
        complete(writer, prompt, **pinned)
        store.assert_readable(name, stored)
        assert puts_since(store, before) == (stored, 0)
        complete(writer, prompt, cache_salt="reused", **pinned)
        reference = complete(writer, sibling, cache_salt="reused", **pinned)
        assert reference.cached_tokens == fetched
    finally:
        shut_down(writer)
    cell_trace(tmp_path / "writer", cell)

    new_blocks = len(sibling) // TOKENS_PER_BLOCK - fetched // TOKENS_PER_BLOCK
    new_objects = cell.objects_per_block * new_blocks
    reader = cell_server(monkeypatch, cell, tmp_path / "reader", sharing(store, name), **llm_args)
    try:
        held, _ = store.readable_objects(name)  # the writer's salted prompts too
        before = store.master_metrics()
        output = complete(reader, sibling, **pinned)
        store.assert_readable(name, held + new_objects)
        puts = puts_since(store, before)
    finally:
        shut_down(reader)
    assert tokens(output) == tokens(reference)
    assert output.cached_tokens == fetched
    assert puts == (new_objects, 0)
    trace = cell_trace(tmp_path / "reader", cell)
    assert len(decisions(trace, "resume")) == 1, trace


@pytest.mark.parametrize("name", CELLS)
def test_a_reader_of_each_model_class_and_layout_fetches_exactly_and_each_object_is_put_once(
    store: Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    check_a_reader_fetches_exactly_and_each_object_is_put_once(
        store, tmp_path, monkeypatch, name, CELLS[name], P, SIBLING, FETCHED
    )


# Each DeepSeek-V3.2 server start loads its whole checkpoint, so the case runs only on request.
DSA_OPT_IN_ENV = "TRTLLM_MOONCAKE_E2E_DSA"


@pytest.mark.skipif(
    os.environ.get(DSA_OPT_IN_ENV) != "1",
    reason=f"starts two DeepSeek-V3.2 servers on the node's four GPUs; set {DSA_OPT_IN_ENV}=1",
)
def test_a_reader_with_deepseek_sparse_attention_fetches_its_mla_and_indexer_caches_exactly(
    store: Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DeepSeek-V3.2 with an FP8 KV cache under attention DP and expert parallelism on four GPUs:
    a writer, then a reader in its place. Each object holds a block's MLA latent and indexer keys,
    and the reader's queries select their top-k among the fetched indexer keys."""
    cell = Cell(DEEPSEEK_V32, 1, attention_dp=True, kv_dtype="fp8", tp=4)
    check_a_reader_fetches_exactly_and_each_object_is_put_once(
        store,
        tmp_path,
        monkeypatch,
        "deepseek-v32-attention-dp4",
        cell,
        LONG_P,
        LONG_SIBLING,
        LONG_FETCHED,
        moe_expert_parallel_size=4,
        max_seq_len=4096,
    )


@dataclass(frozen=True)
class WindowedCell:
    """A model with a sliding window in some layers and full attention in the others, on ``tp``
    GPUs; each rank holds KV heads of its own, so each layer group's block is ``tp`` objects.
    ``window`` is the checkpoint's sliding window, which the KV cache keeps."""

    model: str
    tp: int
    window: int


WINDOWED_CELLS = {
    "gemma-3-1b-tp1": WindowedCell(GEMMA3, tp=1, window=512),  # 5 local layers per global layer
    "gpt-oss-20b-tp2": WindowedCell(GPT_OSS, tp=2, window=128),  # every other layer
}


def window_blocks(window: int, end: int) -> set[int]:
    """The blocks a window of ``window`` tokens keeps at a whole-block ``end``: those holding one
    of the tokens the next token reads."""
    return set(range((end + 1 - window) // TOKENS_PER_BLOCK, end // TOKENS_PER_BLOCK))


@pytest.mark.parametrize("name", WINDOWED_CELLS)
def test_a_reader_of_each_windowed_model_fetches_to_the_writers_last_window_exactly(
    store: Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """The writer publishes P: every block of its full-attention layers and, at the context's end,
    the blocks its window keeps there. A fresh reader of a follow-up of P fetches to P's last whole
    block, where the store holds the window, and serves the tokens of the writer's local reuse; it
    puts its own new blocks once, and re-puts of window blocks the store holds are refused."""
    cell = WINDOWED_CELLS[name]
    window = cell.window
    assert EXTENDS_P[P_END] != P[P_END]  # the follow-up leaves P at P's last whole block

    def windowed_server(env: pytest.MonkeyPatch, trace_dir: Path) -> LLM:
        # Fixed kernels let a reader match its writer exactly (gpt-oss is MoE, as cell_server).
        return server(
            env,
            cell.tp,
            trace_dir,
            model=cell.model,
            enable_autotuner=False,
            kv_sharing=sharing(store, name),
        )

    with pytest.MonkeyPatch.context() as writer_env:
        writer = windowed_server(writer_env, tmp_path / "writer")
    at_p = window_blocks(window, P_END)
    stored = cell.tp * (P_BLOCKS + len(at_p))
    try:
        before = store.master_metrics()
        complete(writer, P)
        store.assert_readable(name, stored)
        assert puts_since(store, before) == (stored, 0)
        complete(writer, P, cache_salt="reused")
        reference = complete(writer, EXTENDS_P, cache_salt="reused")
        assert reference.cached_tokens == P_END
    finally:
        shut_down(writer)
    rank_traces(tmp_path / "writer", cell.tp)

    # The reader puts its new whole blocks of every layer group, and the window it keeps at its
    # prompt's last whole block, of which the store holds the blocks P's window kept already.
    reader_end = len(EXTENDS_P) // TOKENS_PER_BLOCK * TOKENS_PER_BLOCK
    at_reader_end = window_blocks(window, reader_end)
    new = cell.tp * (reader_end // TOKENS_PER_BLOCK - P_BLOCKS + len(at_reader_end - at_p))
    again = cell.tp * len(at_reader_end & at_p)
    reader = windowed_server(monkeypatch, tmp_path / "reader")
    try:
        held, _ = store.readable_objects(name)  # the writer's salted prompts too
        before = store.master_metrics()
        output = complete(reader, EXTENDS_P)
        store.assert_readable(name, held + new)
        puts = puts_since(store, before)
    finally:
        shut_down(reader)
    assert tokens(output) == tokens(reference)
    assert output.cached_tokens == P_END
    assert puts == (new + again, again)
    trace = rank_traces(tmp_path / "reader", cell.tp)
    assert len(decisions(trace, "resume")) == 1, trace

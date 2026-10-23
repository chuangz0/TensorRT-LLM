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
"""KV sharing through a real Mooncake store under tensor parallelism (Llama-3.1-8B, TP2 and TP4).

Each rank publishes its own share of every prompt block, once; a fresh reader of the same layout
fetches every rank's share and serves the tokens of a server that reused the blocks locally, also
when part of the prefix is in its own cache already. A fetch running late drops on every rank and
the request computes its prompt; a cancel or a shutdown with a fetch in flight ends cleanly. Two
instances of one service share system prompts and follow-up turns under concurrent load. Every
rank writes a trace of its rounds (``TRTLLM_KV_SHARING_TRACE_DIR``): the ranks of a server must
write the same one, which proves they entered the same store collectives with the same tables and
page counts.

Needs four GPUs, ``mooncake_master`` on the path, the ``mooncake.store`` SDK and, for the native
driver, the built ``tensorrt_llm.mooncake_store_worker``; fails without them. The store and its
drivers are the module's (``conftest.py``). Servers run their ranks in worker processes.
"""

from __future__ import annotations

import gc
import os
import random
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from conftest import Store, decisions, rank_traces, run_two_turn_workload, wait_until
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.pyexecutor.kv_sharing.startup import TRACE_DIR_ENV
from tensorrt_llm.llmapi import KvCacheConfig

TOKENS_PER_BLOCK = 32
NEW_TOKENS = 32
_tokens = random.Random(20261023)
P = [1000] + [_tokens.randrange(1000, 30000) for _ in range(999)]  # 31 whole blocks
P_BLOCKS = len(P) // TOKENS_PER_BLOCK
# Siblings of P: its first 16 blocks, then 200 tokens of their own; one per reader.
FETCHED = 512
FETCHED_BLOCKS = FETCHED // TOKENS_PER_BLOCK
T_BLOCKS = (FETCHED + 200) // TOKENS_PER_BLOCK  # 22
# TP2 over GPU Direct reads exactly in the follow-up test below.
READERS = [(2, "host_staging"), (4, "auto")]
EXACT = [f"exact-tp{tp}-{data_path}" for tp, data_path in READERS]
COLD = ["late"]
T = {
    name: P[:FETCHED] + [_tokens.randrange(1000, 30000) for _ in range(200)]
    for name in EXACT + COLD
}
# A follow-up turn: P's first 24 blocks (a system prompt of 16 blocks and a first turn of 8, which
# another server computed), then a question of its own.
SYSTEM, FIRST_TURN = 512, 256
FOLLOW_UP = P[: SYSTEM + FIRST_TURN] + [_tokens.randrange(1000, 30000) for _ in range(200)]
GREEDY = SamplingParams(max_tokens=NEW_TOKENS, temperature=0.0)
# The master's count of keys a put asked for, and of those it refused.
PUT_KEYS = "master_batch_put_start_items_total"
PUT_KEYS_REFUSED = "master_batch_put_start_failed_items_total"


def model_path() -> str:
    return str(llm_models_root(check=True) / "llama-3.1-model" / "Llama-3.1-8B-Instruct")


def llama(
    monkeypatch: pytest.MonkeyPatch, tp: int, trace_dir: Path, *, max_batch_size: int = 8, **kwargs
) -> LLM:
    """A Llama-3.1-8B server over ``tp`` GPUs whose ranks trace their rounds into ``trace_dir``."""
    monkeypatch.delenv("TLLM_WORKER_USE_SINGLE_PROCESS", raising=False)
    monkeypatch.setenv(TRACE_DIR_ENV, str(trace_dir))
    return LLM(
        model=model_path(),
        tensor_parallel_size=tp,
        kv_cache_config=KvCacheConfig(
            tokens_per_block=TOKENS_PER_BLOCK, free_gpu_memory_fraction=0.1
        ),
        cuda_graph_config=None,
        max_batch_size=max_batch_size,
        max_seq_len=2048,
        **kwargs,
    )


def sharing(store: Store, key_prefix: str, **knobs) -> dict:
    """``kv_sharing`` sized for these short prompts; readers fetch prefixes of two blocks or
    more."""
    return {
        "backends": [{"type": "mooncake", "config": store.config(key_prefix)}],
        "max_fetch_tokens": 1024,
        "min_fetch_tokens": 2 * TOKENS_PER_BLOCK,
        **knobs,
    }


def complete(llm: LLM, prompt: list[int], **kwargs):
    return llm.generate([prompt], GREEDY, **kwargs)[0]


def tokens(output) -> list[int]:
    return list(output.outputs[0].token_ids)


def shut_down(llm: LLM) -> None:
    llm.shutdown()
    gc.collect()
    torch.cuda.empty_cache()


def puts_during(store: Store, action: Callable[[], None]) -> tuple[float, float]:
    """Keys the master was asked to put while ``action`` ran, and how many it refused."""
    before = store.master_metrics()
    action()
    after = store.master_metrics()
    return after[PUT_KEYS] - before[PUT_KEYS], after[PUT_KEYS_REFUSED] - before[PUT_KEYS_REFUSED]


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
    """P published by a writer of one layout, and that writer's tokens for the readers to match."""

    key_prefix: str
    references: dict[str, list[int]]


def publish_and_reference(store: Store, tp: int, trace_dir: Path) -> Written:
    """A writer over ``tp`` GPUs (GPU Direct, which 'auto' picks here) publishes P; each rank puts
    its share of each block once. Then, under salts nobody reads, it computes the references: the
    siblings with P's blocks reused locally, and computed whole."""
    key_prefix = f"tp{tp}"
    with pytest.MonkeyPatch.context() as monkeypatch:
        llm = llama(monkeypatch, tp, trace_dir, kv_sharing=sharing(store, key_prefix))
    stored = tp * P_BLOCKS
    references = {}
    try:

        def publish_p() -> None:
            references["P"] = tokens(complete(llm, P))
            assert wait_until(lambda: store.readable_objects(key_prefix) == (stored, stored)), (
                f"the store holds {store.readable_objects(key_prefix)}; expected {stored}"
            )

        assert puts_during(store, publish_p) == (stored, 0)
        complete(llm, P, cache_salt="reused")
        for name in EXACT:
            reused = complete(llm, T[name], cache_salt="reused")
            assert reused.cached_tokens == FETCHED
            references[name] = tokens(reused)
        for name in COLD:
            references[name] = tokens(complete(llm, T[name], cache_salt=f"whole-{name}"))
        follow_up = complete(llm, FOLLOW_UP, cache_salt="reused")
        assert follow_up.cached_tokens == SYSTEM + FIRST_TURN
        references["follow-up"] = tokens(follow_up)
    finally:
        shut_down(llm)
    rank_traces(trace_dir, tp)
    return Written(key_prefix, references)


@pytest.fixture(scope="module")
def written(store: Store, tmp_path_factory: pytest.TempPathFactory) -> Callable[[int], Written]:
    """The writer of each layout, run once for the module."""
    made: dict[int, Written] = {}

    def writer(tp: int) -> Written:
        if tp not in made:
            made[tp] = publish_and_reference(store, tp, tmp_path_factory.mktemp(f"writer-tp{tp}"))
        return made[tp]

    return writer


@pytest.mark.parametrize("tp, data_path", READERS, ids=EXACT)
def test_a_tensor_parallel_reader_fetches_exactly_and_each_object_is_put_once(
    store: Store, written, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tp, data_path
) -> None:
    writer = written(tp)
    name = f"exact-tp{tp}-{data_path}"
    llm = llama(
        monkeypatch, tp, tmp_path, kv_sharing=sharing(store, writer.key_prefix, data_path=data_path)
    )
    try:
        before, _ = store.readable_objects(writer.key_prefix)
        fetched = []

        def fetch_and_publish() -> None:
            fetched.append(complete(llm, T[name]))
            # Each rank puts its share of the blocks past the fetched ones.
            published = before + tp * (T_BLOCKS - FETCHED_BLOCKS)
            assert wait_until(
                lambda: store.readable_objects(writer.key_prefix) == (published, published)
            ), f"the store holds {store.readable_objects(writer.key_prefix)}; expected {published}"

        puts = puts_during(store, fetch_and_publish)
        if tp == 4:
            # Every rank's store threads use its own GPU only.
            ranks = {pid: gpus for pid, gpus in gpus_by_process().items() if pid != os.getpid()}
            assert len(ranks) >= tp and all(len(gpus) == 1 for gpus in ranks.values()), ranks
    finally:
        shut_down(llm)
    (output,) = fetched
    assert tokens(output) == writer.references[name]
    assert output.cached_tokens == FETCHED
    assert puts == (tp * (T_BLOCKS - FETCHED_BLOCKS), 0)
    trace = rank_traces(tmp_path, tp)
    if data_path == "auto":  # GPU Direct resumes in the round that returns the lease
        assert len(decisions(trace, "resume")) == 1, trace


def test_a_fetch_past_its_deadline_drops_on_every_rank_and_serving_goes_on(
    store: Store, written, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = written(2)
    # Past by the first round that covers the fetch, on every rank's clock.
    timeout = sharing(store, writer.key_prefix, fetch_timeout_s=1e-6)
    llm = llama(monkeypatch, 2, tmp_path, kv_sharing=timeout)
    try:
        late = complete(llm, T["late"])
        again = complete(llm, P)
    finally:
        shut_down(llm)
    assert tokens(late) == writer.references["late"] and late.cached_tokens == 0
    # P fetches past its first 16 blocks, which the late request left in the cache.
    assert tokens(again) == writer.references["P"] and again.cached_tokens == FETCHED
    assert len(decisions(rank_traces(tmp_path, 2), "drop")) == 2


def test_cancel_and_shutdown_with_a_fetch_in_flight_end_cleanly(
    store: Store, written, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = written(2)
    llm = llama(monkeypatch, 2, tmp_path, kv_sharing=sharing(store, writer.key_prefix))
    try:
        cancelled = llm.generate_async(T["late"], GREEDY)
        cancelled.abort()
        llm.generate_async(P, GREEDY)  # the shutdown finds its fetch admitted or on its way
    finally:
        shut_down(llm)
    rank_traces(tmp_path, 2)


def test_a_prefix_partly_in_the_local_cache_reads_the_rest_from_the_store_exactly(
    store: Store, written, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A follow-up turn: the system prompt is in the reader's own cache, the first turn only in
    the store. The request reuses both, and the store gains only the new question's blocks."""
    writer = written(2)
    new_blocks = len(FOLLOW_UP) // TOKENS_PER_BLOCK - (SYSTEM + FIRST_TURN) // TOKENS_PER_BLOCK
    llm = llama(monkeypatch, 2, tmp_path, kv_sharing=sharing(store, writer.key_prefix))
    try:
        # The system prompt and one token: all 16 of its blocks are read from the store, so the
        # reader's own cache holds them as the other server computed them in one pass (a block
        # computed here over a cached prefix can differ from that in the last bits).
        complete(llm, P[: SYSTEM + 1])
        before, _ = store.readable_objects(writer.key_prefix)
        followed = []

        def follow_up() -> None:
            followed.append(complete(llm, FOLLOW_UP))
            published = before + 2 * new_blocks
            assert wait_until(
                lambda: store.readable_objects(writer.key_prefix) == (published, published)
            ), f"the store holds {store.readable_objects(writer.key_prefix)}; expected {published}"

        puts = puts_during(store, follow_up)
    finally:
        shut_down(llm)
    (output,) = followed
    assert tokens(output) == writer.references["follow-up"]
    assert output.cached_tokens == SYSTEM + FIRST_TURN
    assert puts == (2 * new_blocks, 0)
    rank_traces(tmp_path, 2)


def test_two_instances_of_one_service_share_system_prompts_and_follow_up_turns(
    store: Store, tmp_path: Path
) -> None:
    """Two TP2 instances, with CUDA graphs as served by default, on the node's four GPUs."""
    run_two_turn_workload(store, model_path(), tp=2, gpu_groups=("0,1", "2,3"), run_dir=tmp_path)

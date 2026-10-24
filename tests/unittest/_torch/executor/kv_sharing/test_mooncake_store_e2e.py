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
"""KV sharing through a real Mooncake store, end to end, one GPU per server.

A server publishes the whole blocks of every prompt it computes, and nothing it generates, and
serves the same tokens as a server without the store. A fresh server fetches the stored prefix of a
prompt instead of computing it, and its tokens equal those of a server that reused the same blocks
locally, also when part of the prefix is in its own cache already; a request whose prefix left the
store while it waited computes it. Servers move blocks over GPU Direct, which 'auto' picks here, or
through host staging, and a block either path stores reads exactly on the other. Two instances of
one service share system prompts and follow-up turns under concurrent load. The store and its
drivers are the module's (``conftest.py``).

Needs a GPU (three for the disaggregated test, two for the service), ``mooncake_master`` on the path, the
``mooncake.store`` SDK and, for the native driver, the built ``tensorrt_llm.mooncake_store_worker``,
and fails without them.
"""

from __future__ import annotations

import gc
import json
import os
import random
import subprocess
import threading
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path

import pytest
import torch
from conftest import WAIT_S, Store, run_two_turn_workload, wait_until
from llmapi.apps.openai_server import RemoteDisaggOpenAIServer, RemoteOpenAIServer
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.disaggregation.blob.mooncake import MooncakeDriver
from tensorrt_llm._torch.disaggregation.blob.store import ProbeStatus
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import MemoryKind
from tensorrt_llm._torch.pyexecutor.kv_sharing import data_path, probe
from tensorrt_llm.llmapi import KvCacheConfig

TOKENS_PER_BLOCK = 32
# Fills the last partial block of P and of Q: whole blocks then hold generated tokens.
NEW_TOKENS = 32
CHUNKED_PREFILL = {"enable_chunked_prefill": True, "max_num_tokens": 256}
DATA_PATHS = {"gpu_direct": MemoryKind.DEVICE, "host_staging": MemoryKind.HOST}

_tokens = random.Random(20261022)
# Token ids; P and Q start differently, so they share no block.
P = [1000] + [_tokens.randrange(1000, 30000) for _ in range(999)]  # 31 whole blocks
Q = [1001] + [_tokens.randrange(1000, 30000) for _ in range(899)]  # 28 whole blocks
P_BLOCKS, Q_BLOCKS = len(P) // TOKENS_PER_BLOCK, len(Q) // TOKENS_PER_BLOCK

# Siblings of P: P's first 16 blocks, then 200 tokens of their own, one sibling per test so that
# each test's publications are its own. A server holding P reuses exactly the 16 blocks.
FETCHED = 512
FETCHED_BLOCKS = FETCHED // TOKENS_PER_BLOCK
SIBLINGS = ("exact", "refreshed", "disaggregated")
T = {name: P[:FETCHED] + [_tokens.randrange(1000, 30000) for _ in range(200)] for name in SIBLINGS}
# The aggregated tests compare with a server without the store; the disaggregated one with its own.
REFERENCED = SIBLINGS[:-1]
T_BLOCKS = (FETCHED + 200) // TOKENS_PER_BLOCK  # 22
# A follow-up turn: P's first 24 blocks (a system prompt of 16 blocks and a first turn of 8, which
# another server computed), then a question of its own.
SYSTEM, FIRST_TURN = 512, 256
FOLLOW_UP = P[: SYSTEM + FIRST_TURN] + [_tokens.randrange(1000, 30000) for _ in range(200)]
# The key prefix under which a writer published P for the readers.
FETCH_PREFIX = "fetch"
# Readers fetch prefixes of at least two blocks.
READER = {"min_fetch_tokens": 2 * TOKENS_PER_BLOCK}


def server(
    monkeypatch: pytest.MonkeyPatch, *, in_process: bool, max_batch_size: int = 8, **kwargs
) -> LLM:
    """A Qwen3-0.6B server. In process, a test reads its executor; otherwise the executor runs in a
    worker process, as it does in production."""
    if in_process:
        monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    else:
        monkeypatch.delenv("TLLM_WORKER_USE_SINGLE_PROCESS", raising=False)
    return LLM(
        model=str(llm_models_root(check=True) / "Qwen3" / "Qwen3-0.6B"),
        kv_cache_config=KvCacheConfig(
            tokens_per_block=TOKENS_PER_BLOCK, free_gpu_memory_fraction=0.3
        ),
        max_batch_size=max_batch_size,
        max_seq_len=2048,
        **kwargs,
    )


def sharing(store: Store, key_prefix: str, max_fetch_tokens: int = 1024, **knobs) -> dict:
    """``kv_sharing`` with a small fetch size: these prompts are short. Publications in flight hold
    at most max(max_fetch_tokens, max_num_tokens) tokens. Unless ``knobs`` say otherwise, nothing
    shorter than ``max_fetch_tokens`` is fetched."""
    return {
        "backends": [{"type": "mooncake", "config": store.config(key_prefix)}],
        "max_fetch_tokens": max_fetch_tokens,
        "min_fetch_tokens": max_fetch_tokens,
        **knobs,
    }


GREEDY = SamplingParams(max_tokens=NEW_TOKENS, temperature=0.0)


def complete(llm: LLM, prompt: list[int], **kwargs):
    """``prompt``'s greedy completion, alone in its batch."""
    return llm.generate([prompt], GREEDY, **kwargs)[0]


def generate(llm: LLM, prompt: list[int], **kwargs) -> list[int]:
    """Greedy tokens for ``prompt``, alone in its batch."""
    return list(complete(llm, prompt, **kwargs).outputs[0].token_ids)


def shut_down(llm: LLM) -> None:
    llm.shutdown()
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def references() -> Iterator[dict[str, list[int]]]:
    """Greedy tokens of a server without the store."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        llm = server(monkeypatch, in_process=True)
    try:
        yield {"P": generate(llm, P), "Q": generate(llm, Q)}
    finally:
        shut_down(llm)


def test_publication_stores_whole_prompt_blocks_and_keeps_output(
    store: Store, references: dict[str, list[int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(data_path.logger, "warning", warnings.append)
    llm = server(monkeypatch, in_process=True, kv_sharing=sharing(store, "overlap"))
    executor = llm._executor.engine
    try:
        # 'auto' picks GPU Direct here.
        assert executor.kv_sharing.memory_kind is MemoryKind.DEVICE, warnings
        tokens = {"P": generate(llm, P), "Q": generate(llm, Q)}
        stored = P_BLOCKS + Q_BLOCKS
        assert wait_until(lambda: store.readable_objects("overlap") == (stored, stored)), (
            f"the store holds {store.readable_objects('overlap')} (objects, readable); "
            f"expected {stored}"
        )
        # The executor retires each publication once its store calls returned.
        assert wait_until(lambda: not executor.disagg.has_pending_shared_transfers())
    finally:
        shut_down(llm)
    assert tokens == references
    assert executor.disagg.shared_shutdown_complete()
    assert store.readable_objects("overlap") == (stored, stored)  # no generated block came later
    counters = executor.kv_sharing.counters
    assert counters["publications_started"] == 2
    assert counters["published_objects"] == stored
    assert counters["publications_not_started"] == counters["publications_failed"] == 0


def test_publication_from_the_non_overlap_loop_in_a_worker_process(
    store: Store, references: dict[str, list[int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = server(
        monkeypatch,
        in_process=False,
        disable_overlap_scheduler=True,
        kv_sharing=sharing(store, "non-overlap"),
    )
    try:
        tokens = generate(llm, P)
        assert wait_until(lambda: store.readable_objects("non-overlap") == (P_BLOCKS, P_BLOCKS)), (
            f"the store holds {store.readable_objects('non-overlap')} (objects, readable)"
        )
    finally:
        shut_down(llm)
    assert tokens == references["P"]


def test_chunked_prefill_publishes_each_chunk_and_keeps_output(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = server(monkeypatch, in_process=True, **CHUNKED_PREFILL)
    try:
        reference = generate(llm, P)
    finally:
        shut_down(llm)
    # Each chunk's range is published once its commit lands, its pages locked while later chunks
    # compute.
    llm = server(
        monkeypatch, in_process=True, kv_sharing=sharing(store, "chunked"), **CHUNKED_PREFILL
    )
    executor = llm._executor.engine
    try:
        tokens = generate(llm, P)
        assert wait_until(lambda: store.readable_objects("chunked") == (P_BLOCKS, P_BLOCKS)), (
            f"the store holds {store.readable_objects('chunked')} (objects, readable)"
        )
        assert wait_until(lambda: not executor.disagg.has_pending_shared_transfers())
    finally:
        shut_down(llm)
    assert tokens == reference
    counters = executor.kv_sharing.counters
    assert counters["publications_started"] == 4  # [0, 256), [256, 512), [512, 768), [768, 992)
    assert counters["published_objects"] == P_BLOCKS
    assert counters["publications_not_started"] == counters["publications_failed"] == 0


# Fetch.


def model_path() -> str:
    return str(llm_models_root(check=True) / "Qwen3" / "Qwen3-0.6B")


def block_bytes() -> int:
    """Bytes of one KV cache block of the model: K and V of every layer, in BF16."""
    config = json.loads((Path(model_path()) / "config.json").read_text())
    per_token = 2 * config["num_hidden_layers"] * config["num_key_value_heads"] * config["head_dim"]
    return per_token * TOKENS_PER_BLOCK * 2


def publish_p(store: Store, key_prefix: str, *, gpu_direct_fails: bool = False) -> MemoryKind:
    """A writer server, shut down once done, publishes P's whole blocks under ``key_prefix``; its
    memory kind. With ``gpu_direct_fails`` its GPU Direct preparation fails, and 'auto' falls back
    to host staging, logging why."""
    warnings: list[str] = []
    with pytest.MonkeyPatch.context() as monkeypatch:
        if gpu_direct_fails:
            monkeypatch.setattr(data_path, "prepare_gpu_direct", lambda _: "forced by the test")
            monkeypatch.setattr(data_path.logger, "warning", warnings.append)
        llm = server(monkeypatch, in_process=True, kv_sharing=sharing(store, key_prefix))
    executor = llm._executor.engine
    try:
        generate(llm, P)
        assert wait_until(lambda: store.readable_objects(key_prefix) == (P_BLOCKS, P_BLOCKS))
    finally:
        shut_down(llm)
    fell_back = [w for w in warnings if "host staging" in w and "forced by the test" in w]
    assert len(fell_back) == gpu_direct_fails, warnings
    return executor.kv_sharing.memory_kind


def reader(monkeypatch: pytest.MonkeyPatch, store: Store, key_prefix=FETCH_PREFIX, **knobs) -> LLM:
    """A fresh server, in process, that fetches what the writer of P published."""
    return server(
        monkeypatch, in_process=True, kv_sharing=sharing(store, key_prefix, **READER, **knobs)
    )


def settled(executor) -> bool:
    """Whether every store operation of the in-process executor ended and its lease went back."""
    return not executor.disagg.has_pending_shared_transfers()


@pytest.fixture(scope="module")
def sibling_references() -> dict[str, dict[str, list[int]]]:
    """Greedy tokens of each sibling from a server without the store: computed whole, and with P's
    first 16 blocks reused (P computed first)."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        llm = server(monkeypatch, in_process=True)
    try:
        # A salt of its own leaves the request nothing to reuse.
        whole = {name: complete(llm, T[name], cache_salt=f"whole-{name}") for name in REFERENCED}
        generate(llm, P)
        reused = {name: complete(llm, T[name]) for name in REFERENCED}
        follow_up = complete(llm, FOLLOW_UP)
    finally:
        shut_down(llm)
    assert all(output.cached_tokens == 0 for output in whole.values())
    assert all(output.cached_tokens == FETCHED for output in reused.values())
    assert follow_up.cached_tokens == SYSTEM + FIRST_TURN
    reused["follow-up"] = follow_up
    return {
        kind: {name: list(output.outputs[0].token_ids) for name, output in outputs.items()}
        for kind, outputs in (("whole", whole), ("reused", reused))
    }


@pytest.fixture(scope="module")
def published_p(store: Store) -> None:
    """P published under FETCH_PREFIX by a writer over GPU Direct, which 'auto' picks here."""
    assert publish_p(store, FETCH_PREFIX) is MemoryKind.DEVICE


@pytest.fixture(scope="module")
def staged_p(store: Store) -> str:
    """The key prefix under which a writer that fell back to host staging published P."""
    assert publish_p(store, "fetch-staged", gpu_direct_fails=True) is MemoryKind.HOST
    return "fetch-staged"


@pytest.mark.parametrize("read_on", ["host_staging", "gpu_direct"])
def test_a_fresh_server_fetches_a_prefix_published_on_the_other_data_path_exactly(
    store: Store,
    published_p: None,
    staged_p: str,
    sibling_references: dict,
    monkeypatch: pytest.MonkeyPatch,
    read_on: str,
) -> None:
    # Read through host staging what GPU Direct wrote, and the reverse.
    key_prefix = FETCH_PREFIX if read_on == "host_staging" else staged_p
    llm = reader(monkeypatch, store, key_prefix=key_prefix, data_path=read_on)
    executor = llm._executor.engine
    before, _ = store.readable_objects(key_prefix)
    try:
        fetched = complete(llm, T["exact"])
        # It publishes the blocks it computed past the fetched ones, and only those.
        published = before + T_BLOCKS - FETCHED_BLOCKS
        assert wait_until(lambda: store.readable_objects(key_prefix) == (published, published)), (
            f"the store holds {store.readable_objects(key_prefix)}; expected {published}"
        )
        # Another salt names other blocks, which the store does not hold.
        salted = complete(llm, T["exact"], cache_salt="other-tenant")
        assert wait_until(lambda: settled(executor))
    finally:
        shut_down(llm)
    assert executor.kv_sharing.memory_kind is DATA_PATHS[read_on]
    assert list(fetched.outputs[0].token_ids) == sibling_references["reused"]["exact"]
    # A fresh server holds no block: the cached tokens came from the store.
    assert fetched.cached_tokens == FETCHED
    counters = executor.kv_sharing.counters
    assert counters["fetches_started"] == counters["fetches_resumed"] == 1
    assert counters["fetched_blocks"] == counters["fetched_objects"] == FETCHED_BLOCKS
    assert counters["fetched_bytes"] == FETCHED_BLOCKS * block_bytes()
    assert list(salted.outputs[0].token_ids) == sibling_references["whole"]["exact"]
    assert salted.cached_tokens == 0


def test_a_prompt_of_whole_blocks_fetches_all_but_its_last_block_and_puts_none_again(
    store: Store, published_p: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prompt that ends on a block boundary: the probe names every one of its blocks, the fetch
    stops one block short (the last prompt token is computed), and the reader puts none of its
    blocks again, as the store held them all."""
    aligned = P[:FETCHED]  # 16 whole blocks, all published by the writer of P
    probe = MooncakeDriver.probe
    probed: list[tuple[ProbeStatus, ...]] = []

    def recording_probe(self, keys, budget_s, hint=None):
        probed.append(probe(self, keys, budget_s, hint))
        return probed[-1]

    monkeypatch.setattr(MooncakeDriver, "probe", recording_probe)
    llm = reader(monkeypatch, store)
    executor = llm._executor.engine
    before = store.readable_objects(FETCH_PREFIX)
    try:
        output = complete(llm, aligned)
        assert wait_until(lambda: settled(executor))
    finally:
        shut_down(llm)
    assert probed == [(ProbeStatus.HIT,) * FETCHED_BLOCKS]
    assert output.cached_tokens == FETCHED - TOKENS_PER_BLOCK
    counters = executor.kv_sharing.counters
    assert counters["fetches_resumed"] == 1 and counters["fetched_blocks"] == FETCHED_BLOCKS - 1
    assert counters["publications_started"] == 0
    assert store.readable_objects(FETCH_PREFIX) == before


def test_a_prefix_partly_in_the_local_cache_reads_only_the_rest_from_the_store(
    store: Store, sibling_references: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A follow-up turn: the system prompt is in the server's own cache, the first turn only in
    the store (another server computed it). The request reads the first turn's blocks alone,
    and the store gains only the new question's blocks."""
    publish_p(store, "follow-up")
    llm = reader(monkeypatch, store, key_prefix="follow-up")
    executor = llm._executor.engine
    counters = executor.kv_sharing.counters
    try:
        # The system prompt and one token: all 16 of its blocks are read from the store, so the
        # reader's own cache holds them as the other server computed them in one pass (a block
        # computed here over a cached prefix can differ from that in the last bits).
        complete(llm, P[: SYSTEM + 1])
        assert wait_until(lambda: settled(executor))
        fetched_before = counters["fetched_blocks"]
        before, _ = store.readable_objects("follow-up")
        output = complete(llm, FOLLOW_UP)
        new_blocks = len(FOLLOW_UP) // TOKENS_PER_BLOCK - (SYSTEM + FIRST_TURN) // TOKENS_PER_BLOCK
        published = before + new_blocks
        assert wait_until(lambda: store.readable_objects("follow-up") == (published, published))
    finally:
        shut_down(llm)
    assert list(output.outputs[0].token_ids) == sibling_references["reused"]["follow-up"]
    assert output.cached_tokens == SYSTEM + FIRST_TURN
    assert counters["fetched_blocks"] - fetched_before == FIRST_TURN // TOKENS_PER_BLOCK


def test_two_instances_of_one_service_share_system_prompts_and_follow_up_turns(
    store: Store, tmp_path: Path
) -> None:
    """Two one-GPU instances of one service."""
    run_two_turn_workload(store, model_path(), tp=1, gpu_groups=("0", "1"), run_dir=tmp_path)


def test_a_waiting_request_takes_the_normal_path_after_its_prefix_left_the_store(
    store: Store, sibling_references: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rank 0 measures a waiting request's hit again; the new hit reaches the request before its
    admission, so it computes the prompt instead of fetching what the store lost. (The test
    removes the prefix; in service, the store evicts it once its read lease ran out.)"""
    publish_p(store, "refresh")  # a prefix of its own: the test removes its objects
    monkeypatch.setattr(probe, "REFRESH_AFTER_S", 0.5)
    monkeypatch.setattr(probe, "REFRESH_SCAN_S", 0.1)
    probe_store = MooncakeDriver.probe
    told: list[tuple] = []  # the request's keys and the store's first answer
    answered = threading.Event()

    def probe_and_tell(self, keys, budget_s, hint=None):
        answer = probe_store(self, keys, budget_s, hint)
        if ProbeStatus.HIT in answer and not answered.is_set():
            told.append((keys, answer))
            answered.set()
        return answer

    monkeypatch.setattr(MooncakeDriver, "probe", probe_and_tell)
    llm = server(
        monkeypatch,
        in_process=True,
        max_batch_size=1,
        kv_sharing=sharing(store, "refresh", **READER),
    )
    executor = llm._executor.engine
    counters = executor.kv_sharing.counters
    try:
        # Two long generations hold both seats (the overlap scheduler admits two requests per
        # batch slot), so the request waits in the queue.
        busy = SamplingParams(max_tokens=1000, temperature=0.0, ignore_eos=True)
        occupiers = [llm.generate_async(prompt, busy) for prompt in (Q, [1002] + Q[1:])]
        assert wait_until(lambda: counters["publications_started"] >= 2)
        waiting = llm.generate_async(T["refreshed"], GREEDY)
        assert answered.wait(WAIT_S)
        keys, _ = told[0]
        for key in keys:  # the request was told; now its prefix leaves the store
            store.client.remove(key, True)
        assert wait_until(lambda: counters["hits_refreshed"] >= 1), dict(counters)
        output = waiting.result(timeout=WAIT_S)
        for occupier in occupiers:
            occupier.result(timeout=WAIT_S)
    finally:
        shut_down(llm)
    _, answer = told[0]
    assert answer[:FETCHED_BLOCKS] == (ProbeStatus.HIT,) * FETCHED_BLOCKS
    assert list(output.outputs[0].token_ids) == sibling_references["whole"]["refreshed"]
    assert output.cached_tokens == 0 and counters["fetches_started"] == 0


# Disaggregated serving: the context server shares through the store; P->D is unchanged.

# NIXL's UCX backend needs a CUDA transport; servers on one node need no InfiniBand.
SERVER_UCX_TLS = "cuda_copy,cuda_ipc,sm,self,tcp"


def test_disaggregated_context_server_fetches_computes_the_rest_and_publishes(
    store: Store, published_p: None, tmp_path: Path
) -> None:
    gpus = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True)
    assert len(gpus.stdout.splitlines()) >= 3, "needs three GPUs"
    worker = {
        "kv_cache_config": {"tokens_per_block": TOKENS_PER_BLOCK, "free_gpu_memory_fraction": 0.3},
        "max_batch_size": 8,
        "max_seq_len": 2048,
        "cache_transceiver_config": {"backend": "DEFAULT"},
    }
    context = {**worker, "disable_overlap_scheduler": True}
    servers = {  # name: (role, GPU, extra LLM options)
        "generation": ("generation", "1", worker),
        "context-without-store": ("context", "2", context),
        "context-with-store": (
            "context",
            "0",
            {**context, "kv_sharing": sharing(store, FETCH_PREFIX, **READER)},
        ),
    }
    with ExitStack() as stack:
        started = {}
        for name, (role, gpu, options) in servers.items():
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=gpu,
                UCX_TLS=SERVER_UCX_TLS,
                # 4 KiB host pages: THP compaction can outlast the start-up wait (nvbugs/6888251).
                TLLM_KV_CACHE_MANAGER_V2_THP="0",
            )
            env.pop("TLLM_WORKER_USE_SINGLE_PROCESS", None)
            started[name] = stack.enter_context(
                RemoteOpenAIServer(
                    model_path(),
                    cli_args=["--server_role", role],
                    env=env,
                    extra_config=options,
                    log_path=str(tmp_path / f"{name}.log"),
                    wait=False,
                )
            )
        for name, worker_server in started.items():
            worker_server.wait_for_server(timeout=RemoteOpenAIServer.MAX_SERVER_START_WAIT_S)
        routers = {
            store_use: stack.enter_context(
                RemoteDisaggOpenAIServer(
                    ctx_servers=[f"localhost:{started[f'context-{store_use}'].port}"],
                    gen_servers=[f"localhost:{started['generation'].port}"],
                    log_path=str(tmp_path / f"router-{store_use}.log"),
                )
            )
            for store_use in ("without-store", "with-store")
        }

        def completion(store_use: str, prompt: list[int]):
            client = routers[store_use].get_client()
            return client.completions.create(
                model=model_path(), prompt=prompt, max_tokens=NEW_TOKENS, temperature=0
            )

        # Without the store, the context server reuses P's blocks, which it computed first.
        completion("without-store", P)
        reference = completion("without-store", T["disaggregated"])
        before, _ = store.readable_objects(FETCH_PREFIX)
        shared = completion("with-store", T["disaggregated"])
        # The context server fetched P's blocks, computed the rest and published it.
        published = before + T_BLOCKS - FETCHED_BLOCKS
        assert wait_until(lambda: store.readable_objects(FETCH_PREFIX) == (published, published)), (
            f"the store holds {store.readable_objects(FETCH_PREFIX)}; expected {published}"
        )
    assert shared.choices[0].text == reference.choices[0].text
    assert reference.usage.prompt_tokens_details.cached_tokens == FETCHED
    assert shared.usage.prompt_tokens_details.cached_tokens == FETCHED

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
locally, also when it computes the rest in chunks; a request cancelled while a stuck store holds its
fetch ends at once. Servers move blocks over GPU Direct, which 'auto' picks here, or through host
staging, and a block either path stores reads exactly on the other. Two instances of one service
share system prompts and follow-up turns under concurrent load, with KV caches too small to hold it
all. A disaggregated context server, whose loop runs without the overlap scheduler, fetches,
computes the rest and publishes. When the store fails, requests compute their prompts with the same
tokens: slow gets still serve hits, stuck ones pause fetching, a stalled master pauses probing and
publishing, a full store declines puts without a pause, and a restarted provider only empties the
store. Where two servers provide all of the store's memory, one that stops normally while the other
reads from it ends no reader. The store and its drivers are the module's (``conftest.py``), except
where a test needs a store of its own.

Runs only with ``TRTLLM_MOONCAKE_E2E=1`` (it skips otherwise), from ``tests/unittest``:
``TRTLLM_MOONCAKE_E2E=1 pytest _torch/executor/kv_sharing/test_mooncake_store_e2e.py``. Then it
needs four GPUs (servers of one test start side by side, each on a GPU of its own),
``mooncake_master`` on the path, the ``mooncake.store`` SDK and, for the native driver, the built
``tensorrt_llm.mooncake_store_worker``, and fails without them.
"""

from __future__ import annotations

import json
import os
import random
import signal
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import pytest
from conftest import (
    DRIVER,
    FIRST_MESSAGE_TOKENS,
    SERVER_STOP_S,
    SYSTEM_TOKENS,
    TOKENS_PER_BLOCK,
    Store,
    ask,
    assert_agree,
    deployment_text,
    instance_counters,
    instance_options,
    rank_traces,
    require_environment,
    run_two_turn_workload,
    running_provider,
    running_store,
    send_at_once,
    shut_down,
    skip_unless_opted_in,
    start_instance,
    stop_server,
    wait_until,
    whole_blocks,
    workload_sharing,
)
from llmapi.apps.openai_server import RemoteDisaggOpenAIServer, RemoteOpenAIServer
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.disaggregation.blob.mooncake import MooncakeDriver
from tensorrt_llm._torch.disaggregation.blob.mooncake_setup import (
    MOONCAKE_UNMOUNT_CLEANUP_S,
    UNMOUNT_GRACE_S,
)
from tensorrt_llm._torch.disaggregation.blob.store import ProbeStatus
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import MemoryKind
from tensorrt_llm._torch.pyexecutor.kv_sharing import breaker, data_path
from tensorrt_llm.llmapi import KvCacheConfig

pytestmark = skip_unless_opted_in

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
SIBLINGS = ("exact", "cancelled", "disaggregated")
T = {name: P[:FETCHED] + [_tokens.randrange(1000, 30000) for _ in range(200)] for name in SIBLINGS}
# The aggregated tests compare with a server without the store; the disaggregated one with its own.
REFERENCED = SIBLINGS[:-1]
T_BLOCKS = (FETCHED + 200) // TOKENS_PER_BLOCK  # 22
# A sibling of P with 600 tokens of its own: past P's 16 blocks, three chunks of chunked prefill.
LONG_SIBLING = P[:FETCHED] + [_tokens.randrange(1000, 30000) for _ in range(600)]
LONG_SIBLING_NEW_BLOCKS = len(LONG_SIBLING) // TOKENS_PER_BLOCK - FETCHED_BLOCKS  # 18
# The key prefix under which a writer published P for the readers.
FETCH_PREFIX = "fetch"
# Readers fetch prefixes of at least two blocks.
READER = {"min_fetch_tokens": 2 * TOKENS_PER_BLOCK}

# The failure tests' prompts. Each prompt F[i] has a sibling S[i]: F[i]'s first 16 blocks, then
# 200 tokens of its own; a server holding F[i] reuses exactly the 16 blocks.
_failure_tokens = random.Random(20261024)
F = [[2000 + i] + [_failure_tokens.randrange(1000, 30000) for _ in range(999)] for i in range(9)]
S = [f[:FETCHED] + [_failure_tokens.randrange(1000, 30000) for _ in range(200)] for f in F]
F_BLOCKS = len(F[0]) // TOKENS_PER_BLOCK  # 31, as P


def model_path() -> str:
    return str(llm_models_root(check=True) / "Qwen3" / "Qwen3-0.6B")


def server(monkeypatch: pytest.MonkeyPatch, **kwargs) -> LLM:
    """A Qwen3-0.6B server whose executor runs in this process, for a test to read."""
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    return LLM(
        model=model_path(),
        kv_cache_config=KvCacheConfig(
            tokens_per_block=TOKENS_PER_BLOCK, free_gpu_memory_fraction=0.3
        ),
        max_batch_size=8,
        max_seq_len=2048,
        **kwargs,
    )


def sharing(
    store: Store, key_prefix: str, max_fetch_tokens: int = 1024, driver: str = DRIVER, **knobs
) -> dict:
    """``kv_sharing`` with a small fetch size: these prompts are short. Publications in flight hold
    at most max(max_fetch_tokens, max_num_tokens) tokens. Unless ``knobs`` say otherwise, nothing
    shorter than ``max_fetch_tokens`` is fetched."""
    return {
        "backends": [{"type": "mooncake", "config": store.config(key_prefix, driver)}],
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


def settled(executor) -> bool:
    """Whether every store operation of the in-process executor ended and its lease went back."""
    return not executor.disagg.has_pending_shared_transfers()


def test_publication_stores_whole_prompt_blocks_and_keeps_output(
    store: Store, references: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(data_path.logger, "warning", warnings.append)
    llm = server(monkeypatch, kv_sharing=sharing(store, "overlap"))
    executor = llm._executor.engine
    try:
        # 'auto' picks GPU Direct here.
        assert executor.kv_sharing.memory_kind is MemoryKind.DEVICE, warnings
        tokens = {"P": generate(llm, P), "Q": generate(llm, Q)}
        stored = P_BLOCKS + Q_BLOCKS
        store.assert_readable("overlap", stored)
        # The executor retires each publication once its store calls returned.
        assert wait_until(lambda: settled(executor))
    finally:
        shut_down(llm)
    assert tokens == references["fresh"]
    assert executor.disagg.shared_shutdown_complete()
    assert store.readable_objects("overlap") == (stored, stored)  # no generated block came later
    counters = executor.kv_sharing.counters
    assert counters["publications_started"] == 2
    assert counters["published_objects"] == stored
    assert counters["publications_not_started"] == counters["publications_failed"] == 0


def test_chunked_prefill_publishes_each_chunk_and_a_fetch_computes_the_rest_in_chunks(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A writer under chunked prefill publishes each chunk's range once its commit lands, its pages
    locked while later chunks compute, and serves the tokens of a server without the store. A
    fresh reader of LONG_SIBLING fetches its first 16 blocks, P's, then computes the rest in three
    chunks and publishes each once: the tokens of a server that reused the 16 blocks locally."""
    llm = server(monkeypatch, **CHUNKED_PREFILL)
    try:
        reference = generate(llm, P)
        reused = complete(llm, LONG_SIBLING)
    finally:
        shut_down(llm)
    assert reused.cached_tokens == FETCHED
    llm = server(monkeypatch, kv_sharing=sharing(store, "chunked"), **CHUNKED_PREFILL)
    writer = llm._executor.engine
    try:
        tokens = generate(llm, P)
        store.assert_readable("chunked", P_BLOCKS)
        assert wait_until(lambda: settled(writer))
    finally:
        shut_down(llm)
    llm = server(monkeypatch, kv_sharing=sharing(store, "chunked", **READER), **CHUNKED_PREFILL)
    reader = llm._executor.engine
    try:
        fetched = complete(llm, LONG_SIBLING)
        store.assert_readable("chunked", P_BLOCKS + LONG_SIBLING_NEW_BLOCKS)
        assert wait_until(lambda: settled(reader))
    finally:
        shut_down(llm)
    assert tokens == reference
    counters = writer.kv_sharing.counters
    assert counters["publications_started"] == 4  # [0, 256), [256, 512), [512, 768), [768, 992)
    assert counters["published_objects"] == P_BLOCKS
    assert counters["publications_not_started"] == counters["publications_failed"] == 0
    assert list(fetched.outputs[0].token_ids) == list(reused.outputs[0].token_ids)
    assert fetched.cached_tokens == FETCHED
    counters = reader.kv_sharing.counters
    assert counters["fetches_resumed"] == 1
    assert counters["publications_started"] == 3  # [512, 768), [768, 1024), [1024, 1088)
    assert counters["published_objects"] == LONG_SIBLING_NEW_BLOCKS
    assert counters["publications_not_started"] == counters["publications_failed"] == 0


# Fetch.


def block_bytes() -> int:
    """Bytes of one KV cache block of the model: K and V of every layer, in BF16."""
    config = json.loads((Path(model_path()) / "config.json").read_text())
    per_token = 2 * config["num_hidden_layers"] * config["num_key_value_heads"] * config["head_dim"]
    return per_token * TOKENS_PER_BLOCK * 2


def publish(
    store: Store, key_prefix: str, prompts: list[list[int]], *, gpu_direct_fails: bool = False
) -> MemoryKind:
    """A writer server, shut down once done, publishes the whole blocks of ``prompts`` under
    ``key_prefix``; its memory kind. With ``gpu_direct_fails`` its GPU Direct preparation fails,
    and 'auto' falls back to host staging, logging why."""
    warnings: list[str] = []
    with pytest.MonkeyPatch.context() as monkeypatch:
        if gpu_direct_fails:
            monkeypatch.setattr(data_path, "prepare_gpu_direct", lambda _: "forced by the test")
            monkeypatch.setattr(data_path.logger, "warning", warnings.append)
        llm = server(monkeypatch, kv_sharing=sharing(store, key_prefix))
    executor = llm._executor.engine
    try:
        for prompt in prompts:
            generate(llm, prompt)
        store.assert_readable(key_prefix, len(whole_blocks(prompts)))
    finally:
        shut_down(llm)
    fell_back = [w for w in warnings if "host staging" in w and "forced by the test" in w]
    assert len(fell_back) == gpu_direct_fails, warnings
    return executor.kv_sharing.memory_kind


def reader(monkeypatch: pytest.MonkeyPatch, store: Store, key_prefix=FETCH_PREFIX, **knobs) -> LLM:
    """A fresh server, in process, that fetches what the writer of P published."""
    return server(monkeypatch, kv_sharing=sharing(store, key_prefix, **READER, **knobs))


@pytest.fixture(scope="module")
def references() -> dict:
    """Greedy tokens of one server without the store: "fresh" for P and Q (nothing to reuse),
    "whole" for each sibling under a salt of its own, "reused" for each sibling with P's first 16
    blocks reused (P computed before it); and the failure tests' (``failure_references``)."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        llm = server(monkeypatch)
    failures: dict[str, list[list[int]]] = {"prompt": [], "whole": [], "reused": []}
    try:
        # A salt of its own leaves the request nothing to reuse.
        whole = {name: complete(llm, T[name], cache_salt=f"whole-{name}") for name in REFERENCED}
        prompts = {"P": generate(llm, P), "Q": generate(llm, Q)}  # nothing to reuse yet
        reused = {name: complete(llm, T[name]) for name in REFERENCED}
        for f, s in zip(F, S):
            sibling_whole = complete(llm, s, cache_salt=f"whole-{f[0]}")
            failures["prompt"].append(generate(llm, f))
            sibling_reused = complete(llm, s)
            assert sibling_whole.cached_tokens == 0 and sibling_reused.cached_tokens == FETCHED
            failures["whole"].append(list(sibling_whole.outputs[0].token_ids))
            failures["reused"].append(list(sibling_reused.outputs[0].token_ids))
    finally:
        shut_down(llm)
    assert all(output.cached_tokens == 0 for output in whole.values())
    assert all(output.cached_tokens == FETCHED for output in reused.values())
    return {
        "fresh": prompts,
        **{
            kind: {name: list(output.outputs[0].token_ids) for name, output in outputs.items()}
            for kind, outputs in (("whole", whole), ("reused", reused))
        },
        "failures": failures,
    }


@pytest.fixture(scope="module")
def failure_references(references: dict) -> dict[str, list[list[int]]]:
    """The store-off server's tokens for the failure tests: each F[i] computed whole ("prompt"),
    each S[i] computed whole ("whole") and with F[i]'s 16 blocks reused ("reused")."""
    return references["failures"]


@pytest.fixture(scope="module")
def published_p(store: Store) -> None:
    """P published under FETCH_PREFIX by a writer over GPU Direct, which 'auto' picks here."""
    assert publish(store, FETCH_PREFIX, [P]) is MemoryKind.DEVICE


@pytest.fixture(scope="module")
def staged_p(store: Store) -> str:
    """The key prefix under which a writer that fell back to host staging published P."""
    assert publish(store, "fetch-staged", [P], gpu_direct_fails=True) is MemoryKind.HOST
    return "fetch-staged"


@pytest.mark.parametrize("read_on", ["host_staging", "gpu_direct"])
def test_a_fresh_server_fetches_a_prefix_published_on_the_other_data_path_exactly(
    store: Store,
    published_p: None,
    staged_p: str,
    references: dict,
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
    assert list(fetched.outputs[0].token_ids) == references["reused"]["exact"]
    # A fresh server holds no block: the cached tokens came from the store.
    assert fetched.cached_tokens == FETCHED
    counters = executor.kv_sharing.counters
    assert counters["fetches_started"] == counters["fetches_resumed"] == 1
    assert counters["fetched_blocks"] == counters["fetched_objects"] == FETCHED_BLOCKS
    assert counters["fetched_bytes"] == FETCHED_BLOCKS * block_bytes()
    assert list(salted.outputs[0].token_ids) == references["whole"]["exact"]
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


# Each instance's KV cache in the workload under pressure: about 135 pages, of which about 63 may
# be pinned beside the resume of one max_seq_len request, while a wave publishes eight first
# prompts of 21 pages each at once over GPU Direct.
KV_CACHE_PRESSURE_TOKENS = 4096


def test_two_instances_share_system_prompts_and_follow_up_turns_under_kv_cache_pressure(
    store: Store, tmp_path: Path
) -> None:
    """Two one-GPU instances of one service whose KV caches hold KV_CACHE_PRESSURE_TOKENS tokens
    each, fewer than the workload keeps at once."""
    run_two_turn_workload(
        store,
        model_path(),
        tp=1,
        gpu_groups=("0", "1"),
        run_dir=tmp_path,
        kv_cache_tokens=KV_CACHE_PRESSURE_TOKENS,
    )


def test_a_cancel_while_the_store_holds_a_fetch_ends_the_request_at_once(
    store: Store,
    published_p: None,
    references: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A client cancels a request whose fetch a stuck store holds: the request ends at once, not
    when the store answers, and the server serves on once it does. (The test stalls the Python
    driver's gets for seconds; the access watchdog ends the process only after minutes.)"""
    store_answers = threading.Event()
    get_rows = MooncakeDriver._get_rows

    def stalled_get_rows(self, *args, **kwargs):
        store_answers.wait()
        return get_rows(self, *args, **kwargs)

    monkeypatch.setenv("TRTLLM_KV_SHARING_TRACE_DIR", str(tmp_path))
    llm = reader(monkeypatch, store, driver="python", fetch_timeout_s=60.0)
    # Once started: the startup check reads through the store too.
    monkeypatch.setattr(MooncakeDriver, "_get_rows", stalled_get_rows)
    counters = llm._executor.engine.kv_sharing.counters
    try:
        cancelled = llm.generate_async(T["cancelled"], GREEDY)
        assert wait_until(lambda: counters["fetches_started"] == 1)
        cancelled.abort()
        output = cancelled.result(timeout=5)
        assert not store_answers.is_set()
        store_answers.set()
        assert wait_until(lambda: settled(llm._executor.engine))
        served = complete(llm, T["cancelled"])
    finally:
        store_answers.set()
        shut_down(llm)
    assert output.outputs[0].finish_reason == "cancelled"
    assert list(served.outputs[0].token_ids) == references["reused"]["cancelled"]
    assert served.cached_tokens == FETCHED
    assert counters["fetches_cancelled"] == 1 and counters["fetches_resumed"] == 1
    assert rank_traces(tmp_path, 1).splitlines()[-1].endswith(" open=0")


# Disaggregated serving: the context server shares through the store; P->D is unchanged.

# NIXL's UCX backend needs a CUDA transport; servers on one node need no InfiniBand.
SERVER_UCX_TLS = "cuda_copy,cuda_ipc,sm,self,tcp"


def test_disaggregated_context_server_fetches_computes_the_rest_and_publishes(
    store: Store, published_p: None, tmp_path: Path
) -> None:
    """Each run has a generation server of its own: one that kept the reference run's blocks
    would skip receiving them, and decode from those instead of the fetched ones."""
    gpus = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True)
    assert len(gpus.stdout.splitlines()) >= 4, "needs four GPUs"
    worker = {
        "kv_cache_config": {"tokens_per_block": TOKENS_PER_BLOCK, "free_gpu_memory_fraction": 0.3},
        "max_batch_size": 8,
        "max_seq_len": 2048,
        "cache_transceiver_config": {"backend": "DEFAULT"},
    }
    context = {**worker, "disable_overlap_scheduler": True}
    servers = {  # name: (role, GPU, extra LLM options)
        "generation-without-store": ("generation", "3", worker),
        "context-without-store": ("context", "2", context),
        "generation-with-store": ("generation", "1", worker),
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
                    gen_servers=[f"localhost:{started[f'generation-{store_use}'].port}"],
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


# Failures of the store, on the prompts F and their siblings S.

# The breakers' limits in these tests.
FAILURES_TO_PAUSE, PAUSE_S = 3, 5.0


def short_breakers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(breaker, "FAILURES_TO_PAUSE", FAILURES_TO_PAUSE)
    monkeypatch.setattr(breaker, "PAUSE_S", PAUSE_S)


def test_slow_gets_serve_hits_and_stuck_or_failing_ones_pause_fetches(
    store: Store, failure_references: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gets that take 0.3 s still resume their requests. A get that never answers goes late at
    ``fetch_timeout_s`` and its request computes; with no fetch resumed within that time it is a
    store failure (a slow data plane), as are gets the store fails, whose requests drop and
    compute. After FAILURES_TO_PAUSE such rounds in a row the next request is not even probed.
    Once the pause passed, a request fetches again. (The stuck get returns well within the access
    watchdog's grace.)"""
    short_breakers(monkeypatch)
    publish(store, "slow", F[:8])
    slow, stuck, failing = range(2), 2, range(3, 3 + FAILURES_TO_PAUSE - 1)
    unprobed, after_pause = failing.stop, failing.stop + 1
    store_answers = threading.Event()
    gets = ["slow"]
    get_rows = MooncakeDriver._get_rows

    def misbehaving_get_rows(self, batch, indices, may_reread):
        if gets[0] == "failing":  # the store answers an error: access ended, rows faulted
            batch.fault(indices, "batch_get_into_multi_buffers: RPC_FAIL(-900)")
            return True
        if gets[0] == "stuck":
            store_answers.wait()
        time.sleep(0.3)
        return get_rows(self, batch, indices, may_reread)

    monkeypatch.setenv("TRTLLM_KV_SHARING_TRACE_DIR", str(tmp_path))
    llm = reader(monkeypatch, store, key_prefix="slow", driver="python", fetch_timeout_s=1.0)
    monkeypatch.setattr(MooncakeDriver, "_get_rows", misbehaving_get_rows)
    executor = llm._executor.engine
    counters = executor.kv_sharing.counters  # the probe's counts join it at shutdown
    outputs = {}
    try:
        for i in slow:
            outputs[i] = complete(llm, S[i])
        assert counters["fetches_resumed"] == len(slow)
        gets[0] = "stuck"
        outputs[stuck] = complete(llm, S[stuck])
        assert counters["fetches_dropped_late"] == 1
        gets[0] = "failing"
        for i in failing:
            outputs[i] = complete(llm, S[i])
        assert counters["fetches_dropped_short"] == FAILURES_TO_PAUSE - 1
        started = counters["fetches_started"]
        outputs[unprobed] = complete(llm, S[unprobed])  # probing paused
        assert counters["fetches_started"] == started
        gets[0] = "slow"
        store_answers.set()
        assert wait_until(lambda: settled(executor))
        time.sleep(PAUSE_S)
        outputs[after_pause] = complete(llm, S[after_pause])
    finally:
        store_answers.set()
        shut_down(llm)
    for i, output in outputs.items():
        fetched = i in (*slow, after_pause)
        kind = "reused" if fetched else "whole"
        assert list(output.outputs[0].token_ids) == failure_references[kind][i], i
        assert output.cached_tokens == (FETCHED if fetched else 0), i
    assert counters["fetches_resumed"] == len(slow) + 1 and counters["fetch_pauses"] == 1
    assert counters["probe_pauses"] == 0
    assert rank_traces(tmp_path, 1).splitlines()[-1].endswith(" open=0")


# The stalled-master test: below the master's 10 s client TTL, so no client remounts; requests
# a little more than one RPC timeout apart, so each one's probe is a store call of its own.
MASTER_STALL_S = 6.5
STALL_RPC_TIMEOUT_MS = 1000
STALL_REQUEST_GAP_S = STALL_RPC_TIMEOUT_MS / 1000 + 0.6


def test_a_stalled_master_pauses_probes_and_publications_and_serving_goes_on(
    failure_references: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The master stops answering (``SIGSTOP``). New requests' probe calls and their publications'
    put starts time out (the store config's 1 s RPC timeout): after FAILURES_TO_PAUSE of each,
    probing and publishing pause, the next request leaves unprobed, and every request completes
    with its reference's tokens. Once the master answers and the pauses passed, a publication
    stores its objects and a request fetches."""
    short_breakers(monkeypatch)
    with running_store(tmp_path / "store") as own:
        publish(own, "stall", [F[8]])
        trace_dir = tmp_path / "trace"
        monkeypatch.setenv("TRTLLM_KV_SHARING_TRACE_DIR", str(trace_dir))
        config = sharing(own, "stall", **READER)
        config["backends"][0]["config"] = own.config("stall", rpc_timeout_ms=STALL_RPC_TIMEOUT_MS)
        llm = server(monkeypatch, kv_sharing=config)
        executor = llm._executor.engine
        counters = executor.kv_sharing.counters  # the probe's counts join it at shutdown
        stalled = []
        try:
            os.kill(own.master_pid, signal.SIGSTOP)
            stopped_at = time.monotonic()
            try:
                for i, prompt in enumerate(F[: FAILURES_TO_PAUSE + 1]):  # the last one unprobed
                    time.sleep(max(0.0, stopped_at + i * STALL_REQUEST_GAP_S - time.monotonic()))
                    stalled.append(complete(llm, prompt))
            finally:
                time.sleep(max(0.0, stopped_at + MASTER_STALL_S - time.monotonic()))
                os.kill(own.master_pid, signal.SIGCONT)
            assert wait_until(lambda: settled(executor))
            time.sleep(PAUSE_S)
            computed = complete(llm, F[6])
            own.assert_readable("stall", 2 * F_BLOCKS)  # F[8]'s blocks and F[6]'s
            fetched = complete(llm, S[8])
        finally:
            os.kill(own.master_pid, signal.SIGCONT)
            shut_down(llm)
    for i, output in enumerate(stalled):
        assert list(output.outputs[0].token_ids) == failure_references["prompt"][i], i
    assert list(computed.outputs[0].token_ids) == failure_references["prompt"][6]
    assert list(fetched.outputs[0].token_ids) == failure_references["reused"][8]
    assert fetched.cached_tokens == FETCHED and counters["fetches_resumed"] == 1
    assert counters["probe_pauses"] == counters["publication_pauses"] == 1
    # Each probed request outlasted the hold; the last one went unprobed.
    assert counters["held_past_limit"] == FAILURES_TO_PAUSE
    assert counters["publications_failed"] >= FAILURES_TO_PAUSE
    assert rank_traces(trace_dir, 1).splitlines()[-1].endswith(" open=0")


def test_a_full_store_declines_puts_without_pausing(
    failure_references: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store of about 70 of this model's objects, which evicts as Mooncake does by default,
    takes eight prompts of 31 blocks each: puts the full store declines are skipped, not failed,
    and publishing never pauses. A fresh reader of the last prompt fetches the leading blocks the
    store kept of it, as the store answered its probe."""
    short_breakers(monkeypatch)
    probe = MooncakeDriver.probe
    probed: list[tuple[ProbeStatus, ...]] = []

    def recording_probe(self, keys, budget_s, hint=None):
        probed.append(probe(self, keys, budget_s, hint))
        return probed[-1]

    with running_store(tmp_path / "store", eviction_ratio=0.05, provider_bytes=256 << 20) as own:
        llm = server(monkeypatch, kv_sharing=sharing(own, "full"))
        executor = llm._executor.engine
        try:
            computed = [generate(llm, prompt) for prompt in F[:8]]
            assert wait_until(lambda: settled(executor))
        finally:
            shut_down(llm)
        counters = executor.kv_sharing.counters  # the driver's counts join it at shutdown
        llm = reader(monkeypatch, own, key_prefix="full")
        reader_counters = llm._executor.engine.kv_sharing.counters
        monkeypatch.setattr(MooncakeDriver, "probe", recording_probe)
        try:
            last = complete(llm, F[7])
        finally:
            shut_down(llm)
    assert computed == failure_references["prompt"][:8]
    assert counters["driver_rows_declined_pool_full"] > 0
    assert counters["publication_pauses"] == counters["publications_failed"] == 0
    # The master kept a leading run of the last prompt's blocks; the reader fetched exactly it, up
    # to the whole blocks before the last prompt token.
    (answer,) = probed
    kept = next((b for b, status in enumerate(answer) if status is not ProbeStatus.HIT), F_BLOCKS)
    assert kept * TOKENS_PER_BLOCK >= READER["min_fetch_tokens"], f"the store kept {kept} blocks"
    fetchable = (len(F[7]) - 1) // TOKENS_PER_BLOCK
    assert last.cached_tokens == min(kept, fetchable) * TOKENS_PER_BLOCK
    assert reader_counters["fetches_resumed"] == 1
    assert list(last.outputs[0].token_ids) == failure_references["prompt"][7]


def test_a_restarted_provider_empties_the_store_and_serving_goes_on(
    failure_references: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The store's only provider stops while servers are idle and a new one starts: a request
    served meanwhile has its puts declined, a sibling of what the old provider held misses and
    computes, and what is published afterwards reads back exactly. Nothing pauses and the server
    serves throughout."""
    short_breakers(monkeypatch)
    run_dir = tmp_path / "store"
    with running_store(run_dir, provider_bytes=0) as own, ExitStack() as providers:
        old = providers.enter_context(running_provider(own.master, run_dir, 1 << 30))
        llm = server(monkeypatch, kv_sharing=sharing(own, "restart", **READER))
        executor = llm._executor.engine
        try:
            computed = [generate(llm, F[0])]
            own.assert_readable("restart", F_BLOCKS)
            assert wait_until(lambda: settled(executor))
            providers.close()  # the old provider stops (nothing in flight)
            assert old.returncode is not None
            computed.append(generate(llm, F[1]))  # no segment: its puts are declined
            assert wait_until(lambda: settled(executor))
            # A directory of its own: the old provider's readiness marker stays in run_dir.
            providers.enter_context(running_provider(own.master, run_dir / "new", 1 << 30))
            computed.append(generate(llm, F[2]))
            assert wait_until(lambda: settled(executor))
        finally:
            shut_down(llm)
        counters = executor.kv_sharing.counters
        # F[2]'s blocks only: F[0]'s are gone.
        assert own.readable_objects("restart") == (F_BLOCKS, F_BLOCKS), dict(counters)
        llm = reader(monkeypatch, own, key_prefix="restart")
        try:
            missed = complete(llm, S[0])
            fetched = complete(llm, S[2])
        finally:
            shut_down(llm)
    assert computed == failure_references["prompt"][:3]
    assert counters["driver_rows_declined_pool_full"] >= F_BLOCKS
    assert counters["publication_pauses"] == counters["publications_failed"] == 0
    assert list(missed.outputs[0].token_ids) == failure_references["whole"][0]
    assert missed.cached_tokens == 0
    assert list(fetched.outputs[0].token_ids) == failure_references["reused"][2]
    assert fetched.cached_tokens == FETCHED


# Servers that provide all of the store's memory.
SEGMENT_BYTES = 4 << 30
DISTINCT_PROMPTS = 40
# The reader gets one request this often, from four clients, over about 30 s; the other server
# stops STOP_AFTER_S into them.
REQUEST_EVERY_S = 0.75
STOP_AFTER_S = 5.0


def distinct_prompts(model: str) -> list[list[int]]:
    """DISTINCT_PROMPTS prompts, each a system prompt of 16 blocks and a user message of 4:
    consecutive pieces of the docs."""
    from transformers import AutoTokenizer

    tokens = AutoTokenizer.from_pretrained(model).encode(
        deployment_text(), add_special_tokens=False
    )
    length = SYSTEM_TOKENS + FIRST_MESSAGE_TOKENS
    prompts = [tokens[i * length : (i + 1) * length] for i in range(DISTINCT_PROMPTS)]
    assert len(prompts[-1]) == length, "the docs are too short for the prompts"
    return prompts


def test_a_server_providing_memory_that_stops_normally_ends_no_server_reading_it(
    tmp_path: Path,
) -> None:
    """Two one-GPU instances, A and B, both fetch, publish and provide the store 4 GiB each; there
    is no provider. A computes 40 system prompts, whose blocks the master places in A's memory or
    B's. Then four clients send B one request on each system prompt, one every 0.75 s, while A
    stops normally 5 s in. B reads A's blocks before and during A's grace; afterwards those blocks
    miss and B computes them. A's stop waits until Mooncake freed its memory, and its segment
    leaves the store. B's answers agree with a store-off server's, and B leaves no store access
    unproven."""
    require_environment()
    model = model_path()
    prompts = distinct_prompts(model)
    systems = [prompt[:SYSTEM_TOKENS] for prompt in prompts]
    with running_store(tmp_path / "store", provider_bytes=0) as own, ExitStack() as servers:
        providing = workload_sharing(own, "providing")
        providing["backends"][0].update(
            roles=["fetch", "publish", "provide"], segment_size=SEGMENT_BYTES
        )
        # The store-off server for the references starts with A and B, on a GPU of its own.
        off, a, b = (
            servers.enter_context(start_instance(model, 1, gpu, tmp_path / name, options))
            for gpu, name, options in (
                ("2", "store-off", instance_options(None)),
                ("0", "a", instance_options(providing)),
                ("1", "b", instance_options(providing)),
            )
        )
        for instance in (off, a, b):
            instance.wait_for_server(timeout=RemoteOpenAIServer.MAX_SERVER_START_WAIT_S)
        references = send_at_once([off], model, [(0, prompt) for prompt in prompts])
        off.terminate()
        assert len(own.segments()) == 2
        send_at_once([a], model, [(0, system) for system in systems], (False,))
        own.assert_readable("providing", len(whole_blocks(systems)))

        started = time.monotonic()

        def ask_b(index: int):
            time.sleep(max(0.0, started + index * REQUEST_EVERY_S - time.monotonic()))
            return ask(b, model, prompts[index])

        with ThreadPoolExecutor(4) as clients:
            pending = clients.map(ask_b, range(DISTINCT_PROMPTS))
            time.sleep(STOP_AFTER_S)
            a_stopping = time.monotonic()
            a.proc.terminate()  # a normal stop
            statuses = set()
            while a.proc.poll() is None and time.monotonic() < a_stopping + SERVER_STOP_S:
                statuses |= {segment["status"] for segment in own.segments()}
                time.sleep(0.2)
            a_stop_s = time.monotonic() - a_stopping
            assert a.proc.poll() is not None, f"A did not stop within {SERVER_STOP_S:.0f} s"
            answers = list(pending)
        assert wait_until(lambda: len(own.segments()) == 1), own.segments()
        stop_server(b)
    print(f"A stopped in {a_stop_s:.1f} s; the store's segments went through {statuses}")
    assert "GRACEFULLY_UNMOUNTING" in statuses
    assert a_stop_s >= UNMOUNT_GRACE_S + MOONCAKE_UNMOUNT_CLEANUP_S
    assert_agree(references, answers, model)
    read_whole_system_prompt = [answer.cached_tokens >= SYSTEM_TOKENS for answer in answers]
    # Read whole from both servers' memory, then partly once A's blocks left.
    assert any(read_whole_system_prompt) and not all(read_whole_system_prompt), (
        read_whole_system_prompt
    )
    (counters,) = instance_counters(tmp_path / "b")
    assert counters["driver_ops_unprovable"] == 0 and counters["fetches_resumed"] > 0, counters
    assert rank_traces(tmp_path / "b", 1).splitlines()[-1].endswith(" open=0")
    b_log = Path(f"{tmp_path / 'b'}.log").read_text(errors="replace")
    assert "still held KV cache memory" not in b_log  # the store access watchdog's message

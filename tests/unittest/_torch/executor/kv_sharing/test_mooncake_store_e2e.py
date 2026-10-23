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
"""KV sharing through a real Mooncake store, end to end.

A master and one provider (``trtllm-serve mooncake_provider``) run for the module; servers lend the
store no memory. A server publishes the whole blocks of every prompt it computes, and nothing it
generates, and serves the same tokens as a server without the store. A fresh server fetches the
stored prefix of a prompt instead of computing it, and its tokens equal those of a server that
reused the same blocks locally; a fetch that cannot deliver every block, or runs late, leaves the
request to compute its prompt. Servers move blocks over GPU Direct, which 'auto' picks here, or
through host staging, and a block either path stores reads exactly on the other. Every server makes
its store calls through the driver ``TRTLLM_MOONCAKE_E2E_DRIVER`` names: ``native`` (the default,
the C++ store worker; a server refuses to start without it) or ``python``.

Needs a GPU (three for the disaggregated test), ``mooncake_master`` on the path, the
``mooncake.store`` SDK and, for the native driver, the built ``tensorrt_llm.mooncake_store_worker``,
and fails without them. ``TRTLLM_MOONCAKE_PROTOCOL`` (default rdma) and ``TRTLLM_MOONCAKE_DEVICES``
(default: the fastest InfiniBand devices) pick the transport.
"""

from __future__ import annotations

import gc
import json
import os
import random
import shutil
import subprocess
import time
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import yaml
from llmapi.apps.openai_server import RemoteDisaggOpenAIServer, RemoteOpenAIServer
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.disaggregation.blob.mooncake import MooncakeDriver
from tensorrt_llm._torch.disaggregation.blob.mooncake_native import NativeMooncakeDriver
from tensorrt_llm._torch.disaggregation.blob.store import ProbeStatus
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import MemoryKind
from tensorrt_llm._torch.pyexecutor.kv_sharing import data_path
from tensorrt_llm.llmapi import KvCacheConfig

PROTOCOL = os.environ.get("TRTLLM_MOONCAKE_PROTOCOL", "rdma")
DRIVER = os.environ.get("TRTLLM_MOONCAKE_E2E_DRIVER", "native")
assert DRIVER in ("native", "python"), f"TRTLLM_MOONCAKE_E2E_DRIVER={DRIVER!r}"
DEVICES = os.environ.get("TRTLLM_MOONCAKE_DEVICES", "")
TOKENS_PER_BLOCK = 32
PROVIDER_BYTES = 4 << 30
WAIT_S = 60.0
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
SIBLINGS = ("exact", "lost", "late", "cancelled", "disaggregated")
T = {name: P[:FETCHED] + [_tokens.randrange(1000, 30000) for _ in range(200)] for name in SIBLINGS}
# The aggregated tests compare with a server without the store; the disaggregated one with its own.
REFERENCED = SIBLINGS[:-1]
T_BLOCKS = (FETCHED + 200) // TOKENS_PER_BLOCK  # 22
# The key prefix under which a writer published P for the readers.
FETCH_PREFIX = "fetch"
# Readers fetch prefixes of at least two blocks.
READER = {"min_fetch_tokens": 2 * TOKENS_PER_BLOCK}


@dataclass
class Store:
    """The module's Mooncake store, and a client that lends it nothing, to ask what it holds."""

    master: str
    metrics_url: str
    run_dir: Path
    client: object  # mooncake.store.MooncakeDistributedStore

    def config(self, key_prefix: str) -> str:
        """Writes a server's store config whose keys start with ``key_prefix``."""
        mooncake = {
            "master_server_address": self.master,
            "global_segment_size": 0,
            "protocol": PROTOCOL,
            "device_name": DEVICES,
            "driver": DRIVER,
        }
        path = self.run_dir / f"{key_prefix}.yaml"
        path.write_text(yaml.safe_dump({"mooncake": mooncake, "blob": {"key_prefix": key_prefix}}))
        return str(path)

    def readable_objects(self, key_prefix: str) -> tuple[int, int]:
        """How many objects the store has under ``key_prefix``, and how many of them a get could
        read now (an object whose put is still open is not readable)."""
        with urllib.request.urlopen(f"{self.metrics_url}/get_all_keys", timeout=10) as reply:
            keys = [k for k in reply.read().decode().split() if k.startswith(f"{key_prefix}/")]
        readable = sum(answer == 1 for answer in self.client.batch_is_exist(keys)) if keys else 0
        return len(keys), readable


def require_environment() -> None:
    """Fails, not skips, where the test cannot run."""
    assert torch.cuda.is_available(), "needs a GPU"
    assert shutil.which("mooncake_master"), "needs mooncake_master (mooncake-transfer-engine)"
    import mooncake.store  # noqa: F401


def wait_until(condition: Callable[[], bool]) -> bool:
    deadline = time.monotonic() + WAIT_S
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.2)
    return True


def server(monkeypatch: pytest.MonkeyPatch, *, in_process: bool, **kwargs) -> LLM:
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
        max_batch_size=8,
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
def store(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Store]:
    require_environment()
    from test_common.mooncake_utils import running_master_on_free_ports, running_segment_provider

    from tensorrt_llm._torch.disaggregation.blob.mooncake_setup import MooncakeConfig, open_sdk
    from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.gpudirect import (
        select_registration_path,
    )

    # The first Mooncake client of the process fixes how it registers device memory, and the
    # servers in this process register theirs after the test's own client.
    select_registration_path(PROTOCOL, registers_device_memory=True)
    run_dir = tmp_path_factory.mktemp("mooncake")
    with ExitStack() as stack:
        master = stack.enter_context(
            running_master_on_free_ports(str(run_dir), protocol=PROTOCOL, eviction_ratio=0.0)
        )
        stack.enter_context(
            running_segment_provider(
                master.address,
                str(run_dir),
                segment_size=PROVIDER_BYTES,
                protocol=PROTOCOL,
                device_name=DEVICES,
            )
        )
        metrics_port = json.loads((run_dir / "pool.json").read_text())["metrics_port"]
        host = master.address.rsplit(":", 1)[0]
        client = open_sdk(MooncakeConfig(master.address, 0, PROTOCOL, DEVICES, local_buffer_size=0))
        stack.callback(client.close)
        yield Store(master.address, f"http://{host}:{metrics_port}", run_dir, client)


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
    assert counters["published_blocks"] == stored
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
    assert counters["published_blocks"] == P_BLOCKS
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


def slow_store_reads(monkeypatch: pytest.MonkeyPatch, seconds: float) -> None:
    """Every store read in this process ends ``seconds`` after it was queued, as far as the
    server can tell; the executor runs on meanwhile. The Python driver's read sleeps on its
    backend thread; the native driver hands the backend no finished read younger than that."""
    if DRIVER == "python":
        get = MooncakeDriver.get

        def slow_get(self, rows, hint=None):
            time.sleep(seconds)
            return get(self, rows, hint)

        monkeypatch.setattr(MooncakeDriver, "get", slow_get)
        return
    take, waiting = NativeMooncakeDriver.take_finished, NativeMooncakeDriver.finished_waiting_s
    held: dict[NativeMooncakeDriver, list] = {}  # by driver: finished reads not handed out yet

    def take_when_old(self):
        mine = held.setdefault(self, [])
        mine += take(self)
        now = time.monotonic()
        due = [done for done in mine if done.op.is_put or now - done.op.submitted >= seconds]
        mine[:] = [done for done in mine if done not in due]
        return due

    def waiting_while_held(self):  # the backend asks this before it takes
        return float("inf") if held.get(self) else waiting(self)

    monkeypatch.setattr(NativeMooncakeDriver, "take_finished", take_when_old)
    monkeypatch.setattr(NativeMooncakeDriver, "finished_waiting_s", waiting_while_held)


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
    finally:
        shut_down(llm)
    assert all(output.cached_tokens == 0 for output in whole.values())
    assert all(output.cached_tokens == FETCHED for output in reused.values())
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
    assert counters["fetched_blocks"] == FETCHED_BLOCKS
    assert counters["fetched_bytes"] == FETCHED_BLOCKS * block_bytes()
    assert list(salted.outputs[0].token_ids) == sibling_references["whole"]["exact"]
    assert salted.cached_tokens == 0


def test_an_object_lost_after_the_probe_drops_the_fetch_and_computes_the_prompt(
    store: Store, sibling_references: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A prefix of its own: the test removes one of its objects.
    publish_p(store, "lost")
    probe = MooncakeDriver.probe
    lost: list[str] = []

    def probe_then_lose_a_block(self, keys, budget_s, hint=None):
        answer = probe(self, keys, budget_s, hint)
        if not lost:  # the fifth block goes after the store said it holds it
            lost.append(keys[4])
            assert store.client.remove(keys[4], True) == 0
        return answer

    monkeypatch.setattr(MooncakeDriver, "probe", probe_then_lose_a_block)
    llm = reader(monkeypatch, store, key_prefix="lost")
    executor = llm._executor.engine
    try:
        output = complete(llm, T["lost"])
        # The request computed its whole prompt and published it, the lost block included.
        assert wait_until(lambda: settled(executor))
        assert store.client.batch_is_exist(lost) == [1]
    finally:
        shut_down(llm)
    assert list(output.outputs[0].token_ids) == sibling_references["whole"]["lost"]
    assert output.cached_tokens == 0
    counters = executor.kv_sharing.counters
    assert counters["fetches_started"] == counters["fetches_dropped_short"] == 1
    assert counters["fetches_resumed"] == 0


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


def test_a_fetch_past_its_timeout_is_dropped_and_the_request_completes(
    store: Store, published_p: None, sibling_references: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    slow_store_reads(monkeypatch, seconds=2.0)
    llm = reader(monkeypatch, store, fetch_timeout_s=0.5)
    executor = llm._executor.engine
    try:
        output = complete(llm, T["late"])
        # The abandoned read returns later; its lease goes back then.
        assert wait_until(lambda: settled(executor))
    finally:
        shut_down(llm)
    assert list(output.outputs[0].token_ids) == sibling_references["whole"]["late"]
    assert output.cached_tokens == 0
    counters = executor.kv_sharing.counters
    assert counters["fetches_started"] == counters["fetches_dropped_late"] == 1
    assert counters["fetches_resumed"] == 0


def test_a_request_cancelled_while_parked_lets_its_fetch_go(
    store: Store, published_p: None, sibling_references: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    slow_store_reads(monkeypatch, seconds=3.0)
    llm = reader(monkeypatch, store, fetch_timeout_s=30.0)
    executor = llm._executor.engine
    counters = executor.kv_sharing.counters
    try:
        # A long generation keeps the forward running, so cancels are handled every iteration.
        busy = llm.generate_async(
            Q, SamplingParams(max_tokens=1000, temperature=0.0, ignore_eos=True)
        )
        parked = llm.generate_async(T["cancelled"], GREEDY)
        assert wait_until(lambda: counters["fetches_started"] == 1)
        parked.abort()
        # The request ends at once, with no token, while its read still runs.
        assert parked.result(timeout=WAIT_S).outputs[0].token_ids == []
        assert not settled(executor)
        busy.result(timeout=WAIT_S)
        assert wait_until(lambda: settled(executor))
        # The server goes on fetching.
        again = complete(llm, T["cancelled"])
        assert wait_until(lambda: settled(executor))
    finally:
        shut_down(llm)
    assert list(again.outputs[0].token_ids) == sibling_references["reused"]["cancelled"]
    assert again.cached_tokens == FETCHED
    assert counters["fetches_started"] == 2 and counters["fetches_resumed"] == 1


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

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
"""KV sharing through a real Mooncake store, end to end on one GPU.

A master and one provider (``trtllm-serve mooncake_provider``) run for the module; servers lend the
store no memory. A server publishes the whole blocks of every prompt it computes, and nothing it
generates, and serves the same tokens as a server without the store.

Needs a GPU, ``mooncake_master`` on the path and the ``mooncake.store`` SDK, and fails without
them. ``TRTLLM_MOONCAKE_PROTOCOL`` (default rdma) and ``TRTLLM_MOONCAKE_DEVICES`` (default: the
fastest InfiniBand devices) pick the transport.
"""

from __future__ import annotations

import gc
import json
import os
import random
import shutil
import time
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import yaml
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig

PROTOCOL = os.environ.get("TRTLLM_MOONCAKE_PROTOCOL", "rdma")
DEVICES = os.environ.get("TRTLLM_MOONCAKE_DEVICES", "")
TOKENS_PER_BLOCK = 32
PROVIDER_BYTES = 4 << 30
WAIT_S = 60.0
# Fills the last partial block of P and of Q: whole blocks then hold generated tokens.
NEW_TOKENS = 32
CHUNKED_PREFILL = {"enable_chunked_prefill": True, "max_num_tokens": 256}

_tokens = random.Random(20261022)
# Token ids; P and Q start differently, so they share no block.
P = [1000] + [_tokens.randrange(1000, 30000) for _ in range(999)]  # 31 whole blocks
Q = [1001] + [_tokens.randrange(1000, 30000) for _ in range(899)]  # 28 whole blocks
P_BLOCKS, Q_BLOCKS = len(P) // TOKENS_PER_BLOCK, len(Q) // TOKENS_PER_BLOCK


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


def sharing(store: Store, key_prefix: str, max_fetch_tokens: int = 1024) -> dict:
    """``kv_sharing`` with a small fetch size: these prompts are short. Publications in flight hold
    at most max(max_fetch_tokens, max_num_tokens) tokens."""
    return {
        "backends": [{"type": "mooncake", "config": store.config(key_prefix)}],
        "max_fetch_tokens": max_fetch_tokens,
    }


def generate(llm: LLM, prompt: list[int]) -> list[int]:
    """Greedy tokens for ``prompt``, alone in its batch."""
    output = llm.generate([prompt], SamplingParams(max_tokens=NEW_TOKENS, temperature=0.0))[0]
    return list(output.outputs[0].token_ids)


def shut_down(llm: LLM) -> None:
    llm.shutdown()
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Store]:
    require_environment()
    from test_common.mooncake_utils import running_master_on_free_ports, running_segment_provider

    from tensorrt_llm._torch.disaggregation.blob.mooncake_setup import MooncakeConfig, open_sdk

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
    llm = server(monkeypatch, in_process=True, kv_sharing=sharing(store, "overlap"))
    executor = llm._executor.engine
    try:
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
    assert counters["publications_refused"] == counters["publications_failed"] == 0


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
    # Room for one chunk's publication: each later chunk's range waits for the one before it.
    chunk = CHUNKED_PREFILL["max_num_tokens"]
    llm = server(
        monkeypatch,
        in_process=True,
        kv_sharing=sharing(store, "chunked", max_fetch_tokens=chunk),
        **CHUNKED_PREFILL,
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
    assert counters["publications_refused"] == counters["publications_failed"] == 0

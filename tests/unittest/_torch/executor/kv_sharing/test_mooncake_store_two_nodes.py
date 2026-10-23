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
"""KV sharing across two GB200 nodes through one Mooncake store: a tensor-parallel server over both
nodes' eight GPUs (Llama-3.1-8B TP8) reads exactly what an earlier server of that layout published,
and two instances of one service, one per node (TP4 each), share system prompts and follow-up
turns under concurrent load. The store has a provider on each node, so puts and gets cross nodes.

Runs as one pytest task per node of a two-node job (``SLURM_NODEID`` picks the role; four GPUs per
task, its steps ``--overlap``) with ``TRTLLM_KV_SHARING_SHARED_DIR`` naming a directory both nodes
see. Node 0 runs the master, a provider and every check; node 1 a provider and its instance of the
service. An eight-rank server spans both nodes, so the job launches it outside these tasks once
node 0 wrote ``<name>.args``:
``srun -N2 --ntasks-per-node=4 --mpi=pmix trtllm-llmapi-launch trtllm-serve $(cat <name>.args)``
with ``TRTLLM_KV_SHARING_TRACE_DIR=<shared dir>/<name>``. When node 0 writes ``<name>.stop``, the
job sends rank 0's ``trtllm-serve`` SIGTERM (a graceful stop: every rank writes its trace's exit
line, which the test checks), then writes ``<name>.stopped``. Fails unless it runs on two nodes.
"""

from __future__ import annotations

import contextlib
import json
import os
import random
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import openai
import pytest
import requests
from conftest import (
    DEVICES,
    PROTOCOL,
    PROVIDER_BYTES,
    TOKENS_PER_BLOCK,
    Completion,
    Store,
    decisions,
    instance_options,
    puts_since,
    rank_traces,
    require_environment,
    run_two_turn_workload,
    running_store,
    send_at_once,
    start_instance,
)
from llmapi.apps.openai_server import RemoteOpenAIServer
from utils.llm_data import llm_models_root

SHARED_DIR_ENV = "TRTLLM_KV_SHARING_SHARED_DIR"
NEW_TOKENS = 32
TP8 = 8
_tokens = random.Random(20261024)
P = [1000] + [_tokens.randrange(1000, 30000) for _ in range(999)]  # 31 whole blocks
P_BLOCKS = len(P) // TOKENS_PER_BLOCK
# A sibling of P: its first 16 blocks, then 200 tokens of its own (6 more whole blocks).
FETCHED = 512
T = P[:FETCHED] + [_tokens.randrange(1000, 30000) for _ in range(200)]
T_NEW_BLOCKS = len(T) // TOKENS_PER_BLOCK - FETCHED // TOKENS_PER_BLOCK
# How long a server may take to load and warm up.
SERVER_START_S = 1200.0
# How long a node waits for the other's step of the protocol.
PEER_WAIT_S = 3600.0


def model_path() -> str:
    return str(llm_models_root(check=True) / "llama-3.1-model" / "Llama-3.1-8B-Instruct")


def wait_for_file(path: Path, timeout: float = PEER_WAIT_S) -> str:
    """The text of ``path`` once the other node or the job wrote it."""
    deadline = time.monotonic() + timeout
    while not path.exists():
        assert time.monotonic() < deadline, f"{path} did not appear within {timeout:.0f} s"
        time.sleep(1.0)
    return path.read_text()


def write_file(path: Path, text: str = "") -> None:
    """Writes ``path`` whole: the other side never reads a partial file."""
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text)
    partial.rename(path)


@dataclass(frozen=True)
class Remote:
    """A ``trtllm-serve`` this task did not start, at ``host:port``."""

    host: str
    port: int

    def get_client(self) -> openai.OpenAI:
        return openai.OpenAI(
            base_url=f"http://{self.host}:{self.port}/v1", api_key=RemoteOpenAIServer.DUMMY_API_KEY
        )

    def healthy(self) -> bool:
        try:
            return requests.get(f"http://{self.host}:{self.port}/health", timeout=5).ok
        except requests.RequestException:
            return False


@dataclass(frozen=True)
class Node:
    """This task's node and the directory both nodes share."""

    index: int
    shared: Path


@pytest.fixture(scope="module")
def node() -> Node:
    require_environment()
    shared = os.environ.get(SHARED_DIR_ENV)
    assert shared, f"{SHARED_DIR_ENV} must name a directory both nodes see"
    assert os.environ.get("SLURM_NNODES") == "2", "needs one task on each of two nodes"
    index = int(os.environ["SLURM_NODEID"])
    # Each server this task starts is a launch of its own on one node: srun's and MPI's variables
    # (two tasks) must not reach the MPI workers it spawns.
    for name in list(os.environ):
        if name.startswith(("SLURM_", "PMI_", "PMIX_", "OMPI_")):
            del os.environ[name]
    return Node(index, Path(shared))


@pytest.fixture(scope="module")
def store(node: Node) -> Iterator[Store | None]:
    """Node 0: the master, a provider and a client; node 1: a provider, until node 0 is done."""
    from test_common.mooncake_utils import running_segment_provider

    master_file, done = node.shared / "master.json", node.shared / "node0.done"
    if node.index == 0:
        try:
            run_dir = node.shared / "store"
            run_dir.mkdir()
            with running_store(run_dir) as store:
                write_file(master_file, json.dumps({"master": store.master}))
                wait_for_file(node.shared / "provider1.ready")
                yield store
        finally:
            write_file(done)
        return
    master = json.loads(wait_for_file(master_file))["master"]
    with running_segment_provider(
        master,
        str(node.shared / "provider1"),
        segment_size=PROVIDER_BYTES,
        protocol=PROTOCOL,
        device_name=DEVICES,
    ):
        write_file(node.shared / "provider1.ready")
        yield None
        wait_for_file(done)


@contextlib.contextmanager
def tp8_server(shared: Path, name: str, sharing: dict) -> Iterator[Remote]:
    """A Llama-3.1-8B server over both nodes' GPUs, its rank 0 on this node, which the job
    launches from ``<name>.args`` and stops at ``<name>.stop``."""
    from tensorrt_llm._utils import get_free_port

    options = {
        "kv_cache_config": {"tokens_per_block": TOKENS_PER_BLOCK, "free_gpu_memory_fraction": 0.1},
        "cuda_graph_config": None,
        "max_batch_size": 8,
        "max_seq_len": 2048,
        "kv_sharing": sharing,
    }
    options_file = shared / f"{name}.yaml"
    options_file.write_text(json.dumps(options))
    server = Remote("localhost", get_free_port())
    args = [model_path(), "--tp_size", str(TP8), "--port", str(server.port)]
    args += ["--extra_llm_api_options", str(options_file)]
    write_file(shared / f"{name}.args", " ".join(args))
    ended = shared / f"{name}.stopped"
    try:
        deadline = time.monotonic() + SERVER_START_S
        while not server.healthy():
            assert not ended.exists() and time.monotonic() < deadline, (
                f"{name} did not start (see the job's {name} logs)"
            )
            time.sleep(2.0)
        yield server
    finally:
        write_file(shared / f"{name}.stop")
        wait_for_file(ended, SERVER_START_S)


def complete(server: Remote, prompt: list[int]) -> tuple[list[int], int]:
    """Greedy tokens and cached prompt tokens of ``prompt``."""
    reply = server.get_client().completions.create(
        model=model_path(),
        prompt=prompt,
        max_tokens=NEW_TOKENS,
        temperature=0,
        extra_body={"detokenize": False},
    )
    (choice,) = reply.choices
    return list(choice.token_ids), reply.usage.prompt_tokens_details.cached_tokens


def test_a_server_over_both_nodes_reads_what_another_published_exactly(
    node: Node, store: Store | None
) -> None:
    """A TP8 writer publishes P, each rank its share of each block once, then computes P's sibling
    T over P's blocks reused locally (and publishes T's new blocks). A fresh TP8 reader reads P's
    blocks of T from the store (max_fetch_tokens stops it there) and computes the rest, as the
    writer did: the same tokens. Every rank of each server writes the same trace."""
    if node.index != 0:
        return  # node 0 checks; the job runs both servers' ranks on both nodes
    sharing = {
        "backends": [{"type": "mooncake", "config": store.config("tp8")}],
        "min_fetch_tokens": 2 * TOKENS_PER_BLOCK,
        # The completions API takes no cache_salt, so the writer's T is stored whole: the reader
        # stops at P's blocks and computes the rest, as the writer did.
        "max_fetch_tokens": FETCHED,
    }

    with tp8_server(node.shared, "tp8-writer", sharing) as writer:
        stored, before = TP8 * P_BLOCKS, store.master_metrics()
        complete(writer, P)
        store.assert_readable("tp8", stored)
        assert puts_since(store, before) == (stored, 0)
        reference, cached = complete(writer, T)
        assert cached == FETCHED
        store.assert_readable("tp8", stored + TP8 * T_NEW_BLOCKS)
    rank_traces(node.shared / "tp8-writer", TP8)

    with tp8_server(node.shared, "tp8-reader", sharing) as reader:
        before = store.master_metrics()
        tokens, cached = complete(reader, T)
        puts = puts_since(store, before)
    assert tokens == reference
    assert cached == FETCHED
    assert puts == (0, 0)  # the store held every whole block of T already
    trace = rank_traces(node.shared / "tp8-reader", TP8)
    assert len(decisions(trace, "resume")) == 1, trace


class TwoNodeService:
    """Two instances of one service, one per node (the workload's ``Instances``): node 0's runs in
    this task, node 1's in node 1's (``follow_node_0``), started and stopped through files."""

    def __init__(
        self,
        model: str,
        tp: int,
        gpu_groups: tuple[str, str],
        run_dir: Path,
        name: str,
        sharing: dict | None,
    ) -> None:
        self.model = model
        self.trace_dirs = [run_dir / f"{name}-{index}" for index in range(2)]
        self._run_dir, self._name = run_dir, name
        options = instance_options(sharing)
        write_file(run_dir / f"{name}.options.json", json.dumps(options))
        self.servers = [start_instance(model, tp, gpu_groups[0], self.trace_dirs[0], options)]
        try:
            self.servers[0].wait_for_server(timeout=SERVER_START_S)
            host, port = wait_for_file(run_dir / f"{name}.address").split(":")
        except BaseException:
            self.stop()
            raise
        self.servers.append(Remote(host, int(port)))

    def converse(self, requests: list[tuple[int, list[int]]]) -> list[Completion]:
        return send_at_once(self.servers, self.model, requests)

    def stop(self) -> None:
        self.servers[0].terminate()
        write_file(self._run_dir / f"{self._name}.stop")
        wait_for_file(self._run_dir / f"{self._name}.stopped")


def follow_node_0(model: str, tp: int, run_dir: Path) -> None:
    """Node 1: runs its instance of each service node 0 starts, until node 0 stops it."""
    from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.config import local_address

    for name in ("store-off", "store-on"):
        options = json.loads(wait_for_file(run_dir / f"{name}.options.json"))
        host, trace_dir = local_address(), run_dir / f"{name}-1"
        try:
            with start_instance(model, tp, "0,1,2,3", trace_dir, options, host) as server:
                server.wait_for_server(timeout=SERVER_START_S)
                write_file(run_dir / f"{name}.address", f"{host}:{server.port}")
                wait_for_file(run_dir / f"{name}.stop")
        finally:
            write_file(run_dir / f"{name}.stopped")


def test_two_instances_of_one_service_one_per_node_share_system_prompts_and_follow_up_turns(
    node: Node, store: Store | None
) -> None:
    """TP4 instances, with CUDA graphs as served by default, one on each node."""
    run_dir = node.shared / "workload"
    run_dir.mkdir(exist_ok=True)
    if node.index == 1:
        follow_node_0(model_path(), 4, run_dir)
        return
    run_two_turn_workload(
        store,
        model_path(),
        tp=4,
        gpu_groups=("0,1,2,3", "0,1,2,3"),
        run_dir=run_dir,
        service=TwoNodeService,
    )

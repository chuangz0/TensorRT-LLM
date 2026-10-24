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
turns under concurrent load. Disaggregated serving, with the store on the context servers of node
0 and the generation server on node 1: two TP2 context instances run the same workload, one
behind a context-first router and one behind a generation-first router (2P1D); and a TP2 context
server restarted between the turns, as in a rolling upgrade, reads the first turns from the store
while a generation server under attention DP serves on (1P1D, DeepSeek-V3-Lite). The store has a
provider on each node, so puts and gets cross nodes.

Runs as one pytest task per node of a two-node job (``SLURM_NODEID`` picks the role; four GPUs per
task, its steps ``--overlap``) with ``TRTLLM_KV_SHARING_SHARED_DIR`` naming a directory both nodes
see. Node 0 runs the master, a provider and every check; node 1 a provider and the servers node 0
asks it for. An eight-rank server spans both nodes, so the job launches it outside these tasks once
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
from contextlib import ExitStack
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
    assert_agree,
    conversations,
    decisions,
    instance_counters,
    instance_options,
    puts_since,
    rank_traces,
    require_environment,
    run_two_turn_workload,
    running_store,
    send_at_once,
    start_instance,
    whole_blocks,
    workload_sharing,
)
from llmapi.apps.openai_server import RemoteDisaggOpenAIServer, RemoteOpenAIServer
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
# The marker node 0 writes in a test's directory when its part of the test ended, however it ended.
NODE_0_ENDED = "node0.ended"
NODE_GPUS = "0,1,2,3"
# Disaggregated serving: TP2 context servers on node 0, a TP4 generation server on node 1.
CONTEXT_TP, GENERATION_TP = 2, 4
TRANSCEIVER = {"cache_transceiver_config": {"backend": "DEFAULT"}}
# With the autotuner on, each DeepSeek server process picks its own MoE kernels, and two processes'
# outputs differ in the last bits.
NO_AUTOTUNER = {"enable_autotuner": False}


def model_path() -> str:
    return str(llm_models_root(check=True) / "llama-3.1-model" / "Llama-3.1-8B-Instruct")


def deepseek_path() -> str:
    return str(llm_models_root(check=True) / "DeepSeek-V3-Lite" / "bf16")


def wait_for_file(
    path: Path, timeout: float = PEER_WAIT_S, unless: Path | None = None
) -> str | None:
    """The text of ``path`` once the other node or the job wrote it; None if ``unless`` appeared
    first."""
    deadline = time.monotonic() + timeout
    while not path.exists():
        if unless is not None and unless.exists():
            return None
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


@pytest.fixture
def run_dir(node: Node, request: pytest.FixtureRequest) -> Iterator[Path]:
    """The test's directory, which both nodes see. Node 1 serves node 0's requests in it until
    node 0's part of the test ended (``follow_node_0``)."""
    path = node.shared / request.node.name
    path.mkdir(exist_ok=True)
    yield path
    if node.index == 0:
        write_file(path / NODE_0_ENDED)


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


@contextlib.contextmanager
def server_on_node_1(
    run_dir: Path, name: str, tp: int, role: str | None, options: dict
) -> Iterator[Remote]:
    """A server node 1 runs over its four GPUs (``follow_node_0``), its ranks tracing into
    ``run_dir/<name>``, until the context ends."""
    spec = {"tp": tp, "role": role, "options": options}
    write_file(run_dir / f"{name}.server.json", json.dumps(spec))
    try:
        address = wait_for_file(run_dir / f"{name}.address", unless=run_dir / f"{name}.stopped")
        assert address, f"node 1's {name} did not start (see {name}.log)"
        host, port = address.split(":")
        yield Remote(host, int(port))
    finally:
        write_file(run_dir / f"{name}.stop")
        wait_for_file(run_dir / f"{name}.stopped")


def follow_node_0(model: str, run_dir: Path, names: tuple[str, ...]) -> None:
    """Node 1: runs each server node 0 asks for (``server_on_node_1``), in order, until node 0
    stops it or node 0's part of the test ended."""
    from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.config import local_address

    ended = run_dir / NODE_0_ENDED
    for name in names:
        spec = wait_for_file(run_dir / f"{name}.server.json", unless=ended)
        if spec is None:
            return
        server = json.loads(spec)
        trace_dir = run_dir / name
        try:
            with start_instance(
                model,
                server["tp"],
                NODE_GPUS,
                trace_dir,
                server["options"],
                local_address(),
                server["role"],
            ) as instance:
                instance.wait_for_server(timeout=SERVER_START_S)
                write_file(run_dir / f"{name}.address", f"{instance.host}:{instance.port}")
                wait_for_file(run_dir / f"{name}.stop", unless=ended)
        finally:
            write_file(run_dir / f"{name}.stopped")


class TwoNodeService:
    """Two instances of one service, one per node (the workload's ``Instances``): node 0's runs in
    this task, node 1's in node 1's."""

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
        options = instance_options(sharing)
        with ExitStack() as starting:
            local = start_instance(model, tp, gpu_groups[0], self.trace_dirs[0], options)
            starting.enter_context(local)
            remote = starting.enter_context(
                server_on_node_1(run_dir, self.trace_dirs[1].name, tp, None, options)
            )
            local.wait_for_server(timeout=SERVER_START_S)
            self._running = starting.pop_all()
        self.servers = [local, remote]

    def converse(self, requests: list[tuple[int, list[int]]]) -> list[Completion]:
        return send_at_once(self.servers, self.model, requests)

    def stop(self) -> None:
        self._running.close()


def test_two_instances_of_one_service_one_per_node_share_system_prompts_and_follow_up_turns(
    node: Node, store: Store | None, run_dir: Path
) -> None:
    """TP4 instances, with CUDA graphs as served by default, one on each node."""
    if node.index == 1:
        follow_node_0(model_path(), run_dir, ("store-off-1", "store-on-1"))
        return
    run_two_turn_workload(
        store,
        model_path(),
        tp=4,
        gpu_groups=(NODE_GPUS, NODE_GPUS),
        run_dir=run_dir,
        service=TwoNodeService,
    )


# Disaggregated serving.


def context_options(sharing: dict | None) -> dict:
    """A context server of the workload's service: the cache transceiver, and the loop without
    the overlap scheduler."""
    return {**instance_options(sharing), **TRANSCEIVER, "disable_overlap_scheduler": True}


def generation_options() -> dict:
    """A generation server of the workload's service, without the store: production defaults
    (block reuse, the overlap scheduler, CUDA graphs) and the cache transceiver."""
    return {**instance_options(None), **TRANSCEIVER}


# On main 9bb7931d6e a generation-first completion that asks for logprobs fails: the generation
# server receives the first token without its logprobs. So generation-first requests ask for none,
# and store-off deployments send every request context first, for the near-tie rule's logprobs.
CONTEXT_FIRST, GENERATION_FIRST = "context_first", "generation_first"


def router(
    context: RemoteOpenAIServer, generation: Remote, style: str, node_id: int, log: Path
) -> RemoteDisaggOpenAIServer:
    """A router on this node between one context server and the generation server, sending
    ``style`` (CONTEXT_FIRST or GENERATION_FIRST). Routers on one host need their own ``node_id``:
    a request id is the time, the node id and a counter of the router process, so two routers
    could give the generation server equal ids."""
    return RemoteDisaggOpenAIServer(
        ctx_servers=[f"localhost:{context.port}"],
        gen_servers=[f"{generation.host}:{generation.port}"],
        disagg_config={"schedule_style": style, "node_id": node_id},
        log_path=str(log),
    )


class DisaggService:
    """A disaggregated service (the workload's ``Instances``): two context instances on node 0,
    traced at ``<name>-<instance>``, and one TP4 generation server on node 1. With the store,
    instance 0's router sends requests context first and instance 1's generation first; without
    it, both send context first."""

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
        styles = (CONTEXT_FIRST, GENERATION_FIRST if sharing else CONTEXT_FIRST)
        self._logprobs = tuple(style == CONTEXT_FIRST for style in styles)
        options = context_options(sharing)
        with ExitStack() as starting:
            contexts = [
                starting.enter_context(
                    start_instance(model, tp, gpus, trace_dir, options, role="context")
                )
                for gpus, trace_dir in zip(gpu_groups, self.trace_dirs)
            ]
            generation = starting.enter_context(
                server_on_node_1(
                    run_dir, f"{name}-generation", GENERATION_TP, "generation", generation_options()
                )
            )
            for context in contexts:
                context.wait_for_server(timeout=SERVER_START_S)
            self.routers = [
                starting.enter_context(
                    router(context, generation, style, index, run_dir / f"{name}-router{index}.log")
                )
                for index, (context, style) in enumerate(zip(contexts, styles))
            ]
            self._running = starting.pop_all()

    def converse(self, requests: list[tuple[int, list[int]]]) -> list[Completion]:
        return send_at_once(self.routers, self.model, requests, self._logprobs)

    def stop(self) -> None:
        self._running.close()


def test_disaggregated_2p1d_two_context_instances_share_system_prompts_and_follow_up_turns(
    node: Node, store: Store | None, run_dir: Path
) -> None:
    """The two-turn workload on two TP2 context instances and a TP4 generation server. Turn 2
    crosses instances, so both the context-first and the generation-first instance fetch."""
    if node.index == 1:
        follow_node_0(model_path(), run_dir, ("store-off-generation", "store-on-generation"))
        return
    run_two_turn_workload(
        store,
        model_path(),
        tp=CONTEXT_TP,
        gpu_groups=("0,1", "2,3"),
        run_dir=run_dir,
        service=DisaggService,
    )


@contextlib.contextmanager
def context_behind_router(
    model: str,
    run_dir: Path,
    name: str,
    options: dict,
    generation: Remote,
    style: str,
    node_id: int,
) -> Iterator[RemoteDisaggOpenAIServer]:
    """A TP2 context server on GPUs 0 and 1 of this node, traced at ``run_dir/<name>``, behind a
    router sending ``style`` to ``generation``."""
    trace_dir = run_dir / name
    with start_instance(model, CONTEXT_TP, "0,1", trace_dir, options, role="context") as context:
        context.wait_for_server(timeout=SERVER_START_S)
        log = run_dir / f"{name}-router.log"
        with router(context, generation, style, node_id, log) as entry:
            yield entry


def test_disaggregated_1p1d_a_restarted_context_server_reads_the_first_turns_from_the_store(
    node: Node, store: Store | None, run_dir: Path
) -> None:
    """Sixteen conversations at once through one TP2 context server and a generation server under
    attention DP (ADP4), with the store every request generation first. The context server
    restarts between the turns while the generation server serves on: the fresh process reads
    every follow-up's first prompt from the store. Both turns agree with a store-off deployment,
    and each turn puts every new block once. DeepSeek-V3-Lite: MLA, whose latent cache both
    context ranks hold, one object per block. (On main 9bb7931d6e, Llama-3.1-8B under attention
    DP decodes wrong tokens, aggregated too.)"""
    if node.index == 1:
        follow_node_0(deepseek_path(), run_dir, ("store-off-generation", "store-on-generation"))
        return
    model = deepseek_path()
    first, questions = conversations(model)
    attention_dp = {**generation_options(), **NO_AUTOTUNER, "enable_attention_dp": True}
    generation_server = (GENERATION_TP, "generation", attention_dp)

    def converse(
        entry: RemoteDisaggOpenAIServer, prompts: list[list[int]], logprobs: bool
    ) -> list[Completion]:
        return send_at_once([entry], model, [(0, prompt) for prompt in prompts], (logprobs,))

    with server_on_node_1(run_dir, "store-off-generation", *generation_server) as generation:
        options = {**context_options(None), **NO_AUTOTUNER}
        with context_behind_router(
            model, run_dir, "store-off", options, generation, CONTEXT_FIRST, 0
        ) as entry:
            reference_first = converse(entry, first, logprobs=True)
            # From the store-off answers: both deployments send the same second prompts.
            second = [p + a.token_ids + q for p, a, q in zip(first, reference_first, questions)]
            reference_second = converse(entry, second, logprobs=True)

    key_prefix = "disaggregated-1p1d"
    options = {**context_options(workload_sharing(store, key_prefix)), **NO_AUTOTUNER}
    puts = []  # the master's (asked, refused) keys per turn
    # Puts are counted once a context server started: its startup check puts objects too.
    with server_on_node_1(run_dir, "store-on-generation", *generation_server) as generation:
        with context_behind_router(
            model, run_dir, "store-on", options, generation, GENERATION_FIRST, 0
        ) as entry:
            before = store.master_metrics()
            answers_first = converse(entry, first, logprobs=False)
            store.assert_readable(key_prefix, len(whole_blocks(first)))
            puts.append(puts_since(store, before))
        # A rolling upgrade: a fresh context process, the same generation server.
        restarted = "store-on-restarted"
        with context_behind_router(
            model, run_dir, restarted, options, generation, GENERATION_FIRST, 1
        ) as entry:
            before = store.master_metrics()
            answers_second = converse(entry, second, logprobs=False)
            store.assert_readable(key_prefix, len(whole_blocks(first + second)))
            puts.append(puts_since(store, before))
    print(f"keys put/refused per turn: {puts}")
    assert_agree(reference_first + reference_second, answers_first + answers_second, model)
    new_blocks = [whole_blocks(first), whole_blocks(first + second) - whole_blocks(first)]
    assert [asked - refused for asked, refused in puts] == [len(b) for b in new_blocks]
    for prompt, answer in zip(first, answers_second):
        # The restarted process held nothing: it read the first prompt's blocks from the store.
        assert answer.cached_tokens >= len(prompt) // TOKENS_PER_BLOCK * TOKENS_PER_BLOCK
    for name in ("store-on", restarted):
        trace = rank_traces(run_dir / name, CONTEXT_TP)
        assert trace.splitlines()[-1].endswith(" open=0"), f"{name} left store work open"
        counters = instance_counters(run_dir / name)
        assert counters and not any(c.get("publications_failed") for c in counters), counters
    assert decisions(rank_traces(run_dir / restarted, CONTEXT_TP), "resume"), "nothing fetched"

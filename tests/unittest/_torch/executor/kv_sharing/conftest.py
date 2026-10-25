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
"""The real Mooncake store of the end-to-end modules: one master and one provider
(``trtllm-serve mooncake_provider``) per module; servers provide the store no memory unless a
test says so. And the modules' realistic workload: two instances of one service, two-turn
conversations.

``TRTLLM_MOONCAKE_PROTOCOL`` (default rdma) and ``TRTLLM_MOONCAKE_DEVICES`` (default: the fastest
InfiniBand devices) pick the transport; ``TRTLLM_MOONCAKE_E2E_DRIVER`` the servers' store driver:
``native`` (the default, the C++ store worker; a server refuses to start without it) or ``python``.
The modules run only with ``TRTLLM_MOONCAKE_E2E=1`` and skip otherwise.
"""

from __future__ import annotations

import ast
import contextlib
import functools
import itertools
import json
import math
import os
import re
import shutil
import time
import urllib.request
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pytest
import torch
import yaml
from llmapi.apps.openai_server import RemoteOpenAIServer

PROTOCOL = os.environ.get("TRTLLM_MOONCAKE_PROTOCOL", "rdma")
DRIVER = os.environ.get("TRTLLM_MOONCAKE_E2E_DRIVER", "native")
assert DRIVER in ("native", "python"), f"TRTLLM_MOONCAKE_E2E_DRIVER={DRIVER!r}"
DEVICES = os.environ.get("TRTLLM_MOONCAKE_DEVICES", "")
PROVIDER_BYTES = 4 << 30
WAIT_S = 60.0
# How long a server may take to stop: one that provides the store memory stops about 13 s after
# its executor began to, once Mooncake freed the memory.
SERVER_STOP_S = 120.0


@dataclass
class Store:
    """The module's Mooncake store, and a client that gives it no memory, to ask what it holds."""

    master: str
    metrics_url: str
    run_dir: Path
    client: object  # mooncake.store.MooncakeDistributedStore
    master_pid: int

    def config(self, key_prefix: str, driver: str = DRIVER, **mooncake_options) -> str:
        """Writes a server's store config whose keys start with ``key_prefix``; its ``mooncake``
        section takes ``mooncake_options`` too."""
        mooncake = {
            "master_server_address": self.master,
            "global_segment_size": 0,
            "protocol": PROTOCOL,
            "device_name": DEVICES,
            "driver": driver,
            **mooncake_options,
        }
        path = self.run_dir / f"{key_prefix}-{driver}.yaml"
        path.write_text(yaml.safe_dump({"mooncake": mooncake, "blob": {"key_prefix": key_prefix}}))
        return str(path)

    def keys(self, key_prefix: str) -> list[str]:
        """Every key the store has under ``key_prefix``."""
        with urllib.request.urlopen(f"{self.metrics_url}/get_all_keys", timeout=10) as reply:
            return [k for k in reply.read().decode().split() if k.startswith(f"{key_prefix}/")]

    def readable_objects(self, key_prefix: str) -> tuple[int, int]:
        """How many objects the store has under ``key_prefix``, and how many of them a get could
        read now (an object whose put is still open is not readable)."""
        keys = self.keys(key_prefix)
        readable = sum(answer == 1 for answer in self.client.batch_is_exist(keys)) if keys else 0
        return len(keys), readable

    def assert_readable(self, key_prefix: str, objects: int) -> None:
        """Waits until the store has ``objects`` objects under ``key_prefix``, all readable."""
        assert wait_until(lambda: self.readable_objects(key_prefix) == (objects, objects)), (
            f"the store holds {self.readable_objects(key_prefix)}; expected {objects}"
        )

    def segments(self) -> list[dict]:
        """The store's segments, as ``master_segments`` reads them."""
        from test_common.mooncake_utils import master_segments

        return master_segments(self.metrics_url)

    def master_metrics(self) -> dict[str, float]:
        """The master's counters (``/metrics``), by name."""
        with urllib.request.urlopen(f"{self.metrics_url}/metrics", timeout=10) as reply:
            lines = reply.read().decode().splitlines()
        return {
            name: float(value)
            for name, value in (line.rsplit(" ", 1) for line in lines if line and line[0] != "#")
        }


# CI collects this directory. The end-to-end modules start servers on up to four GPUs or two nodes
# and run for tens of minutes, so they run only when opted in.
OPT_IN_ENV = "TRTLLM_MOONCAKE_E2E"
skip_unless_opted_in = pytest.mark.skipif(
    os.environ.get(OPT_IN_ENV) != "1",
    reason=f"starts servers on a real Mooncake store; set {OPT_IN_ENV}=1",
)


def require_environment() -> None:
    """Fails, not skips, where an opted-in test cannot run."""
    assert torch.cuda.is_available(), "needs a GPU"
    assert shutil.which("mooncake_master"), "needs mooncake_master (mooncake-transfer-engine)"
    import mooncake.store  # noqa: F401


def stop_server(server: RemoteOpenAIServer) -> None:
    """Stops ``server`` as an operator would (SIGTERM), waiting up to SERVER_STOP_S for it to
    exit."""
    server.proc.terminate()
    server.proc.wait(timeout=SERVER_STOP_S)
    server.terminate()  # the process is gone; this closes its log


def puts_since(store: Store, before: dict[str, float]) -> tuple[float, float]:
    """Keys the master was asked to put since it reported ``before``, and how many it refused."""
    after = store.master_metrics()
    return after[PUT_KEYS] - before[PUT_KEYS], after[PUT_KEYS_REFUSED] - before[PUT_KEYS_REFUSED]


def wait_until(condition: Callable[[], bool]) -> bool:
    deadline = time.monotonic() + WAIT_S
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.2)
    return True


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Store]:
    require_environment()
    with running_store(tmp_path_factory.mktemp("mooncake")) as store:
        yield store


@contextlib.contextmanager
def running_store(
    run_dir: Path, *, eviction_ratio: float = 0.0, provider_bytes: int = PROVIDER_BYTES
) -> Iterator[Store]:
    """A master and one provider of ``provider_bytes`` (none for 0) on this host, with servers'
    store configs in ``run_dir``. The module's store evicts nothing (``eviction_ratio`` 0)."""
    from test_common.mooncake_utils import running_master_on_free_ports

    from tensorrt_llm._torch.disaggregation.blob.mooncake_setup import MooncakeConfig, open_sdk
    from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.gpudirect import (
        select_registration_path,
    )

    # The first Mooncake client of the process fixes how it registers device memory, and the
    # servers in this process register theirs after the test's own client.
    select_registration_path(PROTOCOL, registers_device_memory=True)
    with ExitStack() as stack:
        master = stack.enter_context(
            running_master_on_free_ports(
                str(run_dir), protocol=PROTOCOL, eviction_ratio=eviction_ratio
            )
        )
        if provider_bytes:
            stack.enter_context(running_provider(master.address, run_dir, provider_bytes))
        metrics_port = json.loads((run_dir / "pool.json").read_text())["metrics_port"]
        host = master.address.rsplit(":", 1)[0]
        client = open_sdk(MooncakeConfig(master.address, 0, PROTOCOL, DEVICES, local_buffer_size=0))
        stack.callback(client.close)
        metrics_url = f"http://{host}:{metrics_port}"
        yield Store(master.address, metrics_url, run_dir, client, master.process.pid)


def running_provider(master: str, run_dir: Path, segment_bytes: int):
    """A provider (``trtllm-serve mooncake_provider``) of the store whose master is ``master``,
    for the body's duration."""
    from test_common.mooncake_utils import running_segment_provider

    return running_segment_provider(
        master, str(run_dir), segment_size=segment_bytes, protocol=PROTOCOL, device_name=DEVICES
    )


# The two-turn workload: system prompts of 16 blocks, each shared by several users whose first
# message is 4 blocks; a follow-up question after the first answer.
TOKENS_PER_BLOCK = 32
SYSTEM_PROMPTS = 4
USERS_PER_PROMPT = 4
SYSTEM_TOKENS = 16 * TOKENS_PER_BLOCK
FIRST_MESSAGE_TOKENS = 4 * TOKENS_PER_BLOCK
FOLLOW_UP_TOKENS = 48
ANSWER_TOKENS = 32
# Greedy tokens change with the batch a request shares, store or not. A run may first take another
# token than the store-off run only where the store-off run gave that token within NEAR_TIE nats of
# its best (e^-0.5: at least 60% as likely); a fetch of wrong bytes diverges where the model is
# sure. Each run reports its TOP_LOGPROBS most likely tokens per position: three can tie.
NEAR_TIE = 0.5
TOP_LOGPROBS = 5
# The master's count of keys a put asked for, and of those it refused (a key that exists already).
PUT_KEYS = "master_batch_put_start_items_total"
PUT_KEYS_REFUSED = "master_batch_put_start_failed_items_total"


def deployment_text() -> str:
    """The repository's feature documentation, as the source of every prompt's words."""
    docs = Path(__file__).resolve().parents[5] / "docs" / "source" / "features"
    return "\n\n".join(path.read_text() for path in sorted(docs.glob("*.md")))


def conversations(model: str) -> tuple[list[list[int]], list[list[int]]]:
    """Every conversation's first prompt (a system prompt shared by USERS_PER_PROMPT users, then
    the user's message) and its follow-up question: consecutive pieces of the docs."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    words = iter(tokenizer.encode(deployment_text(), add_special_tokens=False))

    def take(count: int) -> list[int]:
        return list(itertools.islice(words, count))

    start = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    systems = [start + take(SYSTEM_TOKENS - len(start)) for _ in range(SYSTEM_PROMPTS)]
    users = range(SYSTEM_PROMPTS * USERS_PER_PROMPT)
    first = [systems[user % SYSTEM_PROMPTS] + take(FIRST_MESSAGE_TOKENS) for user in users]
    return first, [take(FOLLOW_UP_TOKENS) for _ in users]


@dataclass
class Completion:
    token_ids: list[int]
    # The most likely tokens per position, by text; None where they were not asked for.
    top_logprobs: list[dict[str, float]] | None
    cached_tokens: int


def ask(
    server: RemoteOpenAIServer, model: str, prompt: list[int], logprobs: bool = True
) -> Completion:
    """``prompt``'s greedy completion; with ``logprobs``, the TOP_LOGPROBS most likely tokens at
    each position."""
    reply = server.get_client().completions.create(
        model=model,
        prompt=prompt,
        max_tokens=ANSWER_TOKENS,
        temperature=0,
        logprobs=TOP_LOGPROBS if logprobs else None,
        extra_body={"detokenize": False},
    )
    (choice,) = reply.choices
    cached = reply.usage.prompt_tokens_details.cached_tokens
    top_logprobs = choice.logprobs.top_logprobs if logprobs else None
    return Completion(list(choice.token_ids), top_logprobs, cached)


def workload_sharing(store: Store, key_prefix: str) -> dict:
    """The ``kv_sharing`` of the workload's service, its keys under ``key_prefix``."""
    return {
        "backends": [{"type": "mooncake", "config": store.config(key_prefix)}],
        "min_fetch_tokens": 2 * TOKENS_PER_BLOCK,
        "max_fetch_tokens": 2048,
    }


def instance_options(sharing: dict | None) -> dict:
    """The LLM options of an instance of the workload's service; with ``sharing``, its
    ``kv_sharing``."""
    return {
        "kv_cache_config": {
            "tokens_per_block": TOKENS_PER_BLOCK,
            "free_gpu_memory_fraction": 0.3,
        },
        "max_batch_size": 16,
        "max_seq_len": 2048,
        **({"kv_sharing": sharing} if sharing else {}),
    }


def start_instance(
    model: str,
    tp: int,
    gpus: str,
    trace_dir: Path,
    options: dict,
    host: str = "localhost",
    role: str | None = None,
) -> RemoteOpenAIServer:
    """One instance in its own ``trtllm-serve`` process over ``gpus``, its ranks tracing their
    rounds into ``trace_dir``; not waited for. ``role`` makes it a disaggregated ``context`` or
    ``generation`` server."""
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=gpus,
        TRTLLM_KV_SHARING_TRACE_DIR=str(trace_dir),
        # 4 KiB host pages: THP compaction can outlast the start-up wait (nvbugs/6888251).
        TLLM_KV_CACHE_MANAGER_V2_THP="0",
    )
    env.pop("TLLM_WORKER_USE_SINGLE_PROCESS", None)
    cli_args = ["--tp_size", str(tp)] + (["--server_role", role] if role else [])
    return RemoteOpenAIServer(
        model,
        cli_args=cli_args,
        host=host,
        env=env,
        extra_config=options,
        log_path=str(trace_dir) + ".log",
        wait=False,
    )


class Service:
    """Two instances of one service, each over ``tp`` GPUs of its own group, in their own
    ``trtllm-serve`` processes; with ``sharing``, both fetch and publish through the store, and
    every rank traces its rounds into ``run_dir/<name>-<instance>``."""

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
        self.trace_dirs = [run_dir / f"{name}-{index}" for index in range(len(gpu_groups))]
        options = instance_options(sharing)
        self.servers: list[RemoteOpenAIServer] = []
        with ExitStack() as starting:
            for gpus, trace_dir in zip(gpu_groups, self.trace_dirs):
                server = start_instance(model, tp, gpus, trace_dir, options)
                self.servers.append(starting.enter_context(server))
            for server in self.servers:
                server.wait_for_server(timeout=RemoteOpenAIServer.MAX_SERVER_START_WAIT_S)
            starting.pop_all()  # both up: stop() ends them from here on

    def converse(self, requests: list[tuple[int, list[int]]]) -> list[Completion]:
        return send_at_once(self.servers, self.model, requests)

    def stop(self) -> None:
        for server in self.servers:
            server.terminate()


class Instances(Protocol):
    """What the workload needs of a service's two instances (``Service``, or a disaggregated
    service's two context instances)."""

    trace_dirs: list[Path]  # each instance's; its log is ``<trace dir>.log``

    def converse(self, requests: list[tuple[int, list[int]]]) -> list[Completion]: ...

    def stop(self) -> None: ...


def send_at_once(
    servers: list,
    model: str,
    requests: list[tuple[int, list[int]]],
    logprobs: tuple[bool, ...] | None = None,
) -> list[Completion]:
    """Sends every ``(instance, prompt)`` to ``servers[instance]`` at once and waits for all;
    ``logprobs[instance]`` False asks that instance for tokens alone."""
    logprobs = logprobs or (True,) * len(servers)
    with ThreadPoolExecutor(len(requests)) as pool:
        return list(pool.map(lambda r: ask(servers[r[0]], model, r[1], logprobs[r[0]]), requests))


def instance_counters(trace_dir: Path) -> list[dict[str, int]]:
    """The kv_sharing counters the ranks of a shut-down instance logged (``<trace dir>.log``)."""
    log = Path(f"{trace_dir}.log").read_text(errors="replace")
    return [ast.literal_eval(found) for found in re.findall(r"kv_sharing counters: (\{.*\})", log)]


def whole_blocks(prompts: list[list[int]]) -> set[tuple[int, ...]]:
    """The distinct whole prompt blocks, each named by the prefix that ends with it."""
    return {
        tuple(prompt[: (block + 1) * TOKENS_PER_BLOCK])
        for prompt in prompts
        for block in range(len(prompt) // TOKENS_PER_BLOCK)
    }


def agrees_with(
    reference: Completion, completion: Completion, token_text: Callable[[int], str]
) -> bool:
    """The store-off run's tokens, or a first different token at a near tie (NEAR_TIE): a token
    this run found most likely there (without logprobs: the token it took) is within NEAR_TIE of
    the store-off run's best. A run that ended earlier (end of sequence) differs at its length,
    where the longer run's two most likely tokens must be a near tie."""
    expected, got = reference.token_ids, completion.token_ids
    shorter = min(len(expected), len(got))
    position = next((i for i in range(shorter) if expected[i] != got[i]), shorter)
    if position == len(expected) == len(got):
        return True
    if position == shorter:
        longer = reference if len(expected) > len(got) else completion
        if longer.top_logprobs is None:
            return False  # nothing tells a near tie
        best, runner_up = sorted(longer.top_logprobs[position].values(), reverse=True)[:2]
        return best - runner_up <= NEAR_TIE
    # Keyed by decoded text: two tokens that decode alike share one entry, which is a tie.
    store_off = reference.top_logprobs[position]
    if completion.top_logprobs is None:
        taken = [token_text(got[position])]
    else:
        this_run = completion.top_logprobs[position]
        this_runs_best = max(this_run.values())
        taken = [text for text, logprob in this_run.items() if logprob == this_runs_best]
    near_best = max(store_off.values()) - NEAR_TIE
    return any(store_off.get(text, -math.inf) >= near_best for text in taken)


def first_difference(
    reference: Completion, completion: Completion, token_text: Callable[[int], str]
) -> str:
    """Where ``completion`` first leaves the store-off run's tokens, with both runs' TOP_LOGPROBS
    most likely tokens there (or the token this run took) and the prompt tokens each found
    cached."""
    expected, got = reference.token_ids, completion.token_ids
    position = next((i for i, (e, g) in enumerate(zip(expected, got)) if e != g), None)
    if position is None:
        return f"lengths {len(expected)} and {len(got)}"
    if completion.top_logprobs is None:
        this_run = f"took {token_text(got[position])!r}"
    else:
        this_run = f"{completion.top_logprobs[position]}"
    return (
        f"position {position}: store-off {reference.top_logprobs[position]}, this run "
        f"{this_run}; cached {reference.cached_tokens} and {completion.cached_tokens} prompt "
        "tokens"
    )


def assert_agree(references: list[Completion], answers: list[Completion], model: str) -> None:
    """Every answer agrees with the store-off run's (``agrees_with``), which reports logprobs."""
    from transformers import AutoTokenizer

    assert all(r.top_logprobs is not None for r in references), "store-off logprobs missing"
    token_text = functools.cache(AutoTokenizer.from_pretrained(model).decode)
    exact = sum(r.token_ids == a.token_ids for r, a in zip(references, answers))
    print(f"{exact} of {len(answers)} requests equal the store-off run's tokens, others near ties")
    disagreeing = {
        i: first_difference(r, a, token_text)
        for i, (r, a) in enumerate(zip(references, answers))
        if not agrees_with(r, a, token_text)
    }
    assert not disagreeing, f"outputs disagree at confident positions: {disagreeing}"


def rank_traces(trace_dir: Path, tp: int) -> str:
    """The trace every rank of a shut-down server wrote; fails unless all are the same."""
    traces = [(trace_dir / f"rank{rank}.txt").read_text() for rank in range(tp)]
    assert traces[0].splitlines()[-1].startswith("exit "), traces[0][-500:]
    for rank, trace in enumerate(traces[1:], 1):
        assert trace == traces[0], f"rank {rank}'s rounds differ from rank 0's"
    return traces[0]


def decisions(trace: str, action: str) -> list[str]:
    """The ``action:seq`` decisions of the trace's rounds."""
    made = []
    for line in trace.splitlines():
        if line.startswith("round "):
            fields = dict(field.split("=", 1) for field in line.split()[1:])
            made += [d for d in fields["decisions"].split(",") if d.startswith(f"{action}:")]
    return made


def run_two_turn_workload(
    store: Store,
    model: str,
    *,
    tp: int,
    gpu_groups: tuple[str, str],
    run_dir: Path,
    service: Callable[..., Instances] = Service,
) -> None:
    """Two instances of one service on one store with concurrent two-turn conversations. Turn 1
    comes in two waves: system prompts 0 and 1 reach both instances at once, so both compute and
    put their blocks (the deployment's one concurrent-writer case), while system prompts 2 and 3
    reach instance 0 only; then their other users reach instance 1, which reads instance 0's
    blocks from the store. A second turn goes to the other instance, which holds the system prompt
    itself and fetches the user's first message from the store.

    Asserts every request completes and agrees with the same workload on a store-off service;
    every second-wave request reads its system prompt from the store and every second turn its
    whole first prompt from the local cache or the store; the store holds tp objects per distinct
    whole prompt block, all readable, and the master refused every other put as a key that exists
    already (some in the first wave); no publication failed (a put of an object the other instance
    was still putting is no failure); both instances shut down with nothing open, identical rank
    traces and fetches done. ``service`` starts the two instances, given ``Service``'s arguments.
    """
    first, questions = conversations(model)
    first_requests = [((user // SYSTEM_PROMPTS) % 2, p) for user, p in enumerate(first)]
    late_users = [
        user
        for user, (instance, _) in enumerate(first_requests)
        if instance == 1 and user % SYSTEM_PROMPTS >= 2
    ]
    waves = ([user for user in range(len(first)) if user not in late_users], late_users)

    def in_waves(send: Callable[[list[tuple[int, list[int]]]], list]) -> list[Completion]:
        """Turn 1's answers in user order, each wave sent once the one before it ended."""
        answers = [None] * len(first)
        for wave in waves:
            for user, answer in zip(wave, send([first_requests[user] for user in wave])):
                answers[user] = answer
        return answers

    store_off = service(model, tp, gpu_groups, run_dir, "store-off", sharing=None)
    try:
        reference_first = in_waves(store_off.converse)
        # From the store-off answers: both runs send the same second prompts.
        second = [p + a.token_ids + q for p, a, q in zip(first, reference_first, questions)]
        second_requests = [(1 - instance, p) for (instance, _), p in zip(first_requests, second)]
        reference_second = store_off.converse(second_requests)
    finally:
        store_off.stop()

    key_prefix = f"service-tp{tp}"
    sharing = workload_sharing(store, key_prefix)
    puts = []  # the master's (asked, refused) keys per wave of turn 1, then turn 2
    sent: list[list[int]] = []  # every prompt the store-on service was sent

    def turn(requests: list[tuple[int, list[int]]]) -> list[Completion]:
        """Sends ``requests``, waits until the store holds every whole block sent so far, and
        records the keys the master was asked to put and refused meanwhile."""
        sent.extend(prompt for _, prompt in requests)
        before = store.master_metrics()
        answers = store_on.converse(requests)
        store.assert_readable(key_prefix, tp * len(whole_blocks(sent)))
        puts.append(puts_since(store, before))
        return answers

    store_on = service(model, tp, gpu_groups, run_dir, "store-on", sharing)
    try:
        answers = in_waves(turn)
        answers += turn(second_requests)
    finally:
        store_on.stop()
    print(f"keys put/refused per wave and turn 2: {puts}")
    assert_agree(reference_first + reference_second, answers, model)
    # Every distinct block was put once; each other put of it was refused as existing already.
    first_wave_blocks = whole_blocks([first[user] for user in waves[0]])
    new_blocks = [  # per wave of turn 1, then turn 2
        first_wave_blocks,
        whole_blocks(first) - first_wave_blocks,
        whole_blocks(first + second) - whole_blocks(first),
    ]
    assert [asked - refused for asked, refused in puts] == [tp * len(b) for b in new_blocks], puts
    assert puts[0][1] > 0, "wave 1's concurrent writers had no put refused"
    for user in late_users:  # instance 0 computed their system prompt
        assert answers[user].cached_tokens >= SYSTEM_TOKENS, (user, answers[user].cached_tokens)
    for prompt, answer in zip(first, answers[len(first) :]):
        # The user's message blocks exist only where the first turn ran: covering them reads the
        # other instance's publications.
        assert answer.cached_tokens >= len(prompt) // TOKENS_PER_BLOCK * TOKENS_PER_BLOCK
    for trace_dir in store_on.trace_dirs:
        trace = rank_traces(trace_dir, tp)
        assert trace.splitlines()[-1].endswith(" open=0"), f"{trace_dir.name} left store work open"
        assert decisions(trace, "resume"), f"{trace_dir.name} fetched nothing"
        counters = instance_counters(trace_dir)
        assert counters and not any(c.get("publications_failed") for c in counters), counters

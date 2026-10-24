# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What KV sharing through a Mooncake store does to serving latency, on real servers.

Each mode starts a Mooncake master and one provider per socket on this host, then
``trtllm-serve`` servers with and without ``kv_sharing``, and prints one JSON line per measurement
and a summary:

- ``ttft``: the time to the first token of a prompt whose prefix another server published (hit)
  against computing it (miss), at each ``--lengths``: a writer publishes prefixes, a fresh reader
  asks each prefix plus 64 tokens of its own and fresh prompts of the same length, and a server
  without the store asks the same misses. Each hit and each store-off miss is asked once more
  ("repeat"): its prefix is then in the server's own cache, so it shows what serving costs without
  the store. With ``--busy``, also while background streams decode, whose gaps between tokens are
  reported for a quiet window and while hits and misses are asked (``--data-path host_staging``
  shows what landing staged fetches costs them). ``--burst`` sends that many longest prompts at
  once, half of them hits.
- ``idle``: decode iteration time while the store has nothing to fetch (every prompt is new, so
  servers only publish), in process, servers with and without the store taking turns.
- ``workload``: a disaggregated service (two context servers, one generation server) under
  conversations that share system prompts and come back with follow-up turns, run without and
  with the store: TTFT per kind of request, TPOT, and the share of prompt tokens served from
  caches against the ideal.

Prompts are token ids (no tokenizer). Run it from ``tests/microbenchmarks`` on a GPU node with
``mooncake_master`` and ``numactl`` on the path, e.g.::

    python3 kv_sharing_serving.py ttft --model <dir> --tp 1 --gpus 0 --run-dir <dir>
"""

from __future__ import annotations

import argparse
import ast
import gc
import json
import os
import random
import re
import statistics
import sys
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import yaml

TESTS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(TESTS), str(TESTS / "unittest")]

from llmapi.apps.openai_server import RemoteDisaggOpenAIServer, RemoteOpenAIServer  # noqa: E402
from test_common.mooncake_utils import (  # noqa: E402
    running_master_on_free_ports,
    running_segment_provider,
)

TOKENS_PER_BLOCK = 32
VOCABULARY = (1000, 100_000)  # token ids prompts are drawn from; every model here has more
# ttft: prompts per cell, a hit's own tokens past the published prefix, the busy phase's decoding
# streams and its window with nothing but them.
REPEATS = 5
OWN_TOKENS = 64
STREAMS, STREAM_TOKENS = 16, 1024
QUIET_S = 15.0
# idle: one batch of new prompts decoding at once.
IDLE_REQUESTS, IDLE_PROMPT, IDLE_OUTPUT = 64, 1024, 256
# workload (spec 3.4): system prompts shared by every SYSTEM_PROMPTS-th conversation, each first
# message a document of its own, then TURNS - 1 follow-up questions after ANSWER-token answers;
# AT_ONCE conversations at a time, closed loop.
SYSTEM_PROMPTS, SYSTEM_TOKENS = 4, 2048
CONVERSATIONS, DOCUMENT_TOKENS = 64, 4096
TURNS, ANSWER_TOKENS, QUESTION_TOKENS = 4, 256, 256
AT_ONCE = 32


def emit(record: dict) -> None:
    print(json.dumps(record, default=str), flush=True)


def tokens(count: int, rng: random.Random) -> list[int]:
    return [rng.randrange(*VOCABULARY) for _ in range(count)]


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def infiniband_by_socket() -> dict[int, list[str]]:
    """The active InfiniBand devices of each NUMA node."""
    found: dict[int, list[str]] = {}
    for device in sorted(Path("/sys/class/infiniband").iterdir()):
        port = device / "ports" / "1"
        if (port / "link_layer").read_text().strip() != "InfiniBand":
            continue
        if "ACTIVE" in (port / "state").read_text():
            node = int((device / "device" / "numa_node").read_text())
            found.setdefault(node, []).append(device.name)
    return found


@contextmanager
def running_store(run_dir: Path, segment_gib: int) -> Iterator[str]:
    """A master and one provider per socket; yields the master's address.

    The providers lend ``segment_gib`` GiB together, each from its socket's memory, on its CPUs
    and InfiniBand devices.
    """
    from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.gpudirect import (
        select_registration_path,
    )

    select_registration_path("rdma", registers_device_memory=True)
    sockets = infiniband_by_socket()
    with ExitStack() as stack:
        master = stack.enter_context(running_master_on_free_ports(str(run_dir), protocol="rdma"))
        for node, devices in sockets.items():
            stack.enter_context(
                running_segment_provider(
                    master.address,
                    str(run_dir / f"provider-{node}"),
                    segment_size=(segment_gib << 30) // len(sockets),
                    protocol="rdma",
                    device_name=",".join(devices),
                    launcher=("numactl", f"--cpunodebind={node}", f"--membind={node}"),
                )
            )
        yield master.address


def sharing(run_dir: Path, master: str, key_prefix: str, **knobs) -> dict:
    """``kv_sharing`` over the store at ``master``, its keys under ``key_prefix``."""
    config = run_dir / f"store-{key_prefix}.yaml"
    mooncake = {"master_server_address": master, "global_segment_size": 0, "protocol": "rdma"}
    config.write_text(yaml.safe_dump({"mooncake": mooncake, "blob": {"key_prefix": key_prefix}}))
    return {"backends": [{"type": "mooncake", "config": str(config)}], **knobs}


@dataclass
class Server:
    """A ``trtllm-serve`` process; its log and kv_sharing traces under ``run_dir/name``."""

    name: str
    run_dir: Path
    model: str
    tp: int
    gpus: str
    options: dict
    role: str | None = None
    process: RemoteOpenAIServer | None = field(default=None, init=False)

    def start(self) -> Server:
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=self.gpus,
            TRTLLM_KV_SHARING_TRACE_DIR=str(self.run_dir / self.name),
            TLLM_KV_CACHE_MANAGER_V2_THP="0",  # 4 KiB host pages: THP compaction slows start-up
        )
        cli_args = ["--tp_size", str(self.tp)] + (["--server_role", self.role] if self.role else [])
        self.process = RemoteOpenAIServer(
            self.model,
            cli_args=cli_args,
            env=env,
            extra_config=self.options,
            log_path=str(self.run_dir / f"{self.name}.log"),
            wait=False,
        )
        return self

    def wait(self) -> Server:
        self.process.wait_for_server(timeout=RemoteOpenAIServer.MAX_SERVER_START_WAIT_S)
        return self

    def stop(self) -> dict:
        """Stops the server; the kv_sharing counters its ranks logged, and each rank's trace exit.

        A trace exit reads ``exit ... open=0`` when every store operation closed.
        """
        self.process.terminate()
        log = (self.run_dir / f"{self.name}.log").read_text(errors="replace")
        counters = [ast.literal_eval(c) for c in re.findall(r"kv_sharing counters: (\{.*\})", log)]
        traces = sorted((self.run_dir / self.name).glob("rank*.txt"))
        return {"counters": counters, "exits": [t.read_text().splitlines()[-1] for t in traces]}


def first_token_seconds(client, model: str, prompt: list[int]) -> tuple[float, int]:
    """Seconds to the first token of ``prompt``, and the prompt tokens served from a cache."""
    started = time.perf_counter()
    reply = client.completions.create(
        model=model, prompt=prompt, max_tokens=1, temperature=0, extra_body={"detokenize": False}
    )
    details = reply.usage.prompt_tokens_details
    return time.perf_counter() - started, details.cached_tokens if details else 0


# ttft


def ttft(args: argparse.Namespace) -> None:
    run_dir, rng = Path(args.run_dir), random.Random(args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    model = Path(args.model).name
    phases = ("idle", "busy") if args.busy else ("idle",)
    # Prompts of their own per phase: the reader's own cache holds the idle phase's hits.
    prefixes = {
        (phase, length): [tokens(length, rng) for _ in range(REPEATS)]
        for phase in phases
        for length in args.lengths
    }
    hits = {key: [p + tokens(OWN_TOKENS, rng) for p in group] for key, group in prefixes.items()}
    misses = {key: [tokens(key[1] + OWN_TOKENS, rng) for _ in range(REPEATS)] for key in prefixes}
    burst = [tokens(max(args.lengths), rng) for _ in range(args.burst // 2)]
    options = {
        "kv_cache_config": {"tokens_per_block": TOKENS_PER_BLOCK, "free_gpu_memory_fraction": 0.85},
        "max_seq_len": max(args.lengths) + 512,
        "max_num_tokens": 8192,
        "max_batch_size": 64,
        "enable_chunked_prefill": True,
    }
    rows: list[dict] = []

    def ask(server: Server, store: str, phase: str, kind: str, prompts: dict) -> None:
        """TTFT of each prompt of ``phase``, one at a time.

        A hit, and a miss without the store, is asked again ("repeat").
        """
        client = server.process.get_client()
        again = kind == "hit" or store == "off"
        for length in args.lengths:
            for prompt in prompts[phase, length]:
                for asked in (kind, "repeat") if again else (kind,):
                    seconds, cached = first_token_seconds(client, args.model, prompt)
                    row = {"model": model, "tp": args.tp, "store": store, "phase": phase}
                    row.update(kind=asked, length=length, ttft_ms=round(1e3 * seconds, 1))
                    rows.append({**row, "cached": cached})
                    emit(rows[-1])

    stopped = {}
    with running_store(run_dir / "store", args.segment_gib) as master:
        knobs = {"max_fetch_tokens": max(args.lengths), "data_path": args.data_path}
        on = {"kv_sharing": sharing(run_dir, master, "ttft", **knobs)}
        writer = Server("writer", run_dir, args.model, args.tp, args.gpus, {**options, **on})
        client = writer.start().wait().process.get_client()
        for prompt in [p for group in prefixes.values() for p in group] + burst:
            first_token_seconds(client, args.model, prompt)
        time.sleep(5)  # the last publications end
        stopped["writer"] = writer.stop()
        for store, server_options in (("on", {**options, **on}), ("off", options)):
            server = Server(
                f"reader-{store}", run_dir, args.model, args.tp, args.gpus, server_options
            )
            first_token_seconds(
                server.start().wait().process.get_client(), args.model, tokens(1024, rng)
            )
            for phase in phases:
                with background_streams(server, args) if phase == "busy" else nullcontext() as gaps:
                    windows = {}
                    if gaps is not None:
                        started = time.monotonic()
                        time.sleep(QUIET_S)
                        windows["quiet"] = (started, time.monotonic())
                    asks = (
                        (("hit", hits), ("miss", misses)) if store == "on" else (("miss", misses),)
                    )
                    for kind, prompts in asks:
                        started = time.monotonic()
                        ask(server, store, phase, kind, prompts)
                        windows[kind] = (started, time.monotonic())
                if gaps is not None:
                    report_decode_gaps(gaps, windows, store, args)
            if args.burst and store == "on":
                hit_prompts = [p + tokens(OWN_TOKENS, rng) for p in burst]
                at_once(server, args, hit_prompts + [tokens(max(args.lengths), rng) for _ in burst])
            stopped[f"reader-{store}"] = server.stop()
    summarize_ttft(rows, args, phases)
    emit({"stopped": stopped})


@contextmanager
def background_streams(server: Server, args: argparse.Namespace) -> Iterator[list]:
    """STREAMS streams decoding STREAM_TOKENS tokens after as many prompt tokens, again and again.

    Yields ``(when, seconds)`` of every gap between two tokens of a stream.
    """
    stop, client = threading.Event(), server.process.get_client()
    rng = random.Random(args.seed + 1)
    gaps: list[tuple[float, float]] = []

    def stream() -> None:
        while not stop.is_set():
            last = None
            for _ in client.completions.create(
                model=args.model,
                prompt=tokens(STREAM_TOKENS, rng),
                max_tokens=STREAM_TOKENS,
                temperature=0,
                stream=True,
                extra_body={"detokenize": False, "ignore_eos": True},
            ):
                now = time.monotonic()
                if last is not None:
                    gaps.append((now, now - last))
                last = now

    with ThreadPoolExecutor(STREAMS) as pool:
        for _ in range(STREAMS):
            pool.submit(stream)
        time.sleep(10)  # every stream decodes
        try:
            yield gaps
        finally:
            stop.set()


def report_decode_gaps(gaps: list, windows: dict, store: str, args: argparse.Namespace) -> None:
    """The background streams' gaps between tokens in each window, p50 and p99."""
    summary = {"summary": "decode_gaps", "store": store, "data_path": args.data_path}
    for name, (start, end) in windows.items():
        inside = [1e3 * seconds for when, seconds in gaps if start <= when <= end]
        summary[name] = {
            "gaps": len(inside),
            "p50_ms": round(percentile(inside, 0.5), 2),
            "p99_ms": round(percentile(inside, 0.99), 2),
        }
    emit(summary)


def at_once(server: Server, args: argparse.Namespace, prompts: list[list[int]]) -> None:
    """Sends ``prompts`` at once: their TTFT, and how many were served from the store."""
    client = server.process.get_client()
    with ThreadPoolExecutor(len(prompts)) as pool:
        results = list(pool.map(lambda p: first_token_seconds(client, args.model, p), prompts))
    seconds = [s for s, _ in results]
    emit(
        {
            "summary": "burst",
            "tp": args.tp,
            "requests": len(prompts),
            "hits": sum(cached > 0 for _, cached in results),
            "ttft_p50_ms": round(1e3 * statistics.median(seconds), 1),
            "ttft_max_ms": round(1e3 * max(seconds), 1),
        }
    )


def summarize_ttft(rows: list[dict], args: argparse.Namespace, phases: tuple[str, ...]) -> None:
    def median(store: str, phase: str, kind: str, length: int) -> float:
        cell = (store, phase, kind, length)
        return statistics.median(
            r["ttft_ms"] for r in rows if (r["store"], r["phase"], r["kind"], r["length"]) == cell
        )

    for phase in phases:
        for length in args.lengths:
            hit, repeat = median("on", phase, "hit", length), median("on", phase, "repeat", length)
            miss_on, miss_off = (
                median("on", phase, "miss", length),
                median("off", phase, "miss", length),
            )
            emit(
                {
                    "summary": "ttft",
                    "model": Path(args.model).name,
                    "tp": args.tp,
                    "phase": phase,
                    "length": length,
                    "hit_ms": hit,
                    # The prefix in the server's own cache: serving's floor, with and without the
                    # store (store off: its misses asked again).
                    "repeat_on_ms": repeat,
                    "repeat_off_ms": median("off", phase, "repeat", length),
                    "store_ms": round(hit - repeat, 1),  # what the store adds over local reuse
                    "miss_on_ms": miss_on,
                    "miss_off_ms": miss_off,
                    "hit/miss(off)": round(hit / miss_off, 3),
                    "miss(on)/miss(off)": round(miss_on / miss_off, 3),
                }
            )


# idle


def decode_iterations_ms(llm, tp: int) -> list[float]:
    """The latencies of the decode-only iterations since the last call.

    In process (one rank) from the executor itself, else from the LLM API's iteration stats.
    """
    if tp == 1:
        return [
            stats.iter_latency_ms
            for stats, *_ in llm._executor.engine.get_latest_iteration_stats()
            if stats.inflight_batching_stats.num_context_requests == 0
            and stats.inflight_batching_stats.num_gen_requests > 0
        ]
    return [
        stats["iterLatencyMS"]
        for stats in llm.get_stats(timeout=2)
        if stats["inflightBatchingStats"]["numContextRequests"] == 0
        and stats["inflightBatchingStats"]["numGenRequests"] > 0
    ]


def idle(args: argparse.Namespace) -> None:
    """Decode-only iteration latency in process, servers with and without the store in turn."""
    if args.tp == 1:  # in process; tensor parallelism needs worker processes
        os.environ["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import KvCacheConfig

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    decode = SamplingParams(max_tokens=IDLE_OUTPUT, temperature=0.0, ignore_eos=True)
    medians: dict[str, list[float]] = {"off": [], "on": []}
    with running_store(run_dir / "store", args.segment_gib) as master:
        on = sharing(run_dir, master, "idle")
        for run, store in enumerate(("off", "on") * args.rounds):
            llm = LLM(
                model=args.model,
                tensor_parallel_size=args.tp,
                kv_cache_config=KvCacheConfig(
                    tokens_per_block=TOKENS_PER_BLOCK, free_gpu_memory_fraction=0.5
                ),
                max_batch_size=IDLE_REQUESTS,
                max_seq_len=IDLE_PROMPT + IDLE_OUTPUT + 64,
                enable_iter_perf_stats=True,
                **({"kv_sharing": on} if store == "on" else {}),
            )
            rng = random.Random(args.seed + run)
            llm.generate([tokens(IDLE_PROMPT, rng) for _ in range(8)], decode)  # warm-up
            decode_iterations_ms(llm, args.tp)  # drops the warm-up's iterations
            llm.generate([tokens(IDLE_PROMPT, rng) for _ in range(IDLE_REQUESTS)], decode)
            latencies = decode_iterations_ms(llm, args.tp)
            medians[store].append(statistics.median(latencies))
            median_ms = round(medians[store][-1], 3)
            emit(
                {
                    "run": run,
                    "store": store,
                    "decode_iterations": len(latencies),
                    "median_ms": median_ms,
                }
            )
            llm.shutdown()
            del llm
            gc.collect()
    off, on_ms = statistics.mean(medians["off"]), statistics.mean(medians["on"])
    emit(
        {
            "summary": "idle",
            "model": Path(args.model).name,
            "tp": args.tp,
            "off_ms": round(off, 3),
            "on_ms": round(on_ms, 3),
            "on/off": round(on_ms / off, 4),
        }
    )


# workload


def workload(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    turns = {
        store: serve_conversations(args, run_dir / store, store == "on") for store in ("off", "on")
    }
    for kind in ("new system prompt", "first turn", "follow-up"):
        off, on = (summary_of(turns[store], kind) for store in ("off", "on"))
        emit(
            {
                "summary": "workload",
                "kind": kind,
                "off": off,
                "on": on,
                "ttft_p50 on/off": round(on["ttft_p50_ms"] / off["ttft_p50_ms"], 3),
                "ttft_p90 on/off": round(on["ttft_p90_ms"] / off["ttft_p90_ms"], 3),
            }
        )
    tpot = {store: [t["tpot_ms"] for t in turns[store]] for store in turns}
    summary = {"summary": "tpot"}
    for store, values in tpot.items():
        summary |= {f"{store}_p50": round(percentile(values, 0.5), 3)}
        summary |= {f"{store}_p99": round(percentile(values, 0.99), 3)}
    emit(summary | {"p99 on/off": round(summary["on_p99"] / summary["off_p99"], 3)})


def conversations(seed: int) -> list[list[list[int]]]:
    """Per conversation, its first prompt and its later turns' questions; alike in both runs."""
    rng = random.Random(seed)
    systems = [tokens(SYSTEM_TOKENS, rng) for _ in range(SYSTEM_PROMPTS)]
    return [
        [systems[c % SYSTEM_PROMPTS] + tokens(DOCUMENT_TOKENS, rng)]
        + [tokens(QUESTION_TOKENS, rng) for _ in range(TURNS - 1)]
        for c in range(CONVERSATIONS)
    ]


def serve_conversations(args: argparse.Namespace, run_dir: Path, with_store: bool) -> list[dict]:
    """Runs the conversations through a 2P1D service; one row per turn."""
    run_dir.mkdir(parents=True, exist_ok=True)
    context_options = {
        "kv_cache_config": {"tokens_per_block": TOKENS_PER_BLOCK, "free_gpu_memory_fraction": 0.6},
        "max_seq_len": 16384,
        "max_num_tokens": 8192,
        "max_batch_size": AT_ONCE,
        "disable_overlap_scheduler": True,
        "cache_transceiver_config": {"backend": "DEFAULT"},
    }
    generation_options = {**context_options, "disable_overlap_scheduler": False}
    with ExitStack() as stack:
        if with_store:
            master = stack.enter_context(running_store(run_dir / "store", args.segment_gib))
            context_options["kv_sharing"] = sharing(
                run_dir, master, "workload", min_fetch_tokens=64
            )
        contexts = [
            Server(f"context-{i}", run_dir, args.model, 1, gpu, context_options, role="context")
            for i, gpu in enumerate(args.context_gpus.split(";"))
        ]
        generation = Server(
            "generation",
            run_dir,
            args.model,
            args.generation_tp,
            args.generation_gpus,
            generation_options,
            role="generation",
        )
        for server in [*contexts, generation]:
            server.start()
        for server in [*contexts, generation]:
            server.wait()
        router = RemoteDisaggOpenAIServer(
            ctx_servers=[f"localhost:{s.process.port}" for s in contexts],
            gen_servers=[f"localhost:{generation.process.port}"],
            log_path=str(run_dir / "router.log"),
        )
        turns = converse(router.get_client(), args.model, conversations(args.seed))
        router.terminate()
        emit(
            {
                "store": "on" if with_store else "off",
                "context_servers": [s.stop() for s in contexts],
            }
        )
        generation.stop()
    return turns


def converse(client, model: str, plan: list[list[list[int]]]) -> list[dict]:
    """Runs AT_ONCE conversations at a time, closed loop.

    Each turn's prompt is the last prompt, its answer and the next question.
    """
    measured: list[dict] = []
    lock = threading.Lock()
    asked_systems: set[int] = set()

    def run_conversation(conversation: int) -> None:
        prompt, *questions = plan[conversation]
        system = conversation % SYSTEM_PROMPTS
        with lock:
            kind = "first turn" if system in asked_systems else "new system prompt"
            asked_systems.add(system)
        # Reusable: the system prompt once asked before, then the previous turn's prompt (its
        # answer was generated on the generation server, so no context server holds it).
        ideal = SYSTEM_TOKENS if kind == "first turn" else 0
        for turn in range(TURNS):
            row = stream_turn(client, model, prompt)
            answer = row.pop("answer")
            row |= {"conversation": conversation, "turn": turn, "kind": kind, "prompt": len(prompt)}
            row["ideal"] = ideal // TOKENS_PER_BLOCK * TOKENS_PER_BLOCK
            with lock:
                measured.append(row)
            if turn < len(questions):
                kind, ideal = "follow-up", len(prompt)
                prompt = prompt + answer + questions[turn]

    with ThreadPoolExecutor(AT_ONCE) as pool:
        list(pool.map(run_conversation, range(len(plan))))
    return measured


def stream_turn(client, model: str, prompt: list[int]) -> dict:
    """Streams one answer: TTFT, TPOT, its tokens and the prompt tokens served from a cache."""
    started = time.perf_counter()
    first, answer, cached = None, [], 0
    stream = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=ANSWER_TOKENS,
        temperature=0,
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"detokenize": False, "ignore_eos": True},
    )
    for chunk in stream:
        if chunk.choices:
            ids = getattr(chunk.choices[0], "token_ids", None) or []
            if ids and first is None:
                first = time.perf_counter()
            answer += ids
        if chunk.usage is not None and chunk.usage.prompt_tokens_details is not None:
            cached = chunk.usage.prompt_tokens_details.cached_tokens
    ended = time.perf_counter()
    first = first or ended
    return {
        "ttft_ms": 1e3 * (first - started),
        "tpot_ms": 1e3 * (ended - first) / max(1, len(answer) - 1),
        "answer": answer,
        "cached": cached,
    }


def summary_of(turns: list[dict], kind: str) -> dict:
    rows = [t for t in turns if t["kind"] == kind]
    ttft = [t["ttft_ms"] for t in rows]
    prompt = sum(t["prompt"] for t in rows)
    return {
        "requests": len(rows),
        "ttft_p50_ms": round(percentile(ttft, 0.5), 1),
        "ttft_p90_ms": round(percentile(ttft, 0.9), 1),
        "ttft_p99_ms": round(percentile(ttft, 0.99), 1),
        "hit_rate": round(sum(t["cached"] for t in rows) / prompt, 3),
        "ideal_hit_rate": round(sum(t["ideal"] for t in rows) / prompt, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    modes = parser.add_subparsers(dest="mode", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", required=True)
    common.add_argument("--run-dir", required=True)
    common.add_argument("--seed", type=int, default=20261024)
    common.add_argument("--segment-gib", type=int, default=160, help="the providers together")
    one = modes.add_parser("ttft", parents=[common])
    one.add_argument("--tp", type=int, default=1)
    one.add_argument("--gpus", default="0")
    one.add_argument("--lengths", type=int, nargs="+", default=[8192, 32768])
    one.add_argument("--busy", action="store_true", help="also while STREAMS streams decode")
    one.add_argument("--data-path", default="auto", choices=("auto", "host_staging"))
    one.add_argument("--burst", type=int, default=0, help="this many longest prompts at once")
    two = modes.add_parser("idle", parents=[common])
    two.add_argument("--tp", type=int, default=1)
    two.add_argument("--gpus", default="0")
    two.add_argument("--rounds", type=int, default=2, help="of off then on")
    three = modes.add_parser("workload", parents=[common])
    three.add_argument("--context-gpus", default="0;1", help="one GPU per context server")
    three.add_argument("--generation-gpus", default="2,3")
    three.add_argument("--generation-tp", type=int, default=2)
    args = parser.parse_args()
    {"ttft": ttft, "idle": idle, "workload": workload}[args.mode](args)


if __name__ == "__main__":
    main()

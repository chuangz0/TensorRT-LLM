# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the store e2e tests: a ``mooncake_master`` and a long-lived segment provider
on free loopback ports, the KV transfer YAML, the model and KV cache settings every engine uses,
and the status dumps written by ``close``.

Mooncake objects live in the client segments; the engines contribute none
(``global_segment_size: 0``), so the provider is the one client whose segment holds the published
blocks, and it must outlive every ``LLM()`` of a test. A test that wants a provider of another
size parametrizes the ``mooncake_cluster`` fixture indirectly with the segment's byte count.
"""

from __future__ import annotations

import glob
import importlib.util
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass

import pytest
import yaml

# Literal copies of ``backends.config.KV_TRANSFER_CONFIG_ENV`` and
# ``assembly.KV_TRANSFER_STATUS_DUMP_ENV``: this module imports ``tensorrt_llm`` only inside
# functions, so the e2e tests collect (and skip) without it; ``backends/test_config_registry.py``
# pins the first literal to the constant.
KV_TRANSFER_CONFIG_ENV = "TRTLLM_KV_TRANSFER_CONFIG"
KV_TRANSFER_STATUS_DUMP_ENV = "TRTLLM_KV_TRANSFER_STATUS_DUMP"

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 5)))
_MODEL_NAME = "TinyLlama-1.1B-Chat-v1.0"


def _model_path() -> str:
    """``$TINYLLAMA_MODEL_PATH``, else the model under ``$LLM_MODELS_ROOT`` (directly or in
    ``llama-models-v2/``, the layout the model root uses), else the repo's ``.models/``."""
    explicit = os.environ.get("TINYLLAMA_MODEL_PATH")
    if explicit:
        return explicit
    root = os.environ.get("LLM_MODELS_ROOT")
    if root:
        for candidate in (
            os.path.join(root, _MODEL_NAME),
            os.path.join(root, "llama-models-v2", _MODEL_NAME),
        ):
            if os.path.isdir(candidate):
                return candidate
    return os.path.join(_REPO_ROOT, ".models", _MODEL_NAME)


MODEL_PATH = _model_path()
MASTER = shutil.which("mooncake_master") or os.path.expanduser("~/.local/bin/mooncake_master")

TOKENS_PER_BLOCK = 32
PROMPT_LEN = 230
"""7 full blocks and a partial tail; ``(230 - 1) // 32 == 7`` nameable blocks."""
NAMEABLE_BLOCKS = (PROMPT_LEN - 1) // TOKENS_PER_BLOCK
MAX_TOKENS = 16

READY_TIMEOUT_S = 30.0
SEGMENT_BYTES = 1 << 30
GENERATE_TIMEOUT_S = 180.0


def timeout_mark(seconds: float):
    """``pytest.mark.timeout`` when the plugin is present; otherwise a no-op marker."""
    if importlib.util.find_spec("pytest_timeout") is not None:
        return pytest.mark.timeout(seconds)
    return pytest.mark.usefixtures()


def prompt_token_ids(seed: int = 1, prompt_len: int = PROMPT_LEN) -> list[int]:
    """A fixed prompt of in-vocabulary ids; content does not matter, determinism does."""
    return [1] + [(seed * 7919 + i * 13 + 1) % 31000 + 100 for i in range(prompt_len - 1)]


def kv_cache_config(free_gpu_memory_fraction: float = 0.2):
    """``free_gpu_memory_fraction`` is of the memory free when the engine starts, so a test that
    keeps several engines up at once lowers it to leave room for the next one."""
    from tensorrt_llm.llmapi import KvCacheConfig

    return KvCacheConfig(
        use_kv_cache_manager_v2=True,
        enable_block_reuse=True,
        free_gpu_memory_fraction=free_gpu_memory_fraction,
        tokens_per_block=TOKENS_PER_BLOCK,
    )


def sampling_params(max_tokens: int = MAX_TOKENS):
    from tensorrt_llm.sampling_params import SamplingParams

    return SamplingParams(max_tokens=max_tokens)


def generate_ids(llm, prompt: list[int], *, max_tokens: int = MAX_TOKENS, **kwargs) -> list[int]:
    """Greedy generation of one prompt; returns the generated token ids."""
    (output,) = llm.generate([prompt], sampling_params(max_tokens), **kwargs)
    return list(output.outputs[0].token_ids)


# ---------------------------------------------------------------------------------------------
# Processes
# ---------------------------------------------------------------------------------------------


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_tcp(port: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return True
        except OSError:
            time.sleep(0.05)
    return False


def kill(proc: subprocess.Popen | None) -> None:
    """Kill the process and everything in its session (both are started with
    ``start_new_session=True``): the Mooncake client forks helpers that outlive their parent."""
    if proc is None:
        return
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
    # Children re-parented to init are not waited for by ``proc``; sweep the group again.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def leftover_pids(marker: str) -> list[int]:
    """Live processes whose command line carries ``marker`` (a port or address unique to this
    test); read from ``/proc`` so no extra dependency is needed."""
    found = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == os.getpid():
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as f:
                cmdline = f.read().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if marker in cmdline:
            found.append(int(entry))
    return found


_PROVIDER_SCRIPT = textwrap.dedent(
    """
    import sys, time
    from mooncake.store import MooncakeDistributedStore
    master, segment_bytes = sys.argv[1], int(sys.argv[2])
    store = MooncakeDistributedStore()
    status = store.setup("127.0.0.1", "P2PHANDSHAKE", segment_bytes, 16 << 20, "tcp", "", master)
    if status != 0:
        print(f"SETUP_FAILED {status}", flush=True)
        sys.exit(1)
    print("READY", flush=True)
    while True:
        time.sleep(1.0)
    """
)


@dataclass
class MooncakeCluster:
    master_address: str
    master: subprocess.Popen
    provider: subprocess.Popen

    def kill_master(self) -> None:
        """The store's master goes away mid-test; every client call fails from here on."""
        kill(self.master)

    def close(self) -> None:
        kill(self.provider)
        kill(self.master)


def start_master() -> tuple[str, subprocess.Popen]:
    rpc, metrics = free_port(), free_port()
    proc = subprocess.Popen(
        [MASTER, f"--rpc_port={rpc}", f"--metrics_port={metrics}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    if not wait_tcp(rpc, READY_TIMEOUT_S):
        kill(proc)
        pytest.fail(f"mooncake_master did not listen on {rpc} within {READY_TIMEOUT_S}s")
    return f"127.0.0.1:{rpc}", proc


def start_segment_provider(
    master_address: str, segment_bytes: int = SEGMENT_BYTES
) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-c", _PROVIDER_SCRIPT, master_address, str(segment_bytes)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    deadline = time.monotonic() + READY_TIMEOUT_S
    line = ""
    while time.monotonic() < deadline:
        line = proc.stdout.readline().strip()
        if line:
            break
    if line != "READY":
        kill(proc)
        pytest.fail(f"segment provider did not come up: {line!r}")
    return proc


@pytest.fixture
def tinyllama_path() -> str:
    """The TinyLlama weights, or a skip; only the tests that run TinyLlama ask for it, so a test
    on a model it builds itself (the tiny Gemma3) runs without any weights on the machine."""
    if not os.path.isdir(MODEL_PATH):
        pytest.skip(f"TinyLlama weights not found at {MODEL_PATH}")
    return MODEL_PATH


@pytest.fixture
def mooncake_cluster(request):
    """A master and one segment provider, killed at teardown whatever happened. Needs a GPU,
    the Mooncake bindings and ``mooncake_master``; no model weights. The provider's segment is
    ``SEGMENT_BYTES`` unless the test parametrizes this fixture indirectly with another size."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("the engines need a GPU")
    pytest.importorskip("mooncake.store")
    if not os.access(MASTER, os.X_OK):
        pytest.skip("mooncake_master binary not found")
    segment_bytes = getattr(request, "param", SEGMENT_BYTES)
    master_address, master = start_master()
    provider = None
    try:
        provider = start_segment_provider(master_address, segment_bytes)
        cluster = MooncakeCluster(master_address, master, provider)
        yield cluster
    finally:
        kill(provider)
        kill(master)
        # The master address is unique to this test: the master carries its port, the provider
        # carries the address; a Mooncake helper that survived would carry one of the two.
        rpc_port = master_address.rsplit(":", 1)[1]
        leftovers = leftover_pids(f"--rpc_port={rpc_port}") + leftover_pids(master_address)
        for pid in leftovers:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        assert not leftovers, f"mooncake processes of this test survived teardown: {leftovers}"


# ---------------------------------------------------------------------------------------------
# Config and status dumps
# ---------------------------------------------------------------------------------------------


def write_kv_transfer_yaml(
    directory,
    master_address: str,
    namespace: str,
    *,
    landing: str | None = None,
    backend_overrides: dict | None = None,
    omit_coordinator_timeouts: bool = False,
    **overrides,
) -> str:
    """The KV transfer config file for a TCP loopback store with one backend.

    ``landing`` is written only when given: left out, the mooncake factory resolves it to
    ``host`` for TCP, which is what the tests assert. ``backend_overrides`` are merged into the
    backend entry (``max_landed_units`` and the other backend options live there, not at the
    top level). The three coordinator timeouts are written explicitly unless
    ``omit_coordinator_timeouts`` asks for the defaults; ``overrides`` go to the top level.
    """
    backend = dict(
        name="shared-store",
        type="mooncake",
        roles=["fetch", "publish"],
        master_server_address=master_address,
        protocol="tcp",
        local_hostname="127.0.0.1",
        metadata_server="P2PHANDSHAKE",
        global_segment_size=0,
        local_buffer_size=256 << 20,
        namespace=namespace,
    )
    if landing is not None:
        backend["landing"] = landing
    backend.update(backend_overrides or {})
    config = dict(backends=[backend])
    if not omit_coordinator_timeouts:
        config.update(fetch_timeout_s=30, publish_timeout_s=60, probe_timeout_s=1.0)
    config.update(overrides)
    path = os.path.join(str(directory), "kv_transfer.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    return path


def dump_template(directory, tag: str) -> str:
    return os.path.join(str(directory), f"kvt-{tag}-{{pid}}.json")


def read_status_dump(directory, tag: str) -> dict:
    """The one dump an engine tagged ``tag`` wrote at shutdown."""
    paths = sorted(glob.glob(os.path.join(str(directory), f"kvt-{tag}-*.json")))
    assert len(paths) == 1, f"expected one status dump for {tag!r}, found {paths}"
    with open(paths[0], encoding="utf-8") as f:
        dump = json.load(f)
    assert set(dump) == {"started_at", "pid", "rank", "coordinator", "backends"}
    assert dump["rank"] == 0 and dump["coordinator"]["plan_authority"] == "VOTED"  # TP=1
    return dump


def counters(dump: dict) -> dict:
    (backend,) = dump["backends"]
    assert backend["type"] == "mooncake" and backend["name"] == "shared-store"
    return backend["counters"]


def landing_of(dump: dict) -> str:
    """The landing the factory resolved for the one backend, as the status dump reports it."""
    (backend,) = dump["backends"]
    return backend["landing"]


def assert_no_leftover_records(dump: dict) -> None:
    coordinator = dump["coordinator"]
    assert coordinator["records"] == [], coordinator
    assert coordinator["finished_pending"] == [], coordinator
    assert coordinator["decided_plans"] == 0, coordinator


# ---------------------------------------------------------------------------------------------
# Multi-rank and RDMA runs (E4, E5)
# ---------------------------------------------------------------------------------------------

RDMA_ENV = "KV_TRANSFER_E2E_RDMA"
"""``1`` enables the RDMA transports; the machine needs an RDMA NIC."""
RDMA_DEVICES_ENV = "KV_TRANSFER_E2E_RDMA_DEVICES"
"""HCAs handed to Mooncake as ``device_name``; empty lets it discover them, which on some hosts
pairs InfiniBand ports with Ethernet ones that cannot reach them."""

_needs_rdma = pytest.mark.skipif(
    os.environ.get(RDMA_ENV) != "1", reason=f"{RDMA_ENV}=1 only: needs an RDMA NIC"
)
TRANSPORTS = [
    pytest.param(("tcp", None), id="tcp"),
    pytest.param(("rdma", "device"), id="rdma_device", marks=_needs_rdma),
    pytest.param(("rdma", "host"), id="rdma_host", marks=_needs_rdma),
]
"""``(protocol, landing)`` for ``transport_store``; a ``None`` landing is the factory's default
(``host`` over TCP). Over RDMA the GPU KV pools are registered through dma-buf, which the Mooncake
wheel does only with ``WITH_NVIDIA_PEERMEM=0``; the tests set it unless the caller did."""

LAYOUTS = {
    "tp2": dict(tensor_parallel_size=2),
    "pp2": dict(pipeline_parallel_size=2),
    "tp2_adp": dict(tensor_parallel_size=2, enable_attention_dp=True),
    "tp2pp2": dict(tensor_parallel_size=2, pipeline_parallel_size=2),
}

MODELS = {
    # name: (path under $LLM_MODELS_ROOT, or None for the tinyllama fixture; prompt length;
    #        max_seq_len; whether the model has sliding-window layer groups)
    "tinyllama": (None, PROMPT_LEN, None, False),
    "llama31_8b": ("llama-3.1-model/Llama-3.1-8B-Instruct", 1000, None, False),
    "qwen3_8b": ("Qwen3/Qwen3-8B", 1000, None, False),
    # W = 1024, L = 2000: stale_end(L) == stale_end(B) == 30, so every live block is fetchable.
    "gemma3_12b": ("gemma/gemma-3-12b-it", 2000, 4096, True),
    # W = 512, L = 1000: window-aligned, as test_store_vswa_tinygemma3's real case.
    "gemma3_1b": ("gemma/gemma-3-1b-it", 1000, 2048, True),
}
SELECTED_MODELS = os.environ.get("KV_TRANSFER_E2E_MODELS", "tinyllama").split(",")

SETTLE_S = float(os.environ.get("KV_TRANSFER_E2E_SETTLE_S", "3"))
"""How long the publishing engine keeps serving after its request finished, so the publish lands
while the loop still runs: the loop stops at shutdown without waiting for transfers in flight, and
with several ranks rank 0 can block idle while a peer's publish of a finished request has not
settled yet."""


def world_size_of(layout: dict) -> int:
    return layout.get("tensor_parallel_size", 1) * layout.get("pipeline_parallel_size", 1)


def model_path_for(request, model: str) -> str:
    """The weights of ``model`` (a ``MODELS`` key), or a skip."""
    rel = MODELS[model][0]
    if rel is None:
        return request.getfixturevalue("tinyllama_path")
    path = os.path.join(os.environ.get("LLM_MODELS_ROOT", ""), rel)
    if not os.path.isdir(path):
        pytest.skip(f"{path} not found")
    return path


def host_ip() -> str:
    return socket.gethostbyname(socket.gethostname())


_TRANSPORT_PROVIDER_SCRIPT = textwrap.dedent(
    """
    import sys, time
    from mooncake.store import MooncakeDistributedStore
    master, segment_bytes, protocol, host, devices = sys.argv[1:6]
    store = MooncakeDistributedStore()
    status = store.setup(host, "P2PHANDSHAKE", int(segment_bytes), 16 << 20, protocol, devices, master)
    if status != 0:
        print(f"SETUP_FAILED {status}", flush=True)
        sys.exit(1)
    print("READY", flush=True)
    while True:
        time.sleep(1.0)
    """
)


@pytest.fixture
def transport_store(request):
    """A master and one segment provider speaking the ``TRANSPORTS`` protocol given by indirect
    parametrization; yields ``(protocol, landing, master_address)``."""
    protocol, landing = request.param
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("the engines need a GPU")
    pytest.importorskip("mooncake.store")
    if not os.access(MASTER, os.X_OK):
        pytest.skip("mooncake_master binary not found")
    master_address, master = start_master()
    host = "127.0.0.1" if protocol == "tcp" else host_ip()
    devices = "" if protocol == "tcp" else os.environ.get(RDMA_DEVICES_ENV, "")
    provider = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _TRANSPORT_PROVIDER_SCRIPT,
            master_address,
            str(4 << 30),
            protocol,
            host,
            devices,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + READY_TIMEOUT_S
        line = ""
        while time.monotonic() < deadline and not line:
            line = provider.stdout.readline().strip()
        if line != "READY":
            pytest.fail(f"{protocol} segment provider did not come up: {line!r}")
        yield protocol, landing, master_address
    finally:
        kill(provider)
        kill(master)


def write_transport_yaml(directory, master_address, namespace, protocol, landing) -> str:
    """``write_kv_transfer_yaml`` for either protocol: over RDMA the engine announces the host's
    own address and the HCAs of ``RDMA_DEVICES_ENV``."""
    backend = dict(
        name="shared-store",
        type="mooncake",
        roles=["fetch", "publish"],
        master_server_address=master_address,
        protocol=protocol,
        local_hostname="127.0.0.1" if protocol == "tcp" else host_ip(),
        metadata_server="P2PHANDSHAKE",
        global_segment_size=0,
        local_buffer_size=256 << 20,
        namespace=namespace,
    )
    if protocol == "rdma":
        backend["device_name"] = os.environ.get(RDMA_DEVICES_ENV, "")
    if landing is not None:
        backend["landing"] = landing
    config = dict(fetch_timeout_s=30, publish_timeout_s=60, probe_timeout_s=1.0, backends=[backend])
    path = os.path.join(str(directory), "kv_transfer.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    return path


def read_rank_dumps(directory, tag: str, world_size: int) -> list[dict]:
    """One dump per rank of the engine tagged ``tag``, ordered by rank."""
    dumps = []
    for path in glob.glob(os.path.join(str(directory), f"kvt-{tag}-*.json")):
        with open(path, encoding="utf-8") as f:
            dumps.append(json.load(f))
    dumps.sort(key=lambda d: d["rank"])
    assert [d["rank"] for d in dumps] == list(range(world_size)), [d["rank"] for d in dumps]
    return dumps


def backend_of(dump: dict) -> dict:
    (backend,) = dump["backends"]
    assert backend["type"] == "mooncake" and backend["name"] == "shared-store"
    return backend

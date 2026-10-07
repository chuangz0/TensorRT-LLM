# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The engines of the store e2e tests: the models and where their weights come from, the
parallel layouts, the prompts and KV cache settings every engine uses, the KV transfer YAML, and
one way to start an ``LLM()`` with the store attached (``start_engine``) or to run one to
completion and read back what it wrote (``run_engine``).

This module imports ``tensorrt_llm`` only inside functions, so the e2e tests collect (and skip)
without it.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass

import pytest
import yaml
from mooncake_cluster import RDMA_DEVICES_ENV, host_ip
from status_dumps import (
    KV_TRANSFER_STATUS_DUMP_ENV,
    dump_template,
    read_rank_dumps,
    read_single_rank_dump,
)

# Literal copy of ``backends.config.KV_TRANSFER_CONFIG_ENV``; ``backends/test_config.py`` pins the
# literal to the constant.
KV_TRANSFER_CONFIG_ENV = "TRTLLM_KV_TRANSFER_CONFIG"

# ---------------------------------------------------------------------------------------------
# Environment knobs (``RDMA_ENV`` and ``RDMA_DEVICES_ENV`` are ``mooncake_cluster``'s)
# ---------------------------------------------------------------------------------------------

TINYLLAMA_PATH_ENV = "KV_TRANSFER_E2E_TINYLLAMA_PATH"
"""The TinyLlama weights; else the model under ``$LLM_MODELS_ROOT``, else the repo's ``.models/``."""
GEMMA3_PATH_ENV = "KV_TRANSFER_E2E_GEMMA3_PATH"
"""The Gemma-3-1b-it weights; else ``gemma/gemma-3-1b-it`` under ``$LLM_MODELS_ROOT``."""
MODELS_ENV = "KV_TRANSFER_E2E_MODELS"
"""``MODELS`` keys the multi-rank and exactness tests run, comma-separated; TinyLlama by default."""
EXACT_LAYOUTS_ENV = "KV_TRANSFER_E2E_EXACT_LAYOUTS"
"""Layouts the exactness test runs, comma-separated: ``tp1`` (one rank) or ``LAYOUTS`` keys."""

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 5)))
TINYLLAMA_NAME = "TinyLlama-1.1B-Chat-v1.0"


def tinyllama_path_from_env() -> str:
    """``$KV_TRANSFER_E2E_TINYLLAMA_PATH``, else the model under ``$LLM_MODELS_ROOT`` (directly or
    in ``llama-models-v2/``, the layout the model root uses), else the repo's ``.models/``."""
    explicit = os.environ.get(TINYLLAMA_PATH_ENV)
    if explicit:
        return explicit
    root = os.environ.get("LLM_MODELS_ROOT")
    if root:
        for candidate in (
            os.path.join(root, TINYLLAMA_NAME),
            os.path.join(root, "llama-models-v2", TINYLLAMA_NAME),
        ):
            if os.path.isdir(candidate):
                return candidate
    return os.path.join(_REPO_ROOT, ".models", TINYLLAMA_NAME)


def gemma3_1b_path_from_env() -> str | None:
    """``$KV_TRANSFER_E2E_GEMMA3_PATH``, else ``gemma/gemma-3-1b-it`` under ``$LLM_MODELS_ROOT``,
    else ``None``."""
    explicit = os.environ.get(GEMMA3_PATH_ENV)
    if explicit:
        return explicit
    root = os.environ.get("LLM_MODELS_ROOT")
    if root:
        candidate = os.path.join(root, "gemma", "gemma-3-1b-it")
        if os.path.isdir(candidate):
            return candidate
    return None


TINYLLAMA_PATH = tinyllama_path_from_env()


@pytest.fixture
def tinyllama_path() -> str:
    """The TinyLlama weights, or a skip; only the tests that run TinyLlama ask for it, so a test
    on a model it builds itself (the tiny Gemma3) runs without any weights on the machine."""
    if not os.path.isdir(TINYLLAMA_PATH):
        pytest.skip(f"TinyLlama weights not found at {TINYLLAMA_PATH}")
    return TINYLLAMA_PATH


# ---------------------------------------------------------------------------------------------
# Prompts, KV cache, sampling
# ---------------------------------------------------------------------------------------------

TOKENS_PER_BLOCK = 32
PROMPT_LEN = 230
"""7 full blocks and a partial tail; ``(230 - 1) // 32 == 7`` nameable blocks."""
NAMEABLE_BLOCKS = (PROMPT_LEN - 1) // TOKENS_PER_BLOCK
MAX_TOKENS = 16


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


def sampling_params(max_tokens: int = MAX_TOKENS, **overrides):
    """Greedy sampling of ``max_tokens``; ``overrides`` are further ``SamplingParams`` fields."""
    from tensorrt_llm.sampling_params import SamplingParams

    return SamplingParams(max_tokens=max_tokens, **overrides)


def generate_ids(llm, prompt: list[int], *, max_tokens: int = MAX_TOKENS, **kwargs) -> list[int]:
    """Greedy generation of one prompt; returns the generated token ids."""
    (output,) = llm.generate([prompt], sampling_params(max_tokens), **kwargs)
    return list(output.outputs[0].token_ids)


# ---------------------------------------------------------------------------------------------
# Models and layouts
# ---------------------------------------------------------------------------------------------

LAYOUTS = {
    "tp2": dict(tensor_parallel_size=2),
    "pp2": dict(pipeline_parallel_size=2),
    "tp2_adp": dict(tensor_parallel_size=2, enable_attention_dp=True),
    "tp2pp2": dict(tensor_parallel_size=2, pipeline_parallel_size=2),
}
"""The multi-rank layouts, as ``LLM()`` keyword arguments."""


@dataclass(frozen=True)
class ModelCase:
    """One model the multi-rank and exactness tests run.

    Attributes:
        models_root_path: Path under ``$LLM_MODELS_ROOT``, or ``None`` for the TinyLlama fixture.
        prompt_len: Prompt length the model is run at.
        max_seq_len: ``max_seq_len`` to start the engine with, or ``None`` for the default.
        has_sliding_window: Whether the model has sliding-window layer groups, which publish
            and fetch the live windowed blocks on top of the full-attention ones.
    """

    models_root_path: str | None
    prompt_len: int
    max_seq_len: int | None
    has_sliding_window: bool


MODELS = {
    "tinyllama": ModelCase(None, PROMPT_LEN, None, False),
    "llama31_8b": ModelCase("llama-3.1-model/Llama-3.1-8B-Instruct", 1000, None, False),
    "qwen3_8b": ModelCase("Qwen3/Qwen3-8B", 1000, None, False),
    # W = 1024, L = 2000: stale_end(L) == stale_end(B) == 30, so every live block is fetchable.
    "gemma3_12b": ModelCase("gemma/gemma-3-12b-it", 2000, 4096, True),
    # W = 512, L = 1000: window-aligned, as test_store_sliding_window's real case.
    "gemma3_1b": ModelCase("gemma/gemma-3-1b-it", 1000, 2048, True),
}
SELECTED_MODELS = os.environ.get(MODELS_ENV, "tinyllama").split(",")
SELECTED_EXACT_LAYOUTS = os.environ.get(EXACT_LAYOUTS_ENV, "tp1").split(",")


def world_size_of(layout: dict | None) -> int:
    layout = layout or {}
    return layout.get("tensor_parallel_size", 1) * layout.get("pipeline_parallel_size", 1)


def model_path_for(request, model: str) -> str:
    """The weights of ``model`` (a ``MODELS`` key), or a skip."""
    rel = MODELS[model].models_root_path
    if rel is None:
        return request.getfixturevalue("tinyllama_path")
    path = os.path.join(os.environ.get("LLM_MODELS_ROOT", ""), rel)
    if not os.path.isdir(path):
        pytest.skip(f"{path} not found")
    return path


# ---------------------------------------------------------------------------------------------
# The KV transfer config file
# ---------------------------------------------------------------------------------------------


def write_kv_transfer_yaml(
    directory,
    master_address: str,
    namespace: str,
    *,
    protocol: str = "tcp",
    landing: str | None = None,
    backend_overrides: dict | None = None,
    omit_coordinator_timeouts: bool = False,
    **overrides,
) -> str:
    """The KV transfer config file for a store with one backend, on loopback over ``tcp``; over
    ``rdma`` the engine announces the host's own address and the HCAs of ``RDMA_DEVICES_ENV``.

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
        protocol=protocol,
        local_hostname="127.0.0.1",
        metadata_server="P2PHANDSHAKE",
        global_segment_size=0,
        local_buffer_size=256 << 20,
        namespace=namespace,
    )
    if protocol == "rdma":
        backend["local_hostname"] = host_ip()
        backend["device_name"] = os.environ.get(RDMA_DEVICES_ENV, "")
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


# ---------------------------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------------------------


def transceiver_config():
    """The disaggregated-serving transceiver the engines pair over: NIXL, Python runtime."""
    from tensorrt_llm.llmapi import CacheTransceiverConfig

    return CacheTransceiverConfig(
        backend="NIXL", transceiver_runtime="PYTHON", kv_transfer_timeout_ms=30000
    )


def start_engine(
    tmp_path,
    monkeypatch,
    tag: str | None,
    model_path: str,
    *,
    layout: dict | None = None,
    max_seq_len: int | None = None,
    cache_transceiver: bool = False,
    free_gpu_memory_fraction: float = 0.2,
    **llm_kwargs,
):
    """One ``LLM()`` on the KV cache manager V2 with block reuse, the overlap scheduler off
    unless ``llm_kwargs`` says otherwise, in the parallel ``layout`` (``LAYOUTS`` value), with the
    disaggregated transceiver when ``cache_transceiver``. Its status dump at shutdown is tagged
    ``tag``; ``None`` is an engine without the store attached, which writes none."""
    from tensorrt_llm import LLM

    if tag is None:
        monkeypatch.delenv(KV_TRANSFER_STATUS_DUMP_ENV, raising=False)
    else:
        monkeypatch.setenv(KV_TRANSFER_STATUS_DUMP_ENV, dump_template(tmp_path, tag))
    llm_kwargs.setdefault("disable_overlap_scheduler", True)
    if max_seq_len is not None:
        llm_kwargs["max_seq_len"] = max_seq_len
    if cache_transceiver:
        llm_kwargs["cache_transceiver_config"] = transceiver_config()
    return LLM(
        model=model_path,
        kv_cache_config=kv_cache_config(free_gpu_memory_fraction),
        **(layout or {}),
        **llm_kwargs,
    )


@dataclass
class EngineRun:
    """What one engine run produced.

    Attributes:
        tokens: The generated token ids, one list per prompt.
        logits: The context logits at the last prompt position, one tensor per prompt (the only
            position both a publishing and a fetching engine compute themselves); ``None`` unless
            asked for.
        dumps: The status dumps the engine wrote at shutdown, one per rank ordered by rank;
            empty for an engine without the store attached.
    """

    tokens: list[list[int]]
    logits: list | None
    dumps: list[dict]

    @property
    def dump(self) -> dict:
        """The one dump of a single-rank engine."""
        (dump,) = self.dumps
        return dump


def run_engine(
    tmp_path,
    monkeypatch,
    tag: str | None,
    model_path: str,
    prompts: list[list[int]],
    *,
    layout: dict | None = None,
    batch: bool = False,
    return_logits: bool = False,
    sampling_overrides: dict | None = None,
    **start_kwargs,
) -> EngineRun:
    """``start_engine``, generate each prompt once (one ``generate`` call per prompt, or all of
    them in one call with ``batch``), shut down, and read the status dumps back."""
    llm = start_engine(tmp_path, monkeypatch, tag, model_path, layout=layout, **start_kwargs)
    try:
        sampling = sampling_params(
            return_context_logits=return_logits, **(sampling_overrides or {})
        )
        if batch:
            outputs = llm.generate(prompts, sampling)
        else:
            outputs = [llm.generate([prompt], sampling)[0] for prompt in prompts]
        tokens = [list(output.outputs[0].token_ids) for output in outputs]
        logits = None
        if return_logits:
            logits = [output.context_logits[-1].detach().float().cpu() for output in outputs]
    finally:
        llm.shutdown()
    if tag is None:
        dumps = []
    elif world_size_of(layout) == 1:
        dumps = [read_single_rank_dump(tmp_path, tag)]
    else:
        dumps = read_rank_dumps(tmp_path, tag, world_size_of(layout))
    return EngineRun(tokens, logits, dumps)

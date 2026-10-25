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
"""Which setups ``kv_sharing`` accepts: the LLM argument's checks where the user builds the args,
and startup's refusal of model facts only the built executor knows."""

import re
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_sharing.startup import start_kv_sharing
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def store_config(tmp_path) -> str:
    path = tmp_path / "store.yaml"
    path.write_text(
        "mooncake:\n  master_server_address: 10.0.0.1:50051\n  global_segment_size: 0\n"
    )
    return str(path)


def llm_args(store_config: str, kv_sharing: dict | None = None, **kwargs) -> TorchLlmArgs:
    sharing = {"backends": [{"type": "mooncake", "config": store_config}], **(kv_sharing or {})}
    return TorchLlmArgs(
        model="/tmp/dummy_model", skip_tokenizer_init=True, kv_sharing=sharing, **kwargs
    )


def test_minimal_kv_sharing_config_is_accepted(store_config: str) -> None:
    sharing = llm_args(store_config).kv_sharing
    assert sharing.backends[0].roles == ["fetch", "publish"]
    assert sharing.data_path == "auto"
    # A disaggregated context server shares through the store too, on either data path.
    llm_args(
        store_config,
        {"data_path": "gpu_direct"},
        cache_transceiver_config={"backend": "NIXL"},
    )
    # Tensor parallelism, attention DP and an FP8 KV cache share through the store, also on a
    # disaggregated context server.
    assert llm_args(store_config, tensor_parallel_size=2).kv_sharing is not None
    llm_args(store_config, tensor_parallel_size=2, enable_attention_dp=True)
    llm_args(store_config, kv_cache_config={"dtype": "fp8"})
    transceiver = {"backend": "NIXL"}
    llm_args(store_config, tensor_parallel_size=2, cache_transceiver_config=transceiver)
    llm_args(
        store_config,
        tensor_parallel_size=2,
        enable_attention_dp=True,
        cache_transceiver_config=transceiver,
    )


UNSUPPORTED = {
    "a KV cache connector": (
        {
            "kv_connector_config": {
                "connector_module": "my_connector",
                "connector_scheduler_class": "Scheduler",
                "connector_worker_class": "Worker",
            }
        },
        "kv_connector_config",
    ),
    "block reuse off": ({"kv_cache_config": {"enable_block_reuse": False}}, "block reuse off"),
    "PP2": ({"pipeline_parallel_size": 2}, "pipeline or context parallelism"),
    "an NVFP4 KV cache": ({"kv_cache_config": {"dtype": "nvfp4"}}, "BF16 or FP8 KV caches only"),
    "speculative decoding": (
        {"speculative_config": {"decoding_type": "NGram", "max_draft_len": 1}},
        "speculative decoding",
    ),
    "fetch limits out of order": (
        {"kv_sharing": {"min_fetch_tokens": 4096, "max_fetch_tokens": 1024}},
        "max_fetch_tokens (1024) is below min_fetch_tokens (4096)",
    ),
    "a missing store config": (
        {"kv_sharing": {"backends": [{"type": "mooncake", "config": "/no/such/store.yaml"}]}},
        "'/no/such/store.yaml' is not a file",
    ),
}


@pytest.mark.parametrize("setup", UNSUPPORTED)
def test_kv_sharing_rejects_setups_it_cannot_serve(store_config: str, setup: str) -> None:
    kwargs, reason = UNSUPPORTED[setup]
    with pytest.raises(ValueError, match=re.escape(reason)):
        llm_args(store_config, **kwargs)


@pytest.mark.parametrize("fact", ["sparse attention", "recurrent state"])
def test_startup_refuses_a_model_with_sparse_attention_or_recurrent_state_before_any_store_work(
    store_config: str, fact: str
) -> None:
    """A DSA checkpoint fills in its sparse attention config only when the model loads, and a
    hybrid model's recurrent state shows in the manager built for it."""
    hybrid = fact == "recurrent state"
    manager = object.__new__(MambaHybridCacheManagerV2 if hybrid else KVCacheManagerV2)
    manager.enable_block_reuse = True
    model_config = SimpleNamespace(sparse_attention_config=None if hybrid else object())
    executor = SimpleNamespace(
        kv_cache_manager=manager,
        scheduler=object.__new__(KVCacheV2Scheduler),
        model_engine=SimpleNamespace(model=SimpleNamespace(model_config=model_config)),
        dist=None,  # no collective, store or checkpoint work may start
    )
    with pytest.raises(ValueError, match=f"this server is refused: the model has {fact}$"):
        start_kv_sharing(executor, llm_args(store_config), checkpoint_dir=None, max_num_tokens=64)

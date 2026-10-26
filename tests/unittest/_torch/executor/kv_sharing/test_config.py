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
startup's refusal of model facts only the built executor knows, the store driver each data path
gets, and the host memory a server provides the store."""

import re
from types import SimpleNamespace
from typing import get_args

import pytest
from _torch.disaggregation.blob.fake_mooncake import FakeMooncakeStore
from transformers import PretrainedConfig

from tensorrt_llm._torch.disaggregation._assembly import ProfileSpec
from tensorrt_llm._torch.disaggregation.blob import mooncake_setup
from tensorrt_llm._torch.disaggregation.blob.mooncake_setup import UNMOUNT_GRACE_S
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import StagingOptions
from tensorrt_llm._torch.pyexecutor.kv_sharing import startup
from tensorrt_llm._torch.pyexecutor.kv_sharing.startup import start_kv_sharing
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler
from tensorrt_llm._torch.weight_sharing.artifact_identity import ArtifactIdentity
from tensorrt_llm.bindings import DataType
from tensorrt_llm.llmapi.llm_args import (
    CSA2SparseAttentionConfig,
    DeepSeekSparseAttentionConfig,
    SparseAttentionConfig,
    TorchLlmArgs,
)

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def store_config(tmp_path) -> str:
    path = tmp_path / "store.yaml"
    path.write_text("mooncake:\n  master_server_address: 10.0.0.1:50051\n")
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
    # FP8 and NVFP4 KV caches share too.
    llm_args(store_config, kv_cache_config={"dtype": "fp8"})
    llm_args(store_config, kv_cache_config={"dtype": "nvfp4"})


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
    "the packed FP8 sparse-MLA KV cache": (
        {"kv_cache_config": {"dtype": "fp8_ds_mla"}},
        "a packed sparse-MLA or mixed-precision KV cache",
    ),
    "KV cache compression": (
        {"kv_cache_compression_config": {"algorithm": "quantization_for_cold_page"}},
        "kv_cache_compression_config (quantization_for_cold_page)",
    ),
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


def providing(store_config: str, roles: list[str], segment_size="32GiB") -> dict:
    """``kv_sharing`` of a server whose ``roles`` give the store ``segment_size`` per rank."""
    backend = {
        "type": "mooncake",
        "config": store_config,
        "roles": roles,
        "segment_size": segment_size,
    }
    return {"backends": [backend]}


def test_the_role_provide_and_segment_size_go_together(store_config: str) -> None:
    sharing = llm_args(store_config, providing(store_config, ["fetch", "publish", "provide"]))
    assert sharing.kv_sharing.backends[0].segment_size == 32 << 30
    with pytest.raises(ValueError, match="'provide' needs segment_size"):
        llm_args(store_config, providing(store_config, ["provide"], segment_size=0))
    with pytest.raises(ValueError, match="segment_size is set but roles lack 'provide'"):
        llm_args(store_config, providing(store_config, ["fetch", "publish"]))
    with pytest.raises(ValueError, match="ambiguous unit 'GB'"):
        llm_args(store_config, providing(store_config, ["provide"], segment_size="32GB"))
    with pytest.raises(ValueError, match="name a role twice"):
        llm_args(store_config, providing(store_config, ["provide", "provide"]))


def test_a_server_that_only_provides_memory_is_refused_only_without_mpi_or_with_a_kv_connector(
    store_config: str,
) -> None:
    """A generation server leaves its KV cache alone, so the setups that refuse a server that
    fetches or publishes serve."""
    provide_only = providing(store_config, ["provide"])
    for setup in ("block reuse off", "PP2", "KV cache compression", "speculative decoding"):
        llm_args(store_config, provide_only, **UNSUPPORTED[setup][0])
    with pytest.raises(ValueError, match="does not support: kv_connector_config: [^;]*$"):
        llm_args(store_config, provide_only, **UNSUPPORTED["a KV cache connector"][0])
    for orchestrator in ("ray", "rpc"):
        only_reason = (
            f"does not support: orchestrator_type='{orchestrator}': only the MPI orchestrator ["
        )
        with pytest.raises(ValueError, match=re.escape(only_reason)):
            llm_args(store_config, provide_only, orchestrator_type=orchestrator)


# Each algorithm of the sparse attention config union.
ALL_SPARSE_ALGORITHMS = [
    get_args(config.model_fields["algorithm"].annotation)[0]
    for config in get_args(get_args(SparseAttentionConfig)[0])
]


@pytest.mark.parametrize("algorithm", ALL_SPARSE_ALGORITHMS)
def test_llm_args_accept_exactly_the_sparse_attention_startup_serves(
    store_config: str, algorithm: str
) -> None:
    """The LLM arguments name the served algorithms themselves, free of runtime imports; a user
    who sets another one learns it before any model loads."""
    config = {"algorithm": algorithm}
    if algorithm in startup.SPARSE_ALGORITHMS:
        llm_args(store_config, sparse_attention_config=config)
    else:
        with pytest.raises(ValueError, match=re.escape(f"{algorithm} sparse attention: ")):
            llm_args(store_config, sparse_attention_config=config)


DSA = DeepSeekSparseAttentionConfig(index_n_heads=64, index_head_dim=128, index_topk=2048)
NVFP4_MLA = "a KV cache of layout mla-trtllm-latent and dtype nvfp4"
# Model facts only the built executor knows: how to build it, and the fact startup names.
REFUSED_MODELS = {
    # A checkpoint fills in its sparse attention config only when the model loads.
    "DeepSeek-V4.1's CSA2": (
        {"sparse_attention_config": CSA2SparseAttentionConfig()},
        "csa2 sparse attention",
    ),
    "a hybrid model": ({"manager_class": MambaHybridCacheManagerV2}, "recurrent state"),
    # TensorRT-LLM turns block reuse off for dense FP4 MLA.
    "dense NVFP4 MLA": ({"dtype": DataType.NVFP4, "mla": True, "block_reuse": False}, NVFP4_MLA),
    # With DSA it keeps block reuse on: refused before rank 0 reads the checkpoint (387 GiB for
    # DeepSeek-V3.2).
    "NVFP4 MLA with DSA": (
        {"dtype": DataType.NVFP4, "mla": True, "sparse_attention_config": DSA},
        NVFP4_MLA,
    ),
}


@pytest.mark.parametrize("model", REFUSED_MODELS)
def test_startup_refuses_a_model_it_cannot_serve_before_store_work(
    store_config: str, model: str
) -> None:
    built_with, fact = REFUSED_MODELS[model]
    # No collective, store or checkpoint work may start.
    built = executor(dist=None, **built_with)
    with pytest.raises(ValueError, match=f"refuses this server: the model has {fact}\\. It"):
        start_kv_sharing(built, llm_args(store_config), checkpoint_dir=None, max_num_tokens=64)


def mla_profile(store_config: str, sparse_attention_config) -> ProfileSpec:
    """The profile startup builds for one GPU with an FP8 MLA cache in one pool group."""
    built = executor(dtype=DataType.FP8, mla=True, sparse_attention_config=sparse_attention_config)
    manager = built.kv_cache_manager
    manager.num_pools, manager.max_attention_window_vec, manager.pp_layers = 1, [None], [0]
    manager.impl = SimpleNamespace(pool_group_descs=[None])
    identity = ArtifactIdentity(1, "checkpoint_manifest_sha256", "a" * 64)
    return startup._profile(built, llm_args(store_config), identity, StagingOptions(32))


def test_a_dsa_server_names_blocks_after_its_sparse_config_not_its_cuda_graphs(
    store_config: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scope carries the whole effective sparse config: a dense server of the same cache,
    another top-k or indexer key dtype name other blocks; capturing CUDA graphs sets
    ``seq_len_threshold`` and names the same ones."""
    monkeypatch.setattr(startup.torch.cuda, "get_device_capability", lambda: (10, 0))
    config = DSA.model_copy()
    scope = mla_profile(store_config, config).scope
    assert mla_profile(store_config, None).scope != scope
    config.needs_separate_short_long_cuda_graphs()
    assert config.seq_len_threshold == 2048
    assert mla_profile(store_config, config).scope == scope
    for change in ({"index_topk": 1024}, {"indexer_k_dtype": "fp4"}):
        assert mla_profile(store_config, config.model_copy(update=change)).scope != scope


@pytest.mark.parametrize(
    "data_path, device_memory", [("host_staging", False), ("gpu_direct", True)]
)
def test_the_store_driver_registers_device_memory_only_under_gpu_direct(
    store_config: str, tmp_path, monkeypatch, data_path: str, device_memory: bool
) -> None:
    """Under host staging the store reads and writes host memory only. One rank, a store that
    opens and checks as the data path asks; startup stops at attach, past the driver it made."""
    why_not_gpu_direct = None if data_path == "gpu_direct" else "host staging was asked for"
    monkeypatch.setattr(startup, "open_store", lambda requested, mooncake: (None, None))
    monkeypatch.setattr(startup, "check_store", lambda store, why: why_not_gpu_direct)
    monkeypatch.setattr(startup.ArtifactIdentity, "from_checkpoint", lambda checkpoint: None)
    drivers = []

    def stop_at_attach(manager, profile_of, memory_kind, fetch_tokens, driver, blob_config):
        drivers.append(driver)
        raise RuntimeError("stopped at attach")

    monkeypatch.setattr(startup, "_attach", stop_at_attach)
    built = executor(dist=SimpleNamespace(rank=0, world_size=1))
    args = llm_args(store_config, {"data_path": data_path})
    with pytest.raises(ValueError, match="rank 0: stopped at attach"):
        start_kv_sharing(built, args, checkpoint_dir=str(tmp_path), max_num_tokens=64)
    (driver,) = drivers
    assert driver.capabilities.device_memory is device_memory


@pytest.fixture
def unmount_waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """The sleeps of store segments waiting to be freed, which take none; ``time.monotonic()`` reads
    0 throughout."""
    slept: list[float] = []
    monkeypatch.setattr(
        mooncake_setup, "time", SimpleNamespace(monotonic=lambda: 0.0, sleep=slept.append)
    )
    return slept


def test_a_context_server_mounts_its_memory_before_its_check_and_unmounts_it_if_startup_fails(
    store_config: str, tmp_path, monkeypatch, unmount_waits: list[float]
) -> None:
    """Where servers provide all of the store's memory, the startup check's object needs the
    server's own segment: it is mounted first. A start that fails afterwards (here at attach)
    unmounts it gracefully, about 13 s, before raising; a failed mount refuses the server."""
    store = FakeMooncakeStore()
    monkeypatch.setattr(startup, "open_store", lambda requested, mooncake: (store, "no GPU"))
    monkeypatch.setattr(
        startup.psutil, "virtual_memory", lambda: SimpleNamespace(available=1 << 40)
    )
    monkeypatch.setattr(startup.ArtifactIdentity, "from_checkpoint", lambda checkpoint: None)

    def stop_at_attach(*args):
        raise RuntimeError("stopped at attach")

    monkeypatch.setattr(startup, "_attach", stop_at_attach)
    built = executor(dist=SimpleNamespace(rank=0, world_size=1, local_world_size=1))
    args = llm_args(store_config, providing(store_config, ["fetch", "publish", "provide"]))
    with pytest.raises(ValueError, match="rank 0: stopped at attach"):
        start_kv_sharing(built, args, checkpoint_dir=str(tmp_path), max_num_tokens=64)
    methods = [call.method for call in store.calls]
    assert methods.index("allocate_and_mount_segment") < methods.index("batch_put_session_start")
    (unmount,) = store.calls_to("unmount_and_free_segment")
    assert unmount.args == (built.store_segment.segment_ids, UNMOUNT_GRACE_S)
    assert unmount_waits == [13.0]  # the grace, Mooncake's cleanup and a margin

    store.mount_answer = -1
    built = executor(dist=SimpleNamespace(rank=0, world_size=1, local_world_size=1))
    mount_failed = "rank 0: mooncake: mounting 34359738368 bytes of host memory failed with -1"
    with pytest.raises(ValueError, match=mount_failed):
        start_kv_sharing(built, args, checkpoint_dir=str(tmp_path), max_num_tokens=64)
    assert built.store_segment is None


def test_a_context_server_provides_only_what_its_node_can_give_with_host_staging(
    store_config: str, tmp_path, monkeypatch
) -> None:
    """Two ranks on a node of 70 GiB available, each providing 16 GiB and staging through up to
    8 GiB: 48 GiB is more than the 38 GiB left after the 32 GiB reserve. Every rank refuses before
    any rank opens the store."""
    gib = 1 << 30
    monkeypatch.setattr(
        startup.psutil, "virtual_memory", lambda: SimpleNamespace(available=70 * gib)
    )
    monkeypatch.setattr(startup, "open_store", fail_if_called)
    monkeypatch.setattr(startup.ArtifactIdentity, "from_checkpoint", lambda checkpoint: None)
    built = executor(dist=SimpleNamespace(rank=0, world_size=1, local_world_size=2))
    args = llm_args(store_config, providing(store_config, ["fetch", "publish", "provide"], "16GiB"))
    refused = (
        "2 rank(s) here would provide 16.0 GiB each and stage through up to 8.0 GiB, 48.0 GiB in "
        "all, but 70.0 GiB is available and 32 GiB of it stays free"
    )
    with pytest.raises(ValueError, match=re.escape(refused)) as error:
        start_kv_sharing(built, args, checkpoint_dir=str(tmp_path), max_num_tokens=64)
    assert "kv_cache_config.host_cache_size" in str(error.value)


def test_a_mount_that_fails_on_one_rank_unmounts_the_other_ranks_memory_gracefully(
    store_config: str, monkeypatch, unmount_waits: list[float]
) -> None:
    """Rank 0 of a two-rank generation server mounts its memory; rank 1's mount fails. Rank 0
    refuses too, after unmounting its segment gracefully: other servers may have used it."""
    store = FakeMooncakeStore()
    monkeypatch.setattr(startup, "open_sdk", lambda mooncake: store)
    monkeypatch.setattr(
        startup.psutil, "virtual_memory", lambda: SimpleNamespace(available=1 << 40)
    )
    mount_failed = "mooncake: mounting 34359738368 bytes of host memory failed with -1"
    rank_1_reports = iter(
        [
            (None, None),  # the memory check passed
            (mount_failed, None),  # the mount failed
        ]
    )
    dist = SimpleNamespace(
        rank=0, world_size=2, local_world_size=2, allgather=lambda own: [own, next(rank_1_reports)]
    )
    rank_0 = SimpleNamespace(dist=dist, store_segment=None)
    args = llm_args(store_config, providing(store_config, ["provide"]))
    with pytest.raises(ValueError, match=f"rank 1: {mount_failed}"):
        start_kv_sharing(rank_0, args, checkpoint_dir=None, max_num_tokens=64)
    (unmount,) = store.calls_to("unmount_and_free_segment")
    assert unmount.args == (rank_0.store_segment.segment_ids, UNMOUNT_GRACE_S)
    assert unmount_waits == [13.0]


def fail_if_called(*args, **kwargs):
    raise AssertionError("called after a refusal")


def executor(
    manager_class: type[KVCacheManagerV2] = KVCacheManagerV2,
    dtype: DataType = DataType.BF16,
    mla: bool = False,
    sparse_attention_config=None,
    block_reuse: bool = True,
    dist=None,
) -> SimpleNamespace:
    """A built executor as startup reads it, with the V2 scheduler; its KV cache manager has no
    memory."""
    manager = object.__new__(manager_class)
    manager.dtype = dtype
    manager.enable_block_reuse = block_reuse
    manager.tokens_per_block = 32
    pretrained_config = PretrainedConfig(kv_lora_rank=512, qk_rope_head_dim=64) if mla else None
    model_config = ModelConfig(
        pretrained_config=pretrained_config, sparse_attention_config=sparse_attention_config
    )
    return SimpleNamespace(
        kv_cache_manager=manager,
        scheduler=object.__new__(KVCacheV2Scheduler),
        model_engine=SimpleNamespace(model=SimpleNamespace(model_config=model_config)),
        dist=dist,
        store_segment=None,
    )

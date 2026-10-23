# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wires KV sharing into a built executor, on its construction thread before its worker starts:
the store client and its data path, the KV cache manager's lender, the transfer manager and the
orchestrator."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

import tensorrt_llm
from tensorrt_llm.bindings import DataType
from tensorrt_llm.logger import logger
from tensorrt_llm.math_utils import ceil_div

from ...disaggregation._assembly import FactoryRejected, OwnerCleanupFn, ProfileSpec, attach_domain
from ...disaggregation.base.backend import CacheKind, SlotRegion, WakeFn
from ...disaggregation.blob.backend import (
    BlobBackend,
    BlobConfig,
    ConstructionRejected,
    RegionBinding,
)
from ...disaggregation.blob.keys import key_stem
from ...disaggregation.blob.mooncake import MooncakeDriver
from ...disaggregation.blob.mooncake_native import NativeMooncakeDriver
from ...disaggregation.blob.mooncake_setup import load_store_config
from ...disaggregation.lifecycle.retirement import QuiescenceFatalEvent, RetirementWatchdog
from ...disaggregation.transfer_manager import LifecycleBinding, TransferManager
from ...weight_sharing.source_identity import SourceIdentity
from ..config_utils import is_mla
from ..hang_detector import propagate_hard_kill
from ..kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from ..kv_cache.sharing import Lender, MemoryKind, StagingOptions
from ..scheduler.scheduler_v2 import KVCacheV2Scheduler
from .data_path import open_store
from .orchestrator import (
    BINDING_ID,
    DOMAIN_ID,
    MAX_OPEN_PUBLICATIONS,
    PROFILE_ID,
    SharedKvOrchestrator,
    SharingPolicy,
)
from .probe import ProbeService

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

    from ..py_executor import PyExecutor

# A store call that still holds KV cache memory this long after it began, plus an equal grace, ends
# the process (240 s): past Mooncake's own 60 s give-up on a get, and well before the master frees an
# unfinished put's space (put_start_release_timeout_sec, 600 s by default).
STORE_ACCESS_DEADLINE_S = 120.0
# Pinned host memory each rank stages fetches and publications through, at most.
HOST_STAGING_BYTES = 8 << 30
# The attention writer layout of the TRTLLM backend, which stored blocks carry.
MHA_LAYOUT = "mha-trtllm-hnd"
# What kv_sharing serves at this milestone, said before the reason a model is refused.
SUPPORTED_MODELS = (
    "kv_sharing serves, for now, one GPU per server with a BF16 KV cache, multi-head attention "
    "with full attention in every layer (no sliding window, MLA, recurrent state or attention "
    "sinks), and under host staging one fetch of max(max_fetch_tokens, max_num_tokens) tokens "
    f"within {HOST_STAGING_BYTES >> 30} GiB"
)
# The watchdog's names for the two directions of a store call.
_STORE_CALLS = {"send": "store publication", "receive": "store fetch"}


def start_kv_sharing(
    py_executor: PyExecutor,
    llm_args: TorchLlmArgs,
    *,
    checkpoint_dir: str | None,
    max_num_tokens: int,
) -> SharedKvOrchestrator:
    """Opens the store, attaches the lender and hands the executor its orchestrator.

    Run it once, on the final executor, before its worker starts.

    Raises:
        ValueError: The executor or the model is not one kv_sharing serves, or the store config
            is invalid.
        TimeoutError: The store's master does not accept connections.
        RuntimeError: The store client or the backend could not be set up.
    """
    config = llm_args.kv_sharing
    (backend,) = config.backends
    manager = py_executor.kv_cache_manager
    if not isinstance(manager, KVCacheManagerV2) or not manager.enable_block_reuse:
        raise ValueError(
            "kv_sharing needs KV cache manager v2 with block reuse; this executor built "
            f"{type(manager).__name__} (block reuse {getattr(manager, 'enable_block_reuse', None)})"
        )
    if not isinstance(py_executor.scheduler, KVCacheV2Scheduler):
        raise ValueError(
            "kv_sharing fetches through the KV cache manager v2 scheduler; this executor built "
            f"{type(py_executor.scheduler).__name__}"
        )
    checkpoint = checkpoint_dir or llm_args.model
    if not os.path.isdir(checkpoint):
        raise ValueError(
            f"kv_sharing names blocks after a local checkpoint; {checkpoint} is not one"
        )
    blob_section, mooncake = load_store_config(backend.config)
    if mooncake.global_segment_size != 0:
        raise ValueError(
            "kv_sharing servers lend no memory to the store: set mooncake.global_segment_size to 0 "
            f"in {backend.config} and run 'trtllm-serve mooncake_provider'"
        )
    try:
        blob_config = BlobConfig(**blob_section)
    except TypeError as error:  # an unknown key
        raise ValueError(f"store config {backend.config}, section 'blob': {error}") from error
    # Before the lender pins its staging memory: an unreachable master fails here, within a minute.
    memory_kind, driver = open_store(config.data_path, mooncake, blob_config)

    tokens_per_block = manager.tokens_per_block
    fetch_tokens = max(config.max_fetch_tokens, max_num_tokens)
    if memory_kind is MemoryKind.DEVICE:
        # Nothing is staged; the lender still sets one block aside per pool group.
        staging = StagingOptions(guaranteed_fetch_tokens=tokens_per_block)
    else:
        # Host staging holds one fetch, and publications the rest of two fetches within the cap.
        staging = StagingOptions(
            guaranteed_fetch_tokens=fetch_tokens,
            capacity_in_fetches=2,
            max_bytes=HOST_STAGING_BYTES,
        )
    try:
        profile = _profile(py_executor, llm_args, checkpoint, staging)
        domain = attach_domain(manager, profile, memory_kind=memory_kind)
    except (TypeError, ValueError) as error:
        # The lender may stay attached; startup fails, so nothing lends through it.
        raise ValueError(f"{SUPPORTED_MODELS}; this server is refused: {error}") from error
    if memory_kind is MemoryKind.DEVICE:
        publication_tokens = None
        path = f"GPU Direct, at most {MAX_OPEN_PUBLICATIONS} publications locking pages at once"
    else:
        publication_tokens = _publication_tokens(domain.lender, fetch_tokens, tokens_per_block)
        staging_bytes = sum(
            r.nbytes for r in domain.lender.regions if r.memory_kind is MemoryKind.HOST
        )
        path = (
            f"host staging of {staging_bytes >> 20} MiB: one fetch of {fetch_tokens} tokens and "
            f"{publication_tokens} tokens for publications"
        )

    transfers = TransferManager(wake=_nothing_to_wake, hold_returns=True)
    watchdog = RetirementWatchdog(_end_process)
    factory = _BlobBackendFactory(driver, domain.lender)
    transfers.assemble(
        BINDING_ID,
        domain.namespace,
        PROFILE_ID,
        factory,
        domain_id=DOMAIN_ID,
        lender=domain.lender,
        lifecycle=LifecycleBinding(watchdog, STORE_ACCESS_DEADLINE_S),
        provider_cleanup_complete=factory.hold_release_posted,
        config=blob_config,
    )
    watchdog.start()
    py_executor.disagg.attach_shared_transfer_manager(transfers)

    policy = SharingPolicy(
        memory_kind=memory_kind,
        fetch="fetch" in backend.roles,
        publish="publish" in backend.roles,
        min_fetch_tokens=config.min_fetch_tokens,
        max_fetch_tokens=config.max_fetch_tokens,
        fetch_timeout_s=config.fetch_timeout_s,
        publication_tokens=publication_tokens,
        max_util_for_resume=llm_args.kv_cache_config.max_util_for_resume,
    )
    probe = None
    if policy.fetch and py_executor.dist.rank == 0:
        probe = ProbeService(
            domain.lender,
            driver,
            key_stem(blob_config.key_prefix),
            tokens_per_block=tokens_per_block,
            # A fetch leaves the prompt's last token to compute.
            min_prompt_tokens=config.min_fetch_tokens + 1,
            hold_s=config.probe_hold_ms / 1000,
        )
        py_executor.executor_request_queue.probe_on_enqueue = probe.submit
    orchestrator = SharedKvOrchestrator(
        policy=policy,
        kv_cache_manager=manager,
        lender=domain.lender,
        transfers=transfers,
        coordinator=py_executor.disagg,
        execution_stream=py_executor.execution_stream,
        watchdog=watchdog,
        probe=probe,
    )
    py_executor.kv_sharing = orchestrator
    py_executor.scheduler.store_fetches = orchestrator
    store_calls = "native" if isinstance(driver, NativeMooncakeDriver) else "python"
    logger.info(
        f"kv_sharing: store master {mooncake.master_server_address} over {mooncake.protocol}, "
        f"roles {backend.roles}, {store_calls} store driver, {path}"
    )
    return orchestrator


def _publication_tokens(lender: Lender, fetch_tokens: int, tokens_per_block: int) -> int:
    """Tokens open publications may hold, whole blocks: host staging less one fetch.

    Raises:
        ValueError: Host staging holds no block beside one fetch.
    """
    regions = [region for region in lender.regions if region.memory_kind is MemoryKind.HOST]
    # A block takes one slot of its pool group's region for each layer group the region serves.
    staged_blocks = min(region.slot_count // len(region.layer_group_indices) for region in regions)
    fetch_blocks = ceil_div(fetch_tokens, tokens_per_block)
    if staged_blocks <= fetch_blocks:
        staged_gib = sum(region.nbytes for region in regions) / (1 << 30)
        raise ValueError(
            f"kv_sharing: {staged_gib:.1f} GiB of host staging hold one fetch of {fetch_tokens} "
            "tokens, max(max_fetch_tokens, max_num_tokens), and no block for publications; "
            "lower max_fetch_tokens or max_num_tokens"
        )
    return (staged_blocks - fetch_blocks) * tokens_per_block


def _profile(
    py_executor: PyExecutor, llm_args: TorchLlmArgs, checkpoint: str, staging: StagingOptions
) -> ProfileSpec:
    """The facts that scope the stored blocks' names: the checkpoint and its effective config,
    the release and the GPU that computed the bytes, the writer layout and the cache dtype."""
    manager = py_executor.kv_cache_manager
    model_config = py_executor.model_engine.model.model_config
    mapping = model_config.mapping
    major, minor = torch.cuda.get_device_capability()
    compatibility = (
        f"trtllm-{tensorrt_llm.__version__}/tp{mapping.tp_size}-pp{mapping.pp_size}"
        f"-cp{mapping.cp_size}-adp{int(mapping.enable_attention_dp)}/sm{major}{minor}"
    )
    logger.info(f"kv_sharing: reading every file of {checkpoint} to name its stored blocks")
    return ProfileSpec(
        id=PROFILE_ID,
        lifecycle_profile_id="executor-fatal",
        source_identity=SourceIdentity.from_model_config(model_config, checkpoint_dir=checkpoint),
        compatibility_id=compatibility,
        model_layout="mla" if is_mla(model_config.pretrained_config) else MHA_LAYOUT,
        dtype="bfloat16" if manager.dtype == DataType.BF16 else str(manager.dtype),
        topology=(mapping.tp_size, 1, mapping.pp_size, mapping.cp_size),
        state_kinds=(CacheKind.PAGED,) * manager.num_pools,
        full_attention=True,
        sparse_attention=llm_args.sparse_attention_config is not None,
        sink_token_length=llm_args.kv_cache_config.sink_token_length or 0,
        lora_enabled=False,
        pool_count=len(manager.impl.pool_group_descs),
        speculative_mode="none",
        model_role="target",
        draft_pool=None,
        joint_reuse=False,
        staging=staging,
    )


class _BlobBackendFactory:
    """Builds the blob backend over the store driver when the transfer manager assembles it: takes
    the lender's region hold and registers the lender's staging regions with the store."""

    def __init__(self, driver: MooncakeDriver, lender: Lender) -> None:
        self._driver = driver
        self._lender = lender
        self._backend: _HoldReleasingBackend | None = None

    def __call__(
        self,
        regions: tuple[SlotRegion, ...],
        wake: WakeFn,
        config: BlobConfig,
        post_cleanup: OwnerCleanupFn,
    ) -> BlobBackend:
        binding = RegionBinding(regions, self._driver, self._lender.hold_regions())
        try:
            binding.register()
        except ConstructionRejected as error:  # nothing registered, the hold released
            raise FactoryRejected(str(error)) from error
        self._backend = _HoldReleasingBackend(binding, wake, config, post_cleanup)
        return self._backend

    def hold_release_posted(self) -> bool:
        """Whether the backend handed the region hold's release to the executor thread."""
        return self._backend is not None and self._backend.hold_release_posted


class _HoldReleasingBackend(BlobBackend):
    """The blob backend, which also gives the lender's region hold back once it shut down with
    nothing registered or accessed. The lender takes holds back only on the executor thread, so the
    release is posted there."""

    def __init__(
        self,
        binding: RegionBinding,
        wake: WakeFn,
        config: BlobConfig,
        post_cleanup: OwnerCleanupFn,
    ) -> None:
        super().__init__(binding, wake, config)
        self._region_binding = binding
        self._post_cleanup = post_cleanup
        # Set once on the transfer manager's shutdown thread, read on the executor thread.
        self.hold_release_posted = False

    def shutdown(self, timeout: float) -> frozenset[str]:
        labels = super().shutdown(timeout)
        if not labels and not self.hold_release_posted:
            self.hold_release_posted = self._post_cleanup(self._region_binding.release_hold)
        return labels


def _nothing_to_wake() -> None:
    """The transfer manager's wake. Nothing sleeps on store work: while any is pending, the
    executor loop polls the request queue instead of waiting on it."""


def _end_process(event: QuiescenceFatalEvent) -> None:
    """A store call may still write KV cache memory past its deadline; only ending the process is
    safe. Runs on the watchdog thread or inside the executor loop, so it takes no logging lock."""
    elapsed = event.expired_at - event.started_at
    # The transfer manager watches its backend's construction and teardown as request -1.
    call = "store backend teardown" if event.request_id < 0 else _STORE_CALLS[event.direction]
    message = (
        f"kv_sharing: a {call} still held KV cache memory {elapsed:.0f} s after it began "
        f"({event.reason}); ending the process.\n"
    )
    os.write(2, message.encode())
    propagate_hard_kill(diagnostics=False)

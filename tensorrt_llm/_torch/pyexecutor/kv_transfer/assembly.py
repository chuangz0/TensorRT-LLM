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
"""Assembling the KV transfer layer onto one ``PyExecutor`` (design §7.4, plan §7-§9).

``attach_kv_transfer`` runs once at creation when ``TRTLLM_KV_TRANSFER_CONFIG`` is set: it checks
the engine is in scope, builds the resource reader and region resolver, builds the configured
backends and registers the KV pools with those that need it, builds the coordinator, and attaches
the ``KVTransferEngineBinding`` to the executor and its scheduler.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Sequence

from tensorrt_llm.logger import logger

from ...disaggregation.backends.config import KVTransferConfig, load_kv_transfer_config
from ...disaggregation.backends.registry import (
    BackendBuildContext,
    BackendHandle,
    build_backends,
    close_backends,
)
from ...disaggregation.orchestration.kv_transfer.coordinator import KVTransferCoordinator
from ...disaggregation.orchestration.kv_transfer.interfaces import FetchSource, GroupKind
from ...disaggregation.remote_cache import Planner
from ...disaggregation.resource.kv_extractor import build_page_table_from_manager
from ...disaggregation.resource.kv_v2_reader import KVv2ResourceReader
from ...disaggregation.resource.region import (
    KVv2RegionResolver,
    layout_fingerprint,
    parallel_shard_tag,
)
from ..kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from .binding import KVTransferEngineBinding
from .effects import EngineDist, EngineWorkQueue, PyExecutorKVTransferEffects

if TYPE_CHECKING:
    from ..py_executor import PyExecutor

__all__ = ["KV_TRANSFER_STATUS_DUMP_ENV", "attach_kv_transfer", "check_engine_supports_kv_transfer"]

KV_TRANSFER_STATUS_DUMP_ENV = "TRTLLM_KV_TRANSFER_STATUS_DUMP"
"""Test seam: a path (``{pid}`` replaced by the worker's pid) where ``close`` writes a JSON dump."""


def _refuse(reason: str) -> None:
    raise ValueError(
        f"KV transfer backends are configured but the engine cannot host them: {reason}"
    )


def check_engine_supports_kv_transfer(
    executor: PyExecutor,
    *,
    mapping,
    spec_config,
    kv_connector_manager,
    max_beam_width: int,
) -> None:
    """The scope guard of plan §1 item 7: raise ``ValueError`` naming the first unmet condition."""
    from ..scheduler.scheduler_v2 import KVCacheV2Scheduler

    kv_cache_manager = executor.kv_cache_manager
    if not isinstance(kv_cache_manager, KVCacheManagerV2):
        _refuse("kv_cache_config.use_kv_cache_manager_v2=True is required")
    if not kv_cache_manager.enable_block_reuse:
        _refuse("kv_cache_config.enable_block_reuse=True is required")
    if mapping.cp_size != 1:
        _refuse(
            f"CP=1 is required, got cp={mapping.cp_size}: context parallelism splits the "
            "sequence, so block ordinals do not name the same content on every rank"
        )
    if mapping.pp_size != 1:
        _refuse(f"PP=1 is required, got pp={mapping.pp_size}")
    if spec_config is not None:
        _refuse("speculative decoding is not supported")
    if kv_connector_manager is not None:
        _refuse("a KV connector is attached; remove kv_connector_config")
    if executor.draft_kv_cache_manager is not None:
        _refuse("a draft KV cache manager is present")
    if max_beam_width != 1:
        _refuse("beam search is not supported")
    if getattr(executor, "enable_kv_pool_rebalance", False):
        _refuse(
            "kv_cache_config.enable_kv_pool_rebalance=True is not supported: a rebalance moves "
            "pages under a transfer still in flight"
        )
    if not isinstance(executor.scheduler, KVCacheV2Scheduler):
        _refuse(
            f"the scheduler must be KVCacheV2Scheduler, got {type(executor.scheduler).__name__}"
        )


def _check_layer_groups_are_full_attention(reader: KVv2ResourceReader) -> None:
    for group_spec in reader.group_specs():
        if group_spec.kind is not GroupKind.PAGED or group_spec.window_size is not None:
            _refuse("only full-attention models are supported (no sliding window, no SSM)")


def _register_kv_pools(backends: Sequence[BackendHandle], resolver: KVv2RegionResolver) -> None:
    """Hand every KV pool to each backend whose transport needs it registered."""
    try:
        for handle in backends:
            if handle.pool_registrar is None:
                continue
            for address, size in resolver.pool_memory_spans():
                handle.pool_registrar.register_pool(address, size)
    except Exception:
        close_backends(backends)
        raise


def _build_coordinator(
    config: KVTransferConfig,
    backends: Sequence[BackendHandle],
    reader: KVv2ResourceReader,
    effects: PyExecutorKVTransferEffects,
    dist,
    *,
    attention_dp: bool,
) -> KVTransferCoordinator:
    fetch_sources = [
        FetchSource(handle.name, handle.fetcher, handle.hint_key)
        for handle in backends
        if handle.fetcher is not None
    ]
    publishers = [handle.publisher for handle in backends if handle.publisher is not None]
    # The wall-clock budget alone governs the wait for a store lookup; the planner's clock is
    # left at its default, which is the clock the binding passes to ``advance`` (plan §10 #15).
    planner = Planner(
        fetch_sources,
        reader,
        reader.tokens_per_block,
        probe_budget_rounds=None,
        probe_timeout_s=config.probe_timeout_s,
    )
    return KVTransferCoordinator(
        fetch_sources,
        publishers,
        planner,
        reader,
        effects,
        EngineWorkQueue(),
        dist,
        fetch_timeout_s=config.fetch_timeout_s,
        publish_timeout_s=config.publish_timeout_s,
        unlaunched_timeout_s=config.unlaunched_timeout_s,
        attention_dp=attention_dp,
    )


def _status_dump_path() -> str | None:
    template = os.environ.get(KV_TRANSFER_STATUS_DUMP_ENV)
    if not template:
        return None
    return template.replace("{pid}", str(os.getpid()))


def attach_kv_transfer(
    executor: PyExecutor,
    config_path: str,
    *,
    mapping,
    spec_config,
    kv_connector_manager,
    max_beam_width: int,
) -> KVTransferEngineBinding:
    """Guard, build, register, attach. Raises ``ValueError`` for a configuration out of scope or
    a malformed config file.

    Disaggregated serving may be on at the same time: the transceiver keeps owning gen-init
    receives and context sends; this layer adds store fetch and publish for context requests.
    """
    started_at = time.time()
    check_engine_supports_kv_transfer(
        executor,
        mapping=mapping,
        spec_config=spec_config,
        kv_connector_manager=kv_connector_manager,
        max_beam_width=max_beam_width,
    )
    config = load_kv_transfer_config(config_path)

    kv_cache_manager = executor.kv_cache_manager
    page_table = build_page_table_from_manager(kv_cache_manager)
    reader = KVv2ResourceReader(kv_cache_manager, page_table)
    _check_layer_groups_are_full_attention(reader)
    resolver = KVv2RegionResolver(page_table)
    build_context = BackendBuildContext(
        resolver=resolver,
        layout_fingerprint=layout_fingerprint(
            kv_cache_manager, page_table, parallel_shard=parallel_shard_tag(mapping)
        ),
        max_unit_bytes=resolver.max_unit_bytes(),
        device_index=executor.device_id,
    )
    backends = build_backends(config, build_context)
    _register_kv_pools(backends, resolver)

    effects = PyExecutorKVTransferEffects(executor)
    coordinator = _build_coordinator(
        config,
        backends,
        reader,
        effects,
        EngineDist(executor.dist, mapping),
        attention_dp=mapping.enable_attention_dp,
    )
    binding = KVTransferEngineBinding(
        executor,
        coordinator,
        effects,
        reader,
        backends,
        config.backends,
        close_timeout_s=config.close_timeout_s,
        status_dump_path=_status_dump_path(),
        started_at=started_at,
    )
    executor.scheduler.kv_transfer_planner = binding
    executor.kv_transfer = binding
    logger.info(
        "KV transfer attached: backends=%s pools=%d layout=%s",
        [(entry.name, entry.type, sorted(entry.roles)) for entry in config.backends],
        len(resolver.pool_memory_spans()),
        build_context.layout_fingerprint.hex(),
    )
    return binding

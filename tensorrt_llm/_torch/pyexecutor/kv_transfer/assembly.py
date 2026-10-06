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
"""Assembling the KV transfer layer onto one ``PyExecutor``.

``attach_kv_transfer`` runs once at creation when ``TRTLLM_KV_TRANSFER_CONFIG`` is set: it checks
the engine is in scope, builds the resource reader and region resolver, names the model for the
store's namespace, builds the configured backends and registers the KV pools with those that
need it, builds the coordinator, forwards the import-light layers' stdlib logging to the TRT-LLM
logger, and attaches the ``KVTransferHooks`` to the executor and its scheduler.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import TYPE_CHECKING, Callable, NamedTuple, Sequence, TypeVar

from tensorrt_llm.logger import logger

from ...disaggregation.backends.config import load_kv_transfer_config
from ...disaggregation.backends.registry import (
    BackendBuildContext,
    BackendHandle,
    build_backends,
    close_backends,
)
from ...disaggregation.base.backend import CacheKind
from ...disaggregation.orchestration.kv_transfer.build import build_coordinator
from ...disaggregation.orchestration.kv_transfer.interfaces import PlanAuthority
from ...disaggregation.resource.kv_extractor import build_page_table_from_manager
from ...disaggregation.resource.kv_v2_reader import KVv2ResourceReader
from ...disaggregation.resource.naming import GROUP_TAG_BYTES
from ...disaggregation.resource.region import (
    KVv2RegionResolver,
    layout_fingerprint,
    parallel_shard_tag,
)
from ..kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from .effects import EngineCollective, EngineWorkQueue, PyExecutorKVTransferEffects
from .hooks import KVTransferHooks

if TYPE_CHECKING:
    from ..py_executor import PyExecutor

__all__ = [
    "KV_TRANSFER_STATUS_DUMP_ENV",
    "attach_kv_transfer",
    "check_engine_supports_kv_transfer",
    "install_log_forwarding",
    "model_identity_for",
    "plan_authority_for",
]

KV_TRANSFER_STATUS_DUMP_ENV = "TRTLLM_KV_TRANSFER_STATUS_DUMP"
"""Test seam: a path (``{pid}`` replaced by the worker's pid) where ``close`` writes a JSON dump."""

_DISAGGREGATION_LOGGER_NAME = "tensorrt_llm._torch.disaggregation"
"""The stdlib logger namespace of the import-light layers below the engine (coordinator, backends).
They log through ``logging`` because they do not import ``tensorrt_llm``; the engine forwards
their WARNING+ records to the TRT-LLM logger so a store outage shows up in the engine's log."""


class _ForwardToTrtllmLogger(logging.Handler):
    """Re-emits each stdlib record it receives through ``tensorrt_llm.logger.logger`` at the
    matching severity. Installed once per process by ``install_log_forwarding``."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - a malformed record must not break the caller's log
            self.handleError(record)
            return
        if record.levelno >= logging.CRITICAL:
            logger.critical(message)
        elif record.levelno >= logging.ERROR:
            logger.error(message)
        else:
            logger.warning(message)


def install_log_forwarding() -> None:
    """Attach the forwarding handler to the disaggregation logger namespace unless one is there
    already. Process-wide and idempotent: the handler outlives any one engine, so nothing
    removes it."""
    target = logging.getLogger(_DISAGGREGATION_LOGGER_NAME)
    if not any(isinstance(handler, _ForwardToTrtllmLogger) for handler in target.handlers):
        target.addHandler(_ForwardToTrtllmLogger())


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
    """The scope guard: raise ``ValueError`` naming the first unmet condition."""
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


def plan_authority_for(mapping) -> PlanAuthority:
    """Who plans on this rank, from the loop the engine runs.

    Without pipeline parallelism every rank runs the scheduler and the answers are voted. With
    it, the rank that calls ``_schedule`` in ``_pp_schedule_and_propagate`` owns the answers:
    rank 0, or under attention DP every first pipeline rank (one per replica, whose collective
    is its own pipeline group). Every other rank, the owner's tensor-parallel peers included,
    receives the schedule and follows.

    The owner judges a store answer over its own layer groups; a follower on a pipeline stage
    whose groups differ (one holding only windowed layers, say) rebuilds the ask over its own and
    may name units the store never held. That fetch comes up short and, after one retry, the
    request computes locally; no stage hangs. Gathering every stage's group set at attach time
    to refuse such a model would trade an occasional local compute for a refused engine.
    """
    if mapping.pp_size == 1:
        return PlanAuthority.VOTED
    schedules_for_its_replica = mapping.enable_attention_dp and mapping.tp_size > 1
    if mapping.rank == 0 or (mapping.pp_rank == 0 and schedules_for_its_replica):
        return PlanAuthority.OWNER
    return PlanAuthority.FOLLOWER


_CONFIG_KEYS_NOT_PART_OF_THE_MODEL = ("transformers_version", "_name_or_path", "_commit_hash")
"""Keys of a Hugging Face config that change without the model changing."""


def _pretrained_config_of(executor: PyExecutor):
    """The Hugging Face config behind the engine's model, or ``None`` on an engine without one."""
    model = getattr(getattr(executor, "model_engine", None), "model", None)
    return getattr(getattr(model, "model_config", None), "pretrained_config", None)


def _digest_of_config(config) -> str:
    """A stable digest of a Hugging Face config's contents, naming the model when its path
    does not."""
    contents = {
        key: value
        for key, value in config.to_dict().items()
        if key not in _CONFIG_KEYS_NOT_PART_OF_THE_MODEL
    }
    serialized = json.dumps(contents, sort_keys=True, default=str).encode()
    return hashlib.blake2b(serialized, digest_size=8).hexdigest()


def model_identity_for(executor: PyExecutor) -> str:
    """What names the model whose bytes the store holds, for the layout fingerprint: two models
    of one KV geometry must not read each other's pages under one namespace.

    A checkpoint the config knows the commit hash of (a hub revision) is named by the last
    component of its name or path plus that hash, so nodes with different ``HF_HOME`` snapshot
    paths agree on it. One without a commit hash is named by its name or path as given: the
    identity is then per path, so nodes that should share a store must share the checkpoint
    layout, or the backend must be given an explicit namespace. Without a name, the
    architecture plus a digest of the config. An engine without a model config gives the empty
    identity, which the fingerprint treats as unknown: such models share keys, so it is warned
    about.
    """
    config = _pretrained_config_of(executor)
    if config is None:
        logger.warning(
            "KV transfer: the engine has no model config; the store namespace does not name "
            "the model"
        )
        return ""
    name = getattr(config, "_name_or_path", None) or ""
    commit = getattr(config, "_commit_hash", None)
    if name and commit:
        return f"{os.path.basename(os.path.normpath(name))}@{commit}"
    if name:
        return name
    architecture = (getattr(config, "architectures", None) or ["unknown"])[0]
    return f"{architecture}#{_digest_of_config(config)}"


def _check_layer_groups_are_paged(reader: KVv2ResourceReader) -> None:
    """Every layer group must be paged: full attention or sliding window (with or without sink
    blocks), whose blocks the store path names and fetches per group. A recurrent group is
    refused because nothing publishes its state snapshots.

    A model without sliding attention given windows through ``kv_cache_config.max_attention_window``
    (a Llama, say) passes too: KV v2 drops pages by that window while attention masks nothing,
    and the store path follows the cache's own life cycle exactly as the local path does.
    """
    for group_spec in reader.group_specs():
        if group_spec.kind is not CacheKind.PAGED:
            _refuse(
                "SSM/recurrent layer groups are not supported: the store path names no state "
                "snapshots"
            )


def _check_followers_can_rebuild_plans(mapping, backends: Sequence[BackendHandle]) -> None:
    """Under PP the followers rebuild every plan from ``(token_end, source)`` alone; a routed
    backend's plan also needs the request's hint, which is not on the wire."""
    if mapping.pp_size > 1 and any(handle.hint_key is not None for handle in backends):
        _refuse(
            "PP>1 with a routed (hint) backend: a follower cannot rebuild a routed plan from "
            "(token_end, source)"
        )


def _register_kv_pools(backends: Sequence[BackendHandle], resolver: KVv2RegionResolver) -> None:
    """Hand every KV pool to each backend whose transport needs it registered."""
    for handle in backends:
        if handle.pool_registrar is None:
            continue
        for address, size in resolver.pool_memory_spans():
            handle.pool_registrar.register_pool(address, size)


T = TypeVar("T")


def _closing_backends_on_failure(backends: Sequence[BackendHandle], assemble: Callable[[], T]) -> T:
    """Run the assembly steps that follow ``build_backends``; a failure in any of them closes
    the built backends before it propagates, so no transport thread outlives a refused engine."""
    try:
        return assemble()
    except Exception:
        close_backends(backends)
        raise


def _unit_bytes_by_name(
    reader: KVv2ResourceReader, resolver: KVv2RegionResolver
) -> Callable[[bytes], int]:
    """Size of a unit from its name: the name starts with its layer group's tag, and every unit
    of a group is one page of that group's pools. For a backend that lands units in its own
    memory before their pages exist."""
    bytes_by_tag = {
        spec.tag: sum(size for _, size in resolver(spec.local_group, 0))
        for spec in reader.group_specs()
    }

    def unit_bytes(name: bytes) -> int:
        try:
            return bytes_by_tag[name[:GROUP_TAG_BYTES]]
        except KeyError:
            raise KeyError(f"unit {name.hex()} belongs to no layer group of this rank") from None

    return unit_bytes


def _status_dump_path() -> str | None:
    template = os.environ.get(KV_TRANSFER_STATUS_DUMP_ENV)
    if not template:
        return None
    return template.replace("{pid}", str(os.getpid()))


class _ResourceViews(NamedTuple):
    """This rank's KV cache as the transfer layer sees it: read by the coordinator through
    ``reader``, addressed by the backends through ``resolver`` and ``build_context``."""

    reader: KVv2ResourceReader
    resolver: KVv2RegionResolver
    model_identity: str
    build_context: BackendBuildContext


def _build_resource_views(executor: PyExecutor, config, mapping) -> _ResourceViews:
    """Build the reader and the region resolver over the engine's KV cache manager, refusing a
    model with a recurrent layer group, and gather what every backend factory needs from them."""
    kv_cache_manager = executor.kv_cache_manager
    page_table = build_page_table_from_manager(kv_cache_manager)
    reader = KVv2ResourceReader(kv_cache_manager, page_table)
    _check_layer_groups_are_paged(reader)
    resolver = KVv2RegionResolver(page_table)
    model_identity = model_identity_for(executor)
    build_context = BackendBuildContext(
        resolver=resolver,
        layout_fingerprint=layout_fingerprint(
            kv_cache_manager,
            page_table,
            parallel_shard=parallel_shard_tag(mapping),
            model_identity=model_identity,
        ),
        max_unit_bytes=resolver.max_unit_bytes(),
        device_index=executor.device_id,
        unit_bytes_of=_unit_bytes_by_name(reader, resolver),
        max_request_blocks=-(-executor.max_seq_len // reader.tokens_per_block),
        landing_wait_timeout_s=config.landing_wait_timeout_s,
    )
    return _ResourceViews(reader, resolver, model_identity, build_context)


def _build_hooks(
    executor: PyExecutor,
    config,
    views: _ResourceViews,
    backends: Sequence[BackendHandle],
    *,
    mapping,
    started_at: float,
) -> KVTransferHooks:
    """Check the built backends fit the engine's parallelism, register the KV pools with those
    that need it, build the coordinator over them, and attach the hooks to the executor and its
    scheduler."""
    _check_followers_can_rebuild_plans(mapping, backends)
    _register_kv_pools(backends, views.resolver)
    effects = PyExecutorKVTransferEffects(executor)
    coordinator = build_coordinator(
        config,
        backends,
        views.reader,
        effects,
        EngineWorkQueue(),
        EngineCollective(executor.dist, mapping),
        plan_authority=plan_authority_for(mapping),
    )
    install_log_forwarding()
    hooks = KVTransferHooks(
        executor,
        coordinator,
        effects,
        backends,
        config.backends,
        close_timeout_s=config.close_timeout_s,
        status_dump_path=_status_dump_path(),
        started_at=started_at,
        rank=mapping.rank,
    )
    executor.scheduler.kv_transfer_hooks = hooks
    executor.kv_transfer = hooks
    logger.info(
        "KV transfer attached: backends=%s pools=%d model=%r layout=%s plan_authority=%s",
        [(entry.name, entry.type, sorted(entry.roles)) for entry in config.backends],
        len(views.resolver.pool_memory_spans()),
        views.model_identity,
        views.build_context.layout_fingerprint.hex(),
        coordinator.plan_authority.value,
    )
    return hooks


def attach_kv_transfer(
    executor: PyExecutor,
    config_path: str,
    *,
    mapping,
    spec_config,
    kv_connector_manager,
    max_beam_width: int,
) -> KVTransferHooks:
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
    views = _build_resource_views(executor, config, mapping)
    backends = build_backends(config, views.build_context)
    return _closing_backends_on_failure(
        backends,
        lambda: _build_hooks(
            executor, config, views, backends, mapping=mapping, started_at=started_at
        ),
    )

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wires KV sharing into a built executor, on its construction thread before its worker starts:
the store client and its data path, the store segment of host memory the server provides, the KV
cache manager's lender, the transfer manager and the orchestrator. A server that only provides
memory (roles [provide]) gets the store client and its segment alone.

Every rank of a tensor-parallel server runs it. Each step that can fail on one rank alone ends in
one collective every rank reaches, and every rank then raises naming each failure: a rank raising
alone would leave its peers waiting in the next collective.
"""

from __future__ import annotations

import functools
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, TextIO, TypeVar

import psutil
import torch

import tensorrt_llm
from tensorrt_llm.bindings import DataType
from tensorrt_llm.logger import logger
from tensorrt_llm.math_utils import ceil_div

from ...disaggregation._assembly import (
    SPARSE_ALGORITHMS,
    SUPPORTED_CACHES,
    FactoryRejected,
    OwnerCleanupFn,
    ProfileSpec,
    attach_domain,
)
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
from ...disaggregation.blob.mooncake_setup import (
    MooncakeConfig,
    StoreSegment,
    load_store_config,
    make_store_driver,
    open_sdk,
)
from ...disaggregation.lifecycle.retirement import QuiescenceFatalEvent, RetirementWatchdog
from ...disaggregation.transfer_manager import LifecycleBinding, TransferManager
from ...distributed.communicator import ReduceOp
from ...weight_sharing.artifact_identity import ArtifactIdentity
from ...weight_sharing.source_identity import SourceIdentity
from ..config_utils import is_mla
from ..connectors.mooncake_store.gpudirect import allocation_range
from ..connectors.mooncake_store.validation import NODE_BUDGET_RESERVE_BYTES
from ..hang_detector import propagate_hard_kill
from ..kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from ..kv_cache.mamba_cache_manager import MambaHybridCacheManagerV2
from ..kv_cache.sharing import Lender, MemoryKind, StagingOptions
from ..scheduler.scheduler_v2 import KVCacheV2Scheduler
from .data_path import check_store, choose_data_path, open_store
from .orchestrator import (
    BINDING_ID,
    DOMAIN_ID,
    MAX_OPEN_PUBLICATIONS,
    PROFILE_ID,
    SharedKvOrchestrator,
    SharingPolicy,
    StagingBudget,
)
from .probe import ProbeService
from .replicas import ReplicaSet, replica_set

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import BaseSparseAttentionConfig, TorchLlmArgs

    from ...distributed import Distributed
    from ..py_executor import PyExecutor

_Local = TypeVar("_Local")
_Shared = TypeVar("_Shared")

# A store call that still holds KV cache memory this long after it began, plus an equal grace, ends
# the process (240 s): past Mooncake's own 60 s give-up on a get, and well before the master frees an
# unfinished put's space (put_start_release_timeout_sec, 600 s by default).
STORE_ACCESS_DEADLINE_S = 120.0
# Pinned host memory each rank stages fetches and publications through, at most.
HOST_STAGING_BYTES = 8 << 30
# The attention writer layouts of the TRTLLM backend, which stored blocks carry.
MHA_LAYOUT = "mha-trtllm-hnd"
MLA_LAYOUT = "mla-trtllm-latent"
# The cache dtypes stored blocks carry, by the KV cache manager's dtype.
_CACHE_DTYPES = {
    DataType.BF16: "bfloat16",
    DataType.HALF: "float16",
    DataType.FP8: "fp8",
    DataType.NVFP4: "nvfp4",
}
# Prefixes the reason a model is refused.
SUPPORTED_MODELS = (
    "kv_sharing serves one GPU, tensor parallelism or attention DP, with multi-head "
    "attention (BF16, FP16, FP8 or NVFP4 KV cache) with full attention or sliding windows, or MLA "
    "(BF16 or FP8 KV cache) with full attention or DeepSeek sparse attention (DSA); no recurrent "
    "state, other sparse attention or attention sink tokens; and under host staging one fetch of "
    f"max(max_fetch_tokens, max_num_tokens) tokens within {HOST_STAGING_BYTES >> 30} GiB"
)
# Where each rank writes its round trace (rank<r>.txt), for multi-GPU tests to compare.
TRACE_DIR_ENV = "TRTLLM_KV_SHARING_TRACE_DIR"
# The watchdog's names for the two directions of a store call.
_STORE_CALLS = {"send": "store publication", "receive": "store fetch"}


def start_kv_sharing(
    py_executor: PyExecutor,
    llm_args: TorchLlmArgs,
    *,
    checkpoint_dir: str | None,
    max_num_tokens: int,
) -> None:
    """Shares KV cache blocks through the store or, on a server whose only role is 'provide',
    gives the store host memory and nothing else.

    Run it once, on the final executor, before its worker starts, on every rank. Other servers may
    use a rank's store segment as soon as it is mounted, so a start that fails afterwards unmounts
    it gracefully before raising, about 13 s later.

    Raises:
        ValueError: The executor or the model is not one kv_sharing serves, the store config is
            invalid, the node has too little host memory, or a step failed on some rank (every
            rank raises, naming each failure).
    """
    try:
        if llm_args.kv_sharing.only_provides:
            _provide_store_memory(py_executor, llm_args)
        else:
            _share_blocks(
                py_executor, llm_args, checkpoint_dir=checkpoint_dir, max_num_tokens=max_num_tokens
            )
    except Exception:
        segment = py_executor.store_segment
        if segment is not None:
            segment.begin_unmount()
            segment.wait_until_freed()
        raise


def _share_blocks(
    py_executor: PyExecutor,
    llm_args: TorchLlmArgs,
    *,
    checkpoint_dir: str | None,
    max_num_tokens: int,
) -> None:
    """Opens the store (mounting this rank's segment under 'provide'), attaches the lender and
    hands the executor its orchestrator."""
    config = llm_args.kv_sharing
    (backend,) = config.backends
    manager = py_executor.kv_cache_manager
    dist = py_executor.dist
    if not isinstance(manager, KVCacheManagerV2):
        raise ValueError(
            f"kv_sharing needs KV cache manager v2; this executor built {type(manager).__name__}"
        )
    if not isinstance(py_executor.scheduler, KVCacheV2Scheduler):
        raise ValueError(
            "kv_sharing fetches through the KV cache manager v2 scheduler; this executor built "
            f"{type(py_executor.scheduler).__name__}"
        )
    # Every rank reads these model facts alike, so each refuses on its own, before any collective,
    # checkpoint read or store work.
    unserved: list[str] = []
    layout, cache_dtype = _stored_cache(py_executor)
    if (layout, cache_dtype) not in SUPPORTED_CACHES:
        unserved.append(f"a KV cache of layout {layout} and dtype {cache_dtype}")
    sparse = py_executor.model_engine.model.model_config.sparse_attention_config
    if sparse is not None and sparse.algorithm not in SPARSE_ALGORITHMS:
        unserved.append(f"{sparse.algorithm} sparse attention")
    if isinstance(manager, MambaHybridCacheManagerV2):
        unserved.append("recurrent state")
    if unserved:
        raise ValueError(
            f"{SUPPORTED_MODELS}; this server is refused: the model has {' and '.join(unserved)}"
        )
    if not manager.enable_block_reuse:
        raise ValueError(
            "kv_sharing needs block reuse, which TensorRT-LLM turned off for this model; a warning "
            "in its log says why"
        )
    checkpoint = checkpoint_dir or llm_args.model

    def prepare() -> tuple[tuple[MooncakeConfig, BlobConfig], Optional[ArtifactIdentity]]:
        mooncake, blob_config = _store_config(backend.config)
        if backend.segment_size:
            # Host staging is not chosen yet: count its cap.
            _check_node_memory(
                backend.segment_size,
                staging_bytes=HOST_STAGING_BYTES,
                ranks_on_node=dist.local_world_size,
            )
        if dist.rank != 0:
            return (mooncake, blob_config), None
        if not os.path.isdir(checkpoint):
            raise ValueError(
                f"kv_sharing names blocks after a local checkpoint; {checkpoint} is not one"
            )
        logger.info(f"kv_sharing: reading every file of {checkpoint} to name its stored blocks")
        return (mooncake, blob_config), ArtifactIdentity.from_checkpoint(checkpoint)

    def open_and_check() -> tuple[Any, Optional[str]]:
        store, gpu_direct_error = open_store(config.data_path, mooncake)
        if backend.segment_size:
            # Before the check, whose object needs memory somewhere in the store, maybe only here.
            # Recorded at once: start_kv_sharing unmounts it if a later step fails.
            py_executor.store_segment = StoreSegment.mount(
                store, backend.segment_size, mooncake.protocol
            )
        return store, check_store(store, gpu_direct_error)

    # Three collectives every rank reaches, whatever failed on it: prepare (which checks the node's
    # memory before any rank mounts its segment), opening and checking the store, attach. Every
    # check after them reads gathered facts, so it raises on every rank.
    (mooncake, blob_config), identity_by_rank = _agree(dist, prepare)
    # Before the lender pins its staging memory: an unreachable master fails here, within a minute.
    store, gpu_direct_errors = _agree(dist, open_and_check)
    memory_kind = choose_data_path(config.data_path, gpu_direct_errors)

    tokens_per_block = manager.tokens_per_block
    fetch_tokens = max(config.max_fetch_tokens, max_num_tokens)

    def attach() -> tuple[tuple[_Attached, Optional[TextIO]], _RankFacts]:
        driver = make_store_driver(
            store,
            mooncake,
            blob_config,
            allocation_range=allocation_range if memory_kind is MemoryKind.DEVICE else None,
        )
        profile_of = functools.partial(_profile, py_executor, llm_args, identity_by_rank[0])
        attached, facts = _attach(
            manager, profile_of, memory_kind, fetch_tokens, driver, blob_config
        )
        # Last, so a failed attach leaves no trace file behind.
        return (attached, _open_trace(dist.rank)), facts

    (attached, trace), facts_by_rank = _agree(dist, attach)
    if len({facts.scope for facts in facts_by_rank}) > 1:
        raise ValueError("kv_sharing: the ranks would name blocks differently")
    attention_dp = py_executor.enable_attention_dp
    # Attention DP needs no replica set: each rank serves requests of its own with every KV head,
    # so two ranks put one block only as two servers can, and the master refuses the later put.
    replicas = ReplicaSet()
    min_over_ranks = None
    if dist.tp_size > 1 and not attention_dp:  # tensor parallelism: one table, closed together
        try:
            replicas = replica_set([facts.shards for facts in facts_by_rank], dist.rank)
        except ValueError as error:  # alike on every rank: the shards are gathered
            raise ValueError(f"{SUPPORTED_MODELS}; this server is refused: {error}") from error
        # Creating the TP communicator is itself a collective (MPI Create_group): do it here,
        # where every rank is, not lazily inside the first round.
        dist.tp_comm  # noqa: B018
        min_over_ranks = functools.partial(dist.tp_allreduce_int64, op=ReduceOp.MIN)

    attached.watchdog.start()
    py_executor.disagg.attach_shared_transfer_manager(attached.transfers)
    policy = SharingPolicy(
        memory_kind=memory_kind,
        fetch="fetch" in backend.roles,
        publish="publish" in backend.roles,
        min_fetch_tokens=config.min_fetch_tokens,
        max_fetch_tokens=config.max_fetch_tokens,
        fetch_timeout_s=config.fetch_timeout_s,
        staging=attached.staging,
        max_util_for_resume=llm_args.kv_cache_config.max_util_for_resume,
        max_parked_fetches=py_executor.max_num_active_requests - 1 if attention_dp else None,
    )
    if policy.fetch and policy.max_parked_fetches == 0 and dist.rank == 0:
        logger.warning(
            "kv_sharing: under attention DP with max_num_active_requests 1 no request fetches "
            "from the store: its one seat stays free for the padding request"
        )
    probe = None
    if policy.fetch and dist.rank == 0:
        probe = ProbeService(
            attached.lender,
            attached.driver,
            key_stem(blob_config.key_prefix),
            tokens_per_block=tokens_per_block,
            min_fetch_tokens=config.min_fetch_tokens,
            shares_per_group=[count for count, _ in facts_by_rank[0].shards],
            windowed=any(
                group.kept_window is not None for group in attached.lender.page_layout.layer_groups
            ),
            hold_s=config.probe_hold_ms / 1000,
            exit_wait_s=mooncake.rpc_timeout_ms / 1000 + 1,
        )
        py_executor.executor_request_queue.probe_on_enqueue = probe.submit
    orchestrator = SharedKvOrchestrator(
        policy=policy,
        kv_cache_manager=manager,
        lender=attached.lender,
        transfers=attached.transfers,
        execution_stream=py_executor.execution_stream,
        watchdog=attached.watchdog,
        driver=attached.driver,
        probe=probe,
        min_over_ranks=min_over_ranks,
        replicas=replicas,
        decides_publishing=min_over_ranks is None or dist.rank == 0,
        trace=trace,
    )
    py_executor.kv_sharing = orchestrator
    py_executor.scheduler.store_fetches = orchestrator
    if memory_kind is MemoryKind.DEVICE:
        path = f"GPU Direct, at most {MAX_OPEN_PUBLICATIONS} publications locking pages at once"
    else:
        staging_bytes = sum(
            r.nbytes for r in attached.lender.regions if r.memory_kind is MemoryKind.HOST
        )
        most = attached.staging.longest
        blocks = "any number of" if most == sys.maxsize else f"at most {most}"
        path = (
            f"host staging of {staging_bytes >> 20} MiB: one fetch of {fetch_tokens} tokens and "
            f"publications of {blocks} blocks"
        )
    store_calls = "native" if isinstance(attached.driver, NativeMooncakeDriver) else "python"
    providing = (
        f", providing {backend.segment_size >> 20} MiB per rank" if backend.segment_size else ""
    )
    logger.info(
        f"kv_sharing: store master {mooncake.master_server_address} over {mooncake.protocol}, "
        f"roles {backend.roles}, {store_calls} store driver, {path}{providing}"
    )


def _provide_store_memory(py_executor: PyExecutor, llm_args: TorchLlmArgs) -> None:
    """Gives the store ``segment_size`` bytes of each rank's host memory and does nothing else with
    it, for a server whose only role is 'provide', such as a generation server. No rank fetches or
    publishes, and the KV cache manager gets no lender."""
    (backend,) = llm_args.kv_sharing.backends
    dist = py_executor.dist

    def prepare() -> tuple[MooncakeConfig, None]:
        mooncake, _ = _store_config(backend.config)
        _check_node_memory(
            backend.segment_size, staging_bytes=0, ranks_on_node=dist.local_world_size
        )
        return mooncake, None

    def open_and_mount() -> tuple[None, None]:
        store = open_sdk(mooncake)
        # Recorded at once: start_kv_sharing unmounts it if another rank failed.
        py_executor.store_segment = StoreSegment.mount(
            store, backend.segment_size, mooncake.protocol
        )
        return None, None

    # Two collectives, so that every rank checks the node's memory before any rank mounts its
    # segment.
    mooncake, _ = _agree(dist, prepare)
    _agree(dist, open_and_mount)
    logger.info(
        f"kv_sharing: providing {backend.segment_size >> 20} MiB per rank to the store at "
        f"{mooncake.master_server_address}; this server neither fetches nor publishes"
    )


@dataclass(frozen=True)
class _RankFacts:
    """What every rank must see of each rank's setup.

    Attributes:
        scope: Names the stored blocks, with the layout; must be equal on every rank.
        shards: ``(count, index)`` of the share of each layer group this rank holds.
    """

    scope: bytes
    shards: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _Attached:
    """What one rank set up for sharing before the ranks agreed on it. ``staging``: host
    staging's budget for open publications, a function of the layout and so equal on every rank;
    None under GPU Direct."""

    driver: MooncakeDriver
    lender: Lender
    transfers: TransferManager
    watchdog: RetirementWatchdog
    staging: Optional[StagingBudget]


def _agree(
    dist: Distributed, attempt: Callable[[], tuple[_Local, _Shared]]
) -> tuple[_Local, list[_Shared]]:
    """Runs ``attempt`` on every rank: its local result, and every rank's shared one in rank
    order. If it failed on any rank, every rank raises naming each failure. One rank gathers
    nothing.

    Raises:
        ValueError: ``attempt`` raised on some rank.
    """
    failure: Optional[Exception] = None
    local = shared = None
    try:
        local, shared = attempt()
    except Exception as error:  # noqa: BLE001 - every rank must reach the gather below
        failure = error
    own = (None if failure is None else str(failure), shared)
    results = [own] if dist.world_size == 1 else dist.allgather(own)
    ranks_by_error: dict[str, list[str]] = {}
    for rank, (error, _) in enumerate(results):
        if error is not None:
            ranks_by_error.setdefault(error, []).append(str(rank))
    if ranks_by_error:
        raise ValueError(
            "kv_sharing startup failed: "
            + "; ".join(
                f"rank {', '.join(ranks)}: {error}" for error, ranks in ranks_by_error.items()
            )
        ) from failure
    return local, [shared for _, shared in results]


def _store_config(path: str) -> tuple[MooncakeConfig, BlobConfig]:
    """The ``mooncake`` and ``blob`` sections of the store config at ``path``.

    Raises:
        ValueError: The config is invalid or gives the store memory itself.
    """
    blob, mooncake = load_store_config(path)
    if mooncake.global_segment_size != 0:
        raise ValueError(
            f"leave mooncake.global_segment_size at 0 in {path}; a server provides the store "
            "memory through its role 'provide' and segment_size, 'trtllm-serve "
            "mooncake_provider' through its --segment_size"
        )
    return mooncake, blob


def _check_node_memory(segment_bytes: int, *, staging_bytes: int, ranks_on_node: int) -> None:
    """Refuses to give the store more host memory than this node can give: each of the server's
    ranks on it provides ``segment_bytes`` and stages through up to ``staging_bytes``, and the
    connector's NODE_BUDGET_RESERVE_BYTES stay free. Run on every rank before any rank mounts its
    segment. It sees this server's ranks and the host's available memory only: neither other
    servers starting on the node at the same time nor a cgroup memory limit (Slurm's).

    Raises:
        ValueError: The node's available memory is too small.
    """
    claimed = ranks_on_node * (segment_bytes + staging_bytes)
    available = psutil.virtual_memory().available
    if claimed <= available - NODE_BUDGET_RESERVE_BYTES:
        return
    gib = 1 << 30
    staging = f" and stage through up to {staging_bytes / gib:.1f} GiB" if staging_bytes else ""
    raise ValueError(
        f"kv_sharing: this node cannot give the store the host memory asked for: {ranks_on_node} "
        f"rank(s) here would provide {segment_bytes / gib:.1f} GiB each{staging}, "
        f"{claimed / gib:.1f} GiB in all, but {available / gib:.1f} GiB is available and "
        f"{NODE_BUDGET_RESERVE_BYTES // gib} GiB of it stays free. Lower segment_size, or set "
        "kv_cache_config.host_cache_size to leave room: unset, KV cache manager V2's host tier "
        "takes up to half of the node's available memory when the server starts"
    )


def _attach(
    manager: KVCacheManagerV2,
    profile_of: Callable[[StagingOptions], ProfileSpec],
    memory_kind: MemoryKind,
    fetch_tokens: int,
    driver: MooncakeDriver,
    blob_config: BlobConfig,
) -> tuple[_Attached, _RankFacts]:
    """Attaches the lender over the KV cache manager and assembles the transfer manager with the
    store backend, which registers this rank's memory with the store."""
    tokens_per_block = manager.tokens_per_block
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
        profile = profile_of(staging)
        domain = attach_domain(manager, profile, memory_kind=memory_kind)
    except (TypeError, ValueError) as error:
        # The lender may stay attached; startup fails, so nothing lends through it.
        raise ValueError(f"{SUPPORTED_MODELS}; this server is refused: {error}") from error
    staging_budget = None
    if memory_kind is MemoryKind.HOST:
        staging_budget = _staging_budget(domain.lender, fetch_tokens, tokens_per_block)

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
    shards = tuple((g.shard.count, g.shard.index) for g in domain.lender.page_layout.layer_groups)
    attached = _Attached(driver, domain.lender, transfers, watchdog, staging_budget)
    return attached, _RankFacts(profile.scope, shards)


def _open_trace(rank: int) -> Optional[TextIO]:
    """This rank's round trace, line-buffered, where TRACE_DIR_ENV names a directory."""
    directory = os.environ.get(TRACE_DIR_ENV)
    if not directory:
        return None
    os.makedirs(directory, exist_ok=True)
    return open(os.path.join(directory, f"rank{rank}.txt"), "w", buffering=1)


def _staging_budget(lender: Lender, fetch_tokens: int, tokens_per_block: int) -> StagingBudget:
    """The rows open publications may hold in each host staging region: its slots less one
    fetch's. A block takes one slot of its pool group's region for each layer group the region
    serves; a windowed group at most its sinks and window, whatever the range.

    Raises:
        ValueError: Host staging holds no publication beside one fetch.
    """
    layer_groups = lender.page_layout.layer_groups
    fetch_blocks = ceil_div(fetch_tokens, tokens_per_block)
    regions = [region for region in lender.regions if region.memory_kind is MemoryKind.HOST]
    free_rows, rows_per_block, window_rows = [], [], []
    for region in regions:
        served = [layer_groups[lg] for lg in region.layer_group_indices]
        windows = [
            group.num_sink_blocks + ceil_div(group.kept_window, tokens_per_block)
            for group in served
            if group.kept_window is not None
        ]
        full = len(served) - len(windows)
        # One fetch's slots, as the lender sizes them (slots_per_fetch).
        fetch_rows = full * fetch_blocks + sum(min(fetch_blocks, rows) for rows in windows)
        free_rows.append(region.slot_count - fetch_rows)
        rows_per_block.append(full)
        window_rows.append(sum(windows))
    budget = StagingBudget(tuple(free_rows), tuple(rows_per_block), tuple(window_rows))
    if not budget.fits(1, []):
        staged_gib = sum(region.nbytes for region in regions) / (1 << 30)
        raise ValueError(
            f"kv_sharing: {staged_gib:.1f} GiB of host staging hold one fetch of {fetch_tokens} "
            "tokens, max(max_fetch_tokens, max_num_tokens), and no block for publications; "
            "lower max_fetch_tokens or max_num_tokens"
        )
    return budget


def _profile(
    py_executor: PyExecutor,
    llm_args: TorchLlmArgs,
    artifact_identity: ArtifactIdentity,
    staging: StagingOptions,
) -> ProfileSpec:
    """The facts that scope the stored blocks' names: the checkpoint (``artifact_identity``, rank
    0's reading of it) and its effective config, the release, the parallel layout and the GPU
    that computed the bytes, the writer layout and the cache dtype."""
    manager = py_executor.kv_cache_manager
    model_config = py_executor.model_engine.model.model_config
    layout, cache_dtype = _stored_cache(py_executor)
    mapping = model_config.mapping
    dp_size = mapping.tp_size if mapping.enable_attention_dp else 1
    major, minor = torch.cuda.get_device_capability()
    compatibility = (
        f"trtllm-{tensorrt_llm.__version__}/tp{mapping.tp_size}-pp{mapping.pp_size}"
        f"-cp{mapping.cp_size}-adp{int(mapping.enable_attention_dp)}/sm{major}{minor}"
    )
    return ProfileSpec(
        id=PROFILE_ID,
        lifecycle_profile_id="executor-fatal",
        source_identity=SourceIdentity.from_model_config(
            model_config, artifact_identity=artifact_identity
        ),
        compatibility_id=compatibility,
        model_layout=layout,
        dtype=cache_dtype,
        # Under attention DP each tensor-parallel rank is a data-parallel rank of its own.
        topology=(mapping.tp_size, dp_size, mapping.pp_size, mapping.cp_size),
        state_kinds=(CacheKind.PAGED,) * manager.num_pools,
        attention_windows=_attention_windows(manager),
        sparse_attention=_scoped_sparse_config(model_config.sparse_attention_config),
        sink_token_length=llm_args.kv_cache_config.sink_token_length or 0,
        lora_enabled=False,
        pool_count=len(manager.impl.pool_group_descs),
        speculative_mode="none",
        model_role="target",
        draft_pool=None,
        joint_reuse=False,
        staging=staging,
    )


def _stored_cache(py_executor: PyExecutor) -> tuple[str, str]:
    """The attention writer layout and the cache dtype of the blocks this server stores."""
    manager = py_executor.kv_cache_manager
    pretrained_config = py_executor.model_engine.model.model_config.pretrained_config
    layout = MLA_LAYOUT if is_mla(pretrained_config) else MHA_LAYOUT
    return layout, _CACHE_DTYPES.get(manager.dtype, str(manager.dtype))


def _scoped_sparse_config(config: Optional[BaseSparseAttentionConfig]) -> Optional[dict]:
    """The effective sparse attention config that names stored blocks, as JSON values; None for
    dense attention. Leaves out ``seq_len_threshold``: it only picks CUDA graphs, and DSA sets it to
    its top-k when they are captured."""
    if config is None:
        return None
    return config.model_dump(mode="json", exclude={"seq_len_threshold"})


def _attention_windows(manager: KVCacheManagerV2) -> tuple[tuple[int, int], ...]:
    """``(global layer, window)`` of each layer whose KV cache keeps a sliding window."""
    return tuple(
        (int(manager.pp_layers[local]), int(window))
        for local, window in enumerate(manager.max_attention_window_vec)
        if window is not None
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
    """The transfer manager's wake. Rank 0 does not wait on the request queue while store work is
    pending, and an idle pass waits at most the transfer manager's 1 ms heartbeat."""


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

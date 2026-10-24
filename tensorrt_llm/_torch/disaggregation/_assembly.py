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
"""Private runtime assembly shared by transfer and Store integrations.

The initial profile consumes NAMED leases of one memory kind, HOST (staged) or
DEVICE (in place), from the lender's complete mixed memory namespace. Model
provenance and writer facts derive its canonical scope; public layout and region
facts validate the serving profile. Neither a provider name nor a caller-supplied
hint establishes compatibility or isolation.
"""

from __future__ import annotations

import hashlib
import json
import numbers
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from tensorrt_llm._torch.pyexecutor.kv_cache.page_layout import PageLayout
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import (
    Lease,
    LeaseView,
    Lender,
    MemoryKind,
    MemoryRegion,
    ResumeBounds,
    StagingOptions,
    attach_lender,
)
from tensorrt_llm._torch.weight_sharing.artifact_identity import ArtifactIdentity
from tensorrt_llm._torch.weight_sharing.source_identity import (
    SOURCE_IDENTITY_FORMAT_VERSION,
    SourceIdentity,
)

from .base.backend import Backend, CacheKind, SlotRegion, WakeFn

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2

WriteReturnFn = Callable[[Lease, tuple[np.ndarray, ...], bool], ResumeBounds | None]
"""Owner-thread endpoint for one consuming write return to its originating lender.

The holder supplies the fixed Outcome masks after coordinated return permission
and ended backend access. Keep the endpoint rooted through return and cleanup.
"""


def bind_write_return(lender: Lender) -> WriteReturnFn:
    """Bind only the lender association for a consuming write return.

    Call the endpoint on the manager thread after every participating rank settles
    its Outcome and ends backend access. The endpoint neither authorizes the
    return nor waits for local copies. It forwards the exact masks and reuse
    permission once and propagates errors; the caller must not retry a return.

    Args:
        lender: Originating lender, kept alive by the returned endpoint.

    Returns:
        A callable yielding the lender's single result, possibly pending bounds.
    """

    def return_write(
        lease: Lease, arrived: tuple[np.ndarray, ...], reusable: bool
    ) -> ResumeBounds | None:
        (bounds,) = lender.return_writes([(lease, arrived, reusable)])
        return bounds

    return return_write


OwnerCleanupFn = Callable[[Callable[[], None]], bool]
"""Thread-safe, nonblocking posting of bounded manager-thread finalizers.

True accepts exactly-once execution; False retains provider ownership for retry.
A callback is posted only after required deregistration/access-end evidence.
Acceptance or queue drain does not prove cleanup. Failed finalizers retain their
fault and resources. Blocking SDK work must run outside the manager thread.
"""

ProviderCleanupCompleteFn = Callable[[], bool]
"""Private, nonblocking and thread-safe closure of provider cleanup production.

True is monotonic: no further finalizers will be posted and no owner-cleanup
obligation remains outside the accepted dispatcher queue. False keeps provider
ownership, including finalizers whose posting was rejected, for later retry.
This proves neither ended access nor completion of accepted callbacks. Retirement
also requires empty public shutdown labels, a fault-free drained callback queue,
and all lease/result/Lifecycle obligations. Uncertain construction without a
Backend still requires independent access-end evidence. Public shutdown labels
retain their exact access/registration meaning.
"""

BackendFactory = Callable[[tuple[SlotRegion, ...], WakeFn, object, OwnerCleanupFn], Backend]
"""Private provider construction hook receiving the complete region namespace.

The runtime roots this callable, its ProviderCleanupCompleteFn and its cleanup
dispatcher before invocation. A stateful factory roots partial registration and
RegionHold state before exposure, including failures before a Backend returns.
It captures lender access privately; hold acquisition runs on the manager thread.
It registers only supported/accessed regions without renumbering them, and posts
bounded hold finalizers through the dispatcher. Rejected posts retain provider
ownership for retry through provider progress or repeated shutdown. Public labels,
private cleanup closure and accepted callback completion are separate retirement
conditions. No public Backend cleanup API is added.
"""


class FactoryRejected(RuntimeError):
    """Construction certifies no remaining access, registration or region hold."""


# The tested cells: an attention writer layout with the dtype of its cache.
_TESTED_CELLS = frozenset(
    {("mha-trtllm-hnd", "bfloat16"), ("mha-trtllm-hnd", "fp8"), ("mla-trtllm-latent", "bfloat16")}
)
# The public layout's dtype and quantization of each cell dtype.
_LAYOUT_DTYPES = {"bfloat16": ("BF16", None), "fp8": ("FP8", "fp8")}


@dataclass(frozen=True)
class ProfileSpec:
    """Trusted immutable facts for one of the tested compatibility cells.

    The model/executor assembly supplies facts for the actual writer and serving
    manager. Attachment checks the manager facts against its public page layout.
    This is not user-facing configuration and does not qualify a provider.
    Build source_identity from the effective ModelConfig and a precomputed
    ArtifactIdentity after resolving model overrides. No checkpoint I/O occurs
    here. Supply the actual writer and runtime attention facts; a model name or
    a manager's declared layout alone does not establish compatibility.

    Attributes:
        id: Local profile identifier.
        lifecycle_profile_id: Selected, separately qualified Lifecycle profile.
        source_identity: Immutable loaded-checkpoint and effective-config provenance.
        compatibility_id: Qualified KV semantics ABI, changed for incompatible releases.
        scope: Derived immutable byte identity; callers cannot supply a raw scope.
        model_layout: Actual attention-writer layout identifier: ``mha-trtllm-hnd``
            or ``mla-trtllm-latent``.
        dtype: Actual cache dtype supplied by model assembly: ``bfloat16``, or
            ``fp8`` for the MHA layout.
        topology: Tensor, data, pipeline and context parallel sizes; any tensor
            parallel size, data parallelism 1 or equal to it (attention DP), no
            pipeline or context parallelism.
        state_kinds: Canonical cache kinds from the layout's layer groups.
        attention_windows: ``(global layer id, tokens)`` of each layer whose attention reads a
            sliding window, sorted; ``()`` when every layer reads its whole context. Windowed
            MLA is not a tested cell.
        sparse_attention: Whether runtime sparse attention is enabled; must be false.
        sink_token_length: Runtime sink length; this cell requires zero.
        lora_enabled: Whether adapters are enabled; this cell excludes LoRA.
        pool_count: Number of physical pool groups in the selected layout.
        speculative_mode: Explicit speculative mode; only none is supported here.
        model_role: Target or draft role; this cell accepts only the target.
        draft_pool: Separate draft-pool provenance; this cell requires None.
        joint_reuse: Whether reuse requires another manager's draft state.
        staging: Canonical capacity options for the pinned lender baseline.
        backend_config: Opaque provider-owned factory configuration.

    Raises:
        TypeError: A fact has an invalid type.
        ValueError: A required identity is empty or the cell is unsupported.
    """

    id: str
    lifecycle_profile_id: str
    source_identity: SourceIdentity
    compatibility_id: str
    model_layout: str
    dtype: str
    topology: tuple[int, int, int, int]
    state_kinds: tuple[CacheKind, ...]
    attention_windows: tuple[tuple[int, int], ...]
    sparse_attention: bool
    sink_token_length: int
    lora_enabled: bool
    pool_count: int
    speculative_mode: str
    model_role: Literal["target", "draft"]
    draft_pool: SourceIdentity | None
    joint_reuse: bool
    staging: StagingOptions
    backend_config: object = None
    scope: bytes = field(init=False)

    def __post_init__(self) -> None:
        """Reject unsupported facts before any memory or backend exposure."""
        for name, value in (("id", self.id), ("lifecycle_profile_id", self.lifecycle_profile_id)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty string")
        if not isinstance(self.staging, StagingOptions):
            raise TypeError("profile staging must be StagingOptions")
        if (self.model_layout, self.dtype) not in _TESTED_CELLS:
            raise ValueError(
                f"profile cell {self.model_layout}/{self.dtype} is not one of the tested cells "
                f"{sorted(_TESTED_CELLS)}"
            )
        topology = tuple(self.topology)
        if any(
            isinstance(size, bool) or not isinstance(size, numbers.Integral) for size in topology
        ):
            raise TypeError("topology sizes must be integers")
        tp, dp, pp, cp = topology
        if tp < 1 or dp not in (1, tp) or (pp, cp) != (1, 1):
            raise ValueError("profile requires TPn with DP1 or DPn (attention DP), PP1 and CP1")
        kinds = tuple(self.state_kinds)
        if not kinds or any(not isinstance(kind, CacheKind) for kind in kinds):
            raise TypeError("state kinds must contain canonical CacheKind values")
        if any(kind != CacheKind.PAGED for kind in kinds):
            raise ValueError("initial profile supports paged attention state only")
        windows = [tuple(pair) for pair in self.attention_windows]
        if any(
            len(pair) != 2
            or any(isinstance(v, bool) or not isinstance(v, numbers.Integral) for v in pair)
            for pair in windows
        ):
            raise TypeError("attention windows must be (layer, tokens) integer pairs")
        windows = tuple(sorted((int(layer), int(window)) for layer, window in windows))
        if any(layer < 0 or window <= 0 for layer, window in windows):
            raise ValueError("attention windows need a nonnegative layer and a positive window")
        if len({layer for layer, _ in windows}) != len(windows):
            raise ValueError("attention windows name a layer twice")
        if windows and self.model_layout == "mla-trtllm-latent":
            raise ValueError("windowed MLA is not a tested cell")
        if self.sparse_attention is not False:
            raise ValueError("initial profile excludes sparse attention")
        if isinstance(self.sink_token_length, bool) or not isinstance(
            self.sink_token_length, numbers.Integral
        ):
            raise TypeError("sink_token_length must be an integer")
        if self.sink_token_length != 0:
            raise ValueError("initial profile excludes attention sinks")
        if self.lora_enabled is not False:
            raise ValueError("initial profile excludes LoRA")
        if isinstance(self.pool_count, bool) or not isinstance(self.pool_count, numbers.Integral):
            raise TypeError("pool_count must be an integer")
        if (
            self.speculative_mode != "none"
            or self.model_role != "target"
            or self.joint_reuse is not False
            or self.draft_pool is not None
        ):
            raise ValueError("initial profile excludes speculative and joint draft reuse")
        object.__setattr__(self, "topology", topology)
        object.__setattr__(self, "state_kinds", kinds)
        object.__setattr__(self, "attention_windows", windows)
        object.__setattr__(self, "scope", derive_kv_scope(self))


def derive_kv_scope(profile: ProfileSpec) -> bytes:
    """Derive the common Runtime/Store scope for the supported semantic profile.

    SourceIdentity supplies checkpoint, effective model, quantization and compute
    backend fingerprints. Its placement, shard, discovery name and parallel
    fingerprints do not belong here. The lender adds physical layout, content
    shard and request-prefix identity when it builds object keys. Transfer
    providers, staging capacity and Lifecycle policy do not affect KV meaning.

    The compatibility ID names the qualified writer/model semantics across
    releases. It must change when those semantics change incompatibly. Provider
    names, local paths and hints are not substitutes for checkpoint provenance.

    Args:
        profile: Validated initial-profile facts from trusted model assembly.

    Returns:
        A versioned, deterministic byte scope, independent of local placement.

    Raises:
        TypeError: The profile or checkpoint provenance has the wrong type.
        ValueError: A required identity is missing, malformed or unsupported.
    """
    if not isinstance(profile, ProfileSpec):
        raise TypeError("profile must be ProfileSpec")
    source = profile.source_identity
    if not isinstance(source, SourceIdentity):
        raise TypeError("source_identity must be SourceIdentity")
    if (
        type(source.format_version) is not int
        or source.format_version != SOURCE_IDENTITY_FORMAT_VERSION
    ):
        raise ValueError("unsupported source identity format version")
    if not isinstance(source.artifact_identity, ArtifactIdentity):
        raise TypeError("source identity requires ArtifactIdentity checkpoint provenance")
    if not isinstance(profile.compatibility_id, str) or not profile.compatibility_id.strip():
        raise ValueError("compatibility_id must be a nonempty string")
    fingerprints = {}
    for name, value in (
        ("model_fingerprint", source.model_fingerprint),
        ("quant_fingerprint", source.quant_fingerprint),
        ("backend_fingerprint", source.backend_fingerprint),
    ):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdefABCDEF" for char in value)
        ):
            raise ValueError(f"{name} must be a SHA-256 hex digest")
        fingerprints[name] = value.lower()
    payload = {
        "source_format": source.format_version,
        "artifact": source.artifact_identity.to_dict(),
        "model": fingerprints,
        "transform_abi": source.transform_abi_id,
        "compatibility_id": profile.compatibility_id,
        "writer_layout": profile.model_layout,
        "cache_dtype": profile.dtype,
        "attention": _attention_payload(profile.attention_windows),
        "lora_enabled": profile.lora_enabled,
        "speculative_mode": profile.speculative_mode,
        "model_role": profile.model_role,
        "draft_pool": profile.draft_pool,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return b"trtllm-kv-scope-v1\0" + hashlib.sha256(encoded.encode("utf-8")).digest()


def _attention_payload(windows: tuple[tuple[int, int], ...]) -> dict:
    """The scope's attention facts: full attention keeps its payload; windows name each windowed
    layer and its window, so another window setting names other blocks."""
    if not windows:
        return {"full": True, "sparse": False, "sink_tokens": 0}
    return {
        "full": False,
        "sparse": False,
        "sink_tokens": 0,
        "windows": [[layer, window] for layer, window in windows],
    }


@dataclass(frozen=True)
class RegionNamespace:
    """Validated whole namespace and the operation's selected memory kind.

    Preserve every HOST and DEVICE region and its original lender index. Factories
    register only supported regions; an operation may reference only its selected
    kind. A provider owns any registration hold, not this immutable descriptor.

    Attributes:
        regions: Complete namespace in the lender's original index order.
        memory_kind: Selected operation kind.
        allowed_region_indices: Precomputed indices usable by this operation profile.

    Raises:
        ValueError: The memory kind or region geometry is unsupported.
        TypeError: A region is not the canonical immutable MemoryRegion.
    """

    regions: tuple[MemoryRegion, ...]
    memory_kind: MemoryKind
    allowed_region_indices: frozenset[int] = field(init=False)

    def __post_init__(self) -> None:
        """Check namespace overlap and selection; canonical regions validate their own geometry."""
        if not isinstance(self.memory_kind, MemoryKind):
            raise TypeError("selected memory kind must be MemoryKind")
        regions = tuple(self.regions)
        intervals: dict[MemoryKind, list[tuple[int, int]]] = {}
        for region in regions:
            if not isinstance(region, MemoryRegion):
                raise TypeError("region namespace requires canonical MemoryRegion records")
            intervals.setdefault(region.memory_kind, []).append(
                (region.address, region.address + region.nbytes)
            )
        for same_kind in intervals.values():
            same_kind.sort()
            if any(left[1] > right[0] for left, right in zip(same_kind, same_kind[1:])):
                raise ValueError("regions of the same memory kind overlap")
        object.__setattr__(self, "regions", regions)
        object.__setattr__(
            self,
            "allowed_region_indices",
            frozenset(
                index
                for index, region in enumerate(regions)
                if region.memory_kind is self.memory_kind
            ),
        )


@dataclass(frozen=True)
class DomainBinding:
    """Orchestration's lender and region association for one serving manager.

    The executor owns the manager; this binding does not retain it. Orchestration
    keeps the manager and this lender association alive through transfer cleanup.
    For normal shutdown, stop submissions and confirm ended backend access before
    returning writes with their arrived rows or releasing reads and discarded writes.
    Keep lender progress running for pending copies, then finish deregistration, hold
    release and accepted owner-thread finalizers before manager shutdown. Missing
    evidence requires retention under the selected containment policy.
    The holder receives ``namespace`` and ``lender``, never this association. The
    lender remains the only memory-access authority.

    Attributes:
        lender: Canonical access-granting interface.
        profile: Trusted immutable compatibility/identity facts.
        namespace: Complete validated region descriptors for holder/provider use.
    """

    lender: Lender
    profile: ProfileSpec
    namespace: RegionNamespace


def _validate_layout(profile: ProfileSpec, layout: PageLayout, namespace: RegionNamespace) -> None:
    """Check the selected profile against public manager layout and the selected kind's regions."""
    if not isinstance(layout, PageLayout):
        raise TypeError("lender page_layout must be PageLayout")
    topology = layout.topology
    if (topology.tp_size, topology.dp_size, topology.pp_size, topology.cp_size) != profile.topology:
        raise ValueError("lender topology does not match the selected profile")
    if (layout.dtype, layout.quant_mode) != _LAYOUT_DTYPES[profile.dtype]:
        raise ValueError(
            f"lender layout {layout.dtype} (quantization {layout.quant_mode}) does not match the "
            f"profile's {profile.dtype} cache"
        )
    kinds = {"attention": CacheKind.PAGED, "state": CacheKind.STATE}
    if tuple(kinds.get(group.kind) for group in layout.layer_groups) != profile.state_kinds:
        raise ValueError("lender layer kinds do not match the selected profile")
    windows = dict(profile.attention_windows)
    for group in layout.layer_groups:
        if group.kept_window != group.read_window:
            raise ValueError("a layer group keeps a wider window than its attention reads")
        if any(windows.get(layer) != group.read_window for layer in group.layers):
            raise ValueError("lender windows do not match the selected profile")
    if any(group.num_sink_blocks != 0 for group in layout.layer_groups):
        raise ValueError("initial profile excludes attention sinks")
    if len(layout.pool_slot_bytes) != profile.pool_count:
        raise ValueError("lender pool groups do not match the selected profile")
    if namespace.memory_kind is MemoryKind.HOST:
        _validate_staging_regions(layout, namespace)
    else:
        _validate_device_regions(layout, namespace)


def _layer_groups_of(layout: PageLayout, pool_group: int) -> tuple[int, ...]:
    return tuple(i for i, group in enumerate(layout.layer_groups) if group.pool_group == pool_group)


def _validate_staging_regions(layout: PageLayout, namespace: RegionNamespace) -> None:
    """A staging region per pool group, holding the pool group's slots back to back."""
    if not layout.staging_region_index or set(layout.staging_region_index) != set(
        layout.pool_slot_bytes
    ):
        raise ValueError("lender has no complete HOST/NAMED capability for this profile")
    for pool_group, index in layout.staging_region_index.items():
        if not 0 <= index < len(namespace.regions):
            raise ValueError("staging region is outside the lender namespace")
        region = namespace.regions[index]
        if (
            region.memory_kind is not MemoryKind.HOST
            or region.layer_group_indices != _layer_groups_of(layout, pool_group)
            or region.slot_bytes != sum(layout.pool_slot_bytes[pool_group])
        ):
            raise ValueError("staging region does not match the public HOST layout")


def _validate_device_regions(layout: PageLayout, namespace: RegionNamespace) -> None:
    """A device region per pool of every pool group, slots of the pool's width."""
    for pool_group, widths in layout.pool_slot_bytes.items():
        for pool, width in enumerate(widths):
            index = layout.device_region_index.get((pool_group, pool))
            if index is None or not 0 <= index < len(namespace.regions):
                raise ValueError("lender has no complete DEVICE/NAMED capability for this profile")
            region = namespace.regions[index]
            if (
                region.memory_kind is not MemoryKind.DEVICE
                or region.layer_group_indices != _layer_groups_of(layout, pool_group)
                or region.slot_bytes != width
            ):
                raise ValueError("device region does not match the public DEVICE layout")


def attach_domain(
    manager: KVCacheManagerV2, profile: ProfileSpec, *, memory_kind: MemoryKind = MemoryKind.HOST
) -> DomainBinding:
    """Attach the canonical lender without duplicating manager policy checks.

    The caller supplies facts derived from its actual writer/serving-manager
    assembly. The lender validates attachment and exposes available memory kinds;
    this helper checks its public layout and the NAMED capability of ``memory_kind``:
    staging regions for HOST, a region per device pool for DEVICE. The scope does
    not depend on the kind. Backend-specific pinning requirements belong to its
    factory.
    This helper neither activates a backend nor certifies provider qualification.
    Unexpected attach failures retain the canonical lender's safety meaning.
    The caller retains the manager through cleanup and shuts it down last.
    A failed profile check leaves the lender attached. The caller must perform
    normal shutdown, without retrying attachment or detaching the lender.

    Args:
        manager: Serving manager, never a sizing probe.
        profile: Trusted compatibility facts and canonical byte scope.
        memory_kind: The kind every operation of the domain lends.

    Returns:
        An orchestration-owned association and full namespace for ``memory_kind``
        operations.

    Raises:
        TypeError: Profile is not ProfileSpec.
        ValueError: The canonical lender refuses attachment or region validation fails.
        MemoryError: The lender cannot allocate its staging capacity.
    """
    if not isinstance(profile, ProfileSpec):
        raise TypeError("profile must be ProfileSpec")
    lender = attach_lender(manager, scope=profile.scope, staging=profile.staging)
    namespace = RegionNamespace(lender.regions, memory_kind=memory_kind)
    _validate_layout(profile, lender.page_layout, namespace)
    return DomainBinding(lender, profile, namespace)


def check_view_compatibility(lease_view: LeaseView, namespace: RegionNamespace) -> None:
    """Check operation placement against the profile's permitted regions.

    Canonical LeaseView already implements View. The lender guarantees array
    shape, immutability, group mapping and valid placement. Check only placement
    selection per group, addresses for HOST and slots for DEVICE; do not inspect
    rows or create a replacement View.
    Orchestration must pair the lease with its lender's binding. Region indices
    alone do not prove lease origin, and this check grants no memory access.

    Args:
        lease_view: Ready view from the canonical lender for this namespace.
        namespace: Complete namespace validated once during assembly.

    Raises:
        ValueError: The view uses unsupported memory or row placement.
    """
    kind = namespace.memory_kind
    for blocks in lease_view.groups:
        if kind is MemoryKind.HOST:
            regions = None if blocks.addresses is None else (blocks.region_index,)
        else:
            regions = None if blocks.slots is None else blocks.region_indices
        if blocks.object_keys is None or regions is None:
            raise ValueError(f"initial profile requires keys and rows placed in {kind.name} memory")
        if not namespace.allowed_region_indices.issuperset(regions):
            raise ValueError(f"row region is not permitted by the selected {kind.name} profile")

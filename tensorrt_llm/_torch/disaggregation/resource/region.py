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
"""Where a KV Cache Manager V2 page lives in memory, and a digest of that layout.

``KVv2RegionResolver`` turns a unit's local coordinates ``(layer group, page index)`` into the
memory segments a backend moves, lists the pool spans a backend registers, and sizes the largest
unit. ``layout_fingerprint`` digests everything those bytes assume, so two processes whose bytes
would not mean the same thing miss each other in a store instead of reading each other's pages.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

from .kv_extractor import build_page_table_from_manager
from .page import KVCachePageTable

__all__ = ["KVv2RegionResolver", "layout_fingerprint", "parallel_shard_tag"]


class KVv2RegionResolver:
    """``RegionResolver`` over a V2 page table: ``(local_group, page index)`` -> memory segments.

    A page is one slot in each pool of the layer group's pool group, in pool order. That order is
    fixed by the page table and is part of what ``layout_fingerprint`` covers.
    """

    def __init__(self, page_table: KVCachePageTable) -> None:
        self._page_table = page_table

    def __call__(self, local_group: int, local: int) -> Sequence[tuple[int, int]]:
        layer_groups = self._page_table.layer_groups
        if not 0 <= local_group < len(layer_groups):
            raise KeyError(f"no layer group {local_group}")
        pool_group = self._page_table.pool_groups[layer_groups[local_group].pool_group_idx]
        segments = []
        for pool in pool_group.pools:
            if not 0 <= local < pool.num_slots:
                raise KeyError(f"page {local} outside a pool of {pool.num_slots} slots")
            slot_bytes = int(pool.slot_bytes)
            segments.append((int(pool.base_address) + local * slot_bytes, slot_bytes))
        return segments

    def pool_memory_spans(self) -> list[tuple[int, int]]:
        """``(address, size)`` of every distinct pool; what ``RegistersPools`` is handed."""
        size_by_address: dict[int, int] = {}
        for pool_group in self._page_table.pool_groups:
            for pool in pool_group.pools:
                size_by_address[int(pool.base_address)] = int(pool.slot_bytes) * int(pool.num_slots)
        return sorted(size_by_address.items())

    def max_unit_bytes(self) -> int:
        """The largest unit any layer group produces; sizes one host staging slot."""
        return max(
            sum(int(pool.slot_bytes) for pool in pool_group.pools)
            for pool_group in self._page_table.pool_groups
        )


def parallel_shard_tag(mapping) -> str:
    """Which slice of the model's KV heads this rank's units hold, for ``layout_fingerprint``.

    A rank that holds every head, whether it runs alone (TP=1) or as an attention-DP replica,
    gets the one shared tag, so such workers share store entries. Under tensor parallelism each
    rank holds its own head slice: the tag names the slice so two ranks' units, identical in
    layout, never take each other's place in a store. The tag is a rank index, not a head range:
    the page table exposes the per-rank head count (``kv_head_num_per_rank``) but not which
    heads, so when a model has fewer KV heads than TP ranks the duplicated heads get different
    tags and give up a share they could have had.
    """
    if mapping.tp_size == 1 or mapping.enable_attention_dp:
        return "heads=all"
    return f"heads={mapping.tp_rank}/{mapping.tp_size}"


def layout_fingerprint(
    kv_cache_manager, page_table: KVCachePageTable | None = None, *, parallel_shard: str = ""
) -> bytes:
    """Digest of the memory layout a unit's bytes assume (design appendix B).

    Covers block geometry, per-pool slot width, which layers and roles share a slot and in what
    order, head count and dtype, and the ``parallel_shard`` tag (``parallel_shard_tag``) that
    names which heads this rank holds. Base addresses are left out on purpose: they differ
    between two processes whose bytes mean the same thing.
    """
    page_table = (
        page_table if page_table is not None else build_page_table_from_manager(kv_cache_manager)
    )
    digest = hashlib.blake2b(digest_size=16)
    digest.update(f"parallel_shard={parallel_shard};".encode())
    digest.update(f"tokens_per_block={page_table.tokens_per_block};".encode())
    digest.update(f"dtype={getattr(kv_cache_manager, 'dtype', None)};".encode())
    digest.update(f"head_dim={getattr(kv_cache_manager, 'head_dim', None)};".encode())
    for pool_group in page_table.pool_groups:
        digest.update(b"pool_group:")
        for pool in pool_group.pools:
            digest.update(f"{int(pool.slot_bytes)},".encode())
    for layer_group in page_table.layer_groups:
        digest.update(f"layer_group:{layer_group.pool_group_idx}:".encode())
        digest.update(f"kv_heads={getattr(layer_group, 'kv_head_num_per_rank', None)};".encode())
        digest.update(f"window={getattr(layer_group, 'sliding_window_size', None)};".encode())
        for layer in layer_group.local_layers:
            digest.update(f"layer{layer.global_layer_id},".encode())
        for view in layer_group.pool_views:
            digest.update(f"view:{view.pool_idx}:{','.join(sorted(view.pool_role))}:".encode())
            digest.update(np.ascontiguousarray(view.buffer_entries).tobytes())
    return digest.digest()

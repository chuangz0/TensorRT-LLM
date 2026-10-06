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
"""KV Cache Manager V2 as the KV transfer coordination layer reads it (design §8.2, appendix B).

``KVv2ResourceReader`` is the ``ResourceReader``: which blocks a prompt names, which layer groups
this rank holds, which pages a fetch lands in, which committed pages a publish offers. It is the
only thing between the coordination layer and the KV v2 wrapper, and it reaches the wrapper through
its methods alone; nothing outside ``resource/`` reads a page object or an address.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

import numpy as np

from ..base.backend import CacheKind
from ..base.cache_backend import CacheExtent, Unit
from ..base.views import GroupSpec
from .kv_extractor import build_page_table_from_manager
from .naming import group_tag, units_for_group
from .page import KVCachePageTable, MambaLayerGroup

__all__ = ["KVv2ResourceReader"]

_BLOCK_KEY_CACHE_SIZE = 4096
"""Requests whose block keys the reader remembers; the planner asks for them several times."""


class KVv2ResourceReader:
    """``ResourceReader`` over a ``KVCacheManagerV2`` wrapper.

    Args:
        kv_cache_manager: The engine's primary ``KVCacheManagerV2``.
        page_table: Its page table from ``build_page_table_from_manager``; shared with the region
            resolver so that both describe the same pools.
    """

    def __init__(self, kv_cache_manager, page_table: KVCachePageTable | None = None) -> None:
        self._kv_cache_manager = kv_cache_manager
        self._page_table = (
            page_table
            if page_table is not None
            else build_page_table_from_manager(kv_cache_manager)
        )
        self._tokens_per_block = int(kv_cache_manager.tokens_per_block)
        self._group_specs = self._describe_layer_groups()
        self._block_keys_by_request: OrderedDict[int, tuple[int, list[bytes]]] = OrderedDict()
        """``request_id -> (prompt_len, keys)``; a prompt is hashed once per request, not per ask."""

    @property
    def tokens_per_block(self) -> int:
        return self._tokens_per_block

    def local_reuse_tokens(self, request) -> int:
        """The prefix the local radix tree serves, or what the request already holds."""
        probed_tokens = self._kv_cache_manager.probe_context_reuse(request)
        if probed_tokens is not None:
            return int(probed_tokens)
        kv_cache = self._kv_cache_manager.kv_cache_map.get(request.py_request_id)
        return int(kv_cache.num_committed_tokens) if kv_cache is not None else 0

    def block_keys(self, request) -> list[bytes]:
        """One radix-tree key per full prompt block, by block ordinal."""
        request_id = request.py_request_id
        remembered = self._block_keys_by_request.get(request_id)
        if remembered is not None and remembered[0] == request.prompt_len:
            self._block_keys_by_request.move_to_end(request_id)
            return remembered[1]
        keys = self._kv_cache_manager.context_block_keys(request)
        self._block_keys_by_request[request_id] = (request.prompt_len, keys)
        while len(self._block_keys_by_request) > _BLOCK_KEY_CACHE_SIZE:
            self._block_keys_by_request.popitem(last=False)
        return keys

    def group_specs(self) -> Sequence[GroupSpec]:
        return self._group_specs

    def gen_first_ready(self, request) -> bool:
        """Always ready: the store path has no context request waiting on a generation side."""
        return True

    def fetch_extent(self, request, plan) -> tuple[CacheExtent, frozenset[bytes]]:
        """Units for the pages the scheduler reserved with ``reserve_transfer_pages``, and the
        names of the plan's blocks that reservation found committed locally.

        The reservation matches the prompt against the radix tree first, so a block another
        request committed since the plan was made is already in a shared page: it is not
        fetched over (its name is returned instead, to count as served). A block the reservation
        has no page for is left out and not returned: the fetch comes up short and is retried.
        """
        kv_cache = self._kv_cache_manager.kv_cache_map[request.py_request_id]
        committed_blocks = int(kv_cache.num_committed_tokens) // self._tokens_per_block
        units: list[Unit] = []
        committed_names: set[bytes] = set()
        for group_plan in plan.group_plans:
            page_indices = self._page_indices_by_ordinal(kv_cache, group_plan.spec.local_group)
            to_fetch = []
            for ordinal in group_plan.ordinals:
                if ordinal < committed_blocks:
                    if ordinal < len(plan.block_keys):
                        committed_names.add(group_plan.spec.tag + plan.block_keys[ordinal])
                elif ordinal < len(page_indices):
                    to_fetch.append(ordinal)
            units.extend(self._name_units(group_plan.spec, plan.block_keys, page_indices, to_fetch))
        extent = CacheExtent(
            name=f"fetch:{request.py_request_id}".encode(), units=tuple(units), is_last=True
        )
        return extent, frozenset(committed_names)

    def publish_extent_and_chunk(self, request) -> tuple[CacheExtent, None]:
        """Every committed full block the request still holds, in the pages it holds right now.

        A windowed group holds ``[0, sink) | [stale_end(history), committed)``, with ``history``
        read at publish time: after an unchunked prefill that is ``prompt_len``, one block past a
        fetcher's largest target ``B = (prompt_len - 1) // tpb * tpb``. With ``window % tpb == 0``
        the two stale ends differ exactly when ``prompt_len % tpb`` is ``0`` or ``tpb - 1``; the
        window's first block at ``B`` is then already dropped here, and since the stale end grows
        with the target no larger target does without it either. A fetcher's ``servable_blocks``
        falls to what the sink blocks alone serve (nothing, without sinks) and the request
        computes locally.

        Read after the step was committed: committing may swap a block's pages for a concurrent
        committer's (``allow_seq_rebasing``). The positional chunk is ``None``: the store path
        names whole blocks and never moves a partial tail by position.
        """
        kv_cache = self._kv_cache_manager.kv_cache_map[request.py_request_id]
        keys = self.block_keys(request)
        num_committed_blocks = min(
            int(kv_cache.num_committed_tokens) // self._tokens_per_block, len(keys)
        )
        history_length = int(kv_cache.history_length)
        units: list[Unit] = []
        for group_spec in self._group_specs:
            if group_spec.kind is not CacheKind.PAGED:
                continue
            page_indices = self._page_indices_by_ordinal(kv_cache, group_spec.local_group)
            stale_begin, stale_end = self._kv_cache_manager.stale_block_range(
                group_spec.local_group, history_length
            )
            live_ordinals = [
                ordinal
                for ordinal in range(min(num_committed_blocks, len(page_indices)))
                if not stale_begin <= ordinal < stale_end
            ]
            units.extend(self._name_units(group_spec, keys, page_indices, live_ordinals))
        extent = CacheExtent(
            name=f"publish:{request.py_request_id}".encode(),
            units=tuple(units),
            is_last=request.context_remaining_length == 0,
        )
        return extent, None

    def forget_request(self, request_id: int) -> None:
        self._block_keys_by_request.pop(request_id, None)

    # ---- helpers ----

    @staticmethod
    def _page_indices_by_ordinal(kv_cache, local_group: int) -> list[int]:
        """One page index per block ordinal; ``BAD_PAGE_INDEX`` where the group has no page."""
        return list(kv_cache.get_aggregated_page_indices(local_group, valid_only=False))

    @staticmethod
    def _name_units(
        group_spec: GroupSpec,
        block_keys: Sequence[bytes],
        page_indices: Sequence[int],
        ordinals: Sequence[int],
    ) -> list[Unit]:
        return units_for_group(
            block_keys=block_keys,
            region_ids=np.asarray([page_indices[ordinal] for ordinal in ordinals], dtype=np.int64),
            ordinals=ordinals,
            tag=group_spec.tag,
            local_group=group_spec.local_group,
        )

    def _describe_layer_groups(self) -> tuple[GroupSpec, ...]:
        """One ``GroupSpec`` per layer group: kind, shared tag, window and sink blocks."""
        group_specs = []
        for local_group, layer_group in enumerate(self._page_table.layer_groups):
            pool_roles = frozenset().union(*(view.pool_role for view in layer_group.pool_views))
            if isinstance(layer_group, MambaLayerGroup):
                group_specs.append(
                    GroupSpec(local_group, CacheKind.STATE, group_tag(pool_roles, None))
                )
                continue
            window_size = layer_group.sliding_window_size
            num_sink_blocks = (
                self._kv_cache_manager.num_sink_blocks(local_group)
                if window_size is not None
                else 0
            )
            group_specs.append(
                GroupSpec(
                    local_group,
                    CacheKind.PAGED,
                    group_tag(pool_roles, window_size),
                    window_size,
                    num_sink_blocks,
                )
            )
        return tuple(group_specs)

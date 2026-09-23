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
"""The two read-only views the KV transfer coordination layer consumes (design §6.2, §7.1).

``RequestView`` is what the layer reads of a request; ``GroupSpec`` and ``ResourceReader`` are what
it reads of this rank's cache. The engine side (``pyexecutor``) provides the request view (an
``LlmRequest`` satisfies it structurally), ``resource/`` provides the reader, and tests pass a plain
dataclass and a table. Nothing here imports the engine or the coordination layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Protocol, Sequence

from .backend import CacheKind
from .cache_backend import CacheExtent

if TYPE_CHECKING:
    from .backend import Chunk

__all__ = ["GroupSpec", "RequestView", "ResourceReader"]


class RequestView(Protocol):
    """The subset of ``LlmRequest`` the coordination layer reads.

    Duck-typed: an ``LlmRequest`` satisfies it as-is, and tests pass a plain dataclass. The
    coordination layer never writes to a request; every write goes through ``KVTransferEffects``.
    """

    @property
    def py_request_id(self) -> int: ...

    @property
    def prompt_len(self) -> int: ...

    @property
    def is_gen_init(self) -> bool:
        """A generation-side disagg request that must fetch its whole prompt KV from a context
        worker before it can run. Its plan is a short-circuit (§7.2 step 1)."""
        ...

    @property
    def is_gen_first_context(self) -> bool:
        """A context-side request whose generation side arrived first and must be ready before
        this side may be scheduled (§7.2 step 2)."""
        ...

    @property
    def route_hints(self) -> Mapping[str, Mapping[str, object]]:
        """Routing hints attached by the deployment, keyed by ``FetchSource.hint_key``.

        Each value is the opaque ``hint`` a worker backend's ``open_route`` takes. A request with
        no hints has an empty mapping.
        """
        ...


@dataclass(frozen=True)
class GroupSpec:
    """One layer group as the merge rule needs to see it (design §6.2).

    Attributes:
        local_group: This rank's ordinal for the group. Local only; never sent.
        kind: How the group's units are read (``CacheKind.PAGED`` or ``STATE``).
        tag: ``resource.naming.group_tag`` of the group; prefixed to every unit name.
        window_size: Sliding window in tokens, or ``None`` for full attention. Tokens rather than
            blocks, so that the stale range is computed with the exact formula KV v2 uses
            (``AttnLifeCycle.get_stale_range``).
        sink_blocks: Number of leading sink blocks a windowed group keeps.
    """

    local_group: int
    kind: CacheKind
    tag: bytes
    window_size: int | None = None
    sink_blocks: int = 0


class ResourceReader(Protocol):
    """Read-only view of this rank's cache resources, for planning and for building extents.

    Provided by ``resource/``; it is the only thing between the coordination layer and KV v2. The
    ``plan`` argument type is ``remote_cache.FetchPlan``; it is left untyped here to avoid a cycle.
    """

    @property
    def tokens_per_block(self) -> int: ...

    def local_reuse_tokens(self, request: RequestView) -> int:
        """Tokens the local radix tree can already serve for this prompt (advisory, no pages)."""
        ...

    def block_keys(self, request: RequestView) -> list[bytes]:
        """One key per *full* prompt block, by block ordinal."""
        ...

    def group_specs(self) -> Sequence[GroupSpec]:
        """Every layer group this rank holds."""
        ...

    def gen_first_ready(self, request: RequestView) -> bool:
        """For a gen-first context request: whether the generation side is ready."""
        ...

    def fetch_extent(self, request: RequestView, plan: object) -> CacheExtent:
        """The extent for a fetch the scheduler has already allocated pages for."""
        ...

    def publish_description(self, request: RequestView) -> tuple[CacheExtent, Chunk | None]:
        """What this context step made available: named units, and the positional chunk for
        backends that place pieces. Built from committed pages."""
        ...

    def forget_request(self, request_id: int) -> None:
        """The request is gone: drop whatever the reader remembers about it."""
        ...

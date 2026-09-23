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
"""What the KV transfer coordination layer needs from its surroundings.

Everything here is a ``Protocol`` or a small value type. The coordinator and planner depend on
these and nothing else, so they can be driven by fakes: a scripted ``Fetches``, an effects recorder,
a single-rank ``DistLike``, a table-backed ``ResourceReader``.

The engine adapter (``pyexecutor``) provides the concrete ``RequestView`` (an ``LlmRequest``
satisfies it structurally), the ``KVTransferEffects``, the ``EngineQueue`` and the ``DistLike``.
``resource/`` provides the ``ResourceReader``. Nothing in this module imports the engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Callable, Literal, Mapping, Protocol, Sequence, runtime_checkable

from ...base.cache_backend import Attempt, CacheExtent, Fetches

if TYPE_CHECKING:
    from ...base.backend import Chunk

__all__ = [
    "DEFER",
    "CarriesAux",
    "Defer",
    "DistLike",
    "EngineQueue",
    "FetchSource",
    "GroupKind",
    "GroupSpec",
    "KVTransferEffects",
    "PlacesPieces",
    "RequestView",
    "ResourceReader",
    "Scope",
]

Scope = Literal["world", "pp"]
"""Which ranks take part in one collective: the whole world, or this rank's pipeline group."""


class Defer:
    """The answer "do not schedule this request this round; ask again next round".

    A class rather than a sentinel string so that ``isinstance`` works and so that it can never be
    confused with a plan or with ``None`` (which means "compute locally, no fetch").
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "DEFER"


DEFER = Defer()
"""The single ``Defer`` instance. Compare with ``is``."""


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


class KVTransferEffects(Protocol):
    """Engine-owned side effects the coordinator may trigger (design §7.3).

    These are the coordinator's complete dependency on the engine. Every request-state write
    happens inside one of them; the coordinator itself never touches a request.
    """

    def park_for_fetch(self, requests: Sequence[RequestView]) -> None:
        """Requests whose fetch is in flight: state -> ``KV_FETCH_IN_PROGRESS``."""
        ...

    def unpark(
        self,
        request: RequestView,
        token_end: int,
        no_local_fallback: bool,
        aux: Mapping[str, object] | None,
    ) -> None:
        """A fetch landed. Settle the context cursor at ``token_end`` (or further, if local reuse
        already reached past it), commit the pages on the resource managers the assembly hosts,
        apply ``aux`` if the assembly has a consumer for it, and return the request to
        ``CONTEXT_INIT``. ``no_local_fallback`` marks a gen-init landing, which goes to the
        "landed, awaiting activation" state instead; an assembly that does not host gen-init
        requests may refuse it."""
        ...

    def give_back_fetch_pages(self, requests: Sequence[RequestView]) -> None:
        """Revert the context allocation the scheduler made for a fetch; request ->
        ``CONTEXT_INIT``. Called only after ``quiesce`` answered true for the attempts that
        named those pages."""
        ...

    def prepare_fetch_resources(self, requests: Sequence[RequestView]) -> None:
        """Prepare the non-KV resource managers (spec, draft KV) for requests about to fetch."""
        ...

    def hold_for_transfer(self, requests: Sequence[RequestView]) -> None:
        """Requests that ended while a transfer (either direction) is still in flight: state ->
        ``KV_PUBLISH_IN_PROGRESS``; release seq slot and spec resources; keep the pages, which a
        backend may still be reading or writing. ``terminate_request`` follows once every record
        of the request is released."""
        ...

    def terminate_request(self, request: RequestView) -> None:
        """Final release of a request whose transfers are all ``RELEASED``."""
        ...

    def stage_transfer_response(self, request: RequestView) -> None:
        """Context side: queue the response that tells the peer the publish is complete."""
        ...

    def fail_requests(self, requests: Sequence[RequestView], reason: str) -> None:
        """Fail requests through the engine's error path."""
        ...

    def fail_fatal(self, exc: BaseException) -> None:
        """Mark the engine fatal. Used only when a backend cannot say the caller's memory is
        untouched (``quiesce`` answered false): the pages cannot be reused or freed."""
        ...


@dataclass(frozen=True)
class FetchSource:
    """One fetch backend in the assembly table (design §7.4).

    Attributes:
        name: Stable identifier; recorded on plans and attempts, and used as the allgather key.
        backend: The backend itself.
        hint_key: Which routing hint on a request this backend reads. ``None`` for a backend whose
            destination is unique (a store), which then never gets ``open_route``.
    """

    name: str
    backend: Fetches
    hint_key: str | None


class EngineQueue(Protocol):
    """Work a backend needs done on the engine thread (design §3.1 thread rule).

    A backend ``post``s a callable from its own thread; the coordinator ``drain``s the queue at the
    head of each ``advance`` under a per-round budget. The callable may only reach KV v2 through
    ``resource/``.
    """

    def post(self, fn: Callable[[], None]) -> None: ...

    def drain(self, budget: int) -> int:
        """Run up to ``budget`` posted callables; return how many ran."""
        ...


class DistLike(Protocol):
    """The one collective the coordinator uses. It needs no rank or world size: the gathered
    list's length is the participant count, and every reduction is symmetric."""

    def allgather(self, obj: object, scope: Scope) -> list:
        """Gather ``obj`` from every rank in ``scope``; the result is indexed by rank within that
        scope. Must be called the same number of times on every participating rank."""
        ...


class GroupKind(IntEnum):
    """How a layer group's units are read; mirrors ``base.CacheKind`` without importing the
    heavier module that defines it.

    PAGED: one unit per block, the end of a range is an upper bound (attention).
    STATE: one unit standing for the whole request, the end is an exact checkpoint (recurrent).
    """

    PAGED = 0
    STATE = 1


@dataclass(frozen=True)
class GroupSpec:
    """One layer group as the merge rule needs to see it (design §6.2).

    Attributes:
        local_group: This rank's ordinal for the group. Local only; never sent.
        kind: How the group's units are read.
        tag: ``resource.naming.group_tag`` of the group; prefixed to every unit name.
        window_size: Sliding window in tokens, or ``None`` for full attention. Tokens rather than
            blocks, so that the stale range is computed with the exact formula KV v2 uses
            (``AttnLifeCycle.get_stale_range``).
        sink_blocks: Number of leading sink blocks a windowed group keeps.
    """

    local_group: int
    kind: GroupKind
    tag: bytes
    window_size: int | None = None
    sink_blocks: int = 0


class ResourceReader(Protocol):
    """Read-only view of this rank's cache resources, for planning and for building extents.

    Provided by ``resource/``; it is the only thing between the coordination layer and KV v2. The
    ``plan`` argument type is ``planner.FetchPlan``; it is left untyped here to avoid a cycle.
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


@runtime_checkable
class PlacesPieces(Protocol):
    """Optional backend capability: move the parts that have no name (trailing partial block, live
    recurrent state) by position. Implementers receive every chunk of a series (design §7.5)."""

    def place(self, chunk: Chunk) -> Attempt: ...


@runtime_checkable
class CarriesAux(Protocol):
    """Optional ``Attempt`` capability: side data that rode along with a delivery (first token,
    draft tokens, context usage). The coordinator hands it to ``unpark`` unread."""

    def aux(self) -> Mapping[str, object]: ...

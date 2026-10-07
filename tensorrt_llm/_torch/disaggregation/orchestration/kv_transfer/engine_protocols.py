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
"""What the KV transfer coordinator needs from the engine side.

Everything here is a ``Protocol`` or an enum. The engine adapter (``pyexecutor``) provides the
``KVTransferEffects``, the ``EngineQueue`` and the ``Collective``, so the coordinator can be driven
by fakes: an effects recorder and a single-rank collective. The read-only views it consumes
(``RequestView``, ``GroupSpec``, ``ResourceView``) live in ``base/views.py``; the optional
backend capabilities it recognises (``LandsOnHost``, ``PlacesPieces``, ``CarriesAux``) in
``base/capabilities.py``; the planner's answers and its source table (``DEFER``, ``FetchSource``)
in ``remote_cache.py``. Nothing in this module imports the engine.
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, Mapping, Protocol, Sequence

from ...base.views import RequestView

__all__ = [
    "Collective",
    "EngineQueue",
    "KVTransferEffects",
    "PlanAuthority",
]


class PlanAuthority(Enum):
    """Who decides a request's plan answer on this rank (design §7.1 "齐").

    ALL_RANKS: every rank plans and the collective reduces the answers; the loop where every rank
        runs the scheduler.
    OWNER: this rank plans alone and hands its answers to the other ranks with the schedule
        (``export_plan_answers``); the rank that schedules in the pipeline-parallel loop.
    FOLLOWER: this rank never plans or probes; it takes the owner's answers
        (``adopt_plan_answers``) and builds its own plans from them.

    Arrivals and expiries travel through the collective in every mode.
    """

    ALL_RANKS = "ALL_RANKS"
    OWNER = "OWNER"
    FOLLOWER = "FOLLOWER"


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
        """Requests that ended while a record of theirs remains (a transfer in flight in either
        direction, or a record waiting for the ranks' agreement): move the request to the
        engine's "held for transfer" state (the engine side names it ``KV_PUBLISH_IN_PROGRESS``,
        an alias of an existing state; a fetch hold uses the same value); release seq slot and
        spec resources; keep the pages, which a backend may still be reading or writing.
        ``terminate_request`` follows once every record of the request is released."""
        ...

    def terminate_request(self, request: RequestView) -> None:
        """Final release of a held request, once every record of it is gone."""
        ...

    def fail_requests(self, requests: Sequence[RequestView], reason: str) -> None:
        """Fail requests through the engine's error path."""
        ...

    def fail_fatal(self, exc: BaseException) -> None:
        """Mark the engine fatal. Used only when a backend cannot say the caller's memory is
        untouched (``quiesce`` answered false): the pages cannot be reused or freed."""
        ...


class EngineQueue(Protocol):
    """Work a backend needs done on the engine thread.

    A backend ``post``s a callable from its own thread; the coordinator ``drain``s the queue at the
    head of each ``advance`` under a per-round budget, so the callable runs on the engine thread,
    the only thread that may touch KV v2 (through the ``ResourceView``). No backend in this
    tree posts to it yet; the engine side provides one so the contract is complete.
    """

    def post(self, fn: Callable[[], None]) -> None: ...

    def drain(self, budget: int) -> int:
        """Run up to ``budget`` posted callables; return how many ran."""
        ...


class Collective(Protocol):
    """The one collective the coordinator uses, over the ranks that plan together (the world, or
    this rank's pipeline group under attention DP; the assembly decides which). It needs no rank
    or world size: the gathered list's length is the participant count, and every reduction is
    symmetric."""

    def allgather(self, obj: object) -> list:
        """Gather ``obj`` from every participating rank; the result is indexed by rank within the
        group. Must be called the same number of times on every participating rank."""
        ...

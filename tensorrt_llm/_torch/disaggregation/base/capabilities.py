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
"""Optional backend capabilities beside the contract in ``cache_backend.py``.

A backend implements the contract (``Fetches``, ``Publishes``) and may implement one of these in
addition; the coordination layer recognises each with ``isinstance`` and never branches on a
backend's type otherwise. ``PlacesPieces`` moves the parts that have no name; ``CarriesAux`` lets
an ``Attempt`` bring side data along; ``LandsOnHost`` with its ``Landing`` is the host-first fetch,
where a fetch lands in the backend's own host memory before any page is involved.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Mapping, Protocol, Sequence, runtime_checkable

from .cache_backend import Attempt, CacheExtent, Outcome

if TYPE_CHECKING:
    # ``backend``: Chunk/CacheKind only; ``cache_backend``: the contract.
    from .backend import Chunk

__all__ = ["CarriesAux", "Landing", "LandsOnHost", "PlacesPieces"]


@runtime_checkable
class PlacesPieces(Protocol):
    """Optional backend capability: move the parts that have no name (trailing partial block, live
    recurrent state) by position. Implementers receive every chunk of a series (design §7.5)."""

    def place_piece(self, chunk: Chunk) -> Attempt: ...


@runtime_checkable
class CarriesAux(Protocol):
    """Optional ``Attempt`` capability: side data that rode along with a delivery (first token,
    draft tokens, context usage). The coordinator hands it to ``unpark`` unread."""

    def aux(self) -> Mapping[str, object]: ...


@runtime_checkable
class Landing(Protocol):
    """One host-first landing: the content of a plan's units in the backend's own memory.

    ``poll`` reports how the landing ended; ``Delivered.served`` names the units that arrived.
    ``place`` copies the units of ``extent`` from the landing into the caller's pages, on the
    backend's own thread and stream, and reports through the returned ``Attempt`` once the copy
    has completed. ``close`` hands the landing back: its memory is no longer needed.
    """

    def poll(self) -> Outcome | None:
        """Non-blocking. ``None`` while the units are still on their way to the backend."""
        ...

    def place(self, extent: CacheExtent) -> Attempt:
        """Start copying the units named by ``extent`` into the caller's memory. Only
        ``SubmissionRejected`` may be raised, and only while nothing has been touched; any other
        failure is reported through the attempt. An empty extent completes at once."""
        ...

    def close(self) -> None:
        """Give the landing's memory back. Runs on the caller's thread and must not block or
        wait for I/O; it may be called before ``poll`` has an outcome, and calling it again, or
        after the backend has closed, does nothing."""
        ...


@runtime_checkable
class LandsOnHost(Protocol):
    """Optional backend capability: a fetch lands in the backend's own host memory first, and
    is copied into the caller's pages afterwards, so no page is held while the network is slow.

    ``Delivered`` from the landing means the content reached the backend, not the caller. Like
    ``Fetches`` it probes, quiesces and settles; unlike it there is no ``fetch`` or ``open_route``,
    so a backend is one or the other. ``quiesce`` and ``settle`` only ever receive the attempts
    ``Landing.place`` returned: those are the only ones that touch the caller's memory.
    """

    def fetch_to_host(self, units: Sequence[bytes]) -> Landing:
        """Start landing ``units`` in the backend's memory. Non-blocking. Only
        ``SubmissionRejected`` may be raised; short capacity is not an error but a wait the
        backend absorbs (the caller bounds it with its own clock)."""
        ...

    def probe(self, name: bytes, units: Sequence[bytes]) -> frozenset[bytes] | None:
        """As ``Fetches.probe``."""
        ...

    def quiesce(self, attempts: Iterable[Attempt]) -> bool:
        """As ``Fetches.quiesce``, for placement attempts."""
        ...

    def settle(self, attempts: Iterable[Attempt]) -> None:
        """As ``Fetches.settle``, for placement attempts."""
        ...

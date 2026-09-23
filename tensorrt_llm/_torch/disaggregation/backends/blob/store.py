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
"""The face a blob store driver implements for ``BlobStoreBackend``.

A blob store keeps byte objects under string keys. The backend asks it, in batches and
synchronously, whether it holds keys, to write objects out of memory and to read them into
memory. A call that fails as a whole (unreachable store, an answer of the wrong length, a
registration refused) raises ``BlobStoreError``; a call that went through answers one status per
key, as ``PutStatus`` or ``GetStatus``. The backend branches on those kinds only; the store's own
codes and messages stay inside the driver, which logs them.

A driver lives in ``drivers/`` and translates its store's conventions into this protocol.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol, Sequence

from ...base.region import Segment

__all__ = ["BlobStore", "BlobStoreError", "GetStatus", "PutStatus"]


class BlobStoreError(RuntimeError):
    """A store call failed as a whole: nothing it answered can be trusted.

    Subclassing ``RuntimeError`` is part of the contract: the backend folds a lookup that raised
    this into ``Failed("store lookup failed: ...")``, and ``probe`` re-raises it as a
    ``RuntimeError``.
    """


class PutStatus(Enum):
    """Per-key answer of ``BlobStore.put``."""

    STORED = "stored"
    """The store now holds the object under this key."""
    DECLINED = "declined"
    """The store chose not to hold this key. Not an error: the backend then asks ``holds`` to learn
    whether the key is held all the same."""
    FAILED = "failed"
    """This key could not be written."""


class GetStatus(Enum):
    """Per-key answer of ``BlobStore.get``."""

    HIT = "hit"
    """The whole object was written into the key's buffers."""
    MISS = "miss"
    """The store does not hold this key; the key's buffers were not touched."""
    FAILED = "failed"
    """Anything else: a short read, a size mismatch, a transport error on this key. The buffers may
    be partly written."""


class BlobStore(Protocol):
    """Batch, synchronous access to one blob store.

    One key is one object. A unit spread over several memory segments is one object all the
    same: ``buffers[i]`` lists that key's ``(address, size)`` segments, in order, and the store
    treats their concatenation as the object. This is a requirement of the backend, whose units
    are stored whole or not at all (contract §6.3). A store that takes one buffer per key gathers
    the segments itself inside its driver.

    Every segment handed to ``put`` or ``get`` lies inside a span that ``register_span`` accepted.

    Calls arrive concurrently: the backend runs ``num_workers`` delivery threads and one probe
    thread, and any of them may be inside ``holds``, ``put`` or ``get`` at the same time, while
    ``register_span`` / ``unregister_span`` come from the engine thread. A driver whose client is
    not thread-safe serialises the calls or pools connections itself.
    """

    def register_span(self, address: int, size: int) -> None:
        """Let the store read and write ``[address, address + size)``.

        Raises ``BlobStoreError`` when the store refuses. A store that reaches memory without
        registration implements this as a no-op.
        """
        ...

    def unregister_span(self, address: int, size: int) -> None:
        """Undo ``register_span`` for the same span. Raises ``BlobStoreError`` on failure; a no-op
        for a store that needs no registration."""
        ...

    def holds(self, keys: Sequence[str]) -> Sequence[bool]:
        """Whether the store holds each key, in order.

        Raises ``BlobStoreError`` when it cannot tell (transport failure, an answer of the wrong
        length). It never answers ``False`` for a key it could not look up: the backend must tell
        a miss from an outage (contract §5.2).
        """
        ...

    def put(self, keys: Sequence[str], buffers: Sequence[Sequence[Segment]]) -> Sequence[PutStatus]:
        """Write one object per key out of its segments. One status per key, in order."""
        ...

    def get(self, keys: Sequence[str], buffers: Sequence[Sequence[Segment]]) -> Sequence[GetStatus]:
        """Read one object per key into its segments. One status per key, in order.

        ``HIT`` means the object filled the segments exactly; an object of another size or a
        short read is ``FAILED``; a key the store does not hold is ``MISS`` and its segments are
        left as they were.
        """
        ...

    def close(self) -> None:
        """Release the connection. Idempotent; does not raise."""
        ...

    def describe(self) -> str:
        """One line naming the store for an operator: the driver and where it connects to.

        The backend prefixes its log lines with it.
        """
        ...

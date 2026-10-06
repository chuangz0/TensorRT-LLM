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
"""The in-process driver: ``type: memory`` keeps objects in a dictionary of this process.

It reaches no other process, so it shares nothing between engines; it exists so that the blob
path can be run and tested end to end without a store service, and as the store behind the test
fakes. An entry of this type takes the ``BlobStoreConfig`` keys and no others.
"""

from __future__ import annotations

import ctypes
import threading
from typing import Any, Mapping, Sequence

from ....base.region import Segment
from ...config import BackendEntry
from ...registry import BackendBuildContext, BackendHandle
from ..factory import build_blob_backend
from ..store import BlobStoreError, GetStatus, PutStatus

__all__ = ["MemoryBlobStore", "build_memory_backend"]


class MemoryBlobStore:
    """``BlobStore`` over a dictionary. ``objects`` and ``registered`` are open for tests to read.

    Segments must lie inside a registered span, as they must for a store that reaches memory over
    a transport; a key whose segments do not is ``FAILED`` and nothing is copied. A put of a key
    already held keeps the first object and still answers ``STORED``, as a Mooncake master does.
    """

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.registered: dict[int, int] = {}
        """address -> size of every live registration."""
        self._lock = threading.Lock()

    def register_span(self, address: int, size: int) -> None:
        with self._lock:
            self.registered[address] = size

    def unregister_span(self, address: int, size: int) -> None:
        with self._lock:
            if self.registered.pop(address, None) is None:
                raise BlobStoreError(f"[{address:#x}, {address + size:#x}) is not registered")

    def holds(self, keys: Sequence[str]) -> list[bool]:
        with self._lock:
            return [key in self.objects for key in keys]

    def put(self, keys: Sequence[str], buffers: Sequence[Sequence[Segment]]) -> list[PutStatus]:
        results = []
        for key, segments in zip(keys, buffers):
            if not self._covers(segments):
                results.append(PutStatus.FAILED)
                continue
            data = b"".join(ctypes.string_at(address, size) for address, size in segments)
            with self._lock:
                self.objects.setdefault(key, data)
            results.append(PutStatus.STORED)
        return results

    def get(self, keys: Sequence[str], buffers: Sequence[Sequence[Segment]]) -> list[GetStatus]:
        results = []
        for key, segments in zip(keys, buffers):
            with self._lock:
                data = self.objects.get(key)
            if data is None:
                results.append(GetStatus.MISS)
                continue
            # The segments must add up to the object exactly; nothing is written otherwise.
            if not self._covers(segments) or sum(size for _, size in segments) != len(data):
                results.append(GetStatus.FAILED)
                continue
            offset = 0
            for address, size in segments:
                ctypes.memmove(address, data[offset : offset + size], size)
                offset += size
            results.append(GetStatus.HIT)
        return results

    def close(self) -> None:
        pass

    def describe(self) -> str:
        return "memory (this process only)"

    def _covers(self, segments: Sequence[Segment]) -> bool:
        with self._lock:
            spans = tuple(self.registered.items())
        return all(
            any(base <= address and address + size <= base + span for base, span in spans)
            for address, size in segments
        )


def build_memory_backend(entry: BackendEntry, context: BackendBuildContext) -> BackendHandle:
    """Factory for ``type: memory``."""

    def open_store(options: Mapping[str, Any]) -> MemoryBlobStore:
        return MemoryBlobStore()

    return build_blob_backend(entry, context, open_store=open_store, driver_fields=())

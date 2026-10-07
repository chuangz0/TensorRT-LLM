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
"""Copies between the caller's device memory and a backend's host memory.

Every backend that stages through host memory moves bytes the same way: asynchronous copies
issued on the calling thread, then one wait for that thread's copies. ``Copier`` is that surface;
``CudaCopier`` is the CUDA runtime implementation. Nothing here imports torch or CUDA at module
load: the runtime binding is imported when a ``CudaCopier`` is made.
"""

from __future__ import annotations

import threading
from typing import Literal, Protocol, Sequence

__all__ = ["Copier", "CopyKind", "CudaCopier"]

CopyKind = Literal["d2h", "h2d"]


class Copier(Protocol):
    """Asynchronous copies between the caller's memory and host slots, on the calling thread."""

    def copy(self, dst: int, src: int, size: int, kind: CopyKind) -> None: ...

    def wait_for_copies(self) -> None:
        """Block until every copy this thread issued has completed."""
        ...


class CudaCopier:
    """``cudaMemcpyAsync`` on a per-thread stream, on the rank's device.

    Torch's current device is thread-local, so each worker thread that copies first selects the
    device the pools live on; a stream created before that would belong to device 0.
    """

    def __init__(self, device_index: int | None) -> None:
        try:
            from cuda.bindings import runtime as cudart
        except ImportError:
            from cuda import cudart
        self._cudart = cudart
        self._device_index = device_index
        self._local = threading.local()
        self._kinds = {
            "d2h": cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            "h2d": cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        }

    def _stream(self) -> int:
        stream = getattr(self._local, "stream", None)
        if stream is None:
            if self._device_index is not None:
                self._check(self._cudart.cudaSetDevice(self._device_index))
            status, stream = self._cudart.cudaStreamCreate()
            self._check((status,))
            self._local.stream = stream
        return stream

    def _check(self, result: Sequence[object]) -> None:
        status = result[0]
        if status != self._cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA runtime call failed with {status}")

    def copy(self, dst: int, src: int, size: int, kind: CopyKind) -> None:
        status = self._cudart.cudaMemcpyAsync(
            int(dst), int(src), int(size), self._kinds[kind], self._stream()
        )[0]
        if status != self._cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(
                f"cudaMemcpyAsync({kind}) failed with {status}: dst={int(dst):#x} "
                f"src={int(src):#x} size={size} device={self._device_index}"
            )

    def wait_for_copies(self) -> None:
        self._check(self._cudart.cudaStreamSynchronize(self._stream()))

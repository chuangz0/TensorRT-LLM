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
"""Pinned host slots a unit passes through when the store cannot reach the caller's memory.

The direct path registers the caller's pools with the store, which needs GPUDirect RDMA for device
memory. Staging instead registers one pinned host buffer: a unit is gathered into a slot before a
put and scattered out of one after a get. A slot holds the unit's segments concatenated in resolver
order, which is the same byte string the direct path produces from the same segments, so a pool
written by either path is readable by the other.

Nothing here imports torch or CUDA at module load. Copies go through a ``Copier``; the default one
is created lazily by ``open_default_staging``.
"""

from __future__ import annotations

import threading
from typing import Literal, Protocol, Sequence

from .client import StoreClient
from .regions import Segment

__all__ = ["Copier", "HostStagingPool", "open_default_staging", "plan_slot_geometry"]

CopyKind = Literal["d2h", "h2d"]


class Copier(Protocol):
    """Asynchronous copies between the caller's memory and host slots, on the calling thread."""

    def copy(self, dst: int, src: int, size: int, kind: CopyKind) -> None: ...

    def sync(self) -> None:
        """Block until every copy this thread issued has completed."""
        ...


def plan_slot_geometry(
    max_unit_bytes: int, transfer_batch_size: int, budget_bytes: int
) -> tuple[int, int]:
    """Slot width and slot count: one slot per unit of a batch, within ``budget_bytes``.

    A slot must hold the largest unit, so that size is a floor on the allocation; a budget below one
    unit yields one slot rather than a refusal.
    """
    if max_unit_bytes <= 0:
        raise ValueError(f"max_unit_bytes must be > 0, got {max_unit_bytes}")
    if transfer_batch_size <= 0:
        raise ValueError(f"transfer_batch_size must be > 0, got {transfer_batch_size}")
    affordable = budget_bytes // max_unit_bytes
    return max_unit_bytes, max(1, min(transfer_batch_size, affordable))


class HostStagingPool:
    """A registered host buffer cut into equal slots, shared by the backend's worker threads.

    Args:
        base: Address of the buffer. It must already be registered with the store.
        slot_bytes: Width of one slot; a unit larger than this cannot be staged.
        num_slots: How many units may be staged at once.
        copier: Issues and waits for the copies.
        keepalive: Whatever owns the buffer's memory; held so it outlives the pool.
    """

    def __init__(
        self,
        base: int,
        slot_bytes: int,
        num_slots: int,
        copier: Copier,
        *,
        keepalive: object = None,
    ) -> None:
        if slot_bytes <= 0 or num_slots <= 0:
            raise ValueError(f"need positive slot geometry, got {slot_bytes} x {num_slots}")
        self._base = int(base)
        self._slot_bytes = int(slot_bytes)
        self._num_slots = int(num_slots)
        self._copier = copier
        self._keepalive = keepalive
        self._free = list(range(self._num_slots))
        self._cond = threading.Condition()
        self._shutdown = False

    @property
    def slot_bytes(self) -> int:
        return self._slot_bytes

    @property
    def num_slots(self) -> int:
        return self._num_slots

    def fits(self, total: int) -> bool:
        return 0 < total <= self._slot_bytes

    def slot_address(self, slot: int) -> int:
        if not 0 <= slot < self._num_slots:
            raise IndexError(f"slot {slot} out of range [0, {self._num_slots})")
        return self._base + slot * self._slot_bytes

    def acquire(self, count: int) -> list[int]:
        """Take ``count`` slots, blocking until that many are free. All or nothing.

        Raises ``RuntimeError`` once the pool is shut down, including for a waiter already parked.
        """
        if not 0 < count <= self._num_slots:
            raise ValueError(f"cannot acquire {count} of {self._num_slots} slots")
        with self._cond:
            while len(self._free) < count:
                if self._shutdown:
                    raise RuntimeError("staging pool is shut down")
                self._cond.wait(timeout=1.0)
            if self._shutdown:
                raise RuntimeError("staging pool is shut down")
            taken, self._free = self._free[:count], self._free[count:]
            return taken

    def release(self, slots: Sequence[int]) -> None:
        with self._cond:
            self._free.extend(slots)
            self._cond.notify_all()

    def shutdown(self) -> None:
        """Wake every waiter with an error and refuse further ``acquire`` calls."""
        with self._cond:
            self._shutdown = True
            self._cond.notify_all()

    def _check_fits(self, segments: Sequence[Segment]) -> int:
        total = sum(size for _, size in segments)
        if not self.fits(total):
            raise ValueError(f"unit of {total} B exceeds the {self._slot_bytes} B slot")
        return total

    def gather(self, slot: int, segments: Sequence[Segment]) -> int:
        """Copy a unit's segments into ``slot``, concatenated. Returns the byte total."""
        total = self._check_fits(segments)
        destination = self.slot_address(slot)
        offset = 0
        for address, size in segments:
            self._copier.copy(destination + offset, address, size, "d2h")
            offset += size
        return total

    def scatter(self, slot: int, segments: Sequence[Segment]) -> None:
        """Inverse of :meth:`gather`: copy ``slot`` back out to the unit's segments."""
        self._check_fits(segments)
        source = self.slot_address(slot)
        offset = 0
        for address, size in segments:
            self._copier.copy(address, source + offset, size, "h2d")
            offset += size

    def sync(self) -> None:
        """Wait for this thread's copies to land."""
        self._copier.sync()


class _CudaCopier:
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

    def sync(self) -> None:
        self._check(self._cudart.cudaStreamSynchronize(self._stream()))


def open_default_staging(
    client: StoreClient,
    *,
    slot_bytes: int,
    num_slots: int,
    device_index: int | None = None,
) -> HostStagingPool:
    """Allocate a pinned buffer with torch, register it with ``client`` and wrap it in a pool.

    Imports torch here and nowhere else in the package.
    """
    import torch

    pinned = torch.cuda.is_available()
    buffer = torch.empty(slot_bytes * num_slots, dtype=torch.uint8, pin_memory=pinned)
    base = int(buffer.data_ptr())
    status = client.register_buffer(base, buffer.numel())
    if status != 0:
        raise RuntimeError(
            f"register_buffer failed with status {status} for the staging buffer at "
            f"[{base:#x}, {base + buffer.numel():#x})"
        )
    return HostStagingPool(base, slot_bytes, num_slots, _CudaCopier(device_index), keepalive=buffer)

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
memory. A slot pool instead registers one pinned host buffer: a unit is gathered into a slot before
a put, or lands in a slot from a get and is scattered out later. A slot holds the unit's segments
concatenated in resolver order, which is the same byte string the direct path produces from the
same segments, so a pool written by either path is readable by the other.

Slots are handed out two ways. ``acquire`` blocks the calling thread until enough slots are free;
the publish path uses it from a worker thread whose slots go back when its task returns. ``enqueue``
never blocks: a ``SlotWaiter`` is granted its slots at once when they are free, otherwise it waits
in a queue and the thread that later returns enough slots grants it from inside ``release``. The
host-first fetch path uses it, because its slots are held across scheduler rounds and a worker
parked for one would be a worker lost. One pool serves one of the two styles; the backend keeps a
pool per style.

Nothing here imports torch or CUDA at module load. Copies go through a ``Copier``
(``backends/host_copy.py``); the default one is created lazily by ``open_pinned_slot_pool``.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Protocol, Sequence

from ...base.region import Segment
from ..host_copy import Copier, CudaCopier
from .store import BlobStore, BlobStoreError

__all__ = ["HostSlotPool", "SlotWaiter", "open_pinned_slot_pool", "plan_slot_geometry"]


def plan_slot_geometry(
    max_unit_bytes: int, max_slots: int | None, budget_bytes: int
) -> tuple[int, int]:
    """Slot width and slot count: one slot per unit, as many as ``budget_bytes`` affords.

    A slot must hold the largest unit, so that size is a floor on the allocation; a budget below one
    unit yields one slot rather than a refusal. ``max_slots`` caps the count when given; ``None``
    takes every slot the budget affords.
    """
    if max_unit_bytes <= 0:
        raise ValueError(f"max_unit_bytes must be > 0, got {max_unit_bytes}")
    if max_slots is not None and max_slots <= 0:
        raise ValueError(f"max_slots must be > 0, got {max_slots}")
    affordable = budget_bytes // max_unit_bytes
    if max_slots is not None:
        affordable = min(max_slots, affordable)
    return max_unit_bytes, max(1, affordable)


class SlotWaiter(Protocol):
    """What ``HostSlotPool.enqueue`` takes: told once how its wait ended.

    Either call may come on the enqueuing thread (slots were free) or on whichever thread later
    returned slots or shut the pool down; neither may block.
    """

    def slots_granted(self, slots: list[int]) -> None:
        """The waiter now holds ``slots``; it returns them with ``release``."""
        ...

    def slots_refused(self, reason: str) -> None:
        """The pool was shut down before the waiter's turn; it will never hold slots."""
        ...


class HostSlotPool:
    """A registered host buffer cut into equal slots, shared by the backend's worker threads.

    Args:
        base: Address of the buffer. It must already be registered with the store.
        slot_bytes: Width of one slot; a unit larger than this cannot pass through the pool.
        num_slots: How many units may sit in slots at once.
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
        self._waiting: deque[tuple[SlotWaiter, int]] = deque()
        """Waiters in arrival order; the head is granted first, so a large ask is not starved."""
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

    # ---- blocking hand-out ----

    def acquire(self, count: int) -> list[int]:
        """Take ``count`` slots, blocking until that many are free. All or nothing.

        Raises ``RuntimeError`` once the pool is shut down, including for a waiter already parked.
        """
        self._check_count(count)
        with self._cond:
            while len(self._free) < count:
                if self._shutdown:
                    raise RuntimeError("slot pool is shut down")
                self._cond.wait()  # ``release`` and ``shutdown`` both notify
            if self._shutdown:
                raise RuntimeError("slot pool is shut down")
            return self._take(count)

    # ---- queued hand-out ----

    def enqueue(self, waiter: SlotWaiter, count: int) -> None:
        """Grant ``count`` slots to ``waiter`` now if they are free and nobody is ahead of it,
        otherwise queue it; never blocks. A shut-down pool refuses it at once."""
        self._check_count(count)
        with self._cond:
            if self._shutdown:
                refused = True
            elif not self._waiting and len(self._free) >= count:
                refused, slots = False, self._take(count)
            else:
                self._waiting.append((waiter, count))
                return
        if refused:
            waiter.slots_refused("slot pool is shut down")
        else:
            waiter.slots_granted(slots)

    def dequeue(self, waiter: SlotWaiter) -> bool:
        """Drop ``waiter`` from the queue. ``False`` when it is not there: it was never queued, has
        been granted (or is about to be, by a thread that popped it and has yet to call
        ``slots_granted``), or was refused."""
        with self._cond:
            for index, (queued, _) in enumerate(self._waiting):
                if queued is waiter:
                    del self._waiting[index]
                    return True
        return False

    def release(self, slots: Sequence[int]) -> None:
        """Give slots back, wake blocked acquirers, and grant queued waiters in order while the
        head's ask fits. Grants run on this thread, after the lock is dropped."""
        with self._cond:
            self._free.extend(slots)
            self._cond.notify_all()
            grants = self._pop_grantable()
        for waiter, granted in grants:
            waiter.slots_granted(granted)

    def shutdown(self) -> None:
        """Wake every blocked acquirer with an error, refuse every queued waiter, and refuse
        further ``acquire`` / ``enqueue`` calls. Slots may still be released afterwards."""
        with self._cond:
            self._shutdown = True
            self._cond.notify_all()
            refused, self._waiting = list(self._waiting), deque()
        for waiter, _ in refused:
            waiter.slots_refused("slot pool is shut down")

    def _check_count(self, count: int) -> None:
        if not 0 < count <= self._num_slots:
            raise ValueError(f"cannot take {count} of {self._num_slots} slots")

    def _take(self, count: int) -> list[int]:
        """Caller holds the lock and has checked that ``count`` slots are free."""
        taken, self._free = self._free[:count], self._free[count:]
        return taken

    def _pop_grantable(self) -> list[tuple[SlotWaiter, list[int]]]:
        """Caller holds the lock. Pops waiters from the head while their ask is satisfiable."""
        grants = []
        while self._waiting and len(self._free) >= self._waiting[0][1]:
            waiter, count = self._waiting.popleft()
            grants.append((waiter, self._take(count)))
        return grants

    # ---- copies ----

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


def open_pinned_slot_pool(
    store: BlobStore,
    *,
    slot_bytes: int,
    num_slots: int,
    device_index: int | None = None,
) -> HostSlotPool:
    """Allocate a pinned buffer with torch, register it with ``store`` and wrap it in a pool.

    Imports torch here and nowhere else in the package.
    """
    import torch

    pinned = torch.cuda.is_available()
    buffer = torch.empty(slot_bytes * num_slots, dtype=torch.uint8, pin_memory=pinned)
    base = int(buffer.data_ptr())
    try:
        store.register_span(base, buffer.numel())
    except BlobStoreError as exc:
        raise BlobStoreError(
            f"{exc} for the slot pool buffer at [{base:#x}, {base + buffer.numel():#x})"
        ) from exc
    return HostSlotPool(base, slot_bytes, num_slots, CudaCopier(device_index), keepalive=buffer)

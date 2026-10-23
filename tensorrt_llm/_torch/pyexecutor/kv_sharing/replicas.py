# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which tensor-parallel ranks hold the same KV cache bytes, and which of them puts each block.

Mooncake refuses a second put of a key whose put is open or done (OBJECT_ALREADY_EXISTS), and after
its put_start_discard_timeout lets the second writer take over the first's open put: ranks putting
the same objects would make each object's fate depend on another rank's put. So each object has
exactly one writer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

from ..kv_cache.sharing import MemoryKind


@dataclass(frozen=True)
class ReplicaSet:
    """The tensor-parallel ranks holding the same KV cache bytes as this one (MLA's latent cache,
    or KV heads several ranks share): this rank's position among them and their number.

    Under GPU Direct every member lends each publication's range, so its pages lock alike on every
    rank, and the member whose position is the block's index modulo the size puts the block. Under
    host staging the first member puts every block and the others lend nothing (a host lease locks
    no page): striping would copy each block out to host memory on every member and take staging
    slots on all of them, where one member's copy suffices.
    """

    position: int = 0
    size: int = 1

    def blocks_put(self, start_block: int, end_block: int, memory_kind: MemoryKind) -> int:
        """How many of the blocks ``[start_block, end_block)`` this rank puts. Under GPU Direct:
        the blocks ``b`` with ``b % size == position``, which ``TransferIntent.put_stripe``
        selects (block ``b`` is page ordinal ``b`` without context parallelism)."""
        if self.size == 1:
            return end_block - start_block
        if memory_kind is MemoryKind.HOST:
            return end_block - start_block if self.position == 0 else 0
        first = start_block + (self.position - start_block) % self.size
        return len(range(first, end_block, self.size))

    def put_stripe(self, memory_kind: MemoryKind) -> Optional[tuple[int, int]]:
        """``(position, size)`` where this rank puts a stripe of the rows it lends (GPU Direct with
        replicas); None where it puts every row it lends."""
        if self.size > 1 and memory_kind is MemoryKind.DEVICE:
            return self.position, self.size
        return None


def replica_set(shards_by_rank: Sequence[Sequence[tuple[int, int]]], rank: int) -> ReplicaSet:
    """``rank``'s place among the ranks that hold its share of every layer group.

    Args:
        shards_by_rank: Each rank's ``(count, index)`` share of each layer group, in rank order.
        rank: This rank.

    Raises:
        ValueError: Layer groups are shared by different sets of ranks: no one set of members
            could split every group's blocks. Every rank raises alike from the same list.
    """
    partitions = {_holders(shares) for shares in zip(*shards_by_rank)}
    if len(partitions) > 1:
        raise ValueError(
            "layer groups are shared by different sets of ranks "
            f"({' vs '.join(str(list(p)) for p in sorted(partitions))}), so no one rank could "
            "put each stored block"
        )
    (partition,) = partitions
    members = next(ranks for ranks in partition if rank in ranks)
    return ReplicaSet(members.index(rank), len(members))


def _holders(shares: Sequence[tuple[int, int]]) -> tuple[tuple[int, ...], ...]:
    """The ranks holding each distinct share of one layer group, given each rank's share."""
    by_share: dict[tuple[int, int], list[int]] = {}
    for rank, share in enumerate(shares):
        by_share.setdefault(share, []).append(rank)
    return tuple(sorted(tuple(ranks) for ranks in by_share.values()))

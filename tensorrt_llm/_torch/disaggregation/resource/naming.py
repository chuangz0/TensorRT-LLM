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
"""Turning this rank's region ids into names both sides compute the same way.

The cache manager hands out one key per *full* block, by block ordinal. The id arrays handed to a
transfer are not indexed by ordinal: a group with no page for a block is skipped, a sliding window
drops blocks off the front, a generation-side ask drops more, and beam tails are appended past the
end. Pairing the two by list position is the mistake this module exists to make impossible -- it
produces well-formed names for the wrong tokens, and nothing downstream can tell.

So the caller carries the block ordinal of every id it passes, and everything else follows from
rules written here rather than at each call site. An offset would not do: with the placeholders for
absent pages suppressed, a group's id list is a *subsequence* of the ordinals rather than a window
into them.

**What does not get a name**, and why each is a decision rather than an omission:

- *The trailing partial block.* Keys exist for full blocks only. Naming it by its prefix plus a
  length would give two requests with the same prefix and the same partial length one name for
  different tokens, which is a name that serves bytes nobody generated. The cache manager draws the
  same line: it commits full blocks.
- *A recurrent group's slot.* One slot stands for the whole request and a slot rolled to another
  position is not a prefix of this one, so there is nothing for a prefix key to mean.
- *A beam tail.* Copies of the last block for beams past the first are the same content under the
  same key; an extent may not carry one name twice.
- *A page that is not there.* Region ids legitimately carry a negative for "no page", and a
  coordinate has to be a real one.

Each of those still moves -- the placement layer's ``Chunk`` carries the ids -- it simply is not
offered to anything that addresses by content.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import FrozenSet, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..base.cache_backend import Unit

__all__ = ["GROUP_TAG_BYTES", "GroupNaming", "group_tag", "units_for_extent", "units_for_group"]

GROUP_TAG_BYTES = 8
"""Width of the group tag prefixed to every unit name.

Fixed width so the two parts stay separable: a name is opaque bytes and carries no delimiter of its
own, and a variable-width prefix would let one pair of (tag, key) spell another.
"""


def group_tag(pool_role: FrozenSet[str], window_size: Optional[int]) -> bytes:
    """A short stand-in for a layer group's *shared* identity.

    One token position has one block key whatever the layer group, so a full-attention group and a
    windowed group covering the same tokens compute the same key. Two units of one extent may not
    share a name, and across a store the two groups would serve each other's bytes -- so the group
    has to be inside the name.

    It has to be the identity **both sides agree on**. ``pool_role`` is that: the paired path
    already matches pools across ranks by comparing role sets, never by ordinal. The window size
    goes in beside it because two groups can share a role and cover different spans. A layer group
    ordinal must never be used here; it is numbered independently on each side.
    """
    material = b"|".join(sorted(role.encode() for role in pool_role))
    material += b"#" + (b"none" if window_size is None else str(int(window_size)).encode())
    return hashlib.blake2b(material, digest_size=GROUP_TAG_BYTES).digest()


def units_for_group(
    *,
    block_keys: Sequence[bytes],
    region_ids: np.ndarray,
    ordinals: Sequence[int],
    tag: bytes,
    local_group: int,
) -> List[Unit]:
    """Name the blocks of one layer group, in the order its ids are given.

    Args:
        block_keys: one key per full block, indexed by block ordinal from the start of the
            sequence. Shorter than the sequence whenever it ends mid-block.
        region_ids: this group's ids, already trimmed the way the transfer will use them.
        ordinals: the block ordinal each entry of ``region_ids`` stands for, same length.
        tag: from :func:`group_tag`.
        local_group: this rank's ordinal for the group, recorded on each unit and never sent.

    Returns:
        One unit per id that has both a page and a key, in input order. Shorter than
        ``region_ids`` whenever something was skipped, and skipping is normal.

    The ordinals are given one per entry rather than as a starting offset, because there is no
    offset that would do. A group's id array already omits the blocks it has no page for before
    any trimming happens, so its positions are not a shifted copy of the block ordinals -- they are
    a subsequence of them.
    """
    if len(tag) != GROUP_TAG_BYTES:
        raise ValueError(f"a group tag is {GROUP_TAG_BYTES} bytes, not {len(tag)}")
    ids = np.asarray(region_ids).reshape(-1).tolist()
    ordinal_list = [int(o) for o in ordinals]
    if len(ordinal_list) != len(ids):
        raise ValueError(f"{len(ids)} ids but {len(ordinal_list)} ordinals")
    if any(o < 0 for o in ordinal_list):
        raise ValueError("negative block ordinal")

    units: List[Unit] = []
    seen: set[bytes] = set()
    for region, ordinal in zip(ids, ordinal_list):
        region = int(region)
        if region < 0:
            # No page for this block in this group. The id stays in the array the transfer uses.
            continue
        if ordinal >= len(block_keys):
            # Past the last full block: the trailing partial, or a beam tail appended after it.
            continue
        name = tag + block_keys[ordinal]
        if name in seen:
            # A beam tail repeating content already named. One name, one unit.
            continue
        seen.add(name)
        units.append(Unit(name=name, local_group=local_group, local=region))
    return units


@dataclass(frozen=True)
class GroupNaming:
    """One layer group's contribution to an extent: which ids, and which blocks they stand for.

    ``local_group`` is this rank's ordinal for the group and is recorded on each unit but never
    sent; ``tag`` is the part both sides compute alike.
    """

    local_group: int
    tag: bytes
    region_ids: np.ndarray
    ordinals: Sequence[int]


def units_for_extent(
    *,
    block_keys: Sequence[bytes],
    groups: Iterable[GroupNaming],
) -> Tuple[Unit, ...]:
    """Every unit of one extent, in layer-group order.

    Groups do not collide with each other by construction -- the tag is inside every name and two
    groups with the same tag would be the same group -- so de-duplication stays within a group.

    A group that contributes nothing contributes no units. That is not the same as a group that
    does not exist, and nothing here records the difference: the placement description alongside
    keeps one entry per group, and it is what the transfer is driven by.
    """
    units: List[Unit] = []
    for group in groups:
        units.extend(
            units_for_group(
                block_keys=block_keys,
                region_ids=group.region_ids,
                ordinals=group.ordinals,
                tag=group.tag,
                local_group=group.local_group,
            )
        )
    return tuple(units)

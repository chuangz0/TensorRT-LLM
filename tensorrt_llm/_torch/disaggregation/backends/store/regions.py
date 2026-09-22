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
"""The mapping from a unit's local coordinates to memory, which the contract leaves to the backend."""

from __future__ import annotations

from typing import Protocol, Sequence

__all__ = ["RegionResolver", "Segment"]

Segment = tuple[int, int]
"""``(address, size)`` of one contiguous piece of a unit."""


class RegionResolver(Protocol):
    """Resolve ``(local_group, local)`` to the segments that hold the unit, in a fixed order.

    The order is part of the stored byte layout: segments are concatenated into one object, so a
    resolver that reorders them between two processes makes their bytes disagree. Whatever fixes
    the order must therefore be folded into the layout fingerprint the key scheme carries.

    Raises ``KeyError`` or ``ValueError`` for coordinates it does not know.
    """

    def __call__(self, local_group: int, local: int) -> Sequence[Segment]: ...

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
"""How a unit's name becomes a store key.

A unit name is opaque bytes; it is hex-encoded, never decoded, so the key stays a reversible,
equality-preserving re-encoding of the name (contract §4.1 invariant 2). The extent name is not part
of the key: a store names content, and the same unit is asked for under many extents.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["KeyScheme"]


@dataclass(frozen=True)
class KeyScheme:
    """``<namespace>/<layout fingerprint hex>/<unit name hex>``.

    Attributes:
        namespace: Deployment-chosen prefix; the only human-readable part.
        layout_fingerprint: Digest of the local memory layout, supplied by whoever derives names
            (``resource/``). Two processes whose bytes would not mean the same thing must
            disagree here, so that they miss each other rather than read each other's bytes.
    """

    namespace: str
    layout_fingerprint: bytes

    def __post_init__(self) -> None:
        if not self.namespace:
            raise ValueError("namespace must not be empty")
        if not self.layout_fingerprint:
            raise ValueError("layout_fingerprint must not be empty")

    @property
    def prefix(self) -> str:
        """The literal string every key of this scheme starts with."""
        return f"{self.namespace}/{self.layout_fingerprint.hex()}"

    def key(self, name: bytes) -> str:
        """The store key holding the unit called ``name``."""
        return f"{self.prefix}/{name.hex()}"

    def name(self, key: str) -> bytes:
        """Inverse of :meth:`key`; raises ``ValueError`` for a key of another scheme."""
        prefix = self.prefix + "/"
        if not key.startswith(prefix):
            raise ValueError(f"key {key!r} is not under {prefix!r}")
        return bytes.fromhex(key[len(prefix) :])

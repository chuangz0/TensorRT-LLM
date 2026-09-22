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
"""The KV transfer configuration file: which backends to assemble, and the coordinator's limits.

Read from the YAML named by ``TRTLLM_KV_TRANSFER_CONFIG`` (integration plan §8). The ``backends``
list is the assembly table of design §7.4: its order is the fetch priority, each entry names a
registered backend type and carries that type's own settings unread. Nothing here imports a backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import yaml

__all__ = [
    "BACKEND_ROLES",
    "KV_TRANSFER_CONFIG_ENV",
    "BackendEntry",
    "KVTransferConfig",
    "load_kv_transfer_config",
]

KV_TRANSFER_CONFIG_ENV = "TRTLLM_KV_TRANSFER_CONFIG"
"""Environment variable naming the YAML file; unset means no KV transfer backends."""

BACKEND_ROLES = ("fetch", "publish")
"""What a backend entry may be used for. A backend with both roles fetches and publishes."""

_ENTRY_KEYS = ("name", "type", "hint_key", "roles")
_COORDINATOR_KEYS = ("fetch_timeout_s", "publish_timeout_s", "probe_timeout_s", "close_timeout_s")
_DEFAULT_PROBE_TIMEOUT_S = 0.05
_DEFAULT_CLOSE_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class BackendEntry:
    """One line of the assembly table.

    Attributes:
        name: The ``FetchSource.name``; unique within the file, recorded on plans and attempts.
        type: Registry key of the backend type.
        hint_key: Which routing hint on a request this backend reads; ``None`` for a backend
            whose destination is unique (a store).
        roles: Subset of ``BACKEND_ROLES``.
        options: The type's own settings, passed to its factory unread.
    """

    name: str
    type: str
    hint_key: str | None
    roles: frozenset[str]
    options: Mapping[str, Any]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> BackendEntry:
        for key in ("name", "type"):
            if not raw.get(key):
                raise ValueError(f"backend entry needs a non-empty {key!r}: {dict(raw)}")
        roles = frozenset(raw.get("roles", BACKEND_ROLES))
        unknown_roles = sorted(roles - set(BACKEND_ROLES))
        if unknown_roles or not roles:
            raise ValueError(
                f"backend {raw['name']!r}: roles must be a non-empty subset of "
                f"{list(BACKEND_ROLES)}, got {sorted(roles)}"
            )
        options = {key: value for key, value in raw.items() if key not in _ENTRY_KEYS}
        return cls(
            name=str(raw["name"]),
            type=str(raw["type"]),
            hint_key=raw.get("hint_key"),
            roles=roles,
            options=options,
        )

    @property
    def fetches(self) -> bool:
        return "fetch" in self.roles

    @property
    def publishes(self) -> bool:
        return "publish" in self.roles


@dataclass(frozen=True)
class KVTransferConfig:
    """The whole file: the assembly table plus the coordinator's limits.

    Attributes:
        backends: In fetch priority order.
        fetch_timeout_s: Deadline for a fetch from launch; ``None`` disables.
        publish_timeout_s: Deadline for a publish from first submission; ``None`` disables.
        probe_timeout_s: Longest a request waits for a store lookup before it computes locally.
        close_timeout_s: Longest ``close`` waits for the backends to finish before it gives them
            up and releases the requests they were holding.
    """

    backends: tuple[BackendEntry, ...]
    fetch_timeout_s: float | None = None
    publish_timeout_s: float | None = None
    probe_timeout_s: float = _DEFAULT_PROBE_TIMEOUT_S
    close_timeout_s: float = _DEFAULT_CLOSE_TIMEOUT_S

    def __post_init__(self) -> None:
        if not self.backends:
            raise ValueError("kv transfer config needs at least one backend")
        names = [entry.name for entry in self.backends]
        if len(set(names)) != len(names):
            raise ValueError(f"backend names must be unique, got {names}")
        for key in ("fetch_timeout_s", "publish_timeout_s"):
            value = getattr(self, key)
            if value is not None and value <= 0:
                raise ValueError(f"{key} must be > 0 or null, got {value}")
        if self.probe_timeout_s < 0:
            raise ValueError(f"probe_timeout_s must be >= 0, got {self.probe_timeout_s}")
        if self.close_timeout_s <= 0:
            raise ValueError(f"close_timeout_s must be > 0, got {self.close_timeout_s}")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> KVTransferConfig:
        unknown = sorted(set(raw) - set(_COORDINATOR_KEYS) - {"backends"})
        if unknown:
            raise ValueError(f"unknown kv transfer config keys: {unknown}")
        entries = raw.get("backends")
        if not isinstance(entries, list):
            raise ValueError("kv transfer config needs a 'backends' list")
        coordinator_settings = {key: raw[key] for key in _COORDINATOR_KEYS if key in raw}
        return cls(
            backends=tuple(BackendEntry.from_dict(entry) for entry in entries),
            **coordinator_settings,
        )


def load_kv_transfer_config(path: str) -> KVTransferConfig:
    """Read, validate and return the file at ``path``. Raises ``ValueError`` naming the problem."""
    with open(path, encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping with a 'backends' list")
    return KVTransferConfig.from_dict(raw)

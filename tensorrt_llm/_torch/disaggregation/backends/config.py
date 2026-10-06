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

The executor creator reads ``TRTLLM_KV_TRANSFER_CONFIG`` and hands the YAML path to the assembly,
which loads it here; this module reads no environment. The ``backends`` list is the assembly
table: its order is the fetch priority, each entry names a registered backend type and carries
that type's own settings unread. The timeouts have their one default each here; the assembly
passes them to the coordinator and, for ``landing_wait_timeout_s``, to the backends. Nothing here
imports a backend.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Mapping, TypeVar

import yaml

__all__ = [
    "BACKEND_ROLES",
    "DEFAULT_LANDING_WAIT_TIMEOUT_S",
    "DEFAULT_UNLAUNCHED_TIMEOUT_S",
    "KV_TRANSFER_CONFIG_ENV",
    "BackendEntry",
    "KVTransferConfig",
    "load_kv_transfer_config",
    "strict_from_dict",
]

KV_TRANSFER_CONFIG_ENV = "TRTLLM_KV_TRANSFER_CONFIG"
"""Environment variable naming the YAML file; unset means no KV transfer backends."""

BACKEND_ROLES = ("fetch", "publish")
"""What a backend entry may be used for. A backend with both roles fetches and publishes."""

_ENTRY_KEYS = ("name", "type", "hint_key", "roles")
# The coordinator's clocks. Every wait on another rank, on the scheduler's pages or on a store has
# a finite default, so a peer that stopped voting or a store that stopped answering cannot park a
# request forever; a ``None`` must be asked for explicitly where it is allowed.
_COORDINATOR_KEYS = (
    "fetch_timeout_s",
    "publish_timeout_s",
    "unlaunched_timeout_s",
    "landing_wait_timeout_s",
    "probe_timeout_s",
    "close_timeout_s",
)
_DEFAULT_FETCH_TIMEOUT_S = 30.0
_DEFAULT_PUBLISH_TIMEOUT_S = 60.0
# Wall-clock, measured on the loop clock from a request's first deferral. A store's existence
# lookup is one RPC per request and runs on the master under the load of every rank's probes, and
# the ranks' clocks are not aligned to the round; 50 ms made loaded stores look empty. A budget
# counted in rounds instead of seconds would be independent of the clock skew; that is a design
# change left for a follow-up.
_DEFAULT_PROBE_TIMEOUT_S = 1.0
_DEFAULT_CLOSE_TIMEOUT_S = 30.0
DEFAULT_UNLAUNCHED_TIMEOUT_S = 30.0
"""The one default for ``unlaunched_timeout_s``; ``KVTransferConfig`` carries it and the assembly
passes it on."""
DEFAULT_LANDING_WAIT_TIMEOUT_S = 30.0
"""The one default for ``landing_wait_timeout_s``; ``KVTransferConfig`` carries it and the
assembly passes it on, to the coordinator and to the backends alike."""

_Dataclass = TypeVar("_Dataclass")


def strict_from_dict(cls: type[_Dataclass], raw: Mapping[str, Any]) -> _Dataclass:
    """Build dataclass ``cls`` from a plain mapping, refusing keys it has no field for. A value
    of the wrong type is a ``ValueError`` from ``cls`` like any other bad value."""
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {unknown}")
    return cls(**raw)


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
    def serves_fetch(self) -> bool:
        return "fetch" in self.roles

    @property
    def serves_publish(self) -> bool:
        return "publish" in self.roles


@dataclass(frozen=True)
class KVTransferConfig:
    """The whole file: the assembly table plus the coordinator's limits.

    Attributes:
        backends: In fetch priority order.
        fetch_timeout_s: Deadline for a fetch from its launch, and for a fetch record kept at
            its request's end until the ranks agree on it. Past it the request fails and the
            rank stops waiting for its peers. ``None`` disables and must be given explicitly.
        publish_timeout_s: Deadline for a publish from its first accepted submission, and for a
            publish record kept at its request's end until the ranks agree on it. Past it the
            rank warns and stops waiting for its peers; the publish still settles on its own
            outcome. ``None`` disables and must be given explicitly.
        unlaunched_timeout_s: Longest a rank may leave a fetch unlaunched (its pages not
            reserved) after another rank has launched it, before the ranks give the fetch up
            and plan again. Independent of ``fetch_timeout_s``; ``None`` disables and must be
            given explicitly.
        landing_wait_timeout_s: Longest a rank waits for the scheduler's pages for a planned
            fetch (from the plan's decision, or from landing on the host) or, host-first, for
            the backend's landing memory (``fetch_to_host`` refused), before it votes the fetch
            failed so that the ranks give it up, release the landing and plan again; out of
            retries the request computes locally. ``None`` disables and must be given
            explicitly.
        probe_timeout_s: Longest a request waits for a store lookup before it plans without
            the store (wall-clock, from its first deferral).
        close_timeout_s: Longest ``close`` waits for the backends to finish before it gives them
            up and releases the requests they were holding.
    """

    backends: tuple[BackendEntry, ...]
    fetch_timeout_s: float | None = _DEFAULT_FETCH_TIMEOUT_S
    publish_timeout_s: float | None = _DEFAULT_PUBLISH_TIMEOUT_S
    unlaunched_timeout_s: float | None = DEFAULT_UNLAUNCHED_TIMEOUT_S
    landing_wait_timeout_s: float | None = DEFAULT_LANDING_WAIT_TIMEOUT_S
    probe_timeout_s: float = _DEFAULT_PROBE_TIMEOUT_S
    close_timeout_s: float = _DEFAULT_CLOSE_TIMEOUT_S

    def __post_init__(self) -> None:
        if not self.backends:
            raise ValueError("kv transfer config needs at least one backend")
        names = [entry.name for entry in self.backends]
        if len(set(names)) != len(names):
            raise ValueError(f"backend names must be unique, got {names}")
        for key in (
            "fetch_timeout_s",
            "publish_timeout_s",
            "unlaunched_timeout_s",
            "landing_wait_timeout_s",
        ):
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

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
"""From a backend entry in the config to a built backend (design §7.4, §10.3).

A backend type registers a factory under its type name; ``build_backends`` walks the config's
assembly table and returns one ``BackendHandle`` per entry. The engine assembly speaks only to
handles, so adding a backend is a new directory plus one line in the built-in table (or one
``register_backend_type`` call). Built-in types are imported on first use, so importing this
module imports no backend.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

from ..base.cache_backend import Fetches, Publishes, RegistersPools
from ..base.capabilities import LandsOnHost
from ..base.region import RegionResolver
from .config import DEFAULT_FETCH_WAIT_TIMEOUT_S, BackendEntry, KVTransferConfig

__all__ = [
    "BackendBuildContext",
    "BackendFactory",
    "BackendHandle",
    "BackendRegistry",
    "build_backends",
    "close_backends",
    "register_backend_type",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackendBuildContext:
    """What a factory may need from the engine, gathered once per assembly.

    Attributes:
        resolver: Maps a unit's local coordinates to memory segments.
        layout_fingerprint: Digest of the local memory layout, for content-addressed names.
        max_unit_bytes: The largest unit any layer group produces; sizes the host slot pools.
        device_index: CUDA device of the KV pools, for backends that copy through host memory.
        unit_bytes_of: Callable giving the byte size of the unit called ``name``, for a backend
            that lands units in its own memory before it knows their pages (``LandsOnHost``).
            ``None`` when the assembly offers no such backend.
        max_request_blocks: Blocks the longest request spans (``max_seq_len`` over
            ``tokens_per_block``), so a backend can warn when its landing memory is short of
            one fetch. ``None`` when unknown.
        fetch_wait_timeout_s: ``KVTransferConfig.fetch_wait_timeout_s``, so a backend that
            queues landings for its own memory bounds that wait with the same clock the
            coordinator uses for pages; ``None`` for no bound. Defaults to the config's default.
    """

    resolver: RegionResolver
    layout_fingerprint: bytes
    max_unit_bytes: int
    device_index: Optional[int] = None
    unit_bytes_of: Optional[Callable[[bytes], int]] = None
    max_request_blocks: Optional[int] = None
    fetch_wait_timeout_s: Optional[float] = DEFAULT_FETCH_WAIT_TIMEOUT_S


@dataclass(frozen=True)
class BackendHandle:
    """A built backend as the assembly sees it: which contract sides it serves, and how to stop it.

    Attributes:
        name: ``FetchSource.name``.
        hint_key: Routing hint the backend reads, or ``None``.
        fetcher: The backend as a ``Fetches``, or ``None`` when the entry has no fetch role.
        publisher: The backend as a ``Publishes``, or ``None`` when it has no publish role.
        pool_registrar: The backend as a ``RegistersPools`` when it needs the KV pools registered
            with its transport; ``None`` when it reaches memory another way.
        close: Stops the backend and releases what it holds. Idempotent.
        counters: The backend's operational counters, for the status dump; empty if it has none.
        landing: Where a fetch lands first: ``device`` (the caller's pages; a ``Fetches``) or
            ``host`` (the backend's own memory; a ``LandsOnHost``). Shown in the status dump.
    """

    name: str
    hint_key: Optional[str]
    fetcher: Optional[Fetches | LandsOnHost]
    publisher: Optional[Publishes]
    pool_registrar: Optional[RegistersPools]
    close: Callable[[], None]
    counters: Callable[[], Mapping[str, int]] = field(default=dict)
    landing: str = "device"


BackendFactory = Callable[[BackendEntry, BackendBuildContext], BackendHandle]
"""Builds one backend from its config entry. Raises ``ValueError`` for an entry it cannot serve."""

_BUILTIN_FACTORIES: Mapping[str, str] = {
    "mooncake": ".blob.drivers.mooncake:build_mooncake_backend",
    "memory": ".blob.drivers.memory:build_memory_backend",
}
"""Type name -> ``<module relative to this package>:<factory name>``; imported on first use."""


def _import_builtin_factory(location: str) -> BackendFactory:
    module_path, factory_name = location.split(":")
    module = importlib.import_module(module_path, package=__package__)
    return getattr(module, factory_name)


class BackendRegistry:
    """Type name -> factory. Explicit registrations first, then the built-in table by name."""

    def __init__(self) -> None:
        self._factories: dict[str, BackendFactory] = {}

    def register_backend_type(self, type_name: str, factory: BackendFactory) -> None:
        if not type_name:
            raise ValueError("a backend type needs a name")
        if type_name in self._factories:
            raise ValueError(f"backend type {type_name!r} is already registered")
        self._factories[type_name] = factory

    def factory_for(self, type_name: str) -> BackendFactory:
        factory = self._factories.get(type_name)
        if factory is not None:
            return factory
        location = _BUILTIN_FACTORIES.get(type_name)
        if location is not None:
            return _import_builtin_factory(location)
        known = sorted(set(self._factories) | set(_BUILTIN_FACTORIES))
        raise ValueError(f"unknown kv transfer backend type {type_name!r}; known: {known}")

    def build_backend(self, entry: BackendEntry, context: BackendBuildContext) -> BackendHandle:
        return self.factory_for(entry.type)(entry, context)


DEFAULT_REGISTRY = BackendRegistry()


def register_backend_type(type_name: str, factory: BackendFactory) -> None:
    """Register an external backend type with the default registry (design §10.3)."""
    DEFAULT_REGISTRY.register_backend_type(type_name, factory)


def build_backends(
    config: KVTransferConfig,
    context: BackendBuildContext,
    registry: BackendRegistry = DEFAULT_REGISTRY,
) -> list[BackendHandle]:
    """One handle per config entry, in config order. A failure closes what was already built."""
    handles: list[BackendHandle] = []
    try:
        for entry in config.backends:
            handles.append(registry.build_backend(entry, context))
    except Exception:
        close_backends(handles)
        raise
    return handles


def close_backends(handles: Sequence[BackendHandle]) -> None:
    """Close every handle. One that fails to close does not keep the others from closing."""
    for handle in handles:
        # Broad on purpose, against CODING_GUIDELINES: a backend's close raises its own transport
        # error type, and a teardown must reach every backend.
        try:
            handle.close()
        except Exception:  # noqa: BLE001
            logger.exception("backend %r failed to close", handle.name)

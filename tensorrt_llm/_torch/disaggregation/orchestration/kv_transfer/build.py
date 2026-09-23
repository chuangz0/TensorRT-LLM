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
"""Assembling a ``KVTransferCoordinator`` from the config and the built backends (design §7.4).

The engine side builds the inputs (backends, reader, effects, work queue, collective) and hands
them here with the config; this module turns the backends into the assembly table, builds the
planner, and returns the coordinator. It knows nothing of ``PyExecutor``.
"""

from __future__ import annotations

from typing import Sequence

from ...backends.config import KVTransferConfig
from ...backends.registry import BackendHandle
from ...base.views import ResourceReader
from ...remote_cache import FetchSource, Planner
from .coordinator import KVTransferCoordinator
from .interfaces import DistLike, EngineQueue, KVTransferEffects, PlanAuthority

__all__ = ["build_coordinator"]


def build_coordinator(
    config: KVTransferConfig,
    backends: Sequence[BackendHandle],
    reader: ResourceReader,
    effects: KVTransferEffects,
    queue: EngineQueue,
    dist: DistLike,
    *,
    plan_authority: PlanAuthority = PlanAuthority.VOTED,
) -> KVTransferCoordinator:
    """One ``FetchSource`` per backend with a fetch role, in config (priority) order; one
    publisher per backend with a publish role; the config's timeouts on the planner and the
    coordinator."""
    fetch_sources = [
        FetchSource(handle.name, handle.fetcher, handle.hint_key)
        for handle in backends
        if handle.fetcher is not None
    ]
    publishers = [handle.publisher for handle in backends if handle.publisher is not None]
    planner = Planner(
        fetch_sources, reader, reader.tokens_per_block, probe_timeout_s=config.probe_timeout_s
    )
    return KVTransferCoordinator(
        fetch_sources,
        publishers,
        planner,
        reader,
        effects,
        queue,
        dist,
        fetch_timeout_s=config.fetch_timeout_s,
        publish_timeout_s=config.publish_timeout_s,
        unlaunched_timeout_s=config.unlaunched_timeout_s,
        plan_authority=plan_authority,
    )

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
"""The ``mooncake`` backend type: ``MooncakeStoreBackend`` over a real Mooncake store client.

The registry's factory for ``type: mooncake``. A config entry's options are the fields of
``MooncakeStoreConfig``; the factory opens the client, sets up host staging when asked to, and
returns a ``BackendHandle`` whose ``counters`` are the store counters.
"""

from __future__ import annotations

import dataclasses

from ..kv_transfer_config import BackendEntry
from ..registry import BackendBuildContext, BackendHandle
from .backend import MooncakeStoreBackend
from .client import StoreClient, open_mooncake_client
from .config import MooncakeStoreConfig
from .staging import HostStagingPool, open_default_staging, plan_slot_geometry

__all__ = ["build_mooncake_backend"]


def build_mooncake_backend(entry: BackendEntry, context: BackendBuildContext) -> BackendHandle:
    """Factory for ``type: mooncake``. A store has one destination, so it takes no ``hint_key``."""
    if entry.hint_key is not None:
        raise ValueError(f"backend {entry.name!r}: a mooncake store takes no hint_key")
    store_config = MooncakeStoreConfig.from_dict(entry.options)
    client = open_mooncake_client(store_config)
    try:
        staging = _open_staging(store_config, context, client)
        backend = MooncakeStoreBackend(
            client, store_config, context.resolver, context.layout_fingerprint, staging=staging
        )
    except Exception:
        client.close()
        raise
    return BackendHandle(
        name=entry.name,
        hint_key=None,
        fetches=backend if entry.fetches else None,
        publishes=backend if entry.publishes else None,
        # Staging copies through a registered host buffer, so the KV pools stay unregistered.
        pool_registrar=None if store_config.stage_through_host else backend,
        close=backend.close,
        counters=lambda: dataclasses.asdict(backend.counters),
    )


def _open_staging(
    store_config: MooncakeStoreConfig, context: BackendBuildContext, client: StoreClient
) -> HostStagingPool | None:
    if not store_config.stage_through_host:
        return None
    slot_bytes, num_slots = plan_slot_geometry(
        context.max_unit_bytes, store_config.transfer_batch_size, store_config.staging_buffer_bytes
    )
    return open_default_staging(
        client, slot_bytes=slot_bytes, num_slots=num_slots, device_index=context.device_index
    )

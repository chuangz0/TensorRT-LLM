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
"""From a config entry to a ``BackendHandle`` over a ``BlobStoreBackend``, for every driver.

A driver's registry factory calls ``build_blob_backend`` with a function that opens its store
from the driver's share of the entry's options. The entry's options are one flat mapping: the
keys ``BlobStoreConfig`` reads go to the backend, the driver's keys go to ``open_store``, and a
key neither knows is refused here, naming the entry and its type.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Collection, Mapping

from ..config import BackendEntry
from ..registry import BackendBuildContext, BackendHandle
from .backend import BlobStoreBackend, BlobStoreConfig
from .staging import HostStagingPool, open_default_staging, plan_slot_geometry
from .store import BlobStore

__all__ = ["build_blob_backend"]

OpenStore = Callable[[Mapping[str, Any]], BlobStore]
"""Opens a driver's store from its share of a config entry's options."""


def build_blob_backend(
    entry: BackendEntry,
    context: BackendBuildContext,
    open_store: OpenStore,
    driver_fields: Collection[str],
) -> BackendHandle:
    """Build the backend for ``entry``: split its options, open the store, wire staging.

    A blob store has one destination, so an entry with a ``hint_key`` is refused. A store that was
    opened is closed again if anything after it fails.
    """
    if entry.hint_key is not None:
        raise ValueError(f"backend {entry.name!r}: a blob store takes no hint_key")
    backend_options, driver_options = _split_options(entry, driver_fields)
    config = BlobStoreConfig.from_dict(backend_options)
    store = open_store(driver_options)
    try:
        staging = _open_staging(config, context, store)
        backend = BlobStoreBackend(
            store, config, context.resolver, context.layout_fingerprint, staging=staging
        )
    except Exception:
        store.close()
        raise
    return BackendHandle(
        name=entry.name,
        hint_key=None,
        fetcher=backend if entry.serves_fetch else None,
        publisher=backend if entry.serves_publish else None,
        # Staging copies through a registered host buffer, so the KV pools stay unregistered.
        pool_registrar=None if config.stage_through_host else backend,
        close=backend.close,
        counters=lambda: dataclasses.asdict(backend.counters),
    )


def _split_options(
    entry: BackendEntry, driver_fields: Collection[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    backend_fields = BlobStoreConfig.fields()
    known = backend_fields | frozenset(driver_fields)
    unknown = sorted(set(entry.options) - known)
    if unknown:
        raise ValueError(
            f"backend {entry.name!r} (type {entry.type}): unknown keys {unknown}; "
            f"known: {sorted(known)}"
        )
    backend_options = {k: v for k, v in entry.options.items() if k in backend_fields}
    driver_options = {k: v for k, v in entry.options.items() if k not in backend_fields}
    return backend_options, driver_options


def _open_staging(
    config: BlobStoreConfig, context: BackendBuildContext, store: BlobStore
) -> HostStagingPool | None:
    if not config.stage_through_host:
        return None
    slot_bytes, num_slots = plan_slot_geometry(
        context.max_unit_bytes, config.transfer_batch_size, config.staging_buffer_bytes
    )
    return open_default_staging(
        store, slot_bytes=slot_bytes, num_slots=num_slots, device_index=context.device_index
    )

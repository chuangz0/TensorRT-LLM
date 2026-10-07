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
"""From a config entry to a ``BackendHandle`` over the blob backend, for every driver.

A driver's registry factory calls ``build_blob_backend`` with a function that opens its store
from the driver's share of the entry's options. The entry's options are one flat mapping: the
keys ``BlobStoreConfig`` reads go to the backend, the driver's keys go to ``open_store``, and a
key neither knows is refused here, naming the entry and its type.

``landing`` picks the shape. ``device`` is a ``BlobStoreBackend`` that registers the caller's
pools. ``host`` is a ``HostLandingBlobBackend`` over one with two pinned host pools, the publish
pool and the landing pool, and registers no pools.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Callable, Collection, Mapping

from ..config import BackendEntry
from ..registry import BackendBuildContext, BackendHandle
from .backend import BlobStoreBackend, BlobStoreConfig
from .host_landing import HostLandingBlobBackend
from .slot_pool import HostSlotPool, open_pinned_slot_pool, plan_slot_geometry
from .store import BlobStore

__all__ = ["build_blob_backend"]

logger = logging.getLogger(__name__)

OpenStore = Callable[[Mapping[str, Any]], BlobStore]
"""Opens a driver's store from its share of a config entry's options."""


def build_blob_backend(
    entry: BackendEntry,
    context: BackendBuildContext,
    open_store: OpenStore,
    driver_fields: Collection[str],
) -> BackendHandle:
    """Build the backend for ``entry``: split its options, open the store, wire the shape.

    A blob store has one destination, so an entry with a ``hint_key`` is refused. A store that was
    opened is closed again if anything after it fails.
    """
    if entry.hint_key is not None:
        raise ValueError(f"backend {entry.name!r}: a blob store takes no hint_key")
    backend_options, driver_options = _split_options(entry, driver_fields)
    config = BlobStoreConfig.from_dict(backend_options)
    if config.lands_on_host and context.unit_bytes_of is None:
        raise ValueError(
            f"backend {entry.name!r}: landing 'host' needs the assembly to size units by name"
        )
    store = open_store(driver_options)
    try:
        if config.lands_on_host:
            return _host_landing_handle(entry, config, context, store)
        return _device_handle(entry, config, context, store)
    except Exception:
        store.close()
        raise


def _device_handle(
    entry: BackendEntry, config: BlobStoreConfig, context: BackendBuildContext, store: BlobStore
) -> BackendHandle:
    backend = BlobStoreBackend(store, config, context.resolver, context.layout_fingerprint)
    return BackendHandle(
        name=entry.name,
        hint_key=None,
        fetcher=backend if entry.serves_fetch else None,
        publisher=backend if entry.serves_publish else None,
        pool_registrar=backend,
        close=backend.close,
        counters=lambda: dataclasses.asdict(backend.counters),
        landing=config.landing,
    )


def _host_landing_handle(
    entry: BackendEntry, config: BlobStoreConfig, context: BackendBuildContext, store: BlobStore
) -> BackendHandle:
    """Two pools, then the inner backend over the publish pool and the landing backend over both.
    The store is registered with pinned host memory only, so the KV pools stay unregistered."""
    publish_pool = _open_pool(
        store, context, config.transfer_batch_size, config.publish_buffer_bytes
    )
    landing_pool = _open_pool(store, context, config.max_landed_units, config.landing_buffer_bytes)
    _log_pool_geometry(entry, publish_pool, landing_pool, context.max_request_blocks)
    inner = BlobStoreBackend(
        store, config, context.resolver, context.layout_fingerprint, publish_pool=publish_pool
    )
    assert context.unit_bytes_of is not None  # checked by the caller before the store was opened
    backend = HostLandingBlobBackend(
        inner,
        landing_pool,
        context.unit_bytes_of,
        fetch_wait_timeout_s=context.fetch_wait_timeout_s,
    )
    return BackendHandle(
        name=entry.name,
        hint_key=None,
        fetcher=backend if entry.serves_fetch else None,
        publisher=inner if entry.serves_publish else None,
        pool_registrar=None,
        close=backend.close,
        counters=lambda: {
            **dataclasses.asdict(backend.counters),
            "landings_held": backend.landings_held(),
        },
        landing=config.landing,
    )


def _open_pool(
    store: BlobStore, context: BackendBuildContext, max_slots: int | None, budget_bytes: int
) -> HostSlotPool:
    slot_bytes, num_slots = plan_slot_geometry(context.max_unit_bytes, max_slots, budget_bytes)
    return open_pinned_slot_pool(
        store, slot_bytes=slot_bytes, num_slots=num_slots, device_index=context.device_index
    )


def _log_pool_geometry(
    entry: BackendEntry,
    publish_pool: HostSlotPool,
    landing_pool: HostSlotPool,
    max_request_blocks: int | None,
) -> None:
    """The two pools' sizes, and a warning when the landing pool cannot hold two fetches of the
    longest request: landings hold their slots until pages are found, so a pool that small
    serialises fetches behind the scheduler."""
    logger.info(
        "backend %r: landing 'host' with a publish pool of %d x %d B and a landing pool of "
        "%d x %d B",
        entry.name,
        publish_pool.num_slots,
        publish_pool.slot_bytes,
        landing_pool.num_slots,
        landing_pool.slot_bytes,
    )
    if max_request_blocks is not None and landing_pool.num_slots < 2 * max_request_blocks:
        logger.warning(
            "backend %r: the landing pool has %d slots, fewer than two requests of "
            "max_seq_len (%d blocks each); raise landing_buffer_bytes to land fetches in parallel",
            entry.name,
            landing_pool.num_slots,
            max_request_blocks,
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

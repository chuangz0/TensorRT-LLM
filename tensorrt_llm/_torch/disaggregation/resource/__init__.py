# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How the KV transfer layers read this rank's KV cache without touching a page object.

``page.py`` describes the cache's memory as a ``KVCachePageTable`` (pools, layer groups, the
roles a pool holds) and ``kv_extractor.py`` builds that table from the engine's cache manager.
Over the table, ``kv_v2_view.py`` (``KVv2ResourceView``) is what the coordination layer reads:
which blocks a prompt names, which layer groups this rank holds, which pages a fetch lands in,
which committed pages a publish offers. ``naming.py`` turns block ordinals into the unit names
both sides compute the same way. ``region.py`` resolves addresses: ``KVv2RegionResolver`` maps a
unit's ``(layer group, page index)`` to its memory segments and lists the pool spans a backend
registers, and ``compute_layout_fingerprint`` digests the layout those bytes assume so that two
processes whose bytes differ miss each other in a store. ``utils.py`` holds the pool-descriptor
helpers the paired path (``native/``, ``transceiver.py``) reads from the same table;
``cache_reuse.py`` is the prefix-reuse interface over KV cache manager V1 and V2.
"""

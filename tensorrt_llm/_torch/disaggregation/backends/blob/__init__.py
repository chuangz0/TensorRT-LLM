# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The blob KIND: a cache backend over a store of byte objects addressed by key.

Two layers. The adaptation layer is store-agnostic: ``store.py`` is the ``BlobStore`` protocol
a store must offer, ``backend.py`` is ``BlobStoreBackend`` (``Fetches`` / ``Publishes`` /
``RegistersPools`` over one ``BlobStore``) and ``HostLandingBlobBackend`` (its ``LandsOnHost``
shape, chosen by ``BlobStoreConfig.landing: host``), ``factory.py`` builds either from a config
entry for every store alike, and ``keys.py`` / ``staging.py`` / ``worker_pool.py`` are their
parts. The drivers under ``drivers/`` each wrap one real store as a ``BlobStore``; a new store is
a new module there and one line in the registry, nothing here.
"""

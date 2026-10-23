# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The blob backend: KV rows shared through a byte store.

Read top down; a fetch or publish flows ``BlobBackend`` -> ``StoreDriver`` -> store SDK:

- ``backend``: ``BlobBackend``, the public ``base.backend.Backend``. It cuts each view into
  batches and hands them to a ``store.StoreDriver`` on worker threads. ``RegionBinding``
  registers the lender's regions the driver can access, under the lender's hold.
- ``store`` and ``keys``: the private driver boundary, which shares only ``MemoryKind`` with the
  public contract, and the store key encoding. How an object is laid out in a store is its
  driver's private format.
- ``mooncake``: ``MooncakeDriver``, a ``StoreDriver`` over a set-up Mooncake SDK client.
  ``BlobBackend`` has no probe: a caller asking which objects exist calls the driver's ``probe``
  directly.
- ``mooncake_native``: ``NativeMooncakeDriver``, the same driver with its store calls on the C++
  store worker's threads (``tensorrt_llm.mooncake_store_worker``), which never hold the GIL.
- ``mooncake_setup``: reads a store config, opens the SDK client and picks the driver
  (``make_store_driver``); ``open_mooncake_driver`` returns the driver and the config's ``blob``
  section, from which ``BlobConfig`` is built.

No module imports torch, ``tensorrt_llm._torch.pyexecutor`` or a store SDK at import time; only
``mooncake_setup.open_sdk`` imports ``mooncake.store``, and ``mooncake_setup`` the store worker,
when called. Each module's imports are checked by
``test_mooncake_setup.py::test_blob_modules_import_only_their_layer``.
"""

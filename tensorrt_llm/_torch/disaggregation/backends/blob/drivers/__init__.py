# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One module per blob store. Each implements ``store.BlobStore`` (``../store.py``) over its
store's own client, translating that client's codes into ``PutStatus`` / ``GetStatus`` /
``BlobStoreError``, and exposes a ``build_<name>_backend`` factory that hands its store to
``factory.build_blob_backend``. The registry's ``_BUILTIN_FACTORIES`` points here, one line per
driver; this package imports none of them, so importing the registry imports no store client.
"""

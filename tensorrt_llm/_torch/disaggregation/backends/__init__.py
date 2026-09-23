# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend implementations.

One KIND is one sub-package; sub-packages must not import each other (README §5 rule one), shared
abstractions go to ``base/``. ``config.py`` maps the assembly-table YAML to ``KVTransferConfig``;
``registry.py`` maps ``type:`` to a factory. Today the only entry is ``type: mooncake`` ->
``.blob.mooncake:build_mooncake_backend``. A new blob DRIVER adds a module under ``blob/`` plus one
line in ``_BUILTIN_FACTORIES``; a new KIND adds a sub-package plus one line there.

``blob/``: the peer implements the byte-store contract; drivers live under ``backends/blob/drivers/``
(once the blob abstraction plan lands). ``worker/`` (future): the peer is an engine worker, after the
paired path migrates. ``kvcr/`` (future): the peer is the KVCR runtime.
"""

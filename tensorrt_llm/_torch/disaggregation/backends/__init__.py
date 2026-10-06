# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend implementations.

One KIND is one sub-package. Sub-packages must not import each other: what two kinds share goes
to ``base/`` when it is a contract, or to a module at this level when it is an implementation
piece (``host_copy.py``: the ``Copier`` protocol and ``CudaCopier``, the device/host copy every
host-landing backend uses). ``config.py`` maps the assembly-table YAML to ``KVTransferConfig``;
``registry.py`` maps ``type:`` to a factory. The built-in entries are ``type: mooncake`` ->
``.blob.drivers.mooncake:build_mooncake_backend`` and ``type: memory`` ->
``.blob.drivers.memory:build_memory_backend``. A new blob DRIVER adds a module under
``blob/drivers/`` plus one line in ``_BUILTIN_FACTORIES``; a new KIND adds a sub-package plus one
line there.

``blob/``: the peer is a store of byte objects (``blob/store.py::BlobStore``); ``blob/backend.py``
is the ``landing: device`` shape, ``blob/host_landing.py`` the ``landing: host`` shape, and the
drivers live under ``blob/drivers/``. ``worker/`` (future): the peer is an engine worker, after
the paired path migrates. ``kvcr/`` (future): the peer is the KVCR runtime.

Tests: ``tests/unittest/_torch/disaggregation/backends/`` (config and registry; ``blob/`` for the
blob backend, its drivers and the store fakes) and ``tests/unittest/_torch/disaggregation/e2e/``
(real Mooncake master, GPU).
"""

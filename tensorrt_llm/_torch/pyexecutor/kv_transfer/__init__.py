# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The three places where the KV transfer layer lands on ``PyExecutor``.

``assembly.py`` is assembled once at creation; ``hooks.py`` is the object the loop calls every
iteration; ``effects.py`` is the only place that writes request transfer state. The paired-path
counterpart is ``pyexecutor/disagg_adapter.py``. Outside this package the layer is reached through
``self.kv_transfer.<hook>(...)`` calls in ``py_executor.py``, each under an
``if self.kv_transfer is not None`` guard, the duck-typed ``kv_transfer`` attribute of
``KVCacheV2Scheduler`` (the same object), and one lazy ``attach_kv_transfer`` import in
``py_executor_creator``;
``tests/unittest/_torch/executor/kv_transfer/test_hook_points.py`` reads those sites off the
source. Do not import submodules from this ``__init__``: ``py_executor`` must load without
loading this layer.
"""

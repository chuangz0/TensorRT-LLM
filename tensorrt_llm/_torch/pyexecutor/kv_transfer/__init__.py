# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The three places where the KV transfer layer lands on ``PyExecutor``.

``assembly.py`` is assembled once at creation; ``binding.py`` is the object the loop calls every
iteration; ``effects.py`` is the only place that writes request transfer state. The paired-path
counterpart is ``pyexecutor/disagg_adapter.py``. Every hook in the rest of the engine is a single
guarded call. Do not import submodules from this ``__init__``: ``py_executor`` must load without
loading this layer.
"""

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Entry points the engine calls every iteration.

The flat ``coordinator.py``, ``transfer_manager.py``, ``admission.py``, ``pp_termination.py`` and
``interfaces.py`` are today's coordination layer of the paired path; ``kv_transfer/`` is the unified
coordination layer, currently carrying the store path and replacing the former once the paired path
migrates onto it (design §12).
"""

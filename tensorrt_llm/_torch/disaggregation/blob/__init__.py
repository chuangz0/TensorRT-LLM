# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The blob backend: KV rows shared through a byte store.

No module here imports torch, ``tensorrt_llm._torch.pyexecutor`` or a store SDK at import time.

- ``store`` (the private driver boundary) imports only the standard library and ``base.memory``,
  whose ``MemoryKind`` it shares with the public contract; ``keys`` imports only the standard
  library. How an object is laid out in a store is its driver's private format.
- ``backend`` (the public backend over a driver) imports only ``store``, ``keys``,
  ``base.backend`` and ``base.memory``, plus numpy and ``tensorrt_llm.logger``.
"""

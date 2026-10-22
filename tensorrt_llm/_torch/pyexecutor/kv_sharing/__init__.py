# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KV sharing: servers share prompt KV cache blocks through a KV store (``kv_sharing``).

- ``startup``: wires the store client, the lender and the transfer manager into a built executor.
- ``orchestrator``: ``SharedKvOrchestrator``, the one object the executor calls.
"""

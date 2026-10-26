# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KV sharing: servers share prompt KV cache blocks through a KV store (``kv_sharing``).

Read in this order:

- ``orchestrator``: ``SharedKvOrchestrator``, the one object the executor and the scheduler call:
  the table of store operations, fetch admission, the round, publication, cancel and shutdown.
- ``store_ops``: the table's entries, each rank's report on them, and ``decide``, the rule that
  closes them.
- ``probe``: rank 0's ``ProbeService``, which asks the store about new requests and holds them
  until it answers.
- ``replicas``: which tensor-parallel rank puts each block several ranks hold.
- ``breaker``: pauses probes, fetches or publications after failures in a row.
- ``startup``: wires the store client, the lender and the transfer manager into a built executor.
- ``data_path``: the startup store check and the choice between GPU Direct and host staging.
"""

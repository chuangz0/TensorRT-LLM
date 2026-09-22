# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Settings for the Mooncake store backend: how to reach the pool and how to move bytes into it."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Mapping

__all__ = ["DEFAULT_METADATA_SERVER", "MooncakeStoreConfig"]

DEFAULT_METADATA_SERVER = "P2PHANDSHAKE"
"""Mooncake's peer-to-peer handshake, which needs no separate metadata process."""

_DEFAULT_GLOBAL_SEGMENT_SIZE = 3355443200
_DEFAULT_LOCAL_BUFFER_SIZE = 1073741824
_DEFAULT_STAGING_BUFFER_BYTES = 536870912


@dataclass(frozen=True)
class MooncakeStoreConfig:
    """Everything the backend needs to open a store handle and to size its own machinery.

    Attributes:
        master_server_address: ``host:port`` of the Mooncake master.
        local_hostname: Address this process is reachable at. ``None`` picks the host's own.
        metadata_server: Mooncake metadata service connstring.
        protocol: Transport protocol, ``"rdma"`` or ``"tcp"``.
        device_name: RDMA device filter handed to ``setup``; empty means any.
        global_segment_size: Bytes this process contributes to the pool.
        local_buffer_size: Bytes of the client's own transfer buffer.
        namespace: Leading component of every key; two deployments share cache only when they agree.
        transfer_batch_size: Units per store call. Bounds one RPC, not one delivery.
        stage_through_host: Pass units through a pinned host buffer instead of registering the
            caller's pools. Costs a copy each way; works without GPUDirect RDMA.
        staging_buffer_bytes: Ceiling on the pinned staging allocation when staging.
        max_inflight_ops: Deliveries that may be queued or running at once. A submission past
            this bound is refused with ``SubmissionRejected``.
        num_workers: Threads that drive store calls.
        probe_ttl_s: Seconds an unconsumed probe answer is kept before it is dropped.
    """

    master_server_address: str
    local_hostname: str | None = None
    metadata_server: str = DEFAULT_METADATA_SERVER
    protocol: str = "rdma"
    device_name: str = ""
    global_segment_size: int = _DEFAULT_GLOBAL_SEGMENT_SIZE
    local_buffer_size: int = _DEFAULT_LOCAL_BUFFER_SIZE
    namespace: str = "trtllm"
    transfer_batch_size: int = 64
    stage_through_host: bool = False
    staging_buffer_bytes: int = _DEFAULT_STAGING_BUFFER_BYTES
    max_inflight_ops: int = 256
    num_workers: int = 2
    probe_ttl_s: float = 30.0

    def __post_init__(self) -> None:
        if not self.master_server_address:
            raise ValueError("master_server_address is required")
        if not self.metadata_server:
            raise ValueError("metadata_server must not be empty")
        if not self.namespace:
            raise ValueError("namespace must not be empty")
        if self.protocol not in ("tcp", "rdma"):
            raise ValueError(f"protocol must be 'tcp' or 'rdma', got {self.protocol!r}")
        if self.global_segment_size < 0:
            raise ValueError("global_segment_size must be >= 0")
        for field in (
            "local_buffer_size",
            "transfer_batch_size",
            "max_inflight_ops",
            "num_workers",
        ):
            if getattr(self, field) <= 0:
                raise ValueError(f"{field} must be > 0")
        if self.stage_through_host and self.staging_buffer_bytes <= 0:
            raise ValueError("staging_buffer_bytes must be > 0 when stage_through_host is set")
        if self.probe_ttl_s <= 0:
            raise ValueError("probe_ttl_s must be > 0")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> MooncakeStoreConfig:
        """Build from a plain mapping, refusing keys this class does not know."""
        known = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"unknown MooncakeStoreConfig keys: {unknown}")
        return cls(**raw)

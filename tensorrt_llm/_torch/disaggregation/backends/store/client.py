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
"""The slice of ``MooncakeDistributedStore`` the backend uses, and how to open the real one.

The backend takes the client by injection so that all of its logic runs against a fake; the real
class is imported only inside ``open_mooncake_client``.
"""

from __future__ import annotations

from typing import Protocol, Sequence

from .config import MooncakeStoreConfig

__all__ = ["OBJECT_NOT_FOUND", "StoreClient", "open_mooncake_client"]

OBJECT_NOT_FOUND = -704
"""Status a get or ``get_size`` answers for a key the store does not hold; nothing is written."""


class StoreClient(Protocol):
    """Return conventions follow ``mooncake.store.MooncakeDistributedStore``.

    Every call returns status integers rather than raising: ``0`` is success for ``setup``,
    ``register_buffer``, ``unregister_buffer``, ``close`` and each key of a put. ``batch_is_exist``
    answers ``1`` present, ``0`` absent, negative for a failed lookup. ``batch_get_into_multi_buffers``
    answers the bytes read per key, or a negative code for a key it could not read, leaving that
    key's destination untouched: ``OBJECT_NOT_FOUND`` when the key is not there, ``-600`` /
    ``-800`` when the buffers add up to less / more than the object.

    One key is one object assembled from several buffers, in order, which is how a unit spread over
    several segments stays one object. Every buffer handed to a put or a get must lie in a span that
    ``register_buffer`` accepted.
    """

    def setup(
        self,
        local_hostname: str,
        metadata_server: str,
        global_segment_size: int,
        local_buffer_size: int,
        protocol: str,
        device_name: str,
        master_server_address: str,
    ) -> int: ...

    def register_buffer(self, buffer_ptr: int, size: int) -> int: ...

    def unregister_buffer(self, buffer_ptr: int) -> int: ...

    def batch_is_exist(self, keys: Sequence[str]) -> Sequence[int]: ...

    def batch_put_from_multi_buffers(
        self,
        keys: Sequence[str],
        all_buffer_ptrs: Sequence[Sequence[int]],
        all_sizes: Sequence[Sequence[int]],
    ) -> Sequence[int]: ...

    def batch_get_into_multi_buffers(
        self,
        keys: Sequence[str],
        all_buffer_ptrs: Sequence[Sequence[int]],
        all_sizes: Sequence[Sequence[int]],
    ) -> Sequence[int]: ...

    def close(self) -> int: ...


def _default_hostname() -> str:
    import socket

    return socket.gethostbyname(socket.gethostname())


def open_mooncake_client(config: MooncakeStoreConfig) -> StoreClient:
    """Connect a real ``MooncakeDistributedStore`` to the master named by ``config``.

    Raises:
        ImportError: The Mooncake Python bindings are not installed.
        RuntimeError: ``setup`` returned a non-zero status.
    """
    try:
        from mooncake.store import MooncakeDistributedStore
    except ImportError as exc:
        raise ImportError(
            "The Mooncake store backend needs the Mooncake Python bindings "
            "(`pip install mooncake-transfer-engine`)."
        ) from exc

    store = MooncakeDistributedStore()
    status = store.setup(
        config.local_hostname or _default_hostname(),
        config.metadata_server,
        config.global_segment_size,
        config.local_buffer_size,
        config.protocol,
        config.device_name,
        config.master_server_address,
    )
    if status != 0:
        raise RuntimeError(
            f"MooncakeDistributedStore.setup failed with status {status} "
            f"(master={config.master_server_address!r}, metadata={config.metadata_server!r}, "
            f"protocol={config.protocol!r})"
        )
    return store

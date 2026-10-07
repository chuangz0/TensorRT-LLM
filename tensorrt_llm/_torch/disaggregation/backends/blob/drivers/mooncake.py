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
"""The Mooncake driver: ``type: mooncake`` as the blob backend over a ``MooncakeDistributedStore``,
in either of its shapes (``BlobStoreBackend`` for ``landing: device``, ``HostLandingBlobBackend``
for ``landing: host``; over TCP the default is ``host``).

Everything Mooncake-specific lives here: ``MooncakeStoreConfig`` (the connection options of a
config entry), ``MooncakeBlobStore`` (the ``BlobStore`` over the bindings, the only place their
status codes are read) and ``build_mooncake_backend``, the registry's factory. The bindings are
imported in ``MooncakeBlobStore.open`` and nowhere else.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ....base.region import Segment
from ...config import BackendEntry, strict_from_dict
from ...registry import BackendBuildContext, BackendHandle
from ..factory import build_blob_backend
from ..store import BlobStoreError, GetStatus, PutStatus

__all__ = [
    "DEFAULT_METADATA_SERVER",
    "LEASE_EXPIRED",
    "NO_AVAILABLE_HANDLE",
    "OBJECT_ALREADY_EXISTS",
    "OBJECT_NOT_FOUND",
    "MooncakeBlobStore",
    "MooncakeStoreConfig",
    "build_mooncake_backend",
]

logger = logging.getLogger(__name__)

DEFAULT_METADATA_SERVER = "P2PHANDSHAKE"
"""Mooncake's peer-to-peer handshake, which needs no separate metadata process."""

OBJECT_NOT_FOUND = -704
"""Status a get answers for a key the store does not hold; nothing is written."""

LEASE_EXPIRED = -707
"""Status a get answers when the lease its own query took ran out before the transfer ended.
The bytes have been written all the same; only the check after them failed. A second get takes
a fresh lease, so the driver asks once more for such keys."""

OBJECT_ALREADY_EXISTS = -705
"""Mooncake's code for a put of a key the store already holds. The client reports such a put as
``0`` and the store keeps the first object (observed against a real master), so this code is
translated for completeness: it means declined, not failed."""

NO_AVAILABLE_HANDLE = -200
"""Status a put answers when the store has no room for the object; nothing is written."""

_PUT_DECLINED = frozenset({OBJECT_ALREADY_EXISTS, NO_AVAILABLE_HANDLE})
"""Put statuses that mean the store chose not to hold the key, not that the call went wrong."""

MEMCPY_BYPASS_ENV = "MC_STORE_MEMCPY"
"""Mooncake client switch that copies an object living in this process's own segment without the
transfer engine: older clients with a plain ``memcpy`` whatever the memory, newer ones by where
the pointer lives. Unset, the client turns it on whenever TCP is the only transport it loaded;
``_keep_memcpy_bypass_off_over_device_memory`` says what this driver does about it."""

_MEMCPY_BYPASS_OFF = ("0", "false", "no", "off")
"""The values the client reads as "off", compared lowercase with nothing trimmed; any other value
is on."""

_DEFAULT_GLOBAL_SEGMENT_SIZE = 3355443200
_DEFAULT_LOCAL_BUFFER_SIZE = 16 * 1024 * 1024


@dataclass(frozen=True)
class MooncakeStoreConfig:
    """How to reach a Mooncake store; the backend's own options are ``BlobStoreConfig``.

    Attributes:
        master_server_address: ``host:port`` of the Mooncake master.
        local_hostname: Address this process is reachable at. ``None`` picks the host's own.
        metadata_server: Mooncake metadata service connstring.
        protocol: Transport protocol, ``"rdma"`` or ``"tcp"``.
        device_name: RDMA device filter handed to ``setup``; empty means any.
        global_segment_size: Bytes this process contributes to the pool.
        local_buffer_size: Bytes of the client's own transfer buffer. Only the bindings' copying
            APIs (``put`` / ``get`` of Python bytes) stage through it; this driver uses the
            zero-copy multi-buffer calls over registered memory, so the bindings' own default of
            16 MiB is plenty.
    """

    master_server_address: str
    local_hostname: str | None = None
    metadata_server: str = DEFAULT_METADATA_SERVER
    protocol: str = "rdma"
    device_name: str = ""
    global_segment_size: int = _DEFAULT_GLOBAL_SEGMENT_SIZE
    local_buffer_size: int = _DEFAULT_LOCAL_BUFFER_SIZE

    def __post_init__(self) -> None:
        if not self.master_server_address:
            raise ValueError("master_server_address is required")
        if not self.metadata_server:
            raise ValueError("metadata_server must not be empty")
        if self.protocol not in ("tcp", "rdma"):
            raise ValueError(f"protocol must be 'tcp' or 'rdma', got {self.protocol!r}")
        if self.global_segment_size < 0:
            raise ValueError("global_segment_size must be >= 0")
        if self.local_buffer_size <= 0:
            raise ValueError("local_buffer_size must be > 0")

    @classmethod
    def fields(cls) -> frozenset[str]:
        """The option keys this driver reads."""
        return frozenset(f.name for f in dataclasses.fields(cls))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> MooncakeStoreConfig:
        """Build from a plain mapping, refusing keys this class does not know."""
        return strict_from_dict(cls, raw)


def _default_hostname() -> str:
    import socket

    return socket.gethostbyname(socket.gethostname())


def _allocation_spans(address: int, size: int) -> list[Segment]:
    """The span cut at the boundaries of the CUDA allocations it covers.

    KV cache manager V2 maps one ``cuMemCreate`` chunk after another into a reserved address
    range, so a KV pool is many allocations. Without nvidia-peermem the Mooncake client registers
    GPU memory through a dma-buf of the one allocation holding the address it is given, and a span
    over several chunks fails to register; chunk by chunk it registers. Memory the driver cannot
    look up (host memory, or no CUDA driver) comes back whole.
    """
    try:
        import cuda.bindings.driver as cuda
    except ImportError:
        logger.debug(
            "registering %s whole: the CUDA driver bindings are unavailable",
            _span_text(address, size),
        )
        return [(address, size)]
    spans = []
    cursor, end = address, address + size
    while cursor < end:
        err, base, length = cuda.cuMemGetAddressRange(cursor)
        if err != cuda.CUresult.CUDA_SUCCESS:
            return [(address, size)]
        stop = min(int(base) + int(length), end)
        if stop <= cursor:  # an allocation that does not reach past the cursor cannot be walked
            return [(address, size)]
        spans.append((cursor, stop - cursor))
        cursor = stop
    return spans


def _span_text(address: int, size: int) -> str:
    """``[start, end)`` in hex, as the backend's own messages write spans."""
    return f"[{address:#x}, {address + size:#x})"


def _split(buffers: Sequence[Sequence[Segment]]) -> tuple[list[list[int]], list[list[int]]]:
    """The bindings take addresses and sizes as two parallel lists per key."""
    ptrs = [[address for address, _ in segments] for segments in buffers]
    sizes = [[size for _, size in segments] for segments in buffers]
    return ptrs, sizes


class MooncakeBlobStore:
    """``BlobStore`` over a ``MooncakeDistributedStore``: translates its status codes.

    The bindings answer integers, never raise; the codes are Mooncake's ``ErrorCode``
    (``mooncake-store/include/types.h``). ``batch_is_exist`` answers ``1`` present, ``0`` absent,
    negative for a lookup that failed. A get answers the bytes read per key, or a negative code
    with the destination untouched: ``OBJECT_NOT_FOUND`` for a key not there, other codes
    (``INVALID_PARAMS`` -600 / ``TRANSFER_FAIL`` -800 when the buffers do not add up to the
    object) for a key it could not read. A put answers ``0`` per key it stored;
    ``OBJECT_ALREADY_EXISTS`` and ``NO_AVAILABLE_HANDLE`` mean the store chose not to hold the
    key and are ``DECLINED``, which the backend follows with a lookup; any other code
    (``INVALID_PARAMS`` -600, ``TRANSFER_FAIL`` -800, ``RPC_FAIL`` -900, ``INTERNAL_ERROR`` -1,
    ...) means the key could not be written and is ``FAILED``, which fails the publish.

    Two behaviours observed against a real master (``test_real_master.py``): a put of a key the
    store already holds answers ``0`` and keeps the first object, so a publish that lost a race
    usually reads as ``STORED`` rather than ``DECLINED`` (the backend is correct either way,
    because it asks ``contains`` before every put); and a master started with
    ``--memory_allocator=cachelib`` stores an object as one allocation of at most one slab (16 MiB
    minus 16 bytes) and answers ``INVALID_PARAMS`` for a unit whose segments add up to more,
    however they are cut, where the default ``offset`` allocator has no such cap. An answer of
    the wrong length from any batch call raises ``BlobStoreError``. A span is registered one
    CUDA allocation at a time (``_allocation_spans``) and unregistered the same way.
    """

    def __init__(self, client: Any, config: MooncakeStoreConfig) -> None:
        self._client = client
        self._config = config
        self._pieces: dict[int, list[Segment]] = {}
        """The pieces each registered span was cut into and still holds, by the span's address."""

    @classmethod
    def open(cls, config: MooncakeStoreConfig) -> MooncakeBlobStore:
        """Connect a real ``MooncakeDistributedStore`` to the master named by ``config``.

        Raises:
            ImportError: The Mooncake Python bindings are not installed.
            BlobStoreError: ``setup`` returned a non-zero status.
        """
        try:
            from mooncake.store import MooncakeDistributedStore
        except ImportError as exc:
            raise ImportError(
                "The Mooncake store backend needs the Mooncake Python bindings "
                "(`pip install mooncake-transfer-engine`)."
            ) from exc

        client = MooncakeDistributedStore()
        status = client.setup(
            config.local_hostname or _default_hostname(),
            config.metadata_server,
            config.global_segment_size,
            config.local_buffer_size,
            config.protocol,
            config.device_name,
            config.master_server_address,
        )
        if status != 0:
            raise BlobStoreError(
                f"MooncakeDistributedStore.setup failed with status {status} "
                f"(master={config.master_server_address!r}, metadata={config.metadata_server!r}, "
                f"protocol={config.protocol!r})"
            )
        return cls(client, config)

    @property
    def raw(self) -> Any:
        """The wrapped ``MooncakeDistributedStore``, for tests that inspect the store directly."""
        return self._client

    def describe(self) -> str:
        return (
            f"mooncake master={self._config.master_server_address} protocol={self._config.protocol}"
        )

    def register_span(self, address: int, size: int) -> None:
        pieces = _allocation_spans(address, size)
        registered = []
        for start, length in pieces:
            status = self._client.register_buffer(start, length)
            if status != 0:
                for done, _ in registered:
                    self._client.unregister_buffer(done)
                where = (
                    f" (piece {len(registered) + 1} of {len(pieces)})" if len(pieces) > 1 else ""
                )
                raise BlobStoreError(
                    f"register_buffer failed with status {status} for "
                    f"{_span_text(start, length)}{where}"
                )
            registered.append((start, length))
        self._pieces[address] = registered
        if len(pieces) > 1:
            logger.info(
                "%s: registered %d bytes as %d CUDA allocations", self.describe(), size, len(pieces)
            )

    def unregister_span(self, address: int, size: int) -> None:
        """Every piece is asked to unregister even after one fails. The pieces that failed stay
        on the books, so a retry asks for those alone; one error names them all."""
        pieces = self._pieces.get(address, [(address, size)])
        failed = []
        for start, length in pieces:
            status = self._client.unregister_buffer(start)
            if status != 0:
                failed.append((start, length, status))
        if not failed:
            self._pieces.pop(address, None)
            return
        self._pieces[address] = [(start, length) for start, length, _ in failed]
        failures = ", ".join(
            f"status {status} for {_span_text(start, length)}" for start, length, status in failed
        )
        where = f" ({len(failed)} of {len(pieces)} pieces)" if len(pieces) > 1 else ""
        raise BlobStoreError(f"unregister_buffer failed with {failures}{where}")

    @staticmethod
    def _check_count(call: str, statuses: Sequence[int], keys: Sequence[str]) -> None:
        """An answer of the wrong length is a failed call: none of it can be trusted."""
        if len(statuses) != len(keys):
            raise BlobStoreError(f"{call} answered {len(statuses)} of {len(keys)} keys")

    def contains(self, keys: Sequence[str]) -> Sequence[bool]:
        statuses = self._client.batch_is_exist(list(keys))
        self._check_count("batch_is_exist", statuses, keys)
        if any(status < 0 for status in statuses):
            raise BlobStoreError(f"batch_is_exist answered {sorted(set(statuses))}")
        return [status == 1 for status in statuses]

    def put(self, keys: Sequence[str], buffers: Sequence[Sequence[Segment]]) -> Sequence[PutStatus]:
        ptrs, sizes = _split(buffers)
        statuses = self._client.batch_put_from_multi_buffers(list(keys), ptrs, sizes)
        self._check_count("batch_put_from_multi_buffers", statuses, keys)
        return [self._put_status(key, status) for key, status in zip(keys, statuses)]

    def _put_status(self, key: str, status: int) -> PutStatus:
        if status == 0:
            return PutStatus.STORED
        if status in _PUT_DECLINED:
            logger.debug("%s: put of %s declined with status %d", self.describe(), key, status)
            return PutStatus.DECLINED
        logger.warning("%s: put of %s failed with status %d", self.describe(), key, status)
        return PutStatus.FAILED

    def get(self, keys: Sequence[str], buffers: Sequence[Sequence[Segment]]) -> Sequence[GetStatus]:
        statuses = list(self._batch_get(keys, buffers))
        expired = [i for i, status in enumerate(statuses) if status == LEASE_EXPIRED]
        if expired:
            logger.info(
                "%s: lease expired during the get of %d of %d keys; asking once more",
                self.describe(),
                len(expired),
                len(keys),
            )
            again = self._batch_get([keys[i] for i in expired], [buffers[i] for i in expired])
            for i, status in zip(expired, again):
                statuses[i] = status
        return [
            self._read_status(key, status, sum(size for _, size in segments))
            for key, status, segments in zip(keys, statuses, buffers)
        ]

    def _batch_get(
        self, keys: Sequence[str], buffers: Sequence[Sequence[Segment]]
    ) -> Sequence[int]:
        ptrs, sizes = _split(buffers)
        statuses = self._client.batch_get_into_multi_buffers(list(keys), ptrs, sizes)
        self._check_count("batch_get_into_multi_buffers", statuses, keys)
        return statuses

    def _read_status(self, key: str, status: int, expected: int) -> GetStatus:
        if status == expected:
            return GetStatus.HIT
        if status == OBJECT_NOT_FOUND:
            return GetStatus.MISS
        logger.warning(
            "%s: get of %s answered %d for %d bytes", self.describe(), key, status, expected
        )
        return GetStatus.FAILED

    def close(self) -> None:
        status = self._client.close()
        if status != 0:
            logger.warning("%s: close answered status %d", self.describe(), status)


def _resolve_landing(entry: BackendEntry) -> BackendEntry:
    """The entry with its ``landing`` decided: over TCP an unset ``landing`` becomes ``host``,
    because the transfer engine's TCP path reaches GPU memory only through a synchronous copy
    per 64 KB chunk, while RDMA writes GPU memory directly and keeps the ``device`` default. An
    explicit ``device`` over TCP is allowed with a warning (and with the memcpy bypass kept off,
    see ``_keep_memcpy_bypass_off_over_device_memory``)."""
    protocol = entry.options.get("protocol", MooncakeStoreConfig.protocol)
    landing = entry.options.get("landing")
    if protocol != "tcp":
        return entry
    if landing is None:
        logger.info(
            "backend %r: protocol tcp with no landing set; landing on host memory first "
            "(landing: host)",
            entry.name,
        )
        return dataclasses.replace(entry, options={**entry.options, "landing": "host"})
    if landing == "device":
        logger.warning(
            "backend %r: landing 'device' over tcp registers GPU memory with a transport that "
            "copies it through host memory in 64 KB steps; 'host' is the intended landing for tcp",
            entry.name,
        )
    return entry


def _keep_memcpy_bypass_off_over_device_memory(entry: BackendEntry) -> None:
    """With ``landing: device`` the backend registers the KV pools, which live on the GPU. Older
    Mooncake clients served the memcpy bypass with a plain ``memcpy`` even for such spans; this
    backend keeps the bypass off for the device landing whatever the client, and refuses an
    explicit on. Left unset, the variable is decided by the client from the transports it
    actually loaded (on when only TCP came up, which an RDMA entry without a reachable HCA does
    too), not from the entry's ``protocol``; so an unset variable is set to off here, before
    ``setup`` reads it. The variable is process-wide and is set only when unset. ``landing:
    host`` registers pinned host memory only, where the bypass is a harmless fast path, and is
    left to the client. Reads the resolved ``landing``."""
    if entry.options.get("landing", "device") != "device":
        return
    value = os.environ.get(MEMCPY_BYPASS_ENV)
    if value is None:
        logger.info(
            "backend %r: %s unset; set to 0 for landing: device", entry.name, MEMCPY_BYPASS_ENV
        )
        os.environ[MEMCPY_BYPASS_ENV] = "0"
        return
    if value.lower() in _MEMCPY_BYPASS_OFF:
        return
    raise ValueError(
        f"backend {entry.name!r}: {MEMCPY_BYPASS_ENV}={value!r} turns the Mooncake client's "
        "memcpy bypass on, which this backend keeps off for landing: device (it registers GPU "
        f"memory, and older clients memcpy'd such spans); set {MEMCPY_BYPASS_ENV}=0 or use "
        "landing: host"
    )


def build_mooncake_backend(entry: BackendEntry, context: BackendBuildContext) -> BackendHandle:
    """Factory for ``type: mooncake``."""
    entry = _resolve_landing(entry)
    _keep_memcpy_bypass_off_over_device_memory(entry)
    return build_blob_backend(
        entry,
        context,
        open_store=lambda options: MooncakeBlobStore.open(MooncakeStoreConfig.from_dict(options)),
        driver_fields=MooncakeStoreConfig.fields(),
    )

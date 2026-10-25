# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opening Mooncake from a store config: the deployment glue around the driver.

A store config has two sections, both read here: ``blob`` into the backend's ``BlobConfig``, and
``mooncake``. ``make_store_driver`` picks the driver: the native one, whose store calls run on C++
threads that never hold the GIL, wherever it can run, or the Python one.

Mooncake reads its own environment variables, some once per process (``MOONCAKE_STORE_CHECKSUM``),
some per client (``MC_STORE_MEMCPY``) and ``MC_TCP_PROTO`` on every transfer; this module logs them
before setup and sets the two RPC timeouts (the call timeout from the config, the connect timeout
to RPC_CONNECT_TIMEOUT_MS) and, unless set, ``MC_HANDSHAKE_MAX_LENGTH``. ``open_sdk`` refuses what
the drivers' rules do not cover: TCP's legacy framing, the local hot cache, the newer transfer
engine and any SDK build other than the pinned ones.

``StoreSegment`` is host memory a process gives the store: a segment that holds any process's
objects.

Over RDMA, a process that used ibverbs before its first Mooncake client (importing
``tensorrt_llm`` does) logs ``RDMA context setup failed: fork compatibility: Invalid argument``
once: ``ibv_fork_init`` only works before any other verbs call. It is harmless on Linux 5.13 and
later, which copy pinned pages on fork, and the devices come up normally.
"""

from __future__ import annotations

import importlib.metadata
import os
import pathlib
import socket
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml

from tensorrt_llm.logger import logger

from .backend import FETCH_BATCHES_PER_PUBLISH_BATCH, BlobConfig
from .mooncake import AllocationRange, MooncakeDriver
from .mooncake_native import NativeMooncakeDriver

__all__ = [
    "MooncakeConfig",
    "StoreSegment",
    "load_store_config",
    "make_store_driver",
    "open_sdk",
]

SDK_DISTRIBUTION = "mooncake-transfer-engine-cuda13"  # the one TensorRT-LLM installs
# The builds of SDK_VERSION the drivers' rules were read for: one source tree, built with CUDA 13,
# CUDA 12 or no CUDA. All three compile the client class the store worker mirrors alike.
PINNED_SDK_DISTRIBUTIONS = (
    SDK_DISTRIBUTION,
    "mooncake-transfer-engine",
    "mooncake-transfer-engine-non-cuda",
)
SDK_VERSION = "0.3.13"
IB_SYSFS_ROOT = "/sys/class/infiniband"
MASTER_WAIT_S = 60.0
MASTER_RETRY_S = 0.5
# How long one connection to the master may take; short, as ``open_sdk`` first waits
# (MASTER_WAIT_S) until the master accepts connections.
RPC_CONNECT_TIMEOUT_MS = 5_000
SECTIONS = ("blob", "mooncake")
DRIVERS = ("auto", "native", "python")
# The tensorrt_llm.mooncake_store_worker API this code speaks (its API_VERSION, bindings.cpp
# kApiVersion): a module built from other sources is refused at startup.
STORE_WORKER_API_VERSION = 5
# The longest handshake message a process accepts. A process's first transfer with another sends
# its transfer descriptor, which lists every buffer it registered: under GPU Direct thousands of KV
# cache pieces. Mooncake drops a longer message than MC_HANDSHAKE_MAX_LENGTH (1 MiB unless set, at
# most 128 MiB), which it reads once per process.
HANDSHAKE_MAX_BYTES = 64 << 20
# How long a segment being unmounted stays readable (StoreSegment.begin_unmount): a get that misses
# after it costs a request its fetch, nothing more.
UNMOUNT_GRACE_S = 2
# Mooncake frees a segment unmounted gracefully this long after its grace ended, once the master
# confirms (client_service.cpp, Client::StartGracefulUnmountTimer): a get that found the segment
# before the grace ended has that long to finish.
MOONCAKE_UNMOUNT_CLEANUP_S = 10
# Slack after Mooncake's cleanup, for the master's confirmation and the free itself.
_UNMOUNT_MARGIN_S = 1
# A segment's location hint: Mooncake reads the NUMA node from the memory itself.
_ANY_LOCATION = "*"

# Logged before setup, as Mooncake will read them.
_LOGGED_ENVIRONMENT = (
    "MC_RPC_TIMEOUT_MS",
    "MC_RPC_CONNECT_TIMEOUT_MS",
    "MC_HANDSHAKE_MAX_LENGTH",
    "MC_STORE_MEMCPY",
    "MOONCAKE_STORE_CHECKSUM",
    "WITH_NVIDIA_PEERMEM",
    "MC_TCP_SLICE_SIZE",
    "MC_TCP_PROTO",
    "MC_STORE_LOCAL_HOT_CACHE_SIZE",
)
# Reserved for documentation (RFC 5737): a datagram socket connected to it reveals the route off
# this host without sending anything.
_ROUTE_PROBE_ENDPOINT = ("192.0.2.1", 9)


@dataclass(frozen=True)
class MooncakeConfig:
    """The ``mooncake`` section of a store config.

    Attributes:
        master_server_address: The master's ``host:port``.
        global_segment_size: Bytes of this process's memory that hold pool objects (its
            segment). Required: the objects in it go when the process does; 0 leaves holding
            objects to other processes.
        protocol: ``"rdma"`` or ``"tcp"``.
        device_name: Comma-separated RDMA devices; empty picks every active InfiniBand device at
            the highest rate.
        local_hostname: The address peers reach this process at; empty takes the address the
            route to the master leaves from.
        metadata_server: Where segments are described; ``P2PHANDSHAKE`` needs no server.
        local_buffer_size: Bytes for the SDK's copying calls, which the driver never makes; 0 is
            allowed, and the default is the SDK's.
        rpc_timeout_ms: How long one call to the master may take. It bounds the store calls
            that take no deadline of their own, an existence check say.
        driver: ``native``: store calls run on C++ threads that never hold the GIL, and the server
            refuses to start where they cannot; ``python``: on the blob backend's Python threads,
            where each call's return waits for the GIL; ``auto``: native where it can run.
    """

    master_server_address: str
    global_segment_size: int
    protocol: str = "rdma"
    device_name: str = ""
    local_hostname: str = ""
    metadata_server: str = "P2PHANDSHAKE"
    local_buffer_size: int = 16 << 20
    rpc_timeout_ms: int = 10_000
    driver: str = "auto"

    def __post_init__(self) -> None:
        for name in ("master_server_address", "device_name", "local_hostname", "metadata_server"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be a str, got {getattr(self, name)!r}")
        if self.driver not in DRIVERS:
            raise ValueError(f"driver must be one of {DRIVERS}, got {self.driver!r}")
        if self.protocol not in ("rdma", "tcp"):
            raise ValueError(f"protocol must be rdma or tcp, got {self.protocol!r}")
        for name, least in (
            ("global_segment_size", 0),
            ("local_buffer_size", 0),
            ("rpc_timeout_ms", 1),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < least:
                raise ValueError(f"{name} must be an int of at least {least}, got {value!r}")


def load_store_config(config: str | os.PathLike | Mapping) -> tuple[BlobConfig, MooncakeConfig]:
    """Reads a store config: a YAML file, or the mapping parsed from one.

    Returns:
        The ``blob`` section (defaults when absent) and the ``mooncake`` section.

    Raises:
        ValueError: An unknown section or key, a missing ``mooncake`` section or required key, or
            a value of the wrong type.
    """
    if isinstance(config, Mapping):
        document, name = config, "store config"
    else:
        with open(config) as file:
            document = yaml.safe_load(file)
        name = f"store config {config}"
    if not isinstance(document, Mapping):
        raise ValueError(f"{name} is not a mapping")
    unknown = sorted(set(document) - set(SECTIONS))
    if unknown:
        raise ValueError(f"{name}: unknown sections {unknown}; it has {list(SECTIONS)}")
    blob, section = document.get("blob") or {}, document.get("mooncake")
    if not isinstance(blob, Mapping) or not isinstance(section, Mapping):
        raise ValueError(f"{name}: 'mooncake' is a required mapping, 'blob' an optional one")
    try:
        mooncake = MooncakeConfig(**section)
    except (TypeError, ValueError) as error:  # an unknown or missing key, or a bad value
        raise ValueError(f"{name}, section 'mooncake': {error}") from error
    try:
        return BlobConfig(**blob), mooncake
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name}, section 'blob': {error}") from error


def open_sdk(config: MooncakeConfig) -> Any:
    """Opens a set-up ``MooncakeDistributedStore``, on the builder thread: ``setup`` holds the GIL
    while it mounts this process's segment. Sets the process's ``MC_RPC_TIMEOUT_MS`` from
    ``config``, ``MC_RPC_CONNECT_TIMEOUT_MS`` and, unless set, ``MC_HANDSHAKE_MAX_LENGTH`` first:
    every process may receive another's descriptor, one with a segment from each process that
    reads it.

    Raises:
        ValueError: TCP would use its legacy framing, the local hot cache is on, or the newer
            transfer engine is selected.
        ImportError: ``mooncake.store`` does not import, or is not a pinned build of SDK_VERSION.
        TimeoutError: The master does not accept connections.
        RuntimeError: ``setup`` failed; the half-open client is closed.
    """
    _refuse_legacy_tcp_framing(config.protocol)
    _refuse_hot_cache()
    _refuse_newer_transfer_engine()
    # Mooncake reads both when a client is created; unset, each is 30 s.
    os.environ["MC_RPC_TIMEOUT_MS"] = str(config.rpc_timeout_ms)
    os.environ["MC_RPC_CONNECT_TIMEOUT_MS"] = str(RPC_CONNECT_TIMEOUT_MS)
    os.environ.setdefault("MC_HANDSHAKE_MAX_LENGTH", str(HANDSHAKE_MAX_BYTES))
    _log_environment()
    try:
        import mooncake.store as sdk
    except ImportError as error:
        raise ImportError(_sdk_missing(error)) from error
    _require_pinned_sdk(sdk)
    _wait_for_master(config.master_server_address)
    store = sdk.MooncakeDistributedStore()
    code = store.setup(
        config.local_hostname or _local_address(config.master_server_address),
        config.metadata_server,
        config.global_segment_size,
        config.local_buffer_size,
        config.protocol,
        _devices(config),
        config.master_server_address,
    )
    if code != 0:
        store.close()  # no registration and no call in flight yet, unlike a client in service
        raise RuntimeError(
            f"Mooncake setup failed with {code} (master {config.master_server_address}, "
            f"protocol {config.protocol})"
        )
    return store


@dataclass
class StoreSegment:
    """Host memory this process gives the store: a segment of fresh, pinned memory that the master
    places objects in and other clients read and write directly.

    Unmounting takes two calls. ``begin_unmount``: no new object lands here, and the objects here
    stay readable for UNMOUNT_GRACE_S; after that the master drops them and gets miss. Mooncake
    keeps the memory registered MOONCAKE_UNMOUNT_CLEANUP_S longer for gets already under way, then
    frees it. ``wait_until_freed`` returns after that; then the process may exit.

    Rules: never ``close()`` the client, nor end the process before ``wait_until_freed`` returns:
    either frees the memory under a reader's get, that get fails without proof that its transfers
    ended, and the reader's process ends. Mount at most once per client: a later segment would go
    by the same name (the client's address), where a peer's retried read of an old object could
    land.
    """

    store: Any  # the MooncakeDistributedStore that mounted the memory
    segment_ids: list[str]
    # When Mooncake will have freed the memory (time.monotonic()); None until an unmount was taken.
    _freed_at: Optional[float] = field(default=None, init=False)

    @classmethod
    def mount(cls, store: Any, nbytes: int, protocol: str) -> StoreSegment:
        """Mounts ``nbytes`` of fresh host memory as a segment of ``store``, a client opened with
        no segment of its own (``global_segment_size`` 0). Mooncake rounds the size up to whole
        allocator slabs.

        Raises:
            RuntimeError: Mooncake did not mount it.
        """
        mounted = store.allocate_and_mount_segment(nbytes, protocol, _ANY_LOCATION)
        if mounted["ret"] != 0:
            raise RuntimeError(
                f"mooncake: mounting {nbytes} bytes of host memory failed with {mounted['ret']}"
            )
        return cls(store, list(mounted["segment_ids"]))

    def begin_unmount(self) -> None:
        """Stops new objects landing in the segment; the old ones stay readable for
        UNMOUNT_GRACE_S. If the master refuses, the memory goes when the process exits."""
        code = self.store.unmount_and_free_segment(self.segment_ids, UNMOUNT_GRACE_S)
        if code != 0:
            logger.error(f"mooncake: the master refused to unmount the segment (code {code})")
            return
        self._freed_at = (
            time.monotonic() + UNMOUNT_GRACE_S + MOONCAKE_UNMOUNT_CLEANUP_S + _UNMOUNT_MARGIN_S
        )

    def wait_until_freed(self) -> None:
        """Sleeps until Mooncake has freed the memory; returns at once if no unmount was taken."""
        if self._freed_at is not None:
            time.sleep(max(0.0, self._freed_at - time.monotonic()))


class _NativeDriverUnavailable(Exception):
    """Why the native store driver cannot run here."""


def make_store_driver(
    store: Any,
    config: MooncakeConfig,
    blob: BlobConfig,
    *,
    allocation_range: AllocationRange | None = None,
) -> MooncakeDriver:
    """The serving driver over a set-up ``store`` from ``open_sdk``, which admits only the pinned
    SDK builds, as ``config.driver`` asks: the native driver with ``blob.workers`` store threads
    where it can run, else the Python driver.

    Raises:
        RuntimeError: The driver ``native`` was asked for and cannot run here.
    """
    if config.driver == "python":
        logger.info("mooncake: store calls run on Python threads (driver: python)")
        return MooncakeDriver(store, allocation_range=allocation_range)
    try:
        worker = _start_store_worker(store, blob)
    except Exception as error:  # the one fallback boundary: whatever stops the worker, say why
        reason = str(error) if isinstance(error, _NativeDriverUnavailable) else repr(error)
        if config.driver == "native":
            raise RuntimeError(f"mooncake: driver 'native' cannot run here: {reason}") from error
        logger.warning(
            "mooncake: store calls run on Python threads, where each call's return waits for the "
            f"GIL, as the native store worker cannot run here: {reason}"
        )
        return MooncakeDriver(store, allocation_range=allocation_range)
    logger.info(f"mooncake: store calls run on {blob.workers} C++ threads that never hold the GIL")
    return NativeMooncakeDriver(store, worker, allocation_range=allocation_range)


def _start_store_worker(store: Any, blob: BlobConfig) -> Any:
    """A store worker over ``store``'s client; raises saying why not."""
    try:
        import tensorrt_llm.mooncake_store_worker as store_worker
    except ImportError as error:
        raise _NativeDriverUnavailable(f"the store worker is not built: {error}") from error
    api_version = getattr(store_worker, "API_VERSION", None)
    if api_version != STORE_WORKER_API_VERSION:
        raise _NativeDriverUnavailable(
            f"the store worker module has API version {api_version}, this code needs "
            f"{STORE_WORKER_API_VERSION}: it was built from other sources"
        )
    # Raises where the store has no client or a client the worker does not mirror (by slot).
    return store_worker.StoreWorker(
        store,
        threads=blob.workers,
        put_threads=blob.publish_workers,
        gets_per_put=FETCH_BATCHES_PER_PUBLISH_BATCH,
        rows_per_call=blob.rows_per_batch,
        bytes_per_call=blob.bytes_per_batch,
    )


def _refuse_legacy_tcp_framing(protocol: str) -> None:
    """Under ``MC_TCP_PROTO=1`` a TCP write completes once its last chunk is in the local socket
    buffer, so a put could be READY before its bytes landed. The SDK reads the variable on every
    transfer: setting it after this check goes unnoticed."""
    if protocol == "tcp" and os.environ.get("MC_TCP_PROTO") == "1":
        raise ValueError("MC_TCP_PROTO=1 selects TCP framing that reports writes before they land")


def _refuse_hot_cache() -> None:
    """The SDK copies the slot a get read into its hot cache before it checks the read lease, and
    never drops the copy when the master evicts the object. So a read answered LEASE_EXPIRED, torn
    by a reuse of the object's memory, stays cached; once the key is put again, every get of it
    here is served those bytes as a whole object. Every value but empty or "0" is refused: the SDK
    reads it with ``stoull``, which also turns the cache on for values like "64MB" or " 64"."""
    if os.environ.get("MC_STORE_LOCAL_HOT_CACHE_SIZE", "0").strip() not in ("", "0"):
        raise ValueError("MC_STORE_LOCAL_HOT_CACHE_SIZE turns on a cache that can serve torn reads")


def _refuse_newer_transfer_engine() -> None:
    """Either variable, set to any value, makes the SDK build its transfer engine on TENT, whose
    completion rules the drivers were not read against (``mooncake`` module docstring). The SDK
    reads them when a client is set up."""
    selected = [name for name in ("MC_USE_TENT", "MC_USE_TEV1") if name in os.environ]
    if selected:
        raise ValueError(
            f"{', '.join(selected)} selects a transfer engine the driver does not cover"
        )


def _log_environment() -> None:
    values = ", ".join(f"{name}={os.environ.get(name, '<unset>')}" for name in _LOGGED_ENVIRONMENT)
    logger.info(f"mooncake: environment at setup: {values}")


def _sdk_missing(error: ImportError) -> str:
    hint = f"Install {SDK_DISTRIBUTION}=={SDK_VERSION} and put it before NIXL on the path."
    shadow = getattr(sys.modules.get("mooncake"), "__file__", None)
    if shadow:
        hint = f"The 'mooncake' found is {shadow}; NIXL ships one that shadows the SDK's. {hint}"
    return f"cannot import mooncake.store ({error}). {hint}"


def _require_pinned_sdk(sdk: Any) -> None:
    """The drivers' code mapping and access-end rules were read from SDK_VERSION; on another version
    a put could return before its transfer ended, and a revoke then free space a transfer still
    writes. So the imported ``mooncake.store`` must be a file of one of PINNED_SDK_DISTRIBUTIONS at
    SDK_VERSION, not of another version, a source tree or a shadowing package."""
    file = getattr(sdk, "__file__", None)
    found = []
    for name in PINNED_SDK_DISTRIBUTIONS:
        try:
            build = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        if file is not None and _lists_file(build, file):
            if build.version == SDK_VERSION:
                return
            found.append(f"{name}=={build.version}")
    raise ImportError(
        f"mooncake.store ({file}) comes from {', '.join(found) or 'no pinned distribution'}; the "
        f"driver follows {SDK_VERSION} of {', '.join(PINNED_SDK_DISTRIBUTIONS)} only. Install "
        f"{SDK_DISTRIBUTION}=={SDK_VERSION}."
    )


def _lists_file(build: importlib.metadata.Distribution, file: str) -> bool:
    path = pathlib.Path(file).resolve()
    return any(pathlib.Path(build.locate_file(f)).resolve() == path for f in build.files or ())


# Three deployment helpers, with the same rules as the Mooncake connector's
# (pyexecutor/connectors/mooncake_store: master.py and config.py), which this package cannot import.


def _devices(config: MooncakeConfig) -> str:
    """``device_name``, or for RDMA left empty every active InfiniBand device at the highest rate:
    the compute fabric, without the Ethernet ports that share an HCA."""
    if config.device_name or config.protocol != "rdma":
        return config.device_name
    detected = _fastest_active_ib_devices(IB_SYSFS_ROOT)
    if not detected:
        logger.warning(f"mooncake: no active InfiniBand device under {IB_SYSFS_ROOT}")
    return ",".join(detected)


def _fastest_active_ib_devices(sysfs_root: str) -> list[str]:
    try:
        devices = sorted(os.listdir(sysfs_root))
    except OSError:
        return []
    rates = {}
    for device in devices:
        port = os.path.join(sysfs_root, device, "ports", "1")
        link, state = _read_attribute(port, "link_layer"), _read_attribute(port, "state")
        rate = _read_attribute(port, "rate").split()  # "400 Gb/sec (4X NDR)"
        if link == "InfiniBand" and "ACTIVE" in state and rate and rate[0].isdigit():
            rates[device] = int(rate[0])
    fastest = max(rates.values(), default=None)
    return [device for device, rate in rates.items() if rate == fastest]


def _read_attribute(directory: str, name: str) -> str:
    try:
        with open(os.path.join(directory, name)) as file:
            return file.read().strip()
    except OSError:
        return ""


def _local_address(master: str) -> str:
    """The address the route to the master leaves from. Not ``gethostbyname(gethostname())``,
    which answers 127.0.1.1 where /etc/hosts maps the host's name to loopback."""
    for endpoint in (_split_address(master), _ROUTE_PROBE_ENDPOINT):
        if endpoint is None:
            continue
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                probe.connect(endpoint)
                address = probe.getsockname()[0]
        except OSError:
            continue
        if not address.startswith("127."):
            return address
    return socket.gethostbyname(socket.gethostname())


def _wait_for_master(address: str) -> None:
    """Turns an unreachable master into one line naming it, instead of a bare code from setup."""
    endpoint = _split_address(address)
    if endpoint is None:
        logger.warning(f"mooncake: cannot check master_server_address {address!r}: not host:port")
        return
    deadline = time.monotonic() + MASTER_WAIT_S
    while True:
        try:
            with socket.create_connection(endpoint, timeout=1.0):
                return
        except OSError as error:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"the Mooncake master at {address} did not accept connections within "
                    f"{MASTER_WAIT_S:g} s: {error}"
                ) from error
        time.sleep(MASTER_RETRY_S)


def _split_address(address: str) -> tuple[str, int] | None:
    host, _, port = address.rpartition(":")
    return (host.strip("[]"), int(port)) if host and port.isdigit() else None

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How KV cache blocks move between the store and the KV cache, chosen once at startup.

GPU Direct: the store's NICs read and write the KV cache pages in place. Host staging: blocks are
copied through pinned host memory. Both store a block as the same bytes under the same key, so
servers on either path share blocks.

Deployment assumptions: a provider started otherwise than by ``trtllm-serve mooncake_provider``
sets ``MC_HANDSHAKE_MAX_LENGTH`` as that command does, since a GPU Direct server's first transfer
sends it a descriptor of every registered piece of KV cache memory. Where a peer-memory module is
loaded but the KV cache pools could not be allocated GPUDirect-capable, the check (on an ordinary
device buffer) passes and registering the pools fails: the server is refused, also under 'auto'.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Optional

import torch

from tensorrt_llm.logger import logger

from ...disaggregation.base.memory import MemoryKind
from ...disaggregation.blob.mooncake import MooncakeDriver
from ...disaggregation.blob.mooncake_setup import MooncakeConfig, open_sdk
from ...disaggregation.blob.store import RegionLayout, RowIO, RowStatus, Span
from ..connectors.mooncake_store.gpudirect import (
    PEERMEM_ENV,
    allocation_range,
    select_registration_path,
)

GPU_DIRECT, HOST_STAGING = "gpu_direct", "host_staging"
# The startup check's object: one large GPU page, a caching-allocator block of its own.
STORE_CHECK_BYTES = 2 << 20
# Each half of the check, host and device, may take this long: Mooncake gives a get up to 60 s.
STORE_CHECK_TIMEOUT_S = 60.0
# A check put the store declines for room is tried once more after this: a full store answers
# NO_AVAILABLE_HANDLE until the master's eviction has run.
DECLINED_RETRY_S = 1.0
# The check's objects, apart from every deployment's blocks; the store evicts them in time.
STORE_CHECK_KEY_PREFIX = "kv-sharing-startup-check/"
# The check's memory, registered for the life of the process: a store call that gave up may still
# touch it (4 MiB of host memory, and 2 MiB of device memory under GPU Direct). Registering host
# memory pins it.
_CHECK_MEMORY: list[torch.Tensor] = []


@dataclass(frozen=True)
class PathCheck:
    """What one rank's startup check found.

    Attributes:
        store_error: Why the store does not serve this rank at all, or None.
        gpu_direct_error: Why GPU Direct cannot serve this rank, or None where it can.
    """

    store_error: Optional[str]
    gpu_direct_error: Optional[str]


def open_store(requested: str, mooncake: MooncakeConfig) -> tuple[MemoryKind, MooncakeDriver]:
    """Opens the store client, checks it with one round trip and picks the data path.

    Run it before any other Mooncake client of the process opens: the first one fixes how the
    process registers device memory.

    Returns:
        DEVICE for GPU Direct or HOST for host staging, and a driver over the client that
        registers device memory exactly under GPU Direct.

    Raises:
        ValueError: The store failed the check, or 'gpu_direct' was asked for and is not possible.
    """
    if requested == HOST_STAGING:
        gpu_direct_error = "host staging was asked for"
    else:
        gpu_direct_error = prepare_gpu_direct(mooncake.protocol)
    store = open_sdk(mooncake)
    path = choose_data_path(requested, [check_store(store, gpu_direct_error)])
    if path == HOST_STAGING:
        return MemoryKind.HOST, MooncakeDriver(store)
    return MemoryKind.DEVICE, MooncakeDriver(store, allocation_range=allocation_range)


def prepare_gpu_direct(protocol: str) -> Optional[str]:
    """Readies the process for GPU Direct, before its first Mooncake client: picks how Mooncake
    registers device memory, which that client fixes for the process (``WITH_NVIDIA_PEERMEM``).
    Returns why GPU Direct is not possible here without asking the store, or None."""
    if protocol != "rdma":
        return f"the store's protocol is {protocol}, not rdma"
    if not _gpu_orders_nic_writes(torch.cuda.current_device()):
        return "the GPU does not order the NIC's writes before its own later work"
    try:
        if select_registration_path(protocol, registers_device_memory=True):
            logger.warning(
                f"kv_sharing: no peer-memory module is loaded, so {PEERMEM_ENV}=0 registers "
                "KV cache memory with the NICs through dma-buf"
            )
    except RuntimeError:
        return (
            f"an earlier Mooncake client of this process fixed {PEERMEM_ENV}, which Mooncake "
            "reads once, to a way that cannot register device memory here; set it before the "
            "process starts, or use data_path 'host_staging'"
        )
    return None


def check_store(store: Any, gpu_direct_error: Optional[str]) -> PathCheck:
    """A round trip of a random object through the store from host memory and, unless
    ``gpu_direct_error`` already says why GPU Direct is out, into device memory and back out of
    it. Each half waits at most STORE_CHECK_TIMEOUT_S: a timeout of the device half is GPU
    Direct's error, not the store's."""
    source = torch.randint(0, 256, (STORE_CHECK_BYTES,), dtype=torch.uint8)
    landed = torch.zeros(STORE_CHECK_BYTES, dtype=torch.uint8)
    _CHECK_MEMORY.extend((source, landed))
    driver = MooncakeDriver(store, allocation_range=allocation_range)
    key = f"{STORE_CHECK_KEY_PREFIX}{uuid.uuid4().hex}"
    store_error = _on_a_thread(
        "a round trip through host memory", lambda: _host_round_trip(driver, key, source, landed)
    )
    if store_error is not None or gpu_direct_error is not None:
        return PathCheck(store_error, gpu_direct_error)
    device = torch.zeros(STORE_CHECK_BYTES, dtype=torch.uint8, device="cuda")
    _CHECK_MEMORY.append(device)
    torch.cuda.synchronize()  # the zeroing ran before the NIC writes
    return PathCheck(
        None,
        _on_a_thread(
            "a round trip through device memory",
            lambda: _device_round_trip(driver, key, device, source, landed),
        ),
    )


def choose_data_path(requested: str, checks: Sequence[PathCheck]) -> str:
    """The path every rank takes: the one requested, or for 'auto' GPU Direct where every rank
    can take it and host staging otherwise, logged with each rank's reason.

    Raises:
        ValueError: Some rank's store failed its check, or 'gpu_direct' was asked for and some
            rank cannot take it.
    """
    for rank, check in enumerate(checks):
        if check.store_error is not None:
            raise ValueError(
                f"kv_sharing: the store failed its startup check on rank {rank}: "
                f"{check.store_error}"
            )
    if requested == HOST_STAGING:
        return HOST_STAGING
    reasons = "; ".join(
        f"rank {rank}: {check.gpu_direct_error}"
        for rank, check in enumerate(checks)
        if check.gpu_direct_error is not None
    )
    if not reasons:
        return GPU_DIRECT
    if requested == GPU_DIRECT:
        raise ValueError(f"kv_sharing: data_path 'gpu_direct' is not possible here: {reasons}")
    logger.warning(
        "kv_sharing: data_path 'auto' moves KV cache blocks through host staging, as GPU Direct "
        f"is not possible here: {reasons}"
    )
    return HOST_STAGING


class _CheckFailed(Exception):
    """A step of the check went wrong; the message says how."""


def _on_a_thread(what: str, steps: Callable[[], None]) -> Optional[str]:
    """Runs ``steps`` on a daemon thread bound to this thread's CUDA device: why they failed,
    named ``what``, or None. Gives up waiting after STORE_CHECK_TIMEOUT_S: a store call may block
    for good."""
    device = torch.cuda.current_device()
    errors: list[Optional[str]] = []

    def run() -> None:
        try:
            # Mooncake asks the CUDA driver about the memory it registers; with no device bound
            # the driver would make a context on GPU 0 for it.
            torch.cuda.set_device(device)
            steps()
        except Exception as error:  # noqa: BLE001 - any failure is the check's answer
            errors.append(f"{what}: {error}")
        else:
            errors.append(None)

    thread = threading.Thread(target=run, name="kv-sharing-store-check", daemon=True)
    thread.start()
    thread.join(STORE_CHECK_TIMEOUT_S)
    return errors[0] if errors else f"{what}: no answer within {STORE_CHECK_TIMEOUT_S:g} s"


def _host_round_trip(
    driver: MooncakeDriver, key: str, source: torch.Tensor, landed: torch.Tensor
) -> None:
    """Puts ``source`` and gets it back into ``landed``."""
    _move(driver, driver.put, key, _registered(driver, source, MemoryKind.HOST))
    _move(driver, driver.get, key, _registered(driver, landed, MemoryKind.HOST))
    _require_equal(landed, source)


def _device_round_trip(
    driver: MooncakeDriver,
    key: str,
    device: torch.Tensor,
    source: torch.Tensor,
    landed: torch.Tensor,
) -> None:
    """Gets ``key`` into ``device``, puts that and gets it back into the registered ``landed``."""
    copy = f"{key}-from-device"
    in_device = _registered(driver, device, MemoryKind.DEVICE)
    landed.zero_()
    _move(driver, driver.get, key, in_device)
    _move(driver, driver.put, copy, in_device)
    _move(driver, driver.get, copy, Span(landed.data_ptr(), STORE_CHECK_BYTES))
    _require_equal(landed, source)


def _registered(driver: MooncakeDriver, tensor: torch.Tensor, kind: MemoryKind) -> Span:
    layout = RegionLayout(STORE_CHECK_BYTES, 1, "startup-check", kind)
    driver.register(tensor.data_ptr(), STORE_CHECK_BYTES, layout)
    return Span(tensor.data_ptr(), STORE_CHECK_BYTES)


def _move(driver: MooncakeDriver, call: Callable, key: str, span: Span) -> None:
    """One row through ``call``, the driver's put or get; raises unless it is READY."""
    rows = [RowIO(key, (span,))]
    (result,) = driver.poll(call(rows))
    if result.status is RowStatus.DECLINED:
        time.sleep(DECLINED_RETRY_S)  # for the master's eviction to make room
        (result,) = driver.poll(call(rows))
    if result.status is not RowStatus.READY:
        raise _CheckFailed(f"the {call.__name__} was {result.fault or result.status.value}")


def _require_equal(landed: torch.Tensor, source: torch.Tensor) -> None:
    if not torch.equal(landed, source):
        raise _CheckFailed("the bytes read back differ from those written")


def _gpu_orders_nic_writes(device: int) -> bool:
    """Whether the GPU sees a NIC's completed writes in its own later work without a flush."""
    from cuda.bindings import driver

    status, handle = driver.cuDeviceGet(device)
    if int(status) != 0:
        return False
    attribute = driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING
    status, ordering = driver.cuDeviceGetAttribute(attribute, handle)
    owner = driver.CUGPUDirectRDMAWritesOrdering.CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER
    return int(status) == 0 and int(ordering) >= int(owner)

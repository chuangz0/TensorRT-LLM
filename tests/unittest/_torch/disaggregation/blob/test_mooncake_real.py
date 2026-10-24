# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The blob backend over a real Mooncake pool: a master, and a separate process holding objects.

The tested clients give the pool no segment (``global_segment_size`` 0), so every object sits in
the other process's segment and every transfer goes through the transfer engine; their
``local_buffer_size`` is 0, so a call that needed the SDK's copying path would fail. Runs only with
``TRTLLM_MOONCAKE_REAL=1``, on a host with ``mooncake_master`` and ``mooncake.store``. The
transport is ``TRTLLM_MOONCAKE_PROTOCOL`` (default rdma) over ``TRTLLM_MOONCAKE_DEVICES`` (default:
the host's fastest active InfiniBand devices). The shared ``driver_conformance`` checks run here
too, each over a fresh client and under a key prefix of its own, with a driver that registers
device memory: over RDMA, rows of several spans move into and out of GPU memory in place. They and
the backend-level tests run on both drivers, the Python one and the native one whose store worker
calls the same client from C++ threads, and must give the same results.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import secrets
import shutil
import threading
from dataclasses import replace

import numpy as np
import pytest
import torch
from cpu_fixtures import WAIT_S, Endpoint, masks
from fake_mooncake import GetsOfKeyWait

from tensorrt_llm._torch.disaggregation.base.backend import Outcome
from tensorrt_llm._torch.disaggregation.base.memory import MemoryKind
from tensorrt_llm._torch.disaggregation.blob.backend import BlobConfig
from tensorrt_llm._torch.disaggregation.blob.mooncake import MooncakeDriver
from tensorrt_llm._torch.disaggregation.blob.mooncake_native import NativeMooncakeDriver
from tensorrt_llm._torch.disaggregation.blob.mooncake_setup import (
    MooncakeConfig,
    make_store_driver,
    open_sdk,
)
from tensorrt_llm._torch.disaggregation.testing.driver_conformance import SLOT as CHECK_SLOT
from tensorrt_llm._torch.disaggregation.testing.driver_conformance import DriverConformance
from tensorrt_llm._torch.disaggregation.testing.host_memory import CANARY, GUARD, HostBuffer
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.gpudirect import (
    allocation_range,
    select_registration_path,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("TRTLLM_MOONCAKE_REAL") != "1",
    reason="needs a real Mooncake pool; set TRTLLM_MOONCAKE_REAL=1",
)

PROTOCOL = os.environ.get("TRTLLM_MOONCAKE_PROTOCOL", "rdma")
DEVICES = os.environ.get("TRTLLM_MOONCAKE_DEVICES", "")
POOL_BYTES = 1 << 30
SLOT_BYTES = 64 << 10
DRIVERS = ("python", "native")


def _require_mooncake() -> None:
    if shutil.which("mooncake_master") is None:
        pytest.skip("mooncake_master is not on PATH")
    try:
        found = importlib.util.find_spec("mooncake.store") is not None
    except ImportError:
        found = False
    if not found:
        pytest.skip("mooncake.store does not import")
    # Before the process's first client, which fixes how every client registers device memory.
    select_registration_path(PROTOCOL, registers_device_memory=True)


def _running_pool(run_dir: str, segment_size: int) -> tuple[contextlib.ExitStack, str]:
    """A master and a segment provider: the stack that stops them, and the master's address."""
    from test_common.mooncake_utils import running_master_on_free_ports, running_segment_provider

    stack = contextlib.ExitStack()
    # Ratio 0: the master evicts only above its high watermark, never just because a put failed.
    master = stack.enter_context(
        running_master_on_free_ports(run_dir, protocol=PROTOCOL, eviction_ratio=0.0)
    )
    stack.enter_context(
        running_segment_provider(
            master.address,
            run_dir,
            segment_size=segment_size,
            protocol=PROTOCOL,
            device_name=DEVICES,
        )
    )
    return stack, master.address


@pytest.fixture(scope="module")
def pool(tmp_path_factory):
    _require_mooncake()
    stack, address = _running_pool(str(tmp_path_factory.mktemp("pool")), POOL_BYTES)
    with stack:
        yield address


class Clients:
    """SDK clients of one master that give the pool no segment, and the endpoints built over them.

    ``close`` closes the clients only once every endpoint's shutdown returns no labels: a backend
    thread still inside the SDK would crash on a closed client, so a failed test leaves them open.
    """

    def __init__(self, master: str) -> None:
        self.master = master
        self.opened = []
        self.endpoints: list[Endpoint] = []

    def open_client(self):
        self.opened.append(open_sdk(client_config(self.master)))
        return self.opened[-1]

    def endpoint(
        self,
        names,
        *,
        driver: str = "python",
        workers: int = 2,
        rows_per_batch: int = 64,
        **options,
    ) -> Endpoint:
        """An endpoint over a fresh client and the ``driver`` asked for; a native driver's worker
        cuts its calls as the endpoint's backend would."""
        blob = BlobConfig(workers=workers, rows_per_batch=rows_per_batch)
        mooncake = replace(client_config(self.master), driver=driver)
        store_driver = make_store_driver(self.open_client(), mooncake, blob)
        options = {"slot_bytes": SLOT_BYTES, **options}
        return self.add(
            Endpoint(store_driver, names, workers=workers, rows_per_batch=rows_per_batch, **options)
        )

    def add(self, endpoint: Endpoint) -> Endpoint:
        self.endpoints.append(endpoint)
        return endpoint

    def close(self) -> None:
        if all(endpoint.backend.shutdown(WAIT_S) == frozenset() for endpoint in self.endpoints):
            for store in self.opened:
                store.close()


def client_config(master: str) -> MooncakeConfig:
    """A client with no segment of its own and no buffer for the SDK's copying calls."""
    return MooncakeConfig(master, 0, PROTOCOL, DEVICES, local_buffer_size=0)


@pytest.fixture
def clients(pool):
    opened = Clients(pool)
    yield opened
    opened.close()


def random_rows(endpoint: Endpoint, seed: int) -> dict[str, bytes]:
    generator = np.random.default_rng(seed)
    rows = {name: generator.bytes(endpoint.region.slot_bytes) for name in endpoint.names}
    for name, data in rows.items():
        endpoint.write(name, data)
    return rows


def finish(endpoint: Endpoint, submit, *names: str) -> Outcome:
    attempt = submit(endpoint.view(*names), None)
    outcome = endpoint.wait_outcome(attempt)
    endpoint.wait_quiesced(attempt)
    return outcome


def test_real_auto_runs_the_native_driver_where_the_slot_check_passes(clients) -> None:
    config = client_config(clients.master)
    assert config.driver == "auto"
    driver = make_store_driver(clients.open_client(), config, BlobConfig())
    assert isinstance(driver, NativeMooncakeDriver)


@pytest.mark.parametrize("driver", DRIVERS)
def test_real_store_full_is_declined(tmp_path, request, driver) -> None:
    _require_mooncake()
    # A pool of its own that one publish overfills: a 16 MiB segment (the smallest a TCP client
    # may give, one allocator slab) holds at most 256 of the 512 rows. A PutStart that finds no room
    # answers at once. The master evicts once the pool passes its high watermark, so how many rows
    # land varies, but some are always declined and none fault.
    stack, master = _running_pool(str(tmp_path), segment_size=16 << 20)
    clients = Clients(master)
    with stack:
        names = [f"{request.node.name}-{i}" for i in range(512)]
        writer = clients.endpoint(names, driver=driver, workers=1)
        random_rows(writer, seed=4)
        outcome = finish(writer, writer.backend.publish, *names)
        delivered = int(outcome.delivered_masks[0].sum())
        declined = writer.driver.inner.stats()["rows_declined_pool_full"]
        assert outcome.fault is None
        assert delivered > 0 and declined > 0 and delivered + declined == len(names)
        writer.close()
        clients.close()  # while the master still answers


# The reader's backend keeps polling the get whose access is unproven, as it would in production.
@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("driver", DRIVERS)
def test_real_get_refuses_an_object_of_another_size_before_moving_a_byte(
    clients, request, driver
) -> None:
    # batch_get_into_multi_buffers refuses an object longer than its buffers (INVALID_PARAMS),
    # and its transfer submission one shorter (TRANSFER_FAIL, which proves no access end).
    short, long = f"{request.node.name}-short", f"{request.node.name}-long"
    half = clients.endpoint([short], driver=driver, slot_bytes=SLOT_BYTES // 2)
    double = clients.endpoint([long], driver=driver, slot_bytes=2 * SLOT_BYTES)
    random_rows(half, seed=9)
    random_rows(double, seed=10)
    assert masks(finish(half, half.backend.publish, short)) == [[True]]
    assert masks(finish(double, double.backend.publish, long)) == [[True]]
    reader = clients.endpoint([short, long], driver=driver, fill=0x2D, rows_per_batch=1)
    outcome = finish(reader, reader.backend.fetch, long)
    assert masks(outcome) == [[False]] and "INVALID_PARAMS(-600)" in outcome.fault
    assert reader.read(long) == b"-" * SLOT_BYTES
    attempt = reader.backend.fetch(reader.view(short), None)
    outcome = reader.wait_outcome(attempt)
    assert masks(outcome) == [[False]] and "TRANSFER_FAIL(-800)" in outcome.fault
    assert reader.read(short) == b"-" * SLOT_BYTES
    assert not attempt.quiesced() and reader.driver.inner.stats()["ops_unprovable"] == 1
    for each in (half, double):
        each.close()


class DeviceBuffer:
    """GPU memory with canaries on both sides: a ``Memory`` for the checks' device rows."""

    def __init__(self, nbytes: int, fill: int) -> None:
        canary = torch.frombuffer(bytearray(CANARY), dtype=torch.uint8)
        self._tensor = torch.full((nbytes + 2 * GUARD,), fill, dtype=torch.uint8, device="cuda")
        self._tensor[:GUARD] = canary
        self._tensor[-GUARD:] = canary
        torch.cuda.synchronize()  # written before the NIC touches it
        self.address = self._tensor.data_ptr() + GUARD
        self.nbytes = nbytes

    def read(self, offset: int = 0, nbytes: int | None = None) -> bytes:
        nbytes = self.nbytes - offset if nbytes is None else nbytes
        return self._tensor[GUARD + offset : GUARD + offset + nbytes].cpu().numpy().tobytes()

    def write(self, data: bytes, offset: int = 0) -> None:
        part = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        self._tensor[GUARD + offset : GUARD + offset + len(data)] = part.cuda()
        torch.cuda.synchronize()

    def canaries_intact(self) -> bool:
        whole = self._tensor.cpu().numpy().tobytes()
        return whole[:GUARD] == CANARY and whole[-GUARD:] == CANARY


class TestRealMooncakeConformance(DriverConformance):
    """The shared driver checks over the real pool, one fresh client per check, with the scenarios
    a real master can stage: a key another client is writing, and a get held inside the SDK. The
    driver registers device memory, and the checks' device rows lie in GPU memory."""

    timeout = 30.0

    @pytest.fixture(autouse=True)
    def _master(self, pool):
        self.master = pool

    def make_driver(self) -> MooncakeDriver:
        return MooncakeDriver(
            open_sdk(client_config(self.master)), allocation_range=allocation_range
        )

    def allocate(self, nbytes: int, fill: int, memory_kind: MemoryKind):
        if memory_kind is MemoryKind.DEVICE:
            return DeviceBuffer(nbytes, fill), None
        return HostBuffer(nbytes, fill), None

    def make_held_key(self):
        key = f"held{secrets.token_hex(4)}/v1/H"  # the pool outlives each check
        writer = open_sdk(client_config(self.master))  # another client, still writing the key
        assert writer.batch_put_session_start([key], [CHECK_SLOT]) == [0]

        def release() -> None:
            try:
                assert writer.batch_put_session_revoke([key]) == [0]
            finally:
                writer.close()

        return self.make_driver(), key, release

    def make_blocked_get(self):
        key = f"blocked{secrets.token_hex(4)}/v1/S"
        sdk = GetsOfKeyWait(open_sdk(client_config(self.master)), key)
        return (
            MooncakeDriver(sdk, allocation_range=allocation_range),
            key,
            sdk.entered,
            sdk.release.set,
        )


class TestRealNativeConformance(TestRealMooncakeConformance):
    """The same checks on the native driver, which must refuse to start rather than fall back.
    Its store calls bypass the Python client a test could hold, so a get blocked inside the store
    is staged only over the fake SDK (``test_mooncake_driver.py``)."""

    handles_before_completion = True

    def make_driver(self) -> NativeMooncakeDriver:
        config = replace(client_config(self.master), driver="native")
        store = open_sdk(config)
        return make_store_driver(store, config, BlobConfig(), allocation_range=allocation_range)

    def make_blocked_get(self):
        return None


@pytest.mark.parametrize("driver", DRIVERS)
def test_real_four_workers_round_trip_2000_rows_without_crossed_bytes(
    clients, request, driver
) -> None:
    names = [f"{request.node.name}-{i}" for i in range(2000)]
    writer = clients.endpoint(names, driver=driver, workers=4, rows_per_batch=64)
    reader = clients.endpoint(names, driver=driver, workers=4, rows_per_batch=64, fill=0x2D)
    rows = random_rows(writer, seed=6)
    outcome = finish(writer, writer.backend.publish, *names)
    assert outcome.fault is None and all(masks(outcome)[0])
    outcome = finish(reader, reader.backend.fetch, *names)
    assert outcome.fault is None and all(masks(outcome)[0])
    crossed = [name for name in names if reader.read(name) != rows[name]]
    assert crossed == []
    if driver == "python":  # the backend's threads; a native driver runs its calls itself
        assert writer.driver.peak_in_flight >= 2 and reader.driver.peak_in_flight >= 2
    writer.close()
    reader.close()


class FirstScatterWaits:
    """An SDK client whose first put scatter waits, after its session started, for ``release``."""

    def __init__(self, store) -> None:
        self.store = store
        self.entered = threading.Event()
        self.release = threading.Event()
        self._first = threading.Lock()

    def __getattr__(self, name: str):
        return getattr(self.store, name)

    def batch_put_from_multi_buffer_ranges(self, *args):
        if self._first.acquire(blocking=False):
            self.entered.set()
            self.release.wait(WAIT_S)
        return self.store.batch_put_from_multi_buffer_ranges(*args)


def test_real_concurrent_publishes_of_the_same_keys_from_one_client(clients, request) -> None:
    # The first publish holds every key while its scatter waits, so the driver skips the second
    # publish's rows as another writer's, without a store call; then the first publish delivers
    # them all.
    names = [f"{request.node.name}-{i}" for i in range(64)]
    sdk = FirstScatterWaits(clients.open_client())
    # Three workers: publishes leave one to fetches, so the two publishes run side by side.
    writer = clients.add(Endpoint(MooncakeDriver(sdk), names, slot_bytes=SLOT_BYTES, workers=3))
    rows = random_rows(writer, seed=8)
    first = writer.backend.publish(writer.view(*names), None)
    try:
        assert sdk.entered.wait(WAIT_S)
        second = writer.backend.publish(writer.view(*names), None)
        skipped = writer.wait_outcome(second)
        assert skipped.fault is None and masks(skipped) == [[False] * 64]
    finally:
        sdk.release.set()
    assert masks(writer.wait_outcome(first)) == [[True] * 64]
    for attempt in (first, second):
        writer.wait_quiesced(attempt)
    stats = writer.driver.inner.stats()
    assert stats["rows_not_ready_another_writer"] == 64
    assert stats["rows_not_ready_own_session_open"] == 0
    assert writer.driver.peak_in_flight == 2
    reader = clients.endpoint(names, fill=0x2D)
    assert all(masks(finish(reader, reader.backend.fetch, *names))[0])
    assert all(reader.read(name) == rows[name] for name in names)
    writer.close()
    reader.close()

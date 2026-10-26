# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The blob backend over a real Mooncake pool: a master, and a separate process holding objects;
and a segment a client mounts in the pool, then unmounts gracefully.

The tested clients give the pool no segment (``global_segment_size`` 0), so every object sits in
the other process's segment and every transfer goes through the transfer engine; their
``local_buffer_size`` is 0, so a call that needed the SDK's copying path would fail. Runs only with
``TRTLLM_MOONCAKE_REAL=1``, on a host with ``mooncake_master`` and ``mooncake.store``. The
transport is ``TRTLLM_MOONCAKE_PROTOCOL`` (default rdma) over ``TRTLLM_MOONCAKE_DEVICES`` (default:
the host's fastest active InfiniBand devices). The shared ``driver_conformance`` checks run here
too, each over a fresh client and under a key prefix of its own, with a driver that registers
device memory: over RDMA, rows of several spans move into and out of GPU memory in place. They and
the backend-level tests run on both drivers, the Python one and the native one whose store worker
calls the same client from C++ threads, and must give the same results; a full pool, whose answer
is the SDK's whichever driver asks, runs on the native one.
"""

from __future__ import annotations

import importlib.util
import json
import os
import secrets
import shutil
import time
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
    UNMOUNT_GRACE_S,
    MooncakeConfig,
    StoreSegment,
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


@pytest.fixture(scope="module")
def pool(tmp_path_factory):
    from test_common.mooncake_utils import running_pool

    _require_mooncake()
    run_dir = str(tmp_path_factory.mktemp("pool"))
    with running_pool(run_dir, POOL_BYTES, protocol=PROTOCOL, device_name=DEVICES) as master:
        yield master.address


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
    """A client with no segment of its own and no buffer for the SDK's copying calls (the
    defaults)."""
    return MooncakeConfig(master, protocol=PROTOCOL, device_name=DEVICES)


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


def test_real_auto_runs_the_native_driver(clients) -> None:
    config = client_config(clients.master)
    assert config.driver == "auto"
    driver = make_store_driver(clients.open_client(), config, BlobConfig())
    assert isinstance(driver, NativeMooncakeDriver)


def test_real_store_full_is_declined(tmp_path, request) -> None:
    from test_common.mooncake_utils import running_pool

    _require_mooncake()
    # A pool of its own that one publish overfills: a 16 MiB segment (the smallest a TCP client
    # may give, one allocator slab) holds at most 256 of the 512 rows. A PutStart that finds no room
    # answers at once. The master evicts once the pool passes its high watermark, so how many rows
    # land varies, but some are always declined and none fault. The answer is the SDK's, whichever
    # driver asks; each driver's reading of it is pinned over the fake SDK.
    with running_pool(str(tmp_path), 16 << 20, protocol=PROTOCOL, device_name=DEVICES) as master:
        clients = Clients(master.address)
        names = [f"{request.node.name}-{i}" for i in range(512)]
        writer = clients.endpoint(names, driver="native", workers=1)
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


def test_real_store_segment_being_unmounted_serves_reads_for_the_grace_then_misses_cleanly(
    tmp_path, request
) -> None:
    """A client mounts the pool's only memory and puts 8 objects there. Once it begins unmounting,
    the objects read whole for the grace and a new put is declined; after it, gets miss and the end
    of every get is proven; once Mooncake freed the memory, the master lists no segment."""
    from test_common.mooncake_utils import master_segments, running_master_on_free_ports

    _require_mooncake()
    run_dir = str(tmp_path)
    names = [f"{request.node.name}-{i}" for i in range(8)]
    late = f"{request.node.name}-late"
    with running_master_on_free_ports(run_dir, protocol=PROTOCOL, eviction_ratio=0.0) as master:
        with open(os.path.join(run_dir, "pool.json")) as file:
            metrics_port = json.load(file)["metrics_port"]
        metrics_url = f"http://{master.address.rsplit(':', 1)[0]}:{metrics_port}"
        segment = StoreSegment.mount(open_sdk(client_config(master.address)), 256 << 20, PROTOCOL)
        clients = Clients(master.address)
        writer = clients.endpoint(names + [late])
        rows = random_rows(writer, seed=12)
        # Every endpoint is open before the grace starts: a client takes about a second to open.
        readers = [clients.endpoint(names, fill=0x2D) for _ in range(2)]
        assert masks(finish(writer, writer.backend.publish, *names)) == [[True] * 8]
        (listed,) = master_segments(metrics_url)
        assert listed["allocator_used_bytes"] > 0

        segment.begin_unmount()
        grace_ends = time.monotonic() + UNMOUNT_GRACE_S
        during = finish(readers[0], readers[0].backend.fetch, *names)
        declined = finish(writer, writer.backend.publish, late)
        assert time.monotonic() < grace_ends, "the reads took longer than the grace"
        assert during.fault is None and masks(during) == [[True] * 8]
        assert all(readers[0].read(name) == rows[name] for name in names)
        assert declined.fault is None and masks(declined) == [[False]]
        assert writer.driver.inner.stats()["rows_declined_pool_full"] == 1

        time.sleep(grace_ends + 1 - time.monotonic())  # past the grace: the master dropped them
        after = finish(readers[1], readers[1].backend.fetch, *names)
        assert after.fault is None and masks(after) == [[False] * 8]
        assert readers[1].driver.inner.stats()["ops_unprovable"] == 0
        segment.wait_until_freed()
        # The master drops the segment's record shortly after Mooncake freed the memory.
        deadline = time.monotonic() + 10
        while master_segments(metrics_url) and time.monotonic() < deadline:
            time.sleep(0.5)
        assert master_segments(metrics_url) == []
        segment.store.close()  # it holds no segment any more
        clients.close()  # while the master still answers

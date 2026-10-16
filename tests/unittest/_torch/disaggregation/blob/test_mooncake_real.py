# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The blob backend over a real Mooncake pool: a master, and a separate process holding objects.

The tested clients give the pool no segment (``global_segment_size`` 0), so every object sits in
the other process's segment and every transfer goes through the transfer engine; their
``local_buffer_size`` is 0, so a call that needed the SDK's copying path would fail. Runs only with
``TRTLLM_MOONCAKE_REAL=1``, on a host with ``mooncake_master`` and ``mooncake.store``. The
transport is ``TRTLLM_MOONCAKE_PROTOCOL`` (default rdma) over ``TRTLLM_MOONCAKE_DEVICES`` (default:
the host's fastest active InfiniBand devices). The shared ``driver_conformance`` checks run here
too, each over a fresh client and under a key prefix of its own.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import secrets
import shutil
import threading

import numpy as np
import pytest
from cpu_fixtures import WAIT_S, Endpoint, masks
from fake_mooncake import GetsOfKeyWait

from tensorrt_llm._torch.disaggregation.base.backend import Outcome
from tensorrt_llm._torch.disaggregation.blob.mooncake import MooncakeDriver
from tensorrt_llm._torch.disaggregation.blob.mooncake_setup import MooncakeConfig, open_sdk
from tensorrt_llm._torch.disaggregation.testing.driver_conformance import SLOT as CHECK_SLOT
from tensorrt_llm._torch.disaggregation.testing.driver_conformance import DriverConformance

pytestmark = pytest.mark.skipif(
    os.environ.get("TRTLLM_MOONCAKE_REAL") != "1",
    reason="needs a real Mooncake pool; set TRTLLM_MOONCAKE_REAL=1",
)

PROTOCOL = os.environ.get("TRTLLM_MOONCAKE_PROTOCOL", "rdma")
DEVICES = os.environ.get("TRTLLM_MOONCAKE_DEVICES", "")
POOL_BYTES = 1 << 30
SLOT_BYTES = 64 << 10


def _require_mooncake() -> None:
    if shutil.which("mooncake_master") is None:
        pytest.skip("mooncake_master is not on PATH")
    try:
        found = importlib.util.find_spec("mooncake.store") is not None
    except ImportError:
        found = False
    if not found:
        pytest.skip("mooncake.store does not import")


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

    def endpoint(self, names, *, workers: int = 2, **options) -> Endpoint:
        options = {"slot_bytes": SLOT_BYTES, **options}
        driver = MooncakeDriver(self.open_client())
        return self.add(Endpoint(driver, names, workers=workers, **options))

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


def test_real_store_full_is_declined(tmp_path, request) -> None:
    _require_mooncake()
    # A pool of its own that one publish overfills: a 16 MiB segment (the smallest a TCP client
    # may give, one allocator slab) holds at most 256 of the 512 rows. A PutStart that finds no room
    # answers at once. The master evicts once the pool passes its high watermark, so how many rows
    # land varies, but some are always declined and none fault.
    stack, master = _running_pool(str(tmp_path), segment_size=16 << 20)
    clients = Clients(master)
    with stack:
        names = [f"{request.node.name}-{i}" for i in range(512)]
        writer = clients.endpoint(names, workers=1)
        random_rows(writer, seed=4)
        outcome = finish(writer, writer.backend.publish, *names)
        delivered = int(outcome.delivered_masks[0].sum())
        declined = writer.driver.inner.stats()["rows_declined_pool_full"]
        assert outcome.fault is None
        assert delivered > 0 and declined > 0 and delivered + declined == len(names)
        writer.close()
        clients.close()  # while the master still answers


def test_real_get_lands_a_shorter_object_and_refuses_a_longer_one(clients, request) -> None:
    # batch_get_into reads an object shorter than the buffer into the buffer's head and answers
    # its size; it refuses a longer one before any transfer (INVALID_PARAMS).
    short, long = f"{request.node.name}-short", f"{request.node.name}-long"
    half = clients.endpoint([short], slot_bytes=SLOT_BYTES // 2)
    double = clients.endpoint([long], slot_bytes=2 * SLOT_BYTES)
    head = random_rows(half, seed=9)[short]
    random_rows(double, seed=10)
    assert masks(finish(half, half.backend.publish, short)) == [[True]]
    assert masks(finish(double, double.backend.publish, long)) == [[True]]
    reader = clients.endpoint([short, long], fill=0x2D, rows_per_batch=1)
    outcome = finish(reader, reader.backend.fetch, short)  # access ends: finish waits for it
    assert masks(outcome) == [[False]]
    assert f"object is {SLOT_BYTES // 2} bytes, expected {SLOT_BYTES}" in outcome.fault
    assert reader.read(short) == head + b"-" * (SLOT_BYTES // 2)
    outcome = finish(reader, reader.backend.fetch, long)
    assert masks(outcome) == [[False]] and "INVALID_PARAMS(-600)" in outcome.fault
    assert reader.read(long) == b"-" * SLOT_BYTES
    for each in (half, double, reader):
        each.close()


class TestRealMooncakeConformance(DriverConformance):
    """The shared driver checks over the real pool, one fresh client per check, with the scenarios
    a real master can stage: a key another client is writing, and a get held inside the SDK."""

    timeout = 30.0

    @pytest.fixture(autouse=True)
    def _master(self, pool):
        self.master = pool

    def make_driver(self) -> MooncakeDriver:
        return MooncakeDriver(open_sdk(client_config(self.master)))

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
        return MooncakeDriver(sdk), key, sdk.entered, sdk.release.set


def test_real_four_workers_round_trip_2000_rows_without_crossed_bytes(clients, request) -> None:
    names = [f"{request.node.name}-{i}" for i in range(2000)]
    writer = clients.endpoint(names, workers=4, rows_per_batch=64)
    reader = clients.endpoint(names, workers=4, rows_per_batch=64, fill=0x2D)
    rows = random_rows(writer, seed=6)
    outcome = finish(writer, writer.backend.publish, *names)
    assert outcome.fault is None and all(masks(outcome)[0])
    outcome = finish(reader, reader.backend.fetch, *names)
    assert outcome.fault is None and all(masks(outcome)[0])
    crossed = [name for name in names if reader.read(name) != rows[name]]
    assert crossed == []
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
    # The first publish holds every key's put session while its scatter waits, so the second
    # publish's session start finds them taken by this client (INVALID_PARAMS) and skips its
    # rows; then the first publish delivers them all.
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
    assert writer.driver.inner.stats()["rows_not_ready_own_session_open"] == 64
    assert writer.driver.peak_in_flight == 2
    reader = clients.endpoint(names, fill=0x2D)
    assert all(masks(finish(reader, reader.backend.fetch, *names))[0])
    assert all(reader.read(name) == rows[name] for name in names)
    writer.close()
    reader.close()

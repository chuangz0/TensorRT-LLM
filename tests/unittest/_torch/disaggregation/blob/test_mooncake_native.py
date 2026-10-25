# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The native Mooncake driver's own rules, over the fake SDK through the store worker's test client:
gets go before puts and puts leave a thread for gets, operations are cut into store calls, a cancel
or a drain concludes the calls not started CANCELLED, a fault stops the rest of its operation and of
its attempt, rows as arrays are admitted by column, and an operation naming one object twice is
refused. What both drivers share is pinned in ``test_mooncake_driver.py``."""

from __future__ import annotations

import numpy as np
import pytest
from cpu_fixtures import WAIT_S, CpuView, Endpoint, masks, store_key, wait_until
from fake_mooncake import FakeMooncakeStore, FakePool, GetsOfKeyWait
from test_mooncake_driver import (
    GET,
    PUT_RANGES,
    PUT_START,
    SLOT,
    Allocations,
    key,
    native_driver,
    settled,
)

from tensorrt_llm._torch.disaggregation.base.memory import MemoryKind
from tensorrt_llm._torch.disaggregation.blob import mooncake
from tensorrt_llm._torch.disaggregation.blob.backend import SHUTDOWN_FAULT
from tensorrt_llm._torch.disaggregation.blob.store import (
    DriverRejected,
    FinishedOp,
    RegionLayout,
    RowArrays,
    RowIO,
)
from tensorrt_llm._torch.disaggregation.testing.host_memory import HostBuffer

pytestmark = pytest.mark.cpu_only


def statuses(driver, op) -> list[str]:
    """Once concluded."""
    return [result.status.value for result in settled(driver, op)]


def statuses_now(driver, op) -> list[str]:
    return [result.status.value for result in driver.poll(op)]


def gets_of(sdk: GetsOfKeyWait) -> list[str]:
    """Every key a get reached the store with; a get waiting in ``sdk`` has not yet."""
    return [k for call in sdk.store.calls_to(GET) for k in call.args[0]]


def finished(driver, op) -> FinishedOp:
    """The backend's view of a queued operation: its finish, once the driver hands it out."""
    taken: list[FinishedOp] = []

    def arrived() -> bool:
        taken.extend(driver.take_finished())
        return any(done.op is op for done in taken)

    wait_until(arrived, "the operation to finish")
    return next(done for done in taken if done.op is op)


def test_gets_go_first_and_puts_keep_a_thread_for_gets() -> None:
    fake = FakeMooncakeStore()
    # Two threads, so at most one put at a time; calls of at most four rows.
    driver = native_driver(fake, threads=2, put_threads=1, rows_per_call=4)
    buffer = HostBuffer(SLOT * 12, fill=0x2D)
    driver.register(buffer.address, buffer.nbytes, RegionLayout(SLOT, 12))
    fetched = [f"G{i}" for i in range(10)]
    for name in fetched:
        fake.pool.objects[key(name)] = name.encode().ljust(SLOT, b".")
    gate = fake.block(PUT_RANGES)
    first = driver.put([RowIO(key("P1"), (buffer.span(0, SLOT),))])
    assert gate.entered.wait(WAIT_S)  # the first put holds one thread inside its scatter
    second = driver.put([RowIO(key("P2"), (buffer.span(SLOT, SLOT),))])
    get = driver.get(
        [RowIO(key(name), (buffer.span((2 + i) * SLOT, SLOT),)) for i, name in enumerate(fetched)]
    )
    try:
        assert statuses(driver, get) == ["ready"] * 10  # while both puts wait
        assert [len(call.args[0]) for call in fake.calls_to(GET)] == [4, 4, 2]
        assert [call.args[0] for call in fake.calls_to(PUT_START)] == [[key("P1")]]
        assert statuses_now(driver, second) == ["pending"]
    finally:
        gate.release.set()
    assert statuses(driver, first) == statuses(driver, second) == ["ready"]
    assert [call.args[0] for call in fake.calls_to(PUT_START)] == [[key("P1")], [key("P2")]]


def test_a_cancelled_fetch_never_reaches_the_store_and_leaves_its_slots_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(mooncake.logger, "warning", warnings.append)
    pool = FakePool()
    sdk = GetsOfKeyWait(FakeMooncakeStore(pool), store_key("A"))
    reader = Endpoint(native_driver(sdk, threads=1, put_threads=1), ["A", "B"], fill=0x2D)
    pool.objects[store_key("A")] = b"a" * SLOT
    pool.objects[store_key("B")] = b"b" * SLOT
    first = reader.backend.fetch(reader.view("A"), None)
    assert sdk.entered.wait(WAIT_S)  # the worker's only thread is inside the store
    second = reader.backend.fetch(reader.view("B"), None)
    second.cancel()
    try:
        outcome = reader.wait_outcome(second)
        assert outcome.fault is None and masks(outcome) == [[False]]
        reader.wait_quiesced(second)  # never started: its access ended with the cancel
        assert store_key("B") not in gets_of(sdk)
        assert reader.read("B") == b"-" * SLOT
        assert warnings == []  # a cancelled row neither failed nor was skipped
    finally:
        sdk.release.set()
    assert masks(reader.wait_outcome(first)) == [[True]] and reader.read("A") == b"a" * SLOT
    reader.wait_quiesced(first)
    reader.close()


def test_a_faulted_call_drops_the_calls_of_its_operation_not_started_yet() -> None:
    fake = FakeMooncakeStore()
    driver = native_driver(fake, threads=1, put_threads=1, rows_per_call=1)
    buffer = HostBuffer(3 * SLOT, fill=0x2D)
    driver.register(buffer.address, buffer.nbytes, RegionLayout(SLOT, 3))
    for name in "ABC":
        fake.pool.objects[key(name)] = name.encode() * SLOT
    fake.script(GET, [-900])  # RPC_FAIL for A: a fault
    rows = [RowIO(key(name), (buffer.span(i * SLOT, SLOT),)) for i, name in enumerate("ABC")]
    assert statuses(driver, driver.get(rows)) == ["fault", "cancelled", "cancelled"]
    assert [call.args[0] for call in fake.calls_to(GET)] == [[key("A")]]


def test_an_operation_that_names_one_object_twice_is_refused_before_the_store() -> None:
    """The SDK names get targets by key: a get of one key into two slots would answer READY for
    both and fill one."""
    fake = FakeMooncakeStore()
    driver = native_driver(fake)
    buffer = HostBuffer(2 * SLOT)
    driver.register(buffer.address, buffer.nbytes, RegionLayout(SLOT, 2))
    names = np.array([[0xA5], [0xA5]], dtype=np.uint8)
    starts = np.array([[buffer.address], [buffer.address + SLOT]], dtype=np.int64)
    with pytest.raises(DriverRejected, match="row 1 repeats an earlier row's key"):
        driver.get_rows(RowArrays("t/v1/", names, starts, (SLOT,)))
    assert [call for call in fake.calls if call.method.startswith("batch_")] == []
    assert driver.drain(0.0) is True  # the refused operation holds nothing


def test_rows_as_arrays_are_admitted_by_column_and_cut_where_a_registration_piece_begins() -> None:
    """The backend's path: one bounding span per column admits the rows, though it crosses
    registration pieces (CUDA allocations of device memory); the worker cuts each span there."""
    fake = FakeMooncakeStore()
    region = HostBuffer(3 * SLOT, fill=0x2D)  # stands in for device memory of three allocations
    driver = native_driver(fake, allocation_range=Allocations(region))
    driver.register(region.address, region.nbytes, RegionLayout(SLOT, 3, "kv", MemoryKind.DEVICE))
    region.write(b"0123456789abcdef", SLOT // 2)  # bytes 4 to 20: across both inner boundaries
    name = np.array([[0xA5]], dtype=np.uint8)
    starts = np.array([[region.address + SLOT // 2, region.address]], dtype=np.int64)
    rows = RowArrays("t/v1/", name, starts, (2 * SLOT, SLOT // 2))
    put = finished(driver, driver.put_rows(rows))
    assert put.ready.tolist() == [True] and put.faults == {} and put.access_ended
    (scatter,) = fake.calls_to(PUT_RANGES)
    keys, buffers, sizes, offsets = scatter.args
    assert keys == ["t/v1/a5"] and sizes == [[4, 8, 4, 4]] and offsets == [[0, 4, 12, 16]]
    assert buffers[0][:3] == [region.address + 4, region.address + 8, region.address + 16]
    assert fake.pool.objects["t/v1/a5"] == b"0123456789abcdef----"


def test_a_refused_layer_group_stops_its_fetch_and_cancels_the_groups_queued_before_it() -> None:
    pool = FakePool()
    sdk = GetsOfKeyWait(FakeMooncakeStore(pool), store_key("Z"))
    reader = Endpoint(native_driver(sdk, threads=1, put_threads=1), ["Z", "A", "B", "C"], fill=0x2D)
    for name in ("Z", "A", "B", "C"):
        pool.objects[store_key(name)] = name.encode() * SLOT
    blocker = reader.backend.fetch(reader.view("Z"), None)
    assert sdk.entered.wait(WAIT_S)  # the worker's only thread is inside the store
    get_rows, handed_over = reader.driver.get_rows, []

    def refuse_group_b(rows: RowArrays):
        handed_over.append(rows)
        if len(handed_over) == 2:
            raise DriverRejected("refused by the test")
        return get_rows(rows)

    reader.driver.get_rows = refuse_group_b
    groups = [reader.rows([name], layer_group_index=i) for i, name in enumerate("ABC")]
    fetch = reader.backend.fetch(CpuView(groups), None)
    try:
        outcome = reader.wait_outcome(fetch)
        assert outcome.fault == "group 1 row 0: refused: refused by the test"
        assert masks(outcome) == [[False], [False], [False]]
        reader.wait_quiesced(fetch)
        assert len(handed_over) == 2  # A and B; C never handed over
        assert store_key("A") not in gets_of(sdk)  # cancelled while queued
        assert reader.read("A") == reader.read("C") == b"-" * SLOT
    finally:
        sdk.release.set()
    assert masks(reader.wait_outcome(blocker)) == [[True]]
    reader.wait_quiesced(blocker)
    reader.close()


def test_a_shutdown_cancels_queued_calls_and_finishes_once_the_running_one_returns() -> None:
    pool = FakePool()
    sdk = GetsOfKeyWait(FakeMooncakeStore(pool), store_key("A"))
    reader = Endpoint(native_driver(sdk, threads=1, put_threads=1), ["A", "B"], fill=0x2D)
    pool.objects[store_key("A")] = b"a" * SLOT
    pool.objects[store_key("B")] = b"b" * SLOT
    running = reader.backend.fetch(reader.view("A"), None)
    assert sdk.entered.wait(WAIT_S)
    queued = reader.backend.fetch(reader.view("B"), None)
    try:
        labels = reader.backend.shutdown(0.0)  # forces both outcomes; the queued call is dropped
        assert running.label in labels
        assert reader.wait_outcome(queued).fault == SHUTDOWN_FAULT
        reader.wait_quiesced(queued)  # its cancelled call collected by the progress thread
        assert store_key("B") not in gets_of(sdk) and reader.read("B") == b"-" * SLOT
        assert not running.quiesced()
    finally:
        sdk.release.set()
    reader.wait_quiesced(running)
    assert reader.backend.shutdown(WAIT_S) == frozenset()  # drained and unregistered
    assert len(reader.driver.calls_to("drain")) == 1
    assert reader.driver.calls_to("unregister")

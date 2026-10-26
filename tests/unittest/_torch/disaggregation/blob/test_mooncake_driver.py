# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Both Mooncake drivers over a fake SDK that moves real bytes: the Python driver, and the native
one whose store worker reaches the fake through its test client.

Pins the mapping of SDK 0.3.13 answers to row results, the put session bookkeeping (every session
the driver starts is ended, revoked or left open; only rows written whole are ended), the access-end
rules (a put scatter that returns ends access, whatever it answers; a get ends it unless a row
answers TRANSFER_FAIL; a call that raises or miscounts does not, and its access is unprovable),
registration (device memory one piece per CUDA allocation), drain and close, and a backend round
trip over the fake SDK. Each scenario runs on both drivers and must give the same results. The
shared ``driver_conformance`` checks run over the fake SDK too.
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from dataclasses import replace
from typing import NamedTuple

import pytest
from cpu_fixtures import PROMPT_S, WAIT_S, Endpoint, store_key, wait_until
from fake_mooncake import FakeMooncakeStore, FakePool, GetsOfKeyWait

from tensorrt_llm._torch.disaggregation.base.backend import Attempt, Backend, Capabilities
from tensorrt_llm._torch.disaggregation.base.memory import MemoryKind
from tensorrt_llm._torch.disaggregation.blob import mooncake as mooncake_module
from tensorrt_llm._torch.disaggregation.blob.mooncake import MooncakeDriver
from tensorrt_llm._torch.disaggregation.blob.mooncake_native import NativeMooncakeDriver
from tensorrt_llm._torch.disaggregation.blob.store import (
    DriverCapabilities,
    DriverRejected,
    QueuedStoreDriver,
    RegionLayout,
    RegistrationError,
    RowIO,
    RowStatus,
    Span,
    StoreDriver,
    StoreError,
    concluded,
)
from tensorrt_llm._torch.disaggregation.testing.driver_conformance import DriverConformance
from tensorrt_llm._torch.disaggregation.testing.host_memory import HostBuffer

pytestmark = pytest.mark.cpu_only

SLOT = 8
PUT_START = "batch_put_session_start"
PUT_RANGES = "batch_put_from_multi_buffer_ranges"
PUT_END = "batch_put_session_end"
PUT_REVOKE = "batch_put_session_revoke"
GET = "batch_get_into_multi_buffers"
IS_EXIST = "batch_is_exist"
DATA_CALL = {"put": PUT_RANGES, "get": GET}
CALL_FAILED = "existence check: CALL_FAILED(-100000)"


def key(name: str) -> str:
    return f"t/v1/{name}"


def python_driver(sdk, allocation_range=None) -> MooncakeDriver:
    return MooncakeDriver(sdk, allocation_range=allocation_range)


def native_driver(sdk, allocation_range=None, **overrides) -> NativeMooncakeDriver:
    """The native driver over a fake SDK: the worker's store calls go to ``sdk``'s methods."""
    from tensorrt_llm.mooncake_store_worker import StoreWorker

    options = {
        "threads": 4,
        "put_threads": 3,
        "gets_per_put": 4,
        "rows_per_call": 256,
        "bytes_per_call": 1 << 30,
        **overrides,
    }
    worker = StoreWorker.over_sdk_object(sdk, **options)
    return NativeMooncakeDriver(sdk, worker, allocation_range=allocation_range)


DRIVERS = {"python": python_driver, "native": native_driver}


@pytest.fixture(params=DRIVERS)
def make_driver(request):
    """Builds a driver over a fake SDK: each scenario taking it runs on both drivers."""
    return DRIVERS[request.param]


def settled(driver: StoreDriver, op) -> tuple:
    """The results of ``op`` once every row concluded: at once for the Python driver, whose calls
    return only after the store answered; a native operation's results and access end land
    together."""
    wait_until(lambda: concluded(driver.poll(op)), "the operation to conclude")
    return driver.poll(op)


class Bench:
    """A driver (``make_driver``) over a fake SDK, with ``slots`` registered host slots."""

    def __init__(self, make_driver, slots: int = 4, pool: FakePool | None = None) -> None:
        self.fake = FakeMooncakeStore(pool)
        self.driver = make_driver(self.fake)
        self.buffer = HostBuffer(SLOT * slots, fill=0x2D)
        layout = RegionLayout(SLOT, slots)
        self.handle = self.driver.register(self.buffer.address, self.buffer.nbytes, layout)

    def rows(self, *names: str) -> list[RowIO]:
        return [RowIO(key(n), (self.buffer.span(i * SLOT, SLOT),)) for i, n in enumerate(names)]

    def store(self, **objects: bytes) -> None:
        for name, data in objects.items():
            self.fake.pool.objects[key(name)] = data

    def write(self, slot: int, data: bytes) -> None:
        self.buffer.write(data, slot * SLOT)

    def read(self, slot: int) -> bytes:
        return self.buffer.read(slot * SLOT, SLOT)

    def run(self, direction: str, *names: str):
        """The results and access end of a put or get of ``names``, once concluded; ``last`` is
        its handle."""
        rows = self.rows(*names)
        self.last = self.driver.put(rows) if direction == "put" else self.driver.get(rows)
        return settled(self.driver, self.last), self.driver.access_ended(self.last)

    def fill_sources(self, direction: str, *names: str) -> None:
        """Gives a put of ``names`` its slots' bytes, or a get its objects in the pool."""
        for slot, name in enumerate(names):
            data = name.encode() * SLOT
            self.write(slot, data) if direction == "put" else self.store(**{name: data})

    def keys_of(self, method: str) -> list[str]:
        return [k for call in self.fake.calls_to(method) for k in call.args[0]]


def summary(results) -> list[tuple[str, str | None]]:
    return [(result.status.value, result.fault) for result in results]


def test_the_drivers_follow_the_driver_protocol() -> None:
    python = Bench(python_driver).driver
    native = Bench(native_driver).driver
    assert isinstance(python, StoreDriver) and not isinstance(python, QueuedStoreDriver)
    assert isinstance(native, QueuedStoreDriver)
    assert python.capabilities == DriverCapabilities(
        get=True,
        put=True,
        cancel=False,
        probe=True,
        max_rows_per_op=None,
        hints=False,
        absent_is_miss=True,
        device_memory=False,
        multi_span=True,
    )
    # The native driver cancels the calls it has not started and runs its own queue.
    assert native.capabilities == replace(python.capabilities, cancel=True, queued=True)
    for make_driver in DRIVERS.values():
        device = make_driver(FakeMooncakeStore(), allocation_range=lambda address: None)
        assert device.capabilities.device_memory


class TestMooncakeDriverConformance(DriverConformance):
    """The shared driver checks over the fake SDK, with every scenario it can stage. A probe
    cannot be blocked past its budget: the SDK's existence check takes no deadline."""

    driver_of = staticmethod(python_driver)

    def make_driver(self) -> StoreDriver:
        return self.driver_of(FakeMooncakeStore())

    def make_blocked_get(self):
        sdk = GetsOfKeyWait(FakeMooncakeStore(), key("blocked"))
        return self.driver_of(sdk), key("blocked"), sdk.entered, sdk.release.set

    def make_held_key(self):
        pool = FakePool()
        writer = FakeMooncakeStore(pool)  # another client, still writing the key
        assert writer.batch_put_session_start([key("held")], [SLOT]) == [0]

        def release() -> None:
            writer.batch_put_session_revoke([key("held")])

        return self.driver_of(FakeMooncakeStore(pool)), key("held"), release

    def make_full_store(self):
        fake = FakeMooncakeStore()
        # Room for one more object as the check uses it: its one two-row put finds room for the
        # first only.
        fake.script(PUT_START, [None, -200])
        return self.driver_of(fake)

    def make_crash_after_record(self):
        fake = FakeMooncakeStore()
        fake.script(PUT_START, RuntimeError("the SDK crashed"))
        return self.driver_of(fake)

    def make_failing_presence(self):
        fake = FakeMooncakeStore()
        fake.script(IS_EXIST, [-900])  # the existence check fails with RPC_FAIL
        return self.driver_of(fake), key("failing")

    def make_failing_get(self):
        fake = FakeMooncakeStore()
        fake.script(GET, [-900])  # the next read fails with RPC_FAIL; the check reads the key once
        return self.driver_of(fake), key("failing")

    def make_unprovable_get(self):
        fake = FakeMooncakeStore()
        fake.script(GET, [-800])  # a transfer future gave up while its transfer may still run
        return self.driver_of(fake), lambda op: None  # the get concluded on its own


class TestNativeMooncakeDriverConformance(TestMooncakeDriverConformance):
    """The same checks on the native driver. Its put and get return the handle before the store
    answers, so the checks of a get blocked inside the store and cancelled there run too."""

    driver_of = staticmethod(native_driver)
    handles_before_completion = True


# Answers to row results.


def skipped_counts(bench: Bench) -> dict[str, int]:
    return {name: n for name, n in bench.driver.stats().items() if name.startswith("rows_") and n}


GET_ANSWERS = {
    # name: (the answer to the get (None: real), status, fault, the stat counting a skipped row)
    "object_size": (None, "ready", None, None),  # unscripted
    "short_object": (5, "fault", "object is 5 bytes, expected 8", None),
    "OBJECT_NOT_FOUND": (-704, "miss", None, None),
    "REPLICA_IS_NOT_READY": (-703, "not_ready", None, "rows_not_ready_being_written"),
    # The one code whose meaning depends on the call: a put's session start reads it as the
    # client's own session still open.
    "INVALID_PARAMS": (-600, "fault", f"{GET}: INVALID_PARAMS(-600)", None),
    "RPC_FAIL": (-900, "fault", f"{GET}: RPC_FAIL(-900)", None),
    "TRANSFER_FAIL": (-800, "fault", f"{GET}: TRANSFER_FAIL(-800)", None),
    "unknown_code": (-9999, "fault", f"{GET}: code(-9999)", None),
}


@pytest.mark.parametrize("case", GET_ANSWERS)
def test_get_answers_map_to_row_statuses(make_driver, case: str) -> None:
    answer, status, fault, counted = GET_ANSWERS[case]
    bench = Bench(make_driver)
    bench.store(A=b"aaaaaaaa")
    bench.fake.script(GET, [answer])
    results, access_ended = bench.run("get", "A")
    assert summary(results) == [(status, fault)]
    assert skipped_counts(bench) == ({counted: 1} if counted else {})
    assert (bench.read(0) == b"aaaaaaaa") == (status == "ready")
    assert access_ended is (answer != -800)
    assert bench.driver.access_unprovable(bench.last) is (answer == -800)


PUT_START_ANSWERS = {
    # name: (the session start's answer, status, fault, the stat counting a skipped row)
    "NO_AVAILABLE_HANDLE": (-200, "declined", None, "rows_declined_pool_full"),
    "INVALID_PARAMS": (-600, "not_ready", None, "rows_not_ready_own_session_open"),
    "RPC_FAIL": (-900, "fault", "put session start: RPC_FAIL(-900)", None),
    "unknown_code": (-9999, "fault", "put session start: code(-9999)", None),
}


@pytest.mark.parametrize("case", PUT_START_ANSWERS)
def test_put_session_start_answers_map_to_row_statuses(make_driver, case: str) -> None:
    answer, status, fault, counted = PUT_START_ANSWERS[case]
    bench = Bench(make_driver)
    bench.fake.script(PUT_START, [answer])
    results, access_ended = bench.run("put", "A")
    assert summary(results) == [(status, fault)]
    assert skipped_counts(bench) == ({counted: 1} if counted else {})
    assert access_ended
    (start,) = bench.fake.calls_to(PUT_START)
    assert len(start.args) == 2 and start.kwargs == {}  # never a replicate config
    for method in (PUT_RANGES, PUT_END, PUT_REVOKE, "batch_is_exist"):
        assert bench.fake.calls_to(method) == []  # a session it did not start is not its own


PUT_WRITE_ANSWERS = {
    # case: (the scatter's answer, the end's answer (None: real), status, fault, session revoked)
    "written": (None, None, "ready", None, False),
    "short_write": (5, None, "fault", "put ranges: moved 5 of 8 bytes", True),
    "session_lost": (-600, None, "fault", "put ranges: INVALID_PARAMS(-600)", True),
    "end_refused": (None, -704, "fault", "put session end: OBJECT_NOT_FOUND(-704)", True),
    "end_not_owner": (None, -601, "fault", "put session end: ILLEGAL_CLIENT(-601)", True),
}


@pytest.mark.parametrize("case", PUT_WRITE_ANSWERS)
def test_put_ranges_and_end_answers_map_to_row_statuses(make_driver, case: str) -> None:
    ranges, end, status, fault, revoked = PUT_WRITE_ANSWERS[case]
    bench = Bench(make_driver)
    bench.write(0, b"aaaaaaaa")
    bench.fake.script(PUT_RANGES, [ranges])
    bench.fake.script(PUT_END, [end])
    results, access_ended = bench.run("put", "A")
    assert summary(results) == [(status, fault)]
    assert access_ended
    assert (bench.fake.pool.objects.get(key("A")) == b"aaaaaaaa") == (status == "ready")
    assert bench.keys_of(PUT_REVOKE) == ([key("A")] if revoked else [])


EXISTENCE_ANSWERS = {
    # batch_is_exist's step: the results of the two rows that already existed
    "committed_and_not": ([1, 0], [("ready", None), ("not_ready", None)]),
    "check_failed": ([1, -900], [("ready", None), ("fault", "existence check: RPC_FAIL(-900)")]),
    # A metadata call touches no row memory: its raise fails only its rows.
    "check_raised": (RuntimeError("master gone"), [("fault", CALL_FAILED)] * 2),
}


@pytest.mark.parametrize("case", EXISTENCE_ANSWERS)
def test_only_rows_that_already_exist_are_checked_for_existence(make_driver, case: str) -> None:
    exist, expected = EXISTENCE_ANSWERS[case]
    bench = Bench(make_driver)
    # A starts, B and C exist (OBJECT_ALREADY_EXISTS), D finds the pool full.
    bench.fake.script(PUT_START, [None, -705, -705, -200])
    bench.fake.script("batch_is_exist", exist)
    results, access_ended = bench.run("put", "A", "B", "C", "D")
    assert summary(results) == [("ready", None), *expected, ("declined", None)]
    assert bench.keys_of("batch_is_exist") == [key("B"), key("C")]
    assert access_ended


# Session bookkeeping.


def reply_lost(answers):
    """The end committed every object and dropped the sessions, but the reply never came."""
    raise RuntimeError("reply lost")


class Scenario(NamedTuple):
    """One put of A, B and C."""

    steps: dict  # {method: one FakeMooncakeStore.script step per call, in call order}
    rows: str  # each row ready (R) or faulted (F)
    # Each row's session ended or revoked (-), or left open for drain (D) or to the master (M).
    sessions: str
    ends_access: bool


SESSION_SCENARIOS = {
    "all_written": Scenario({}, "RRR", "---", True),
    "transfer_fail": Scenario({PUT_RANGES: [[None, -800, None]]}, "RFR", "-M-", True),
    "short_write": Scenario({PUT_RANGES: [[None, 5, None]]}, "RFR", "---", True),
    "end_retried": Scenario({PUT_END: [[-900, None, None], [None]]}, "RRR", "---", True),
    "end_refused": Scenario({PUT_END: [[-704, None, None]]}, "FRR", "---", True),
    "end_fails_twice": Scenario({PUT_END: [[-901, None, None], [-900]]}, "FRR", "D--", True),
    "end_retry_refused": Scenario({PUT_END: [[-900, None, None], [-704]]}, "FRR", "D--", True),
    "end_and_revoke_fail": Scenario(
        {PUT_END: [[-704, None, None]], PUT_REVOKE: [[-900]]}, "FRR", "D--", True
    ),
    "end_raises": Scenario({PUT_END: [RuntimeError("rpc lost")]}, "FFF", "DDD", True),
    "end_committed_then_raised": Scenario({PUT_END: [reply_lost]}, "FFF", "DDD", True),
    "ranges_raise": Scenario({PUT_RANGES: [RuntimeError("bad pointer")]}, "FFF", "MMM", False),
    "ranges_miscount": Scenario({PUT_RANGES: [lambda answers: answers[:-1]]}, "FFF", "MMM", False),
}


@pytest.mark.parametrize("name", SESSION_SCENARIOS)
def test_every_started_put_session_is_ended_revoked_or_left_open(make_driver, name: str) -> None:
    scenario = SESSION_SCENARIOS[name]
    bench = Bench(make_driver)
    for slot, data in enumerate((b"aaaaaaaa", b"bbbbbbbb", b"cccccccc")):
        bench.write(slot, data)
    for method, method_steps in scenario.steps.items():
        for step in method_steps:
            bench.fake.script(method, step)
    results, access_ended = bench.run("put", "A", "B", "C")
    assert "".join(r.status.value[0].upper() for r in results) == scenario.rows
    assert access_ended is scenario.ends_access

    session_of = dict(zip(map(key, "ABC"), scenario.sessions))
    # A revoke answering INVALID_PARAMS (-600) found the session already gone: settled too.
    revoked = bench.fake.keys_answered(PUT_REVOKE, 0) + bench.fake.keys_answered(PUT_REVOKE, -600)
    settled = Counter(bench.fake.keys_answered(PUT_END, 0) + revoked)
    assert set(settled.values()) <= {1}, f"a session ended or revoked twice: {settled}"
    assert {k for k, session in session_of.items() if session == "-"} == set(settled)
    stats = bench.driver.stats()
    assert stats["put_sessions_left_for_drain"] == scenario.sessions.count("D")
    assert stats["put_sessions_left_to_master"] == scenario.sessions.count("M")
    # The driver knows every session the master still holds for it; an end that raised after it
    # landed leaves the driver unsure, so it counts the session as open.
    left_open = {k for k, session in session_of.items() if session != "-"}
    assert bench.fake.put_sessions <= left_open
    # Only a row written whole may be ended: its object becomes readable as it stands.
    written = bench.fake.keys_answered(PUT_RANGES, SLOT)
    assert set(bench.keys_of(PUT_END)) <= set(written)
    ready = [key(n) for n, r in zip("ABC", results) if r.status is RowStatus.READY]
    assert set(ready) == set(bench.fake.keys_answered(PUT_END, 0))


def test_a_session_whose_end_may_still_land_stays_open_and_its_key_is_never_put_again(
    make_driver, monkeypatch
) -> None:
    """The end of A times out twice: it may still reach the master, which matches an end by key
    and client. Revoked and put again, A's late end could complete an object never written. So the
    driver leaves A's session open, and a later put of A only asks whether A exists: a skip warned
    of once, when the session was left open."""
    warnings: list[str] = []
    monkeypatch.setattr(mooncake_module.logger, "warning", warnings.append)
    bench = Bench(make_driver)
    bench.fill_sources("put", "A")
    bench.fake.script(PUT_END, [-901])
    bench.fake.script(PUT_END, [-901])  # and its retry
    results, _ = bench.run("put", "A")
    assert summary(results) == [("fault", "put session end: RPC_TIMEOUT(-901)")]
    assert bench.keys_of(PUT_REVOKE) == []
    assert bench.driver.stats()["put_sessions_left_for_drain"] == 1
    assert any("put sessions left open for the shutdown drain" in w for w in warnings)

    warnings.clear()
    results, _ = bench.run("put", "A")
    assert summary(results) == [("not_ready", None)]
    assert skipped_counts(bench) == {"rows_not_ready_session_left_open": 1}
    assert warnings == []
    assert bench.fake.batch_put_session_end([key("A")]) == [0]  # the late end lands
    results, _ = bench.run("put", "A")
    assert summary(results) == [("ready", None)]
    assert bench.keys_of(PUT_START) == [key("A")]
    assert bench.keys_of(IS_EXIST) == [key("A")] * 2

    assert bench.driver.drain(0.0) is True  # its revoke finds the session gone
    assert bench.keys_of(PUT_REVOKE) == [key("A")]
    assert bench.fake.pool.objects[key("A")] == b"A" * SLOT


def test_a_put_row_answered_transfer_fail_is_left_to_the_master_even_by_drain(make_driver) -> None:
    """Over TCP the target of a TRANSFER_FAIL row may still write into the object's space, which a
    revoke would free at once; the master frees it after its put timeout."""
    bench = Bench(make_driver)
    bench.fill_sources("put", "A")
    bench.fake.script(PUT_RANGES, [-800])
    results, access_ended = bench.run("put", "A")
    assert summary(results) == [("fault", "put ranges: TRANSFER_FAIL(-800)")]
    assert access_ended
    assert bench.driver.stats()["put_sessions_left_to_master"] == 1
    assert bench.driver.drain(0.0) is True
    assert bench.keys_of(PUT_REVOKE) == [] and bench.fake.put_sessions == {key("A")}


def test_sessions_left_for_drain_are_revoked_only_in_drain(make_driver) -> None:
    bench = Bench(make_driver)
    bench.write(0, b"aaaaaaaa")
    bench.fake.script(PUT_END, [-704])
    bench.fake.script(PUT_REVOKE, [-900])
    bench.run("put", "A")
    assert bench.driver.stats()["put_sessions_left_for_drain"] == 1
    bench.write(1, b"bbbbbbbb")
    bench.store(C=b"cccccccc")
    bench.run("put", "X", "B")
    bench.run("get", "C")
    assert bench.keys_of(PUT_REVOKE) == [key("A")]  # only its own first try

    bench.fake.script(PUT_REVOKE, [-900])
    assert bench.driver.drain(0.0) is True  # sessions left open hold no lender memory
    assert bench.keys_of(PUT_REVOKE) == [key("A")] * 2
    assert bench.fake.put_sessions == {key("A")}
    assert bench.driver.drain(0.0) is True
    assert bench.fake.put_sessions == set()
    assert bench.driver.drain(0.0) is True  # nothing left to revoke
    assert bench.keys_of(PUT_REVOKE) == [key("A")] * 3
    assert bench.driver.stats()["put_sessions_left_for_drain"] == 1  # sessions ever left open


def test_a_put_that_raises_faults_every_row_it_took_even_those_only_checked_for_existence(
    monkeypatch,
) -> None:
    """A raise inside the driver is a bug, not a store answer. Every row of the put must still
    conclude, or its attempt never lands an outcome."""
    bench = Bench(python_driver)
    bench.fill_sources("put", "A", "B")
    bench.fake.script(PUT_END, [-901])
    bench.fake.script(PUT_END, [-901])  # and its retry: A's session is left for drain
    bench.run("put", "A")

    def raise_bug(*args) -> None:
        raise RuntimeError("bug")

    monkeypatch.setattr(MooncakeDriver, "_confirm_existing", raise_bug)
    op = bench.driver.put(bench.rows("A", "B"))
    assert summary(bench.driver.poll(op)) == [("fault", "put failed: RuntimeError('bug')")] * 2
    assert bench.driver.access_unprovable(op)


def test_drain_leaves_sessions_alone_while_access_is_unproven(make_driver) -> None:
    bench = Bench(make_driver)
    bench.fill_sources("put", "A")
    bench.fake.script(PUT_END, [-901])
    bench.fake.script(PUT_END, [-901])  # and its retry: A's session is left for drain
    bench.run("put", "A")
    bench.fake.script(PUT_RANGES, RuntimeError("bad pointer"))
    _, access_ended = bench.run("put", "B")
    assert not access_ended
    assert bench.driver.drain(0.0) is False
    assert bench.fake.calls_to(PUT_REVOKE) == []  # not even A's


LEASE_EXPIRED_CASES = {
    # the first read's answers, the reread's, the results, whether access ended
    "reread_ready": ([None, -707, None], [None], "RRR", True),
    "expired_again": ([None, -707, None], [-707], "RFR", True),
    "first_read_failed": ([None, -800, -707], [None], "RFR", False),
    "reread_failed": ([None, None, -707], [-800], "RRF", False),
    # The first read already wrote the slot, so the reread's answer cannot make it a MISS or
    # NOT_READY.
    "reread_not_found": ([None, -707, None], [-704], "RFR", True),
    "reread_not_ready": ([None, -707, None], [-703], "RFR", True),
}


@pytest.mark.parametrize("case", LEASE_EXPIRED_CASES)
def test_rows_whose_lease_expired_are_reread_once_and_both_reads_must_end_access(
    make_driver, case: str
) -> None:
    first, reread, statuses, ended = LEASE_EXPIRED_CASES[case]
    bench = Bench(make_driver)
    bench.fill_sources("get", "A", "B", "C")
    bench.fake.script(GET, first)
    bench.fake.script(GET, reread)
    results, access_ended = bench.run("get", "A", "B", "C")
    assert "".join(r.status.value[0].upper() for r in results) == statuses
    expired = [key(name) for name, answer in zip("ABC", first) if answer == -707]
    assert [call.args[0] for call in bench.fake.calls_to(GET)] == [[key(n) for n in "ABC"], expired]
    assert access_ended is ended
    assert bench.driver.drain(0.0) is ended


def test_an_object_being_written_is_not_ready_for_put_and_get(make_driver, monkeypatch) -> None:
    """Another writer's put of the key is still open. A get of it is a skip worth a warning; a put
    of it is not: that writer completes the object, which servers computing one prompt prefix at
    once do in normal traffic."""
    warnings: list[str] = []
    monkeypatch.setattr(mooncake_module.logger, "warning", warnings.append)
    pool = FakePool()
    writer = FakeMooncakeStore(pool)
    assert writer.batch_put_session_start([key("K")], [SLOT]) == [0]  # another client is writing
    bench = Bench(make_driver, pool=pool)
    results, access_ended = bench.run("get", "K")
    assert summary(results) == [("not_ready", None)] and access_ended
    results, access_ended = bench.run("put", "K")
    assert summary(results) == [("not_ready", None)] and access_ended
    stats = bench.driver.stats()
    assert stats["rows_not_ready_being_written"] == stats["rows_not_ready_another_writer"] == 1
    assert any(key("K") in line and "not_ready (being_written)" in line for line in warnings)
    assert [line for line in warnings if "mooncake put" in line] == []


def test_a_put_of_a_key_this_driver_is_putting_skips_it_as_another_writer(
    make_driver, monkeypatch
) -> None:
    """Two requests of one server publishing one block at once: the first put stores it; the
    second skips it without a store call (the SDK keys put sessions per client) and warns
    nothing."""
    warnings: list[str] = []
    monkeypatch.setattr(mooncake_module.logger, "warning", warnings.append)
    bench = Bench(make_driver)
    bench.fill_sources("put", "A", "B")
    gate = bench.fake.block(PUT_RANGES)
    first: list = []
    putter = threading.Thread(target=lambda: first.append(bench.run("put", "A")))
    putter.start()
    try:
        assert gate.entered.wait(WAIT_S)  # the first put holds A
        second = bench.driver.put(bench.rows("A", "B"))
        results = settled(bench.driver, second)
    finally:
        gate.release.set()
        putter.join(WAIT_S)
    assert summary(results) == [("not_ready", None), ("ready", None)]
    assert summary(first[0][0]) == [("ready", None)]
    assert bench.keys_of(PUT_START) == [key("A"), key("B")]
    assert skipped_counts(bench) == {"rows_not_ready_another_writer": 1}
    assert bench.fake.pool.objects[key("A")] == b"A" * SLOT
    assert warnings == []
    bench.run("put", "A")  # the first put returned: this driver puts A again
    assert bench.keys_of(PUT_START)[-1] == key("A")


def test_a_full_store_is_warned_about_once_not_per_operation(make_driver, monkeypatch) -> None:
    warnings: list[str] = []
    once: dict[str, str] = {}
    monkeypatch.setattr(mooncake_module.logger, "warning", warnings.append)
    monkeypatch.setattr(
        mooncake_module.logger, "warning_once", lambda message, key: once.setdefault(key, message)
    )
    bench = Bench(make_driver)
    for name in "AB":
        bench.fake.script(PUT_START, [-200])
        results, _ = bench.run("put", name)
        assert summary(results) == [("declined", None)]
    assert skipped_counts(bench) == {"rows_declined_pool_full": 2}
    assert warnings == [] and len(once) == 1
    assert "lack of space" in next(iter(once.values()))


# Access end.


# Each driver turns every raise into one fault and checks every answer count against the rows.
FAILURES = {
    "raises": RuntimeError("rpc lost"),
    "short": lambda answers: answers[:-1],
}


@pytest.mark.parametrize("failure", FAILURES)
@pytest.mark.parametrize("call", ["put", "get", "get_reread"])
def test_access_stays_open_when_a_data_call_raises_or_miscounts(
    make_driver, call: str, failure: str
) -> None:
    bench = Bench(make_driver)
    direction = "put" if call == "put" else "get"
    bench.fill_sources(direction, "A", "B")
    if call == "get_reread":
        bench.fake.script(GET, [None, -707])
    bench.fake.script(DATA_CALL[direction], FAILURES[failure])
    results, access_ended = bench.run(direction, "A", "B")  # nothing escapes the driver
    assert not access_ended and bench.driver.access_unprovable(bench.last)
    unproven = results[1:] if call == "get_reread" else results
    assert all(r.status is RowStatus.FAULT and " raised " in r.fault for r in unproven)
    if call == "get_reread":
        assert results[0].status is RowStatus.READY  # read whole the first time
    assert bench.driver.stats()["ops_unprovable"] == 1
    assert bench.driver.drain(0.0) is False


def test_a_fault_past_the_sdk_calls_keeps_access_open(monkeypatch) -> None:
    """The Python driver's own bug path; the native one's is its worker's catch-all."""

    def broken(*_):
        raise ValueError("mapping bug")

    bench = Bench(python_driver)
    monkeypatch.setattr(mooncake_module, "_get_result", broken)
    bench.fake.script(GET, [5])  # a row read short goes past the all-whole shortcut
    bench.fill_sources("get", "A")
    results, access_ended = bench.run("get", "A")
    assert summary(results) == [("fault", "get failed: ValueError('mapping bug')")]
    assert not access_ended


# Admission, registration, concurrency.


def test_bad_rows_are_rejected_before_the_sdk(make_driver) -> None:
    bench = Bench(make_driver)
    with pytest.raises(DriverRejected, match="at least one"):
        bench.driver.get([])
    with pytest.raises(DriverRejected, match="at most"):
        bench.driver.put([RowIO(key("A"), (Span(bench.buffer.address, 1 << 31),))])
    assert [c for c in bench.fake.calls if c.method.startswith("batch_")] == []


def test_rows_in_two_registrations_are_admitted_row_by_row(make_driver) -> None:
    bench = Bench(make_driver)
    other = HostBuffer(SLOT)  # a second registration, apart from the first
    bench.driver.register(other.address, SLOT, RegionLayout(SLOT, 1))
    first, _ = bench.rows("A", "B")
    op = bench.driver.get([first, RowIO(key("C"), (other.span(),))])  # the gap is no row's
    assert summary(settled(bench.driver, op)) == [("miss", None), ("miss", None)]
    assert bench.driver.access_ended(op)


def test_register_and_unregister_follow_the_protocol(make_driver) -> None:
    fake = FakeMooncakeStore()
    driver = make_driver(fake)
    region, other = HostBuffer(64), HostBuffer(64)
    handle = driver.register(region.address, 64, RegionLayout(SLOT, 8))
    with pytest.raises(DriverRejected, match="layout needs"):
        driver.register(other.address, SLOT, RegionLayout(SLOT, 2))
    assert len(fake.calls_to("register_buffer")) == 1

    fake.script("register_buffer", [-1])
    with pytest.raises(RegistrationError, match="INTERNAL_ERROR") as refused:
        driver.register(other.address, 64, RegionLayout(SLOT, 8))
    assert refused.value.handle is None
    fake.script("register_buffer", RuntimeError("rpc lost"))
    with pytest.raises(RegistrationError, match="rpc lost") as unknown:
        driver.register(other.address, 64, RegionLayout(SLOT, 8))
    assert unknown.value.handle is not None
    with pytest.raises(StoreError):  # the SDK never kept it, so it cannot revoke it either
        driver.unregister(unknown.value.handle)

    def revocations() -> int:
        return [call.args for call in fake.calls_to("unregister_buffer")].count((region.address,))

    fake.script("unregister_buffer", [-1])
    with pytest.raises(StoreError, match="INTERNAL_ERROR"):
        driver.unregister(handle)
    with pytest.raises(DriverRejected, match="overlaps"):  # the failed revocation left it live
        driver.register(region.address + SLOT, SLOT, RegionLayout(SLOT, 1))
    driver.unregister(handle)  # and retryable
    assert revocations() == 2
    driver.unregister(handle)  # and is idempotent once it succeeded
    assert revocations() == 2


class Allocations:
    """A CUDA allocation lookup over host memory: allocations of ``SLOT`` bytes back to back over
    ``buffer``, and none elsewhere."""

    def __init__(self, buffer: HostBuffer) -> None:
        self.buffer = buffer

    def __call__(self, address: int) -> tuple[int, int] | None:
        offset = address - self.buffer.address
        if not 0 <= offset < self.buffer.nbytes:
            return None
        return self.buffer.address + offset // SLOT * SLOT, SLOT


def test_a_device_region_registers_one_piece_per_allocation_and_rolls_back_a_failed_piece(
    make_driver,
) -> None:
    fake = FakeMooncakeStore()
    region = HostBuffer(3 * SLOT)  # stands in for device memory of three allocations
    driver = make_driver(fake, allocation_range=Allocations(region))
    layout = RegionLayout(SLOT, 3, "kv", MemoryKind.DEVICE)
    pieces = [(region.address + i * SLOT, SLOT) for i in range(3)]
    for answer in (None, None, -1):  # the third piece fails
        fake.script("register_buffer", [answer])
    with pytest.raises(RegistrationError, match="INTERNAL_ERROR") as failed:
        driver.register(region.address, region.nbytes, layout)
    assert [call.args for call in fake.calls_to("register_buffer")] == pieces
    assert failed.value.handle is None and fake.registered == {}  # the first two rolled back
    handle = driver.register(region.address, region.nbytes, layout)
    assert sorted(fake.registered.items()) == pieces
    driver.unregister(handle)
    assert fake.registered == {}


def test_a_span_across_allocations_reaches_the_sdk_as_one_buffer_per_allocation(
    make_driver,
) -> None:
    fake = FakeMooncakeStore()
    region = HostBuffer(3 * SLOT, fill=0x2D)
    driver = make_driver(fake, allocation_range=Allocations(region))
    driver.register(region.address, region.nbytes, RegionLayout(SLOT, 3, "kv", MemoryKind.DEVICE))
    region.write(b"0123456789abcdef", SLOT // 2)  # bytes 4 to 20: across both inner boundaries
    row = RowIO(key("A"), (region.span(SLOT // 2, 2 * SLOT), region.span(0, SLOT // 2)))
    assert summary(settled(driver, driver.put([row]))) == [("ready", None)]
    (scatter,) = fake.calls_to(PUT_RANGES)
    _, buffers, lengths, offsets = scatter.args
    assert lengths == [[4, 8, 4, 4]] and offsets == [[0, 4, 12, 16]]
    assert buffers[0][:3] == [region.address + 4, region.address + 8, region.address + 16]
    assert fake.pool.objects[key("A")] == b"0123456789abcdef----"

    region.write(b"-" * 3 * SLOT)
    assert summary(settled(driver, driver.get([row]))) == [("ready", None)]
    (get,) = fake.calls_to(GET)
    assert get.args[2] == [[4, 8, 4, 4]]
    assert region.read() == b"----0123456789abcdef----"


def wait_until_draining(driver: MooncakeDriver) -> None:
    """Returns once a drain waits: admission, under the drain's lock, refuses a stray row as
    draining rather than as outside every region. Nothing reaches the SDK."""
    stray = HostBuffer(SLOT)
    deadline = time.monotonic() + WAIT_S
    while time.monotonic() < deadline:
        with pytest.raises(DriverRejected) as refused:
            driver.get([RowIO(key("probe"), (stray.span(),))])
        if "draining" in str(refused.value):
            return
    pytest.fail("no drain started")


def test_unregister_is_refused_and_drain_waits_while_a_put_runs(make_driver) -> None:
    bench = Bench(make_driver)
    bench.fill_sources("put", "A")
    gate = bench.fake.block(PUT_RANGES)
    putter = threading.Thread(target=lambda: bench.run("put", "A"))
    drained: list[bool] = []
    drainer = threading.Thread(target=lambda: drained.append(bench.driver.drain(WAIT_S)))
    putter.start()
    try:
        assert gate.entered.wait(WAIT_S)  # inside the scatter
        with pytest.raises(DriverRejected, match="may still access"):
            bench.driver.unregister(bench.handle)
        drainer.start()
        wait_until_draining(bench.driver)
        assert drainer.is_alive()  # still running, not unprovable: the drain waits
    finally:
        released = time.monotonic()
        gate.release.set()
        putter.join(WAIT_S)
    drainer.join(WAIT_S)
    assert drained == [True]
    assert time.monotonic() - released < PROMPT_S  # the put's end woke the drain
    bench.driver.unregister(bench.handle)
    assert len(bench.fake.calls_to("unregister_buffer")) == 1


def test_an_unprovable_operation_pins_its_region_and_fails_drain_at_once(make_driver) -> None:
    bench = Bench(make_driver)
    bench.fill_sources("get", "A", "B")
    bench.fake.script(GET, [-800])
    bench.run("get", "A")
    gate = bench.fake.block(GET)
    ops = []
    blocked = threading.Thread(
        target=lambda: ops.append(bench.driver.get(bench.rows("A", "B")[1:]))
    )
    blocked.start()
    try:
        assert gate.entered.wait(WAIT_S)  # another get is still running
        started = time.monotonic()
        assert bench.driver.drain(WAIT_S) is False  # waits for no running call
        assert time.monotonic() - started < PROMPT_S
    finally:
        gate.release.set()
        blocked.join(WAIT_S)
    # The second get ended: only the unprovable first one pins the region, and the worker no
    # longer writes into bench.buffer.
    settled(bench.driver, ops[0])
    with pytest.raises(DriverRejected, match="may still access"):
        bench.driver.unregister(bench.handle)
    with pytest.raises(DriverRejected, match="drain"):
        bench.driver.close()
    assert bench.fake.calls_to("close") == []


def test_close_needs_a_drain_closes_the_sdk_client_once_and_then_queries_raise(
    make_driver,
) -> None:
    bench = Bench(make_driver)
    bench.fill_sources("put", "A")
    bench.run("put", "A")
    with pytest.raises(DriverRejected, match="drain"):
        bench.driver.close()
    assert bench.driver.drain(0.0) is True
    bench.driver.unregister(bench.handle)
    bench.driver.close()
    bench.driver.close()  # idempotent
    assert len(bench.fake.calls_to("close")) == 1
    with pytest.raises(RuntimeError, match="closed"):
        bench.driver.poll(bench.last)
    with pytest.raises(RuntimeError, match="closed"):
        bench.driver.drain(0.0)


def test_hints_and_foreign_handles_are_programming_errors(make_driver) -> None:
    bench, other = Bench(make_driver), Bench(make_driver)
    with pytest.raises(DriverRejected, match="hints"):
        bench.driver.get(bench.rows("A"), "a peer")
    other.fill_sources("put", "A")
    other.run("put", "A")
    for query in ("poll", "access_ended", "access_unprovable", "cancel"):
        with pytest.raises(ValueError, match="does not belong"):
            getattr(bench.driver, query)(other.last)
    with pytest.raises(ValueError, match="does not belong"):
        bench.driver.unregister(other.handle)
    assert [c for c in bench.fake.calls if c.method.startswith("batch_")] == []


def test_backend_round_trip_over_the_fake_store(make_driver) -> None:
    """With the native driver, the backend's queued path: no worker threads, batches handed over on
    the submitting thread, finished operations taken at each poll."""
    pool = FakePool()  # two clients of one master
    writer = Endpoint(make_driver(FakeMooncakeStore(pool)), ["A", "B"])
    reader = Endpoint(make_driver(FakeMooncakeStore(pool)), ["A", "B", "C", "D"], fill=0x2D)
    assert isinstance(writer.backend, Backend)
    assert writer.backend.capabilities == Capabilities(
        fetch=True, publish=True, cancel=True, slot_rows=True
    )
    writer.write("A", b"aaaaaaaa")
    writer.write("B", b"bbbbbbbb")
    pool.objects[store_key("D")] = b"too long!"

    published = writer.backend.publish(writer.view("A", "B"), None)
    assert isinstance(published, Attempt)
    assert writer.wait_outcome(published).delivered_masks[0].tolist() == [True, True]
    assert pool.objects[store_key("A")] == b"aaaaaaaa"
    writer.wait_quiesced(published)

    view = reader.view("A", "B", "C")
    fetched = reader.backend.fetch(view, None)
    outcome = reader.wait_outcome(fetched)
    outcome.check_mask_shapes(view)
    assert outcome.fault is None
    assert outcome.delivered_masks[0].tolist() == [True, True, False]
    assert [reader.read(name) for name in "ABC"] == [b"aaaaaaaa", b"bbbbbbbb", b"-" * SLOT]
    reader.wait_quiesced(fetched)

    wrong = reader.backend.fetch(reader.view("D"), None)
    outcome = reader.wait_outcome(wrong)
    assert outcome.delivered_masks[0].tolist() == [False]
    assert f"{GET}: INVALID_PARAMS(-600)" in outcome.fault  # refused before any transfer
    reader.wait_quiesced(wrong)
    assert reader.region.buffer.canaries_intact()
    writer.close()
    reader.close()

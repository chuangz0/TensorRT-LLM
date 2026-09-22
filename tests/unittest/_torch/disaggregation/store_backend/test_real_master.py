# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``MooncakeStoreBackend`` over the real ``MooncakeDistributedStore`` and a ``mooncake_master``
started for the test (TCP transport, P2P handshake, loopback). Skipped when the bindings or the
master binary are missing.

One backend stands for both ranks: a unit's coordinates are local, its name is not, so the same
name is published from layer group 0 and fetched into layer group 1 of one process. That keeps
one client per test, whose ``close`` the backend owns.
"""

import ctypes
import importlib.util
import os
import shutil
import socket
import subprocess
import time

import pytest

__extra_import_path__ = ["~/tensorrt_llm/_torch"]
from disaggregation.base.cache_backend import Delivered, Failed, Unit  # noqa: E402
from store_fakes import MemoryArena, extent, make_rank, pattern, wait_until  # noqa: E402

pytest.importorskip("mooncake.store")

MASTER = shutil.which("mooncake_master") or os.path.expanduser("~/.local/bin/mooncake_master")
if not os.access(MASTER, os.X_OK):
    pytest.skip("mooncake_master binary not found", allow_module_level=True)

READY_TIMEOUT_S = 20.0
DEADLINE_S = 120.0

_timeout = (
    pytest.mark.timeout(DEADLINE_S)
    if importlib.util.find_spec("pytest_timeout") is not None
    else pytest.mark.usefixtures("_deadline")
)
pytestmark = _timeout


@pytest.fixture
def _deadline():
    """Fallback bound when ``pytest-timeout`` is absent: the waits below all use it."""
    started = time.monotonic()
    yield
    assert time.monotonic() - started < DEADLINE_S, "real-master test overran its deadline"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_tcp(port: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return True
        except OSError:
            time.sleep(0.05)
    return False


@pytest.fixture
def master_port():
    """A ``mooncake_master`` on free ports, killed at teardown whatever happened."""
    rpc, metrics = _free_port(), _free_port()
    proc = subprocess.Popen(
        [MASTER, f"--rpc_port={rpc}", f"--metrics_port={metrics}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        if not _wait_tcp(rpc, READY_TIMEOUT_S):
            code = proc.poll()
            pytest.fail(
                f"mooncake_master did not listen on {rpc} within {READY_TIMEOUT_S}s "
                f"(exit code {code})"
            )
        yield rpc
    finally:
        proc.kill()
        proc.wait(timeout=10)


@pytest.fixture
def store(master_port):
    from mooncake.store import MooncakeDistributedStore

    client = MooncakeDistributedStore()
    status = client.setup(
        "127.0.0.1", "P2PHANDSHAKE", 64 << 20, 16 << 20, "tcp", "", f"127.0.0.1:{master_port}"
    )
    assert status == 0, f"setup failed with {status}"
    return client  # closed by the backend that owns it


def _rank(store, **overrides):
    """A backend over the real client; source units in group 0, destinations in group 1."""
    rank = make_rank(store, arena_bytes=1 << 16, namespace=f"t{os.getpid()}", **overrides)
    return rank


def test_publish_then_fetch_round_trips_two_segment_units(store):
    with _rank(store) as rank:
        src = [rank.unit(0, i, 96, 32) for i in range(3)]
        for i, u in enumerate(src):
            rank.write(u, pattern(i + 1, 128))
        published = rank.finish(rank.backend.publish(extent(src, name=b"ctx")))
        assert published == Delivered(frozenset(u.name for u in src))
        assert all(store.get_size(rank.key(u)) == 128 for u in src)

        dst = []
        for i, u in enumerate(src):
            rank.resolver.add(1, i, 64, 64)  # the same bytes, cut differently on this side
            dst.append(Unit(name=u.name, local_group=1, local=i))
            rank.fill(dst[-1], 0xEE)
        fetched = rank.finish(rank.backend.fetch(extent(dst, name=b"gen")))
        assert fetched == Delivered(frozenset(u.name for u in dst))
        for i, u in enumerate(dst):
            assert rank.read(u) == pattern(i + 1, 128)
        assert rank.backend.counters.fetch_hits == 3 and rank.backend.counters.failed_attempts == 0


def test_missing_unit_is_not_served_and_left_untouched(store):
    with _rank(store) as rank:
        held = rank.unit(0, 0, 64)
        rank.write(held, pattern(5, 64))
        assert isinstance(rank.finish(rank.backend.publish(extent([held]))), Delivered)
        rank.resolver.add(1, 0, 64)
        rank.resolver.add(1, 1, 64)
        dst_held = Unit(name=held.name, local_group=1, local=0)
        dst_missing = Unit(name=b"never-published", local_group=1, local=1)
        rank.fill(dst_held, 0xEE)
        rank.fill(dst_missing, 0xEE)
        outcome = rank.finish(rank.backend.fetch(extent([dst_held, dst_missing])))
        assert outcome == Delivered(frozenset({held.name}))
        assert rank.read(dst_held) == pattern(5, 64)
        assert rank.read(dst_missing) == bytes([0xEE]) * 64
        assert rank.backend.counters.fetch_misses == 1
        assert store.batch_is_exist([rank.key(dst_missing)]) == [0]


def test_probe_answers_what_the_store_holds(store):
    with _rank(store) as rank:
        a, b = rank.unit(0, 0, 32), rank.unit(0, 1, 32)
        assert isinstance(rank.finish(rank.backend.publish(extent([a]))), Delivered)
        names = [a.name, b.name, b"unknown"]
        assert rank.backend.probe(b"n", names) is None
        counters = rank.backend.counters
        wait_until(lambda: counters.probe_hits + counters.probe_misses == 3, what="probe")
        assert rank.backend.probe(b"n", names) == frozenset({a.name})
        assert (counters.probe_hits, counters.probe_misses) == (1, 2)


def test_unregistered_destination_fails_before_the_store_is_asked(store):
    with _rank(store) as rank:
        held = rank.unit(0, 0, 64)
        rank.write(held, pattern(9, 64))
        assert isinstance(rank.finish(rank.backend.publish(extent([held]))), Delivered)
        elsewhere = MemoryArena(64)  # never registered with the backend
        rank.resolver._table[(1, 0)] = ((elsewhere.address, 64),)
        dst = Unit(name=held.name, local_group=1, local=0)
        ctypes.memset(elsewhere.address, 0xEE, 64)
        attempt = rank.backend.fetch(extent([dst]))
        outcome = attempt.poll()
        assert isinstance(outcome, Failed) and "not registered" in outcome.reason
        assert ctypes.string_at(elsewhere.address, 64) == bytes([0xEE]) * 64
        assert rank.backend.counters.fetch_misses == 0
        # A closed registration is the same story.
        rank.registration.close()
        src_again = rank.backend.fetch(extent([Unit(name=held.name, local_group=0, local=0)]))
        assert isinstance(src_again.poll(), Failed)

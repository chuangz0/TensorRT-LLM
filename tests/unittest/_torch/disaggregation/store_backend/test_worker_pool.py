# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``DaemonWorkerPool``: the executor-like contract the store backend drives its deliveries with.

Daemon threads that start on the first ``submit``, run every queued callable, keep running past a
callable that raises, refuse work after ``shutdown``, and are joined by ``shutdown(wait=True)``.
"""

import threading
import time

import pytest

__extra_import_path__ = ["~/tensorrt_llm/_torch"]
from disaggregation.backends.store.worker_pool import DaemonWorkerPool  # noqa: E402

pytestmark = pytest.mark.cpu_only


def _wait(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.002)


def test_rejects_a_non_positive_worker_count():
    with pytest.raises(ValueError, match="num_workers must be > 0"):
        DaemonWorkerPool(0, "x")


def test_threads_are_daemons_named_by_prefix_and_start_on_first_submit():
    pool = DaemonWorkerPool(2, "kv-store-test")
    assert not any(t.is_alive() for t in pool._threads)
    seen = []
    pool.submit(lambda: seen.append(threading.current_thread()))
    _wait(lambda: len(seen) == 1)
    assert seen[0].daemon and seen[0].name.startswith("kv-store-test-")
    assert all(t.is_alive() for t in pool._threads)
    pool.shutdown(wait=True)
    assert not any(t.is_alive() for t in pool._threads)


def test_runs_every_submission_with_its_arguments_across_workers():
    pool = DaemonWorkerPool(3, "w")
    results, lock = [], threading.Lock()
    gate = threading.Event()

    def work(i, tag):
        gate.wait()
        with lock:
            results.append((i, tag, threading.current_thread().name))

    try:
        for i in range(6):
            pool.submit(work, i, "t")
        gate.set()
        _wait(lambda: len(results) == 6)
        assert sorted(r[:2] for r in results) == [(i, "t") for i in range(6)]
        assert {name for *_, name in results} <= {t.name for t in pool._threads}
    finally:
        gate.set()
        pool.shutdown(wait=True)


def test_a_raising_callable_does_not_kill_the_worker(caplog):
    pool = DaemonWorkerPool(1, "w")
    done = threading.Event()
    try:
        with caplog.at_level("ERROR"):
            pool.submit(lambda: 1 / 0)
            pool.submit(done.set)
            assert done.wait(5.0)
        assert any("unhandled error" in r.getMessage() for r in caplog.records)
    finally:
        pool.shutdown(wait=True)


def test_shutdown_refuses_new_work_lets_queued_work_finish_and_is_idempotent():
    pool = DaemonWorkerPool(1, "w")
    release = threading.Event()
    finished = []
    pool.submit(lambda: (release.wait(), finished.append("first")))
    pool.submit(lambda: finished.append("second"))
    pool.shutdown(wait=False)  # queued work still runs
    with pytest.raises(RuntimeError, match="after shutdown"):
        pool.submit(lambda: None)
    release.set()
    _wait(lambda: finished == ["first", "second"])
    pool.shutdown(wait=True)  # a second shutdown is a no-op ...
    _wait(lambda: not any(t.is_alive() for t in pool._threads))  # ... and the workers exit


def test_shutdown_before_any_submit_starts_nothing():
    pool = DaemonWorkerPool(2, "w")
    pool.shutdown(wait=True)
    assert all(t.ident is None for t in pool._threads)
    with pytest.raises(RuntimeError):
        pool.submit(lambda: None)

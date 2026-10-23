# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mooncake store bandwidth beside a busy Python thread, and what the traffic costs that thread.

The blob backend fetches and publishes through a real Mooncake store with the Python or the native
store driver. An executor stand-in thread keeps two attempts in flight, submitting and polling them
once per ``--poll-ms`` (an executor iteration), as an executor loop would; between polls it either
sleeps (idle) or runs pure bytecode (busy), which never releases the GIL on its own. A busy thread
makes every store call that returns into Python wait for the GIL; the native driver's calls never do.

Each cell moves a fixed volume: gets cycle over objects written first, puts write new objects (the
pool must hold the volume). Each busy cell runs ``--repeats`` times, between windows where the
thread runs alone (alone, busy, alone, busy, ..., alone) with the same timer calls and an empty
poll; each busy window is compared with the mean of its two neighbours, so that drift cancels.
Prints one JSON line per cell, then, for the native driver, PASS or FAIL against the targets
(medians over the repeats): busy bandwidth at least 90% of idle for gets and for puts, and the busy
thread keeping at least 99% of the loop rate it has alone.

Needs a running ``mooncake_master``, a client giving the pool its memory, and a store config whose
``mooncake`` section points at the master::

    python3 mooncake_store_busy_executor.py --store-config store.yaml --driver both
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from tensorrt_llm._torch.disaggregation.base.memory import MemoryKind
from tensorrt_llm._torch.disaggregation.blob.backend import BlobBackend, BlobConfig, RegionBinding
from tensorrt_llm._torch.disaggregation.blob.mooncake_setup import (
    load_store_config,
    make_store_driver,
    open_sdk,
)

MIB = 1 << 20
GIB = 1 << 30
ATTEMPTS_IN_FLIGHT = 2
SPIN_STEPS = 1000  # bytecode steps between clock reads in the busy loop
SPIN_ALONE_S = 2.0
BUSY_BANDWIDTH_TARGET = 0.90
LOOP_RATE_TARGET = 0.99
FIRST_PUT_NAME = 1 << 40  # put cells name their objects from here on, apart from the get objects


@dataclass(frozen=True)
class HostRegion:
    """Host memory the backend registers: the lender's ``SlotRegion``."""

    address: int
    nbytes: int
    slot_bytes: int
    slot_count: int
    layout_label: str = "bench"
    memory_kind: MemoryKind = MemoryKind.HOST


@dataclass
class Rows:
    """One layer group of a view, placed by address."""

    object_keys: np.ndarray
    addresses: np.ndarray
    layer_group_index: int = 0
    region_index: int = 0

    def __len__(self) -> int:
        return len(self.addresses)


@dataclass
class View:
    groups: list[Rows]


class Hold:
    def release(self) -> None:
        pass


class Executor:
    """The executor stand-in.

    It calls ``poll`` once per ``poll_interval_s`` and, when busy, runs bytecode in between;
    ``steps`` counts the bytecode loop's iterations.
    """

    def __init__(
        self, busy: bool, poll: Callable[[], None], cpu: int | None, poll_interval_s: float
    ) -> None:
        self.busy, self.poll, self.cpu, self.poll_interval_s = busy, poll, cpu, poll_interval_s
        self.stop = threading.Event()
        self.steps = 0
        self.poll_cpu_s = 0.0  # this thread's CPU time inside ``poll``: submitting and polling
        self.thread = threading.Thread(target=self._run, name="bench-executor")

    def run_until(self, done: Callable[[], bool]) -> float:
        """Runs until ``done()``; returns the elapsed seconds."""
        started = time.perf_counter()
        self.thread.start()
        while not done():
            time.sleep(0.005)
        elapsed = time.perf_counter() - started
        self.stop.set()
        self.thread.join()
        return elapsed

    def _run(self) -> None:
        if self.cpu is not None:
            os.sched_setaffinity(0, {self.cpu})  # this thread only
        steps, due, poll_cpu_s = 0, time.perf_counter(), 0.0
        while not self.stop.is_set():
            if self.busy:
                for _ in range(SPIN_STEPS):
                    steps += 1
            else:
                time.sleep(self.poll_interval_s)
            if time.perf_counter() >= due:
                started = time.thread_time()
                self.poll()
                poll_cpu_s += time.thread_time() - started
                due = time.perf_counter() + self.poll_interval_s
        self.steps, self.poll_cpu_s = steps, poll_cpu_s


def reserve_executor_cpu() -> int:
    """A CPU for the executor stand-in alone, taken from this process before any thread starts.

    Threads started later inherit the process's CPUs and never run there.
    """
    allowed = os.sched_getaffinity(0)
    cpu = max(allowed)
    os.sched_setaffinity(0, allowed - {cpu})
    return cpu


def loop_rate_alone(cpu: int | None, poll_interval_s: float) -> float:
    """The busy loop's rate with nothing to poll: the same timer calls, an empty poll."""
    executor = Executor(busy=True, poll=lambda: None, cpu=cpu, poll_interval_s=poll_interval_s)
    started = time.perf_counter()
    elapsed = executor.run_until(lambda: time.perf_counter() - started >= SPIN_ALONE_S)
    return executor.steps / elapsed


class Bench:
    """A blob backend over one store driver and two attempts' worth of registered host slots."""

    def __init__(
        self,
        store,
        mooncake,
        blob: BlobConfig,
        row_bytes: int,
        attempt_rows: int,
        cpu: int | None,
        poll_interval_s: float,
    ):
        self.driver = make_store_driver(store, mooncake, blob)
        self.cpu, self.poll_interval_s = cpu, poll_interval_s
        self.row_bytes, self.attempt_rows = row_bytes, attempt_rows
        slot_count = ATTEMPTS_IN_FLIGHT * attempt_rows
        self.memory = np.zeros(slot_count * row_bytes, dtype=np.uint8)  # zeroed: pages touched
        region = HostRegion(self.memory.ctypes.data, self.memory.nbytes, row_bytes, slot_count)
        self.slots = region.address + np.arange(slot_count, dtype=np.int64) * row_bytes
        binding = RegionBinding([region], self.driver, Hold())
        binding.register()
        self.backend = BlobBackend(binding, lambda: None, blob)

    def cell(
        self, direction: str, busy: bool, attempts: int, first_name: int, cycle: int | None = None
    ) -> dict:
        """Moves ``attempts`` attempts, two in flight, beside the executor stand-in.

        Attempt ``n`` names its objects from ``first_name + n * attempt_rows``, with ``n`` modulo
        ``cycle`` when given (gets of the objects written first).
        """
        submit = self.backend.fetch if direction == "get" else self.backend.publish
        in_flight: list[tuple[float, int, object]] = []  # (submitted at, slot range, attempt)
        latencies: list[float] = []
        moved_bytes = done = submitted = 0
        submit_cpu_s = 0.0  # the share of the poll hook's CPU time spent submitting

        def poll() -> None:
            nonlocal moved_bytes, done, submitted, submit_cpu_s
            for entry in list(in_flight):
                started, _, attempt = entry
                outcome = attempt.poll()
                if outcome is None or not attempt.quiesced():
                    continue
                in_flight.remove(entry)
                moved_bytes += int(outcome.delivered_masks[0].sum()) * self.row_bytes
                done += 1
                latencies.append(time.perf_counter() - started)
            while len(in_flight) < ATTEMPTS_IN_FLIGHT and submitted < attempts:
                n = submitted % cycle if cycle else submitted
                used = {slots for _, slots, _ in in_flight}
                slots = min(set(range(ATTEMPTS_IN_FLIGHT)) - used)  # each attempt owns a range
                view = self._view(slots, first_name, n)
                started = time.thread_time()
                attempt = submit(view, None)
                submit_cpu_s += time.thread_time() - started
                in_flight.append((time.perf_counter(), slots, attempt))
                submitted += 1

        executor = Executor(busy, poll, self.cpu, self.poll_interval_s)
        elapsed = executor.run_until(lambda: done >= attempts)
        return {
            "direction": direction,
            "busy": busy,
            "gbps": round(moved_bytes / elapsed / 1e9, 2),
            "gib": round(moved_bytes / GIB, 1),
            "seconds": round(elapsed, 2),
            "attempt_ms_p50": round(1e3 * statistics.median(latencies), 1),
            "loop_rate": round(executor.steps / elapsed) if busy else None,
            "poll_cpu_share": round(executor.poll_cpu_s / elapsed, 4),
            "submit_cpu_share": round(submit_cpu_s / elapsed, 4),
        }

    def _view(self, slots: int, first_name: int, n: int) -> View:
        start = slots * self.attempt_rows
        first = first_name + n * self.attempt_rows
        object_names = (
            np.arange(first, first + self.attempt_rows, dtype=">u8")
            .view(np.uint8)
            .reshape(self.attempt_rows, 8)
        )
        return View([Rows(object_names, self.slots[start : start + self.attempt_rows])])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--store-config", required=True)
    parser.add_argument("--driver", choices=("python", "native", "both"), default="both")
    parser.add_argument("--row-mib", type=float, default=1.0)
    parser.add_argument("--attempt-gib", type=float, default=1.0)
    parser.add_argument("--volume-gib", type=int, default=128, help="each cell moves this much")
    parser.add_argument("--objects-gib", type=int, default=4, help="the objects gets cycle over")
    parser.add_argument("--repeats", type=int, default=3, help="of each busy cell")
    parser.add_argument(
        "--poll-ms", type=float, default=5.0, help="the executor's iteration: one poll each"
    )
    parser.add_argument(
        "--pin-executor",
        action="store_true",
        help="give the executor stand-in a CPU no other thread of this process runs on, so that the "
        "loop rate shows the GIL and the driver's Python, not the store's own polling threads",
    )
    args = parser.parse_args()
    cpu = reserve_executor_cpu() if args.pin_executor else None
    poll_interval_s = args.poll_ms / 1e3

    blob_section, mooncake = load_store_config(args.store_config)
    blob = BlobConfig(**blob_section)
    row_bytes = int(args.row_mib * MIB)
    attempt_rows = int(args.attempt_gib * GIB) // row_bytes
    attempt_bytes = attempt_rows * row_bytes
    object_attempts = args.objects_gib * GIB // attempt_bytes
    cell_attempts = args.volume_gib * GIB // attempt_bytes
    store = open_sdk(mooncake)
    shape = {
        "row_mib": args.row_mib,
        "attempt_gib": args.attempt_gib,
        "workers": blob.workers,
        "pinned": args.pin_executor,
        "poll_ms": args.poll_ms,
    }

    drivers = ("python", "native") if args.driver == "both" else (args.driver,)
    ratios: dict[tuple[str, str], list[tuple[float, float]]] = {}  # (bandwidth, loop rate) ratios
    for driver in drivers:
        bench = Bench(
            store,
            replace(mooncake, driver=driver),
            blob,
            row_bytes,
            attempt_rows,
            cpu,
            poll_interval_s,
        )
        bench.cell("put", False, object_attempts, first_name=0)  # the objects the gets read

        def cell(direction: str, busy: bool) -> dict:
            if direction == "get":
                return bench.cell("get", busy, cell_attempts, 0, cycle=object_attempts)
            record = bench.cell("put", busy, cell_attempts, FIRST_PUT_NAME)
            store.remove_all(True)  # room for the next cell's new objects
            return record

        for direction in ("get", "put"):
            idle = cell(direction, busy=False)
            print(json.dumps({"driver": driver, **shape, **idle}), flush=True)
            alone = [loop_rate_alone(cpu, poll_interval_s)]
            for _ in range(args.repeats):  # alone, busy, alone, busy, ..., alone
                busy = cell(direction, busy=True)
                alone.append(loop_rate_alone(cpu, poll_interval_s))
                neighbours = alone[-2:]
                busy["loop_rate_alone"] = [round(rate) for rate in neighbours]
                print(json.dumps({"driver": driver, **shape, **busy}), flush=True)
                pair = (
                    busy["gbps"] / idle["gbps"],
                    busy["loop_rate"] / statistics.mean(neighbours),
                )
                ratios.setdefault((driver, direction), []).append(pair)
        store.remove_all(True)
        bench.backend.shutdown(30.0)

    for (driver, direction), pairs in ratios.items():
        bandwidth = statistics.median(pair[0] for pair in pairs)
        loop = statistics.median(pair[1] for pair in pairs)
        for what, value, target in (
            ("bandwidth busy/idle", bandwidth, BUSY_BANDWIDTH_TARGET),
            ("busy loop rate / alone", loop, LOOP_RATE_TARGET),
        ):
            verdict = ("PASS" if value >= target else "FAIL") if driver == "native" else "info"
            print(f"{verdict} {driver} {direction} {what} = {value:.3f} (target {target})")


if __name__ == "__main__":
    main()

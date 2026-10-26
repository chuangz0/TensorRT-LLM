# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Running a Mooncake master for a test, on ports picked for it, and the memory behind it.

A deployment names its master's ports, so `running_master` takes them as
given. A test has no pair to name and wants any free one, on a machine whose
other ports belong to whatever else runs there. A port is reserved here by
binding it and letting go, so another process can still take it before the
master binds, and the master then exits before it accepts connections. Each
attempt picks a fresh pair.

The master stores nothing itself: providers, separate `trtllm-serve
mooncake_provider` processes, give a pool its memory.
"""

import contextlib
import json
import os
import subprocess  # nosec B404
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import yaml

from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.master import (
    LaunchedMaster,
    running_master,
)
from tensorrt_llm._utils import get_free_ports

__all__ = [
    "infiniband_by_socket",
    "master_segments",
    "running_master_on_free_ports",
    "running_pool",
    "running_providers_per_socket",
    "running_segment_provider",
]

PORT_ATTEMPTS = 5
# The provider imports tensorrt_llm before it joins the pool, which takes long on a cold network
# file system.
SEGMENT_PROVIDER_TIMEOUT = 300.0
# A provider leaves the pool about 13 s after its SIGTERM, once Mooncake freed its memory.
SEGMENT_PROVIDER_STOP_TIMEOUT = 30.0


def _start_master(
    stack: contextlib.ExitStack, run_dir: str, attempts: int, kwargs: dict
) -> LaunchedMaster:
    """Enter `running_master` on `stack`, on the first pair of ports it keeps.

    A failed start leaves nothing of the master behind, so every attempt
    begins from the state the first one did. The last failure is raised as it
    is, since it names the port that was taken.
    """
    for attempt in range(1, attempts + 1):
        rpc_port, metrics_port = get_free_ports(2)
        try:
            return stack.enter_context(
                running_master(run_dir, rpc_port=rpc_port, metrics_port=metrics_port, **kwargs)
            )
        except (RuntimeError, TimeoutError):
            if attempt == attempts:
                raise
            print(
                f"mooncake-store: the master did not come up on rpc={rpc_port} "
                f"metrics={metrics_port} (attempt {attempt} of {attempts}); "
                "retrying on another pair of ports."
            )
    raise AssertionError("unreachable: the last attempt either returns or raises")


@contextlib.contextmanager
def running_master_on_free_ports(
    run_dir: str, *, attempts: int = PORT_ATTEMPTS, **kwargs
) -> Iterator[LaunchedMaster]:
    """Run a master on free ports for the body's duration.

    Args:
        run_dir: Passed through to `running_master`.
        attempts: Pairs of ports to try.
        kwargs: Everything else `running_master` takes, except the two ports.
    """
    with contextlib.ExitStack() as stack:
        yield _start_master(stack, run_dir, attempts, kwargs)


@dataclass(frozen=True)
class _Provider:
    """One `trtllm-serve mooncake_provider` process.

    Attributes:
        run_dir: Where its store config, log and readiness marker go.
        segment_size: Bytes of its segment.
        device_name: RDMA devices; empty picks the host's fastest InfiniBand ones.
        launcher: A command it runs under, e.g. `numactl` binding it to a socket.
    """

    run_dir: str
    segment_size: int
    device_name: str = ""
    launcher: tuple[str, ...] = ()

    @property
    def ready_file(self) -> str:
        return os.path.join(self.run_dir, "segment_provider.ready")

    @property
    def log_path(self) -> str:
        return os.path.join(self.run_dir, "segment_provider.log")

    def launch(self, master_address: str, protocol: str) -> subprocess.Popen:
        os.makedirs(self.run_dir, exist_ok=True)
        config = os.path.join(self.run_dir, "segment_provider.yaml")
        mooncake = {
            "master_server_address": master_address,
            "protocol": protocol,
            "device_name": self.device_name,
        }
        with open(config, "w") as file:
            yaml.safe_dump({"mooncake": mooncake}, file)
        command = [*self.launcher, sys.executable, "-m", "tensorrt_llm.commands.serve"]
        command += ["mooncake_provider", "--no-telemetry", "--config", config]
        command += ["--segment_size", str(self.segment_size), "--ready_file", self.ready_file]
        with open(self.log_path, "wb") as log:
            return subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)  # nosec B603

    def wait_until_joined(
        self, process: subprocess.Popen, master_address: str, deadline: float
    ) -> None:
        while not os.path.exists(self.ready_file):
            if process.poll() is not None or time.monotonic() > deadline:
                with open(self.log_path, errors="replace") as log:
                    tail = log.read()[-4000:]
                raise RuntimeError(f"the segment provider did not join {master_address}:\n{tail}")
            time.sleep(0.1)


@contextlib.contextmanager
def _running_providers(
    master_address: str, providers: list[_Provider], protocol: str
) -> Iterator[list[subprocess.Popen]]:
    """Every provider in `providers`, for the body's duration.

    They join the pool side by side and leave it side by side: each gets one
    SIGTERM, then a deadline shared by all to exit.
    """
    processes = []
    try:
        for provider in providers:
            processes.append(provider.launch(master_address, protocol))
        deadline = time.monotonic() + SEGMENT_PROVIDER_TIMEOUT
        for provider, process in zip(providers, processes):
            provider.wait_until_joined(process, master_address, deadline)
        yield processes
    finally:
        for process in processes:
            process.terminate()  # the provider unmounts its memory gracefully
        deadline = time.monotonic() + SEGMENT_PROVIDER_STOP_TIMEOUT
        for process in processes:
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


@contextlib.contextmanager
def running_segment_provider(
    master_address: str,
    run_dir: str,
    *,
    segment_size: int,
    protocol: str = "tcp",
    device_name: str = "",
) -> Iterator[subprocess.Popen]:
    """Give the pool a `segment_size`-byte segment for the body's duration.

    The segment comes from `trtllm-serve mooncake_provider`, a separate process.
    A client with no segment of its own then finds every object in another
    process's memory, so each transfer goes through the transfer engine as it
    would across hosts, never through a copy inside one process.

    Args:
        master_address: The pool's master, `host:port`.
        run_dir: Where the provider's store config, log and readiness marker go.
        segment_size: Bytes of the provider's segment.
        protocol: `rdma` or `tcp`.
        device_name: RDMA devices; empty picks the host's fastest InfiniBand ones.
    """
    provider = _Provider(run_dir, segment_size, device_name)
    with _running_providers(master_address, [provider], protocol) as (process,):
        yield process


@contextlib.contextmanager
def running_pool(
    run_dir: str,
    segment_size: int,
    *,
    protocol: str = "tcp",
    device_name: str = "",
    eviction_ratio: float = 0.0,
) -> Iterator[LaunchedMaster]:
    """A master on free ports and a provider of `segment_size` bytes, for the body's duration.

    No provider for `segment_size` 0. With `eviction_ratio` 0 the master evicts
    only above its high watermark, never just because a put failed.
    """
    with contextlib.ExitStack() as stack:
        master = stack.enter_context(
            running_master_on_free_ports(run_dir, protocol=protocol, eviction_ratio=eviction_ratio)
        )
        if segment_size:
            stack.enter_context(
                running_segment_provider(
                    master.address,
                    run_dir,
                    segment_size=segment_size,
                    protocol=protocol,
                    device_name=device_name,
                )
            )
        yield master


def infiniband_by_socket() -> dict[int, list[str]]:
    """The active InfiniBand devices of each NUMA node (Grace socket) of this host."""
    found: dict[int, list[str]] = {}
    for device in sorted(Path("/sys/class/infiniband").iterdir()):
        port = device / "ports" / "1"
        if (port / "link_layer").read_text().strip() != "InfiniBand":
            continue
        if "ACTIVE" in (port / "state").read_text():
            socket = int((device / "device" / "numa_node").read_text())
            found.setdefault(socket, []).append(device.name)
    return found


def running_providers_per_socket(
    master_address: str, run_dir: str, *, total_bytes: int, protocol: str = "rdma"
) -> contextlib.AbstractContextManager[list[subprocess.Popen]]:
    """A provider on each socket of this host with active InfiniBand devices.

    As deployments run them: under `numactl` on the socket's CPUs and memory,
    over its devices, `total_bytes` split between them, each in
    `run_dir/provider-<socket>`.
    """
    sockets = infiniband_by_socket()
    providers = [
        _Provider(
            os.path.join(run_dir, f"provider-{socket}"),
            total_bytes // len(sockets),
            ",".join(devices),
            ("numactl", f"--cpunodebind={socket}", f"--membind={socket}"),
        )
        for socket, devices in sockets.items()
    ]
    return _running_providers(master_address, providers, protocol)


def master_segments(metrics_url: str) -> list[dict]:
    """The segments the master at `metrics_url` (`http://host:port`) lists.

    One per mounted segment (`/get_segments_detail`): `segment_name` (the client's `host:port`),
    `status` and `allocator_used_bytes` among others.
    """
    with urllib.request.urlopen(f"{metrics_url}/get_segments_detail", timeout=10) as reply:
        return json.load(reply)["segments"]

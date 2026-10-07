# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Mooncake processes of the store e2e tests: a ``mooncake_master`` and a long-lived segment
provider on free loopback ports, the fixtures that start and stop them, and the transports
(loopback TCP, RDMA to the device, RDMA to pinned host memory).

Mooncake objects live in the client segments; the engines contribute none
(``global_segment_size: 0``), so the provider is the one client whose segment holds the published
blocks, and it must outlive every ``LLM()`` of a test. A test that wants a provider of another
size parametrizes the ``mooncake_cluster`` fixture indirectly with the segment's byte count.

The engines themselves are ``store_engine.py``, the dumps they write ``status_dumps.py``.
"""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass

import pytest

RDMA_ENV = "KV_TRANSFER_E2E_RDMA"
"""``1`` enables the RDMA transports; the machine needs an RDMA NIC."""
RDMA_DEVICES_ENV = "KV_TRANSFER_E2E_RDMA_DEVICES"
"""HCAs handed to Mooncake as ``device_name``; empty lets it discover them, which on some hosts
pairs InfiniBand ports with Ethernet ones that cannot reach them."""

MASTER = shutil.which("mooncake_master") or os.path.expanduser("~/.local/bin/mooncake_master")

READY_TIMEOUT_S = 30.0
SEGMENT_BYTES = 1 << 30
"""The provider's segment unless a test asks for another size."""
TRANSPORT_SEGMENT_BYTES = 4 << 30
"""The provider's segment for the multi-rank and exactness runs: ``gemma3_12b`` at 2000 tokens
publishes about 750 MiB of KV."""


# ---------------------------------------------------------------------------------------------
# Processes
# ---------------------------------------------------------------------------------------------


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_tcp(port: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return True
        except OSError:
            time.sleep(0.05)
    return False


def host_ip() -> str:
    """The address an RDMA endpoint announces: the host's own, not loopback."""
    return socket.gethostbyname(socket.gethostname())


def kill_process_group(proc: subprocess.Popen | None) -> None:
    """Kill the process and everything in its session (both are started with
    ``start_new_session=True``): the Mooncake client forks helpers that outlive their parent."""
    if proc is None:
        return
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
    # Children re-parented to init are not waited for by ``proc``; sweep the group again.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def leftover_pids(marker: str) -> list[int]:
    """Live processes whose command line carries ``marker`` (a port or address unique to this
    test); read from ``/proc`` so no extra dependency is needed."""
    found = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == os.getpid():
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as f:
                cmdline = f.read().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if marker in cmdline:
            found.append(int(entry))
    return found


_PROVIDER_SCRIPT = textwrap.dedent(
    """
    import sys, time
    from mooncake.store import MooncakeDistributedStore
    master, segment_bytes, protocol, host, devices = sys.argv[1:6]
    store = MooncakeDistributedStore()
    status = store.setup(
        host, "P2PHANDSHAKE", int(segment_bytes), 16 << 20, protocol, devices, master
    )
    if status != 0:
        print(f"SETUP_FAILED {status}", flush=True)
        sys.exit(1)
    print("READY", flush=True)
    while True:
        time.sleep(1.0)
    """
)


@dataclass
class MooncakeCluster:
    master_address: str
    master: subprocess.Popen
    provider: subprocess.Popen

    def kill_master(self) -> None:
        """The store's master goes away mid-test; every client call fails from here on."""
        kill_process_group(self.master)

    def close(self) -> None:
        stop_cluster(self.master_address, self.master, self.provider)


def start_master() -> tuple[str, subprocess.Popen]:
    rpc, metrics = free_port(), free_port()
    proc = subprocess.Popen(
        [MASTER, f"--rpc_port={rpc}", f"--metrics_port={metrics}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    if not wait_tcp(rpc, READY_TIMEOUT_S):
        kill_process_group(proc)
        pytest.fail(f"mooncake_master did not listen on {rpc} within {READY_TIMEOUT_S}s")
    return f"127.0.0.1:{rpc}", proc


def start_segment_provider(
    master_address: str, segment_bytes: int = SEGMENT_BYTES, *, protocol: str = "tcp"
) -> subprocess.Popen:
    """A long-lived Mooncake client holding a segment of ``segment_bytes``, on loopback over
    ``tcp``; over ``rdma`` it announces the host's own address and the HCAs of
    ``RDMA_DEVICES_ENV``."""
    host, devices = "127.0.0.1", ""
    if protocol == "rdma":
        host, devices = host_ip(), os.environ.get(RDMA_DEVICES_ENV, "")
    argv = [master_address, str(segment_bytes), protocol, host, devices]
    proc = subprocess.Popen(
        [sys.executable, "-c", _PROVIDER_SCRIPT, *argv],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    deadline = time.monotonic() + READY_TIMEOUT_S
    line = ""
    while time.monotonic() < deadline:
        line = proc.stdout.readline().strip()
        if line:
            break
    if line != "READY":
        kill_process_group(proc)
        pytest.fail(f"{protocol} segment provider did not come up: {line!r}")
    return proc


def skip_without_cluster_prerequisites() -> None:
    """A GPU for the engines, the Mooncake bindings and ``mooncake_master``; no model weights."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("the engines need a GPU")
    pytest.importorskip("mooncake.store")
    if not os.access(MASTER, os.X_OK):
        pytest.skip("mooncake_master binary not found")


def stop_cluster(
    master_address: str, master: subprocess.Popen, provider: subprocess.Popen | None
) -> None:
    """Kill the provider and the master, sweep what survived, and fail if anything did. The
    master address is unique to the test: the master carries its port, the provider carries the
    address; a Mooncake helper that survived would carry one of the two."""
    kill_process_group(provider)
    kill_process_group(master)
    rpc_port = master_address.rsplit(":", 1)[1]
    leftovers = leftover_pids(f"--rpc_port={rpc_port}") + leftover_pids(master_address)
    for pid in leftovers:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    assert not leftovers, f"mooncake processes of this test survived teardown: {leftovers}"


# ---------------------------------------------------------------------------------------------
# Fixtures and transports
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def mooncake_cluster(request):
    """A master and one segment provider on loopback TCP, killed at teardown whatever happened
    (``stop_cluster``). Needs a GPU, the Mooncake bindings and ``mooncake_master``; no model
    weights. The provider's segment is ``SEGMENT_BYTES`` unless the test parametrizes this
    fixture indirectly with another size."""
    skip_without_cluster_prerequisites()
    segment_bytes = getattr(request, "param", SEGMENT_BYTES)
    master_address, master = start_master()
    provider = None
    try:
        provider = start_segment_provider(master_address, segment_bytes)
        yield MooncakeCluster(master_address, master, provider)
    finally:
        stop_cluster(master_address, master, provider)


_needs_rdma = pytest.mark.skipif(
    os.environ.get(RDMA_ENV) != "1", reason=f"{RDMA_ENV}=1 only: needs an RDMA NIC"
)
TRANSPORTS = [
    pytest.param(("tcp", None), id="tcp"),
    pytest.param(("rdma", "device"), id="rdma_device", marks=_needs_rdma),
    pytest.param(("rdma", "host"), id="rdma_host", marks=_needs_rdma),
]
"""``(protocol, landing)`` for ``transport_store``; a ``None`` landing is the factory's default
(``host`` over TCP). Over RDMA the GPU KV pools are registered through dma-buf, which the Mooncake
wheel does only with ``WITH_NVIDIA_PEERMEM=0``; the tests set it unless the caller did."""


@pytest.fixture
def transport_store(request):
    """A master and one segment provider of ``TRANSPORT_SEGMENT_BYTES`` speaking the
    ``TRANSPORTS`` protocol given by indirect parametrization; yields
    ``(protocol, landing, master_address)``. Same prerequisites and teardown as
    ``mooncake_cluster``."""
    protocol, landing = request.param
    skip_without_cluster_prerequisites()
    master_address, master = start_master()
    provider = None
    try:
        provider = start_segment_provider(
            master_address, TRANSPORT_SEGMENT_BYTES, protocol=protocol
        )
        yield protocol, landing, master_address
    finally:
        stop_cluster(master_address, master, provider)

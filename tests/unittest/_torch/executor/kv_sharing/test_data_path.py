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
"""The data path every rank takes, from each rank's startup check, and the check itself over a
fake store client (its device half needs a GPU)."""

import re
import threading
import time
from types import SimpleNamespace

import pytest
from _torch.disaggregation.blob.fake_mooncake import FakeMooncakeStore

from tensorrt_llm._torch.disaggregation.base.memory import MemoryKind
from tensorrt_llm._torch.disaggregation.blob.mooncake_setup import MooncakeConfig
from tensorrt_llm._torch.pyexecutor.kv_sharing import data_path, startup
from tensorrt_llm._torch.pyexecutor.kv_sharing.data_path import choose_data_path

GET = "batch_get_into_multi_buffers"


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "requested, gpu_direct_errors, chosen",
    [
        ("auto", [None, None], MemoryKind.DEVICE),
        ("auto", [None, "no RDMA here"], MemoryKind.HOST),  # one rank keeps every rank off it
        ("host_staging", [None, None], MemoryKind.HOST),
    ],
)
def test_every_rank_takes_the_path_asked_for_or_the_one_auto_picks(
    requested, gpu_direct_errors, chosen, monkeypatch
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(data_path.logger, "warning", warnings.append)
    assert choose_data_path(requested, gpu_direct_errors) is chosen
    if chosen is MemoryKind.HOST and requested == "auto":
        assert len(warnings) == 1 and "rank 1: no RDMA here" in warnings[0]
    else:
        assert warnings == []


@pytest.mark.cpu_only
def test_gpu_direct_asked_for_in_vain_refuses_the_server() -> None:
    with pytest.raises(ValueError, match=re.escape("'gpu_direct' is not possible here: rank 1")):
        choose_data_path("gpu_direct", [None, "no RDMA here"])


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> FakeMooncakeStore:
    """The client ``open_store`` opens, on a process GPU Direct suits."""
    fake = FakeMooncakeStore()
    monkeypatch.setattr(data_path, "open_sdk", lambda config: fake)
    monkeypatch.setattr(data_path, "prepare_gpu_direct", lambda protocol: None)
    return fake


CONFIG = MooncakeConfig("127.0.0.1:50051", 0)


def check_thread_ended() -> bool:
    deadline = time.monotonic() + 5
    while any(t.name == "kv-sharing-store-check" for t in threading.enumerate()):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


def test_auto_takes_host_staging_when_the_device_round_trip_fails(store, monkeypatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(data_path.logger, "warning", warnings.append)
    store.script(GET, [None])  # the host get reads
    store.script(GET, [-600])  # the get into device memory fails
    client, lent, gpu_direct_error = data_path.open_store("auto", CONFIG, 0)
    assert client is store and lent is None
    assert choose_data_path("auto", [gpu_direct_error]) is MemoryKind.HOST
    (fallback,) = [warning for warning in warnings if "through host staging" in warning]
    assert "a round trip through device memory" in fallback and "INVALID_PARAMS" in fallback


def test_a_store_failing_the_host_round_trip_refuses_the_server(store) -> None:
    store.script(GET, [-900])  # RPC_FAIL
    with pytest.raises(ValueError, match="opened and checked: a round trip through host memory"):
        data_path.open_store("auto", CONFIG, 0)


@pytest.mark.cpu_only
def test_a_client_that_does_not_open_on_one_rank_refuses_the_server_on_every_rank(
    monkeypatch,
) -> None:
    """The rank whose client failed still reaches the gather, so every rank refuses alike."""

    def unreachable(config):
        raise TimeoutError("the master does not answer")

    monkeypatch.setattr(data_path, "open_sdk", unreachable)
    monkeypatch.setattr(data_path, "prepare_gpu_direct", lambda protocol: None)
    # Rank 0 opened its store and can take GPU Direct: _agree gathers (error, shared result).
    rank_1 = SimpleNamespace(world_size=2, allgather=lambda own: [(None, None), own])
    with pytest.raises(ValueError, match="rank 1: kv_sharing: the store could not be opened"):
        startup._agree(rank_1, lambda: (data_path.open_store("auto", CONFIG, 0), None))


def test_a_server_lends_its_memory_before_its_check_so_it_can_be_the_stores_only_memory(
    store,
) -> None:
    client, lent, _ = data_path.open_store("host_staging", CONFIG, 1 << 30)
    assert lent.segment_ids
    methods = [call.method for call in client.calls]
    assert methods.index("allocate_and_mount_segment") < methods.index("batch_put_session_start")
    store.mount_answer = -1
    with pytest.raises(ValueError, match="opened and checked: mooncake: lending 1073741824 bytes"):
        data_path.open_store("host_staging", CONFIG, 1 << 30)


def test_a_host_round_trip_that_does_not_answer_in_time_refuses_the_store(
    store, monkeypatch
) -> None:
    monkeypatch.setattr(data_path, "STORE_CHECK_TIMEOUT_S", 0.2)
    release = store.block("batch_put_session_start").release
    timeout = "a round trip through host memory: no answer within 0.2 s"
    try:
        with pytest.raises(RuntimeError, match=re.escape(timeout)):
            data_path.check_store(store, None)
    finally:
        release.set()
    assert check_thread_ended()


def test_a_device_round_trip_that_does_not_answer_in_time_rules_out_gpu_direct(
    store, monkeypatch
) -> None:
    monkeypatch.setattr(data_path, "STORE_CHECK_TIMEOUT_S", 0.2)
    release = threading.Event()
    monkeypatch.setattr(data_path, "_device_round_trip", lambda *args: release.wait(5))
    timeout = "a round trip through device memory: no answer within 0.2 s"
    try:
        assert data_path.check_store(store, None) == timeout
    finally:
        release.set()
    assert check_thread_ended()


def test_a_step_that_raises_before_the_round_trip_reports_its_own_text(store, monkeypatch) -> None:
    def no_device(device: int) -> None:
        raise RuntimeError("the device is gone")

    monkeypatch.setattr(data_path.torch.cuda, "set_device", no_device)
    with pytest.raises(RuntimeError, match="a round trip through host memory: the device is gone"):
        data_path.check_store(store, None)

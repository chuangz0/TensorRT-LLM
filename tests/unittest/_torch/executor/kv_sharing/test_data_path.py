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

import pytest
from _torch.disaggregation.blob.fake_mooncake import FakeMooncakeStore

from tensorrt_llm._torch.disaggregation.base.memory import MemoryKind
from tensorrt_llm._torch.disaggregation.blob.mooncake_setup import MooncakeConfig
from tensorrt_llm._torch.pyexecutor.kv_sharing import data_path
from tensorrt_llm._torch.pyexecutor.kv_sharing.data_path import (
    GPU_DIRECT,
    HOST_STAGING,
    PathCheck,
    choose_data_path,
)

PASSED = PathCheck(None, None)
NO_GPU_DIRECT = PathCheck(None, "no RDMA here")
GET = "batch_get_into_multi_buffers"


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "requested, checks, chosen",
    [
        ("auto", [PASSED, PASSED], GPU_DIRECT),
        ("auto", [PASSED, NO_GPU_DIRECT], HOST_STAGING),  # one rank keeps every rank off it
        ("host_staging", [PASSED, PASSED], HOST_STAGING),
    ],
)
def test_every_rank_takes_the_path_asked_for_or_the_one_auto_picks(
    requested, checks, chosen, monkeypatch
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(data_path.logger, "warning", warnings.append)
    assert choose_data_path(requested, checks) == chosen
    if chosen == HOST_STAGING and requested == "auto":
        assert len(warnings) == 1 and "rank 1: no RDMA here" in warnings[0]
    else:
        assert warnings == []


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "requested, checks, reason",
    [
        ("gpu_direct", [PASSED, NO_GPU_DIRECT], "'gpu_direct' is not possible here: rank 1"),
        ("host_staging", [PASSED, PathCheck("no answer", None)], "startup check on rank 1"),
    ],
)
def test_a_failed_store_or_gpu_direct_asked_for_in_vain_refuses_the_server(
    requested, checks, reason
) -> None:
    with pytest.raises(ValueError, match=re.escape(reason)):
        choose_data_path(requested, checks)


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
    memory_kind, driver = data_path.open_store("auto", CONFIG)
    assert memory_kind is MemoryKind.HOST and not driver.capabilities.device_memory
    (fallback,) = [warning for warning in warnings if "through host staging" in warning]
    assert "a round trip through device memory" in fallback and "INVALID_PARAMS" in fallback


def test_a_store_failing_the_host_round_trip_refuses_the_server(store) -> None:
    store.script(GET, [-900])  # RPC_FAIL
    with pytest.raises(ValueError, match="on rank 0: a round trip through host memory"):
        data_path.open_store("auto", CONFIG)


@pytest.mark.parametrize("half", ["host", "device"])
def test_a_half_of_the_check_that_does_not_answer_in_time_fails(store, monkeypatch, half) -> None:
    """A stalled host half is the store's failure; a stalled device half only GPU Direct's."""
    monkeypatch.setattr(data_path, "STORE_CHECK_TIMEOUT_S", 0.2)
    release = threading.Event()
    if half == "host":
        release = store.block("batch_put_session_start").release
    else:
        monkeypatch.setattr(data_path, "_device_round_trip", lambda *args: release.wait(5))
    try:
        check = data_path.check_store(store, None)
    finally:
        release.set()
    timeout = f"a round trip through {half} memory: no answer within 0.2 s"
    assert check == (PathCheck(timeout, None) if half == "host" else PathCheck(None, timeout))
    assert check_thread_ended()


def test_a_step_that_raises_before_the_round_trip_reports_its_own_text(store, monkeypatch) -> None:
    def no_device(device: int) -> None:
        raise RuntimeError("the device is gone")

    monkeypatch.setattr(data_path.torch.cuda, "set_device", no_device)
    check = data_path.check_store(store, None)
    assert check == PathCheck("a round trip through host memory: the device is gone", None)

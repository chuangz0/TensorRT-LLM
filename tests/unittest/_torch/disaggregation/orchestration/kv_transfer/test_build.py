# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``build_coordinator``: the config's timeouts and the plan authority reach the coordinator and
its planner. Import-light like the rest of this suite."""

import pytest

__extra_import_path__ = ["~/tensorrt_llm/_torch"]
from disaggregation.backends.config import BackendEntry, KVTransferConfig  # noqa: E402
from disaggregation.backends.registry import BackendHandle  # noqa: E402
from disaggregation.orchestration.kv_transfer.build import build_coordinator  # noqa: E402
from disaggregation.orchestration.kv_transfer.interfaces import PlanAuthority  # noqa: E402
from disaggregation.remote_cache import FetchPlan  # noqa: E402
from fakes import (  # noqa: E402
    TPB,
    FakeDist,
    FakeEngineQueue,
    FakeFetches,
    FakePublishes,
    FakeReader,
    FakeRequest,
    RecordingEffects,
    full_attention,
)

pytestmark = pytest.mark.cpu_only


def build(*, store: FakeFetches, plan_authority: PlanAuthority = PlanAuthority.VOTED):
    reader = FakeReader(groups=[full_attention(0)], tokens_per_block=TPB)
    handle = BackendHandle(
        name="store",
        hint_key=None,
        fetcher=store,
        publisher=FakePublishes(),
        pool_registrar=None,
        close=lambda: None,
    )
    config = KVTransferConfig(
        backends=(BackendEntry.from_dict({"name": "store", "type": "fake"}),),
        probe_timeout_s=0.05,
        fetch_timeout_s=12.5,
        unlaunched_timeout_s=7.5,
    )
    coordinator = build_coordinator(
        config,
        [handle],
        reader,
        RecordingEffects(),
        FakeEngineQueue(),
        FakeDist(),
        plan_authority=plan_authority,
    )
    return coordinator, reader


def test_plan_authority_reaches_the_coordinator():
    coordinator, _ = build(
        store=FakeFetches(name="store", single_destination=True),
        plan_authority=PlanAuthority.FOLLOWER,
    )
    assert coordinator.plan_authority is PlanAuthority.FOLLOWER
    assert coordinator.status_dump()["plan_authority"] == "FOLLOWER"


def test_answered_probe_plans_from_the_store():
    store = FakeFetches(name="store", single_destination=True)
    coordinator, reader = build(store=store)
    req = FakeRequest(1, prompt_len=29)
    store.probe_default = reader.unit_names(req, range(7))
    coordinator.advance([req], 0.0)
    plan = coordinator.plan_fetch(req)
    assert isinstance(plan, FetchPlan)
    assert plan.token_end == 28 and plan.source == "store"

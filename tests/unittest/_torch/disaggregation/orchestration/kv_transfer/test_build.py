# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``build_coordinator``: the config's timeouts and the plan authority reach the coordinator and
its planner. Import-light like the rest of this suite."""

import pytest

__extra_import_path__ = ["~/tensorrt_llm/_torch"]
from disaggregation.backends.config import BackendEntry, KVTransferConfig  # noqa: E402
from disaggregation.backends.registry import BackendHandle  # noqa: E402
from disaggregation.orchestration.kv_transfer.build import build_coordinator  # noqa: E402
from disaggregation.orchestration.kv_transfer.engine_protocols import PlanAuthority  # noqa: E402
from disaggregation.remote_cache import FetchPlan  # noqa: E402
from fakes import (  # noqa: E402
    TPB,
    FakeEngineQueue,
    FakeFetches,
    FakeLandsOnHost,
    FakePublishes,
    FakeReader,
    FakeRequest,
    RecordingEffects,
    SingleRankCollective,
    full_attention,
)

pytestmark = pytest.mark.cpu_only


def build(*, store: FakeFetches, plan_authority: PlanAuthority = PlanAuthority.ALL_RANKS):
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
        SingleRankCollective(),
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
    plan = coordinator.fetch_answer(req)
    assert isinstance(plan, FetchPlan)
    assert plan.token_end == 28 and plan.source == "store"


def test_a_host_landing_backend_with_a_hint_key_is_refused_at_assembly():
    """A ``LandsOnHost`` backend is a store with one destination: it takes no routing hint."""
    reader = FakeReader(groups=[full_attention(0)], tokens_per_block=TPB)
    handle = BackendHandle(
        name="host-store",
        hint_key="ctx",
        fetcher=FakeLandsOnHost(name="host-store"),
        publisher=None,
        pool_registrar=None,
        close=lambda: None,
        landing="host",
    )
    config = KVTransferConfig(
        backends=(BackendEntry.from_dict({"name": "host-store", "type": "fake"}),)
    )
    with pytest.raises(ValueError, match="takes no hint_key"):
        build_coordinator(
            config, [handle], reader, RecordingEffects(), FakeEngineQueue(), SingleRankCollective()
        )

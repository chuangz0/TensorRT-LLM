# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The assembly's scope guard (integration plan §1 item 7) and the coordinator it builds.

``check_engine_supports_kv_transfer`` refuses, with a reason, every engine configuration this
feature does not host; ``attach_kv_transfer`` refuses a malformed config file before it builds
anything; ``_build_coordinator`` hands the planner the wall-clock probe budget alone.
"""

import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from engine_fakes import (
    TPB,
    FakeFetches,
    FakeKVCacheManager,
    FakePublishes,
    FakeReader,
    make_request,
)

from tensorrt_llm._torch.disaggregation.backends.config import BackendEntry, KVTransferConfig
from tensorrt_llm._torch.disaggregation.backends.registry import BackendHandle
from tensorrt_llm._torch.disaggregation.orchestration.kv_transfer.interfaces import DEFER
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_transfer import assembly
from tensorrt_llm._torch.pyexecutor.kv_transfer.assembly import (
    attach_kv_transfer,
    check_engine_supports_kv_transfer,
)
from tensorrt_llm._torch.pyexecutor.kv_transfer.effects import (
    EngineRequestView,
    PyExecutorKVTransferEffects,
)
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler

pytestmark = pytest.mark.cpu_only


def in_scope_executor() -> PyExecutor:
    executor = object.__new__(PyExecutor)
    kv_cache_manager = object.__new__(KVCacheManagerV2)
    kv_cache_manager.enable_block_reuse = True
    executor.kv_cache_manager = kv_cache_manager
    executor.draft_kv_cache_manager = None
    executor.scheduler = object.__new__(KVCacheV2Scheduler)
    executor.enable_kv_pool_rebalance = False
    return executor


def in_scope_mapping(**overrides) -> SimpleNamespace:
    mapping = SimpleNamespace(tp_size=1, pp_size=1, cp_size=1, enable_attention_dp=False)
    for key, value in overrides.items():
        setattr(mapping, key, value)
    return mapping


def guard(executor, mapping=None, **overrides):
    kwargs = dict(spec_config=None, kv_connector_manager=None, max_beam_width=1)
    kwargs.update(overrides)
    check_engine_supports_kv_transfer(
        executor, mapping=mapping if mapping is not None else in_scope_mapping(), **kwargs
    )


def test_in_scope_engine_passes():
    guard(in_scope_executor())


def _v1_manager():
    executor = in_scope_executor()
    executor.kv_cache_manager = Mock(enable_block_reuse=True)
    return executor


def _no_reuse():
    executor = in_scope_executor()
    executor.kv_cache_manager.enable_block_reuse = False
    return executor


def _draft_manager():
    executor = in_scope_executor()
    executor.draft_kv_cache_manager = Mock()
    return executor


def _v1_scheduler():
    executor = in_scope_executor()
    executor.scheduler = Mock()
    return executor


def _pool_rebalance():
    # Rebalancing moves pages between pools while a store backend may still be reading them.
    executor = in_scope_executor()
    executor.enable_kv_pool_rebalance = True
    return executor


@pytest.mark.parametrize(
    "make_executor, mapping, overrides, reason",
    [
        (_v1_manager, None, {}, "kv_cache_config.use_kv_cache_manager_v2=True is required"),
        (_no_reuse, None, {}, "kv_cache_config.enable_block_reuse=True is required"),
        (in_scope_executor, in_scope_mapping(tp_size=2), {}, "TP=PP=CP=1 is required, got tp=2"),
        (in_scope_executor, in_scope_mapping(pp_size=2), {}, "TP=PP=CP=1 is required.*pp=2"),
        (in_scope_executor, in_scope_mapping(cp_size=2), {}, "TP=PP=CP=1 is required.*cp=2"),
        (
            in_scope_executor,
            in_scope_mapping(enable_attention_dp=True),
            {},
            "attention data parallelism is not supported",
        ),
        (in_scope_executor, None, {"spec_config": object()}, "speculative decoding"),
        (in_scope_executor, None, {"kv_connector_manager": object()}, "a KV connector is attached"),
        (_draft_manager, None, {}, "a draft KV cache manager is present"),
        (in_scope_executor, None, {"max_beam_width": 2}, "beam search is not supported"),
        (_v1_scheduler, None, {}, "the scheduler must be KVCacheV2Scheduler, got Mock"),
        (
            _pool_rebalance,
            None,
            {},
            "kv_cache_config.enable_kv_pool_rebalance=True is not supported",
        ),
    ],
    ids=[
        "v1_manager",
        "no_reuse",
        "tp",
        "pp",
        "cp",
        "adp",
        "spec",
        "connector",
        "draft",
        "beam",
        "scheduler",
        "pool_rebalance",
    ],
)
def test_every_out_of_scope_condition_is_refused_with_its_reason(
    make_executor, mapping, overrides, reason
):
    with pytest.raises(ValueError, match="cannot host them: " + reason):
        guard(make_executor(), mapping, **overrides)


def test_attach_refuses_a_malformed_config_before_building_anything(tmp_path):
    config_path = tmp_path / "kv_transfer.yaml"
    config_path.write_text("probe_timeout: 1\nbackends:\n  - {name: a, type: mooncake}\n")
    executor = in_scope_executor()
    with pytest.raises(ValueError, match="unknown kv transfer config keys"):
        attach_kv_transfer(
            executor,
            str(config_path),
            mapping=in_scope_mapping(),
            spec_config=None,
            kv_connector_manager=None,
            max_beam_width=1,
        )
    assert executor.kv_transfer is None
    assert not hasattr(executor.scheduler, "kv_transfer_planner")


def test_attach_refuses_an_out_of_scope_engine_before_reading_the_config(tmp_path):
    with pytest.raises(ValueError, match="beam search"):
        attach_kv_transfer(
            in_scope_executor(),
            str(tmp_path / "does-not-exist.yaml"),
            mapping=in_scope_mapping(),
            spec_config=None,
            kv_connector_manager=None,
            max_beam_width=4,
        )


def test_status_dump_path_replaces_pid(monkeypatch):
    import os

    monkeypatch.setenv(assembly.KV_TRANSFER_STATUS_DUMP_ENV, "/x/kvt-{pid}.json")
    assert assembly._status_dump_path() == f"/x/kvt-{os.getpid()}.json"
    monkeypatch.delenv(assembly.KV_TRANSFER_STATUS_DUMP_ENV)
    assert assembly._status_dump_path() is None


class TestAssembledCoordinator:
    """``_build_coordinator``: the planner gets ``probe_budget_rounds=None`` and the config's
    ``probe_timeout_s``; a request whose probe is never answered is deferred until the wall-clock
    budget runs out, then computed locally."""

    def _build(self, *, probe_timeout_s: float, store: FakeFetches):
        kv = FakeKVCacheManager(TPB)
        executor = object.__new__(PyExecutor)
        executor.kv_cache_manager = kv
        reader = FakeReader(kv)
        effects = PyExecutorKVTransferEffects(executor)
        publisher = FakePublishes()
        handle = BackendHandle(
            name="store",
            hint_key=None,
            fetcher=store,
            publisher=publisher,
            pool_registrar=None,
            close=lambda: None,
        )
        config = KVTransferConfig(
            backends=(BackendEntry.from_dict({"name": "store", "type": "fake"}),),
            probe_timeout_s=probe_timeout_s,
            fetch_timeout_s=12.5,
        )
        coordinator = assembly._build_coordinator(config, [handle], reader, effects)
        return coordinator, reader

    def test_planner_budget_is_the_wall_clock_alone(self):
        coordinator, _ = self._build(probe_timeout_s=0.05, store=FakeFetches(probe_answer=None))
        planner = coordinator._planner
        assert planner._probe_budget_rounds is None
        assert planner._probe_timeout_s == 0.05
        assert planner._clock is time.monotonic  # plan §10 #15: one clock with ``advance``
        assert coordinator._fetch_timeout_s == 12.5 and coordinator._publish_timeout_s is None

    def test_answered_probe_plans_within_the_budget(self):
        coordinator, _ = self._build(probe_timeout_s=0.05, store=FakeFetches(probe_answer="all"))
        view = EngineRequestView(make_request(1, 100))
        coordinator.advance([view], time.monotonic())
        plan = coordinator.plan_fetch(view)
        assert plan is not None and plan is not DEFER
        assert plan.token_end == 96 and plan.source == "store"

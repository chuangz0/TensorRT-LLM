# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The assembly's scope guard and the coordinator it builds.

``check_engine_supports_kv_transfer`` refuses, with a reason, every engine configuration this
feature does not host; ``attach_kv_transfer`` refuses a malformed config file before it builds
anything; ``install_log_forwarding`` puts one forwarding handler on the layers' stdlib logger.
"""

import inspect
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from engine_fakes import FakeFetches
from transformers import PretrainedConfig

from tensorrt_llm._torch.disaggregation.backends.registry import BackendHandle
from tensorrt_llm._torch.disaggregation.base.backend import CacheKind
from tensorrt_llm._torch.disaggregation.base.views import GroupSpec
from tensorrt_llm._torch.disaggregation.orchestration.kv_transfer.engine_protocols import (
    PlanAuthority,
)
from tensorrt_llm._torch.disaggregation.resource.region import layout_fingerprint
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_transfer import assembly
from tensorrt_llm._torch.pyexecutor.kv_transfer.assembly import (
    attach_kv_transfer,
    check_engine_supports_kv_transfer,
    model_identity_for,
    plan_authority_for,
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
    mapping = SimpleNamespace(
        rank=0, tp_size=1, pp_size=1, pp_rank=0, cp_size=1, enable_attention_dp=False
    )
    for key, value in overrides.items():
        setattr(mapping, key, value)
    return mapping


def guard(executor, mapping=None, **overrides):
    kwargs = dict(spec_config=None, kv_connector_manager=None, max_beam_width=1)
    kwargs.update(overrides)
    check_engine_supports_kv_transfer(
        executor, mapping=mapping if mapping is not None else in_scope_mapping(), **kwargs
    )


@pytest.mark.parametrize(
    "mapping",
    [
        in_scope_mapping(),
        in_scope_mapping(tp_size=2),
        in_scope_mapping(tp_size=4, enable_attention_dp=True),
        in_scope_mapping(pp_size=2),
        in_scope_mapping(tp_size=2, pp_size=2, enable_attention_dp=True),
    ],
    ids=["single_rank", "tp", "adp", "pp", "adp_pp"],
)
def test_in_scope_engine_passes(mapping):
    guard(in_scope_executor(), mapping)


@pytest.mark.parametrize(
    "mapping, authority",
    [
        (in_scope_mapping(), PlanAuthority.ALL_RANKS),
        (in_scope_mapping(rank=1, tp_size=2), PlanAuthority.ALL_RANKS),
        (in_scope_mapping(pp_size=2), PlanAuthority.OWNER),
        (in_scope_mapping(rank=1, pp_size=2, pp_rank=1), PlanAuthority.FOLLOWER),
        # The owner's TP peer receives the schedule by tp_broadcast: a follower.
        (in_scope_mapping(rank=1, tp_size=2, pp_size=2), PlanAuthority.FOLLOWER),
        # Under attention DP every first PP rank schedules for its own replica.
        (
            in_scope_mapping(rank=1, tp_size=2, pp_size=2, enable_attention_dp=True),
            PlanAuthority.OWNER,
        ),
        (
            in_scope_mapping(rank=3, tp_size=2, pp_size=2, pp_rank=1, enable_attention_dp=True),
            PlanAuthority.FOLLOWER,
        ),
    ],
    ids=[
        "single_rank",
        "tp_peer_votes",
        "pp_rank0_owns",
        "pp_rank1_follows",
        "tp_peer_of_owner_follows",
        "adp_first_pp_rank_owns",
        "adp_second_pp_rank_follows",
    ],
)
def test_plan_authority_follows_the_rank_that_schedules(mapping, authority):
    """Mirrors the owner rule of ``_pp_schedule_and_propagate``: rank 0, or under attention DP
    with TP>1 every first pipeline rank, runs ``_schedule``; everyone else follows."""
    assert plan_authority_for(mapping) is authority


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
        (in_scope_executor, in_scope_mapping(cp_size=2), {}, "CP=1 is required, got cp=2"),
        (
            in_scope_executor,
            in_scope_mapping(tp_size=2, pp_size=2, cp_size=2),
            {},
            "CP=1 is required",  # TP and PP are hosted; CP alone is refused
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
        "cp",
        "cp_with_tp_pp",
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


def _reader_with(*specs: GroupSpec) -> SimpleNamespace:
    return SimpleNamespace(group_specs=lambda: specs)


def test_layer_group_guard_admits_paged_groups_and_refuses_state_groups():
    """Full attention and sliding windows (with or without sink blocks) are fetched and published
    per group; a recurrent group has no published snapshot to fetch."""
    full = GroupSpec(0, CacheKind.PAGED, b"\0" * 8)
    windowed = GroupSpec(1, CacheKind.PAGED, b"\1" * 8, window_size=128, sink_blocks=1)
    state = GroupSpec(2, CacheKind.STATE, b"\2" * 8)
    assembly._check_layer_groups_are_paged(_reader_with(full, windowed))
    with pytest.raises(ValueError, match="SSM/recurrent layer groups are not supported"):
        assembly._check_layer_groups_are_paged(_reader_with(full, state))


def _handle(name: str, hint_key, closed: list) -> BackendHandle:
    return BackendHandle(
        name=name,
        hint_key=hint_key,
        fetcher=FakeFetches(),
        publisher=None,
        pool_registrar=None,
        close=lambda: closed.append(name),
    )


def test_pp_refuses_a_routed_backend():
    """A follower rebuilds plans from ``(token_end, source)``; a routed backend's plan also
    needs the request's hint, which the schedule does not carry."""
    handles = [_handle("store", None, []), _handle("worker", "ctx", [])]
    with pytest.raises(ValueError, match="PP>1 with a routed \\(hint\\) backend"):
        assembly._check_followers_can_rebuild_plans(in_scope_mapping(pp_size=2), handles)
    assembly._check_followers_can_rebuild_plans(in_scope_mapping(pp_size=1), handles)
    assembly._check_followers_can_rebuild_plans(in_scope_mapping(pp_size=2), handles[:1])


def executor_with_model(pretrained_config) -> SimpleNamespace:
    """The attribute path ``model_identity_for`` walks on a real executor."""
    model_config = SimpleNamespace(pretrained_config=pretrained_config)
    return SimpleNamespace(
        model_engine=SimpleNamespace(model=SimpleNamespace(model_config=model_config))
    )


def test_model_identity_names_the_checkpoint_by_path_or_by_revision():
    """Without a commit hash the identity is the path as given (per path); with one, the last
    path component plus the hash, the same on every node whatever its ``HF_HOME``."""
    config = PretrainedConfig(architectures=["LlamaForCausalLM"])
    config._name_or_path = "meta-llama/Llama-3.1-8B"
    assert model_identity_for(executor_with_model(config)) == "meta-llama/Llama-3.1-8B"
    config._commit_hash = "0e9e39f"
    assert model_identity_for(executor_with_model(config)) == "Llama-3.1-8B@0e9e39f"

    def snapshot_under(hf_home: str) -> PretrainedConfig:
        snapshot = PretrainedConfig(architectures=["LlamaForCausalLM"])
        snapshot._name_or_path = (
            f"{hf_home}/hub/models--meta-llama--Llama-3.1-8B/snapshots/0e9e39f/"
        )
        snapshot._commit_hash = "0e9e39f"
        return snapshot

    node_a = model_identity_for(executor_with_model(snapshot_under("/home/a/.cache/huggingface")))
    node_b = model_identity_for(executor_with_model(snapshot_under("/srv/hf")))
    assert node_a == node_b == "0e9e39f@0e9e39f"


def test_model_identity_without_a_name_digests_the_config():
    """No checkpoint path: the architecture plus a digest of the config tells two models apart,
    and ignores the keys that change without the model changing."""
    small = PretrainedConfig(architectures=["LlamaForCausalLM"], hidden_size=16)
    large = PretrainedConfig(architectures=["LlamaForCausalLM"], hidden_size=32)
    small_again = PretrainedConfig(architectures=["LlamaForCausalLM"], hidden_size=16)
    small_again.transformers_version = "0.0.0"
    identities = [model_identity_for(executor_with_model(c)) for c in (small, large, small_again)]
    assert all(identity.startswith("LlamaForCausalLM#") for identity in identities)
    assert identities[0] != identities[1] and identities[0] == identities[2]


def test_model_identity_is_unknown_on_an_engine_without_a_model_config():
    assert model_identity_for(in_scope_executor()) == ""


def test_two_model_identities_give_two_layout_fingerprints_through_the_assembly_path():
    """What the assembly feeds ``layout_fingerprint`` keeps two models of one KV geometry apart
    in the store; the page table and manager here are the minimum the digest reads."""
    page_table = SimpleNamespace(
        tokens_per_block=32,
        pool_groups=[SimpleNamespace(pools=[SimpleNamespace(slot_bytes=4096)])],
        layer_groups=[
            SimpleNamespace(
                pool_group_idx=0,
                kv_head_num_per_rank=8,
                sliding_window_size=None,
                local_layers=[SimpleNamespace(global_layer_id=0)],
                pool_views=[
                    SimpleNamespace(
                        pool_idx=0, pool_role=("key", "value"), buffer_entries=np.zeros(2, np.int64)
                    )
                ],
            )
        ],
    )
    manager = SimpleNamespace(dtype="bf16", head_dim=128)

    def fingerprint_for(name: str) -> bytes:
        config = PretrainedConfig(architectures=["LlamaForCausalLM"])
        config._name_or_path = name
        identity = model_identity_for(executor_with_model(config))
        return layout_fingerprint(manager, page_table, model_identity=identity)

    assert fingerprint_for("meta-llama/Llama-3.1-8B") != fingerprint_for("mistralai/Mistral-7B")
    assert fingerprint_for("meta-llama/Llama-3.1-8B") == fingerprint_for("meta-llama/Llama-3.1-8B")
    assert "model_identity=model_identity" in inspect.getsource(assembly._build_backend_context)
    assert "model=%r" in inspect.getsource(assembly.attach_kv_transfer)


def test_the_failure_guard_closes_the_built_backends_and_nothing_on_success():
    closed = []
    handles = [_handle("store", None, closed), _handle("worker", "ctx", closed)]

    def refuse():
        raise ValueError("coordinator refused")

    with pytest.raises(ValueError, match="coordinator refused"):
        with assembly._closing_backends_on_failure(handles):
            refuse()
    assert closed == ["store", "worker"]
    with assembly._closing_backends_on_failure(handles):
        pass
    assert closed == ["store", "worker"]


class _FakeReader:
    """The slice of ``KVv2ResourceView`` the assembly reads before and after ``build_backends``."""

    tokens_per_block = 32

    def __init__(self, *args) -> None:
        pass

    def group_specs(self):
        return (GroupSpec(0, CacheKind.PAGED, b"\0" * 8),)


class _FakeResolver:
    def __init__(self, *args) -> None:
        pass

    def pool_memory_spans(self):
        return [(0x1000, 4096)]

    def max_unit_bytes(self) -> int:
        return 4096

    def __call__(self, local_group: int, local: int):
        return [(0x1000, 4096)]


def attachable_executor() -> PyExecutor:
    """An in-scope executor with the attributes the assembly reads past the scope guard."""
    executor = in_scope_executor()
    executor.device_id = 0
    executor.max_seq_len = 1024
    executor.dist = SimpleNamespace(allgather=None, pp_allgather=None)
    return executor


def attach_over_fakes(monkeypatch, tmp_path, executor, *, build_backends, build_coordinator):
    """Run ``attach_kv_transfer`` with every KV v2 prerequisite faked, so that what happens
    around ``build_backends`` is the real assembly code."""
    monkeypatch.setattr(assembly, "build_page_table_from_manager", lambda manager: object())
    monkeypatch.setattr(assembly, "KVv2ResourceView", _FakeReader)
    monkeypatch.setattr(assembly, "KVv2RegionResolver", _FakeResolver)
    monkeypatch.setattr(assembly, "layout_fingerprint", lambda *args, **kwargs: b"\xfe" * 16)
    monkeypatch.setattr(assembly, "parallel_shard_tag", lambda mapping: "heads=all")
    monkeypatch.setattr(assembly, "build_backends", build_backends)
    monkeypatch.setattr(assembly, "build_coordinator", build_coordinator)
    config_path = tmp_path / "kv_transfer.yaml"
    config_path.write_text("backends:\n  - {name: store, type: fake}\n")
    attach_kv_transfer(
        executor,
        str(config_path),
        mapping=in_scope_mapping(world_size=1),  # ``EngineCollective`` sizes its group from this
        spec_config=None,
        kv_connector_manager=None,
        max_beam_width=1,
    )


def test_a_build_backends_that_raises_closes_nothing(monkeypatch, tmp_path):
    """Nothing was built: there is nothing to close, and the error is the backend's own."""
    calls = []

    def build_backends(config, build_context):
        calls.append("build raised")
        raise ValueError("store unreachable")

    def never(*args, **kwargs):
        raise AssertionError("the coordinator must not be built after a failed build")

    executor = attachable_executor()
    with pytest.raises(ValueError, match="store unreachable"):
        attach_over_fakes(
            monkeypatch, tmp_path, executor, build_backends=build_backends, build_coordinator=never
        )
    assert calls == ["build raised"]
    assert executor.kv_transfer is None and not hasattr(executor.scheduler, "kv_transfer")


def test_a_failure_after_build_backends_closes_what_was_built(monkeypatch, tmp_path):
    """The backends were built; a later step (here the coordinator) refusing must close them
    before the error leaves ``attach_kv_transfer``, and the executor keeps no hooks."""
    closed = []
    handles = [_handle("store", None, closed), _handle("worker", None, closed)]

    def refuse(*args, **kwargs):
        raise ValueError("coordinator refused")

    executor = attachable_executor()
    with pytest.raises(ValueError, match="coordinator refused"):
        attach_over_fakes(
            monkeypatch,
            tmp_path,
            executor,
            build_backends=lambda config, build_context: handles,
            build_coordinator=refuse,
        )
    assert closed == ["store", "worker"]
    assert executor.kv_transfer is None and not hasattr(executor.scheduler, "kv_transfer")


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
    assert not hasattr(executor.scheduler, "kv_transfer")


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


def test_two_attaches_install_one_forwarding_handler(monkeypatch):
    """The coordinator and the backends log through stdlib ``logging`` (they do not import
    ``tensorrt_llm``); the assembly forwards their WARNING+ records to the TRT-LLM logger with
    one handler however many engines are assembled in the process."""
    import logging

    from tensorrt_llm.logger import logger as trtllm_logger

    forwarded = Mock()
    monkeypatch.setattr(trtllm_logger, "warning", forwarded)
    namespace = logging.getLogger("tensorrt_llm._torch.disaggregation")
    installed_before = [
        h for h in namespace.handlers if isinstance(h, assembly._ForwardToTrtllmLogger)
    ]
    for handler in installed_before:
        namespace.removeHandler(handler)
    try:
        assembly.install_log_forwarding()
        assembly.install_log_forwarding()
        forwarders = [
            h for h in namespace.handlers if isinstance(h, assembly._ForwardToTrtllmLogger)
        ]
        assert len(forwarders) == 1
        layer_logger = logging.getLogger(
            "tensorrt_llm._torch.disaggregation.orchestration.kv_transfer.coordinator"
        )
        layer_logger.warning("probe on %s failed, answer stays pending: %s", "store", "down")
        forwarded.assert_called_once_with(
            "kv transfer: probe on store failed, answer stays pending: down"
        )
        layer_logger.info("not forwarded: below WARNING")
        assert forwarded.call_count == 1
    finally:
        for handler in list(namespace.handlers):
            if isinstance(handler, assembly._ForwardToTrtllmLogger):
                namespace.removeHandler(handler)
        for handler in installed_before:
            namespace.addHandler(handler)


def test_status_dump_path_replaces_pid(monkeypatch):
    import os

    monkeypatch.setenv(assembly.KV_TRANSFER_STATUS_DUMP_ENV, "/x/kvt-{pid}.json")
    assert assembly._status_dump_path() == f"/x/kvt-{os.getpid()}.json"
    monkeypatch.delenv(assembly.KV_TRANSFER_STATUS_DUMP_ENV)
    assert assembly._status_dump_path() is None

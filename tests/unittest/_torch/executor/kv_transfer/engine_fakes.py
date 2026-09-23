# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fakes for the engine-side KV transfer tests, imported through ``tensorrt_llm``.

``orchestration/kv_transfer/fakes.py`` imports the coordination layer as ``disaggregation.*`` so
that suite runs without ``tensorrt_llm``. The engine-side modules under test here
(``pyexecutor/kv_transfer/``) import it as ``tensorrt_llm._torch.disaggregation.*``, and one
contract must not exist twice in a process. These are the same table-backed fakes, trimmed to what
the engine hooks exercise, over the ``tensorrt_llm`` copy of the contract.
"""

from __future__ import annotations

import hashlib
import threading
from collections import deque
from types import SimpleNamespace
from typing import Iterable, Sequence
from unittest.mock import Mock

from tensorrt_llm._torch.disaggregation.base.backend import CacheKind
from tensorrt_llm._torch.disaggregation.base.cache_backend import (
    CacheExtent,
    Delivered,
    Outcome,
    SubmissionRejected,
    Unit,
)
from tensorrt_llm._torch.disaggregation.base.views import GroupSpec
from tensorrt_llm._torch.disaggregation.resource.naming import group_tag
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm.bindings import SamplingConfig

TPB = 32


def full_attention(local_group: int = 0) -> GroupSpec:
    return GroupSpec(local_group, CacheKind.PAGED, group_tag(frozenset({"full"}), None))


def block_key(seed: str, ordinal: int) -> bytes:
    return hashlib.blake2b(f"{seed}:{ordinal}".encode(), digest_size=8).digest()


def make_request(request_id: int, prompt_len: int = 100, *, max_new_tokens: int = 4) -> LlmRequest:
    """A real ``LlmRequest`` in ``CONTEXT_INIT`` as its first context chunk."""
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=max_new_tokens,
        input_tokens=[(request_id * 7919 + i * 13) % 32000 for i in range(prompt_len)],
        sampling_config=SamplingConfig(1),
        is_streaming=False,
    )


def finish_prefill(request: LlmRequest) -> None:
    """Move the context cursor to the end of the prompt, as one full context step does."""
    request.context_chunk_size = request.context_remaining_length
    request.move_to_next_context_chunk()
    assert request.context_remaining_length == 0


def extent_names(extent: CacheExtent) -> frozenset[bytes]:
    return frozenset(u.name for u in extent.units)


# ---------------------------------------------------------------------------------------------
# Backend fakes
# ---------------------------------------------------------------------------------------------


class FakeAttempt:
    def __init__(self, payload: object, outcome: Outcome | None = None) -> None:
        self.payload = payload
        self._outcome = outcome

    def poll(self) -> Outcome | None:
        return self._outcome

    def finish(self, outcome: Outcome) -> None:
        self._outcome = outcome

    def deliver_all(self) -> None:
        self.finish(Delivered(extent_names(self.payload)))

    def deliver_all_but(self, *missing: bytes) -> None:
        self.finish(Delivered(extent_names(self.payload) - frozenset(missing)))


class FakeFetches:
    """A store: ``probe`` answers ``probe_answer`` (``"all"`` = every unit asked about, ``None`` =
    unanswered, or a frozenset); ``fetch`` returns a pending attempt the test finishes."""

    def __init__(self, *, name: str = "store", probe_answer="all") -> None:
        self.name = name
        self.probe_answer = probe_answer
        self.calls: list[tuple[str, tuple]] = []
        self.attempts: list[FakeAttempt] = []
        self.quiesce_answers: deque[bool] = deque()
        self.reject_next = 0

    def fetch(self, extent: CacheExtent, *, route=None) -> FakeAttempt:
        self.calls.append(("fetch", (extent, route)))
        if self.reject_next > 0:
            self.reject_next -= 1
            raise SubmissionRejected(f"{self.name} rejected")
        attempt = FakeAttempt(extent)
        self.attempts.append(attempt)
        return attempt

    def quiesce(self, attempts: Iterable) -> bool:
        attempts = tuple(attempts)
        answer = self.quiesce_answers.popleft() if self.quiesce_answers else True
        self.calls.append(("quiesce", (attempts, answer)))
        return answer

    def settle(self, attempts: Iterable) -> None:
        self.calls.append(("settle", (tuple(attempts),)))

    def probe(self, name: bytes, units: Sequence[bytes]):
        self.calls.append(("probe", (name, tuple(units))))
        if self.probe_answer == "all":
            return frozenset(units)
        return self.probe_answer

    def open_route(self, hint):
        raise NotImplementedError(f"{self.name} has a single destination")

    def count(self, method: str) -> int:
        return sum(1 for m, _ in self.calls if m == method)


class FakeLanding:
    """A ``Landing`` the test finishes; ``place`` returns a pending attempt; ``releases`` counts."""

    def __init__(self, backend: FakeLandsOnHost, units: Sequence[bytes]) -> None:
        self.backend = backend
        self.units = tuple(units)
        self._outcome: Outcome | None = None
        self.releases = 0

    def poll(self) -> Outcome | None:
        return self._outcome

    def finish(self, outcome: Outcome) -> None:
        self._outcome = outcome

    def deliver_all(self) -> None:
        self.finish(Delivered(frozenset(self.units)))

    def place(self, extent: CacheExtent) -> FakeAttempt:
        self.backend.calls.append(("place", (self, extent)))
        attempt = FakeAttempt(extent)
        self.backend.attempts.append(attempt)
        return attempt

    def release(self) -> None:
        self.releases += 1


class FakeLandsOnHost:
    """A store that lands in its own memory first (``LandsOnHost``); ``probe`` answers every
    unit asked about, ``fetch_to_host`` returns a pending landing the test finishes."""

    def __init__(self, *, name: str = "host") -> None:
        self.name = name
        self.calls: list[tuple[str, tuple]] = []
        self.landings: list[FakeLanding] = []
        self.attempts: list[FakeAttempt] = []

    def fetch_to_host(self, name: bytes, units: Sequence[bytes]) -> FakeLanding:
        self.calls.append(("fetch_to_host", (name, tuple(units))))
        landing = FakeLanding(self, units)
        self.landings.append(landing)
        return landing

    def quiesce(self, attempts: Iterable) -> bool:
        self.calls.append(("quiesce", (tuple(attempts), True)))
        return True

    def settle(self, attempts: Iterable) -> None:
        self.calls.append(("settle", (tuple(attempts),)))

    def probe(self, name: bytes, units: Sequence[bytes]):
        self.calls.append(("probe", (name, tuple(units))))
        return frozenset(units)

    def count(self, method: str) -> int:
        return sum(1 for m, _ in self.calls if m == method)


class FakePublishes:
    def __init__(self, *, name: str = "pub") -> None:
        self.name = name
        self.calls: list[tuple[str, tuple]] = []
        self.attempts: list[FakeAttempt] = []
        self.reject_next = 0

    def publish(self, extent: CacheExtent) -> FakeAttempt:
        self.calls.append(("publish", (extent,)))
        if self.reject_next > 0:
            self.reject_next -= 1
            raise SubmissionRejected(f"{self.name} rejected publish")
        attempt = FakeAttempt(extent)
        self.attempts.append(attempt)
        return attempt

    def quiesce(self, attempts: Iterable) -> bool:
        self.calls.append(("quiesce", (tuple(attempts),)))
        return True

    def settle(self, attempts: Iterable) -> None:
        self.calls.append(("settle", (tuple(attempts),)))

    def count(self, method: str) -> int:
        return sum(1 for m, _ in self.calls if m == method)


class SingleRankDist:
    """``DistLike`` for a world of one rank: the gathered list is the payload itself."""

    def allgather(self, payload: object) -> list:
        return [payload]


class HangingClose:
    """A backend ``close`` that blocks until the test opens the gate."""

    def __init__(self) -> None:
        self.gate = threading.Event()
        self.entered = threading.Event()
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1
        self.entered.set()
        self.gate.wait()


# ---------------------------------------------------------------------------------------------
# Engine-side fakes
# ---------------------------------------------------------------------------------------------


class FakeKVCache:
    def __init__(self, *, history_length: int = 0, num_committed_tokens: int = 0) -> None:
        self.history_length = history_length
        self.num_committed_tokens = num_committed_tokens
        self.capacity = history_length
        self.is_active = True


class FakeKVCacheManager:
    """The slice of ``KVCacheManagerV2`` the effects, the hooks and the reader touch.

    ``commit_to[rid]`` overrides what ``try_commit_blocks`` commits (default: the cursor).
    ``release_index_slot`` is idempotent, as the real wrapper's is (plan §7 #3).
    ``reserve_transfer_pages`` answers ``reserve_answer`` (default ``True``) and, when it does
    reserve, leaves behind a cache declaring history to ``token_end``, as the scheduler's call
    on the real wrapper does.
    """

    def __init__(self, tokens_per_block: int = TPB) -> None:
        self.tokens_per_block = tokens_per_block
        self.enable_block_reuse = True
        self.kv_cache_map: dict[int, FakeKVCache] = {}
        self.commit_to: dict[int, int] = {}
        self.reserve_answer = True
        self.calls: list[tuple[str, int]] = []
        self.index_slots_released: list[int] = []
        self._early_freed: set[int] = set()

    def reserve_transfer_pages(self, request, token_end: int) -> bool:
        self.calls.append(("reserve_transfer_pages", request.py_request_id))
        if not self.reserve_answer:
            return False
        self.kv_cache_map[request.py_request_id] = FakeKVCache(history_length=token_end)
        return True

    def get_history_length(self, request) -> int | None:
        kv_cache = self.kv_cache_map.get(request.py_request_id)
        return None if kv_cache is None else kv_cache.history_length

    def is_request_active(self, request_id: int) -> bool:
        """As the wrapper's: pages on the device, not suspended to a lower tier."""
        kv_cache = self.kv_cache_map.get(request_id)
        return kv_cache is not None and kv_cache.is_active

    def try_commit_blocks(self, request) -> None:
        rid = request.py_request_id
        self.calls.append(("try_commit_blocks", rid))
        kv_cache = self.kv_cache_map.get(rid)
        if kv_cache is not None:
            kv_cache.num_committed_tokens = self.commit_to.get(
                rid, request.context_current_position
            )

    def release_index_slot(self, request_id: int) -> None:
        self.calls.append(("release_index_slot", request_id))
        if request_id in self._early_freed:
            return
        self.index_slots_released.append(request_id)
        self._early_freed.add(request_id)

    def free_resources(self, request, pin_on_release: bool = False) -> None:
        rid = request.py_request_id
        self.calls.append(("free_resources", rid))
        self.kv_cache_map.pop(rid, None)
        self._early_freed.discard(rid)

    def probe_context_reuse(self, request) -> int | None:
        return None

    def count(self, name: str) -> int:
        return sum(1 for n, _ in self.calls if n == name)


class FakeSlotManager:
    """``SlotManager``: freeing an unknown request is a no-op, as in the real one."""

    def __init__(self) -> None:
        self.slots: set[int] = set()
        self.freed: list[int] = []

    def add(self, request) -> None:
        self.slots.add(request.py_request_id)

    def free_resources(self, request) -> None:
        self.freed.append(request.py_request_id)
        self.slots.discard(request.py_request_id)


def make_executor(kv: FakeKVCacheManager, slots: FakeSlotManager) -> PyExecutor:
    """A ``PyExecutor`` with only what the effects, the hooks and the two real methods under
    test (``_terminate_request``, ``_try_cancel_request``) read."""
    executor = object.__new__(PyExecutor)
    executor.kv_cache_manager = kv
    executor.resource_manager = SimpleNamespace(
        resource_managers={
            ResourceManagerType.KV_CACHE_MANAGER: kv,
            ResourceManagerType.SEQ_SLOT_MANAGER: slots,
        }
    )
    executor._revert_ctx_alloc = Mock()
    executor._prepare_disagg_gen_resources = Mock()
    executor._do_terminate_request = Mock()
    executor._free_request_resources = Mock()
    executor._handle_errors = Mock()
    executor._disagg_pp_termination_handler = None
    executor.kv_cache_transceiver = None
    executor._fatal_error = None
    executor.is_shutdown = False
    executor.active_requests = []
    executor.canceled_req_ids = []
    return executor


class FakeReader:
    """``ResourceReader`` over ``FakeKVCacheManager``: keys by request seed, one full-attention
    group, units addressed by block ordinal."""

    def __init__(self, kv_cache_manager: FakeKVCacheManager) -> None:
        self.kv = kv_cache_manager
        self.groups = (full_attention(0),)
        self.reuse_tokens: dict[int, int] = {}
        self.seeds: dict[int, str] = {}
        self.forgotten: list[int] = []
        self.calls: list[tuple[str, int]] = []

    @property
    def tokens_per_block(self) -> int:
        return self.kv.tokens_per_block

    def seed(self, request) -> str:
        return self.seeds.get(request.py_request_id, f"req{request.py_request_id}")

    def local_reuse_tokens(self, request) -> int:
        return self.reuse_tokens.get(request.py_request_id, 0)

    def block_keys(self, request) -> list[bytes]:
        seed = self.seed(request)
        return [
            block_key(seed, o) for o in range((request.prompt_len - 1) // self.tokens_per_block)
        ]

    def group_specs(self) -> Sequence[GroupSpec]:
        return self.groups

    def gen_first_ready(self, request) -> bool:
        return True

    def fetch_extent(self, request, plan) -> tuple[CacheExtent, frozenset[bytes]]:
        """As the real reader: a block the cache already committed is returned by name instead
        of being fetched over."""
        rid = request.py_request_id
        self.calls.append(("fetch_extent", rid))
        kv_cache = self.kv.kv_cache_map.get(rid)
        committed = (
            0 if kv_cache is None else kv_cache.num_committed_tokens // self.tokens_per_block
        )
        keys = plan.block_keys
        units = []
        committed_names = set()
        for g in plan.group_plans:
            for o in g.ordinals:
                if o >= len(keys):
                    continue
                if o < committed:
                    committed_names.add(g.spec.tag + keys[o])
                else:
                    units.append(
                        Unit(name=g.spec.tag + keys[o], local_group=g.spec.local_group, local=o)
                    )
        extent = CacheExtent(name=f"fetch:{rid}".encode(), units=tuple(units), is_last=True)
        return extent, frozenset(committed_names)

    def publish_description(self, request):
        self.calls.append(("publish_description", request.py_request_id))
        keys = self.block_keys(request)
        units = [
            Unit(name=s.tag + keys[o], local_group=s.local_group, local=o)
            for s in self.groups
            for o in range(len(keys))
        ]
        extent = CacheExtent(
            name=f"publish:{request.py_request_id}".encode(),
            units=tuple(units),
            is_last=request.context_remaining_length == 0,
        )
        return extent, None

    def forget_request(self, request_id: int) -> None:
        self.forgotten.append(request_id)

    def unit_names(self, request, ordinals: Iterable[int]) -> frozenset[bytes]:
        keys = self.block_keys(request)
        return frozenset(s.tag + keys[o] for s in self.groups for o in ordinals)

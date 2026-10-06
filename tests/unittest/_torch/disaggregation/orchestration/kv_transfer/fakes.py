# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fakes for the KV transfer coordination layer (design §13).

Everything the coordinator and planner depend on is a ``Protocol``; these are table-backed,
scripted implementations of each. They record every call so tests assert on *what the engine and
the backends were asked to do*, in what order.

The modules under test are imported as ``disaggregation.*`` (see ``conftest.py``); nothing here
imports ``tensorrt_llm``.
"""

from __future__ import annotations

import copy
import hashlib
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence

__extra_import_path__ = ["~/tensorrt_llm/_torch"]
from disaggregation.base.backend import CacheKind  # noqa: E402
from disaggregation.base.cache_backend import (  # noqa: E402
    CacheExtent,
    Delivered,
    Outcome,
    SubmissionRejected,
    Unit,
)
from disaggregation.base.views import GroupSpec  # noqa: E402
from disaggregation.orchestration.kv_transfer.coordinator import KVTransferCoordinator  # noqa: E402
from disaggregation.orchestration.kv_transfer.interfaces import PlanAuthority  # noqa: E402
from disaggregation.orchestration.kv_transfer.records import TransferRecord  # noqa: E402
from disaggregation.remote_cache import (  # noqa: E402
    FetchPlan,
    FetchSource,
    GroupPlan,
    Planner,
    required_ordinals,
    unit_names,
)
from disaggregation.resource.naming import group_tag  # noqa: E402

TPB = 4
"""Tokens per block used throughout the tests."""


# ---------------------------------------------------------------------------------------------
# Layer groups and plans
# ---------------------------------------------------------------------------------------------


def full_attention(local_group: int = 0) -> GroupSpec:
    return GroupSpec(local_group, CacheKind.PAGED, group_tag(frozenset({"full"}), None))


def windowed(
    local_group: int = 1, *, window_blocks: int = 3, sink_blocks: int = 1, tpb: int = TPB
) -> GroupSpec:
    window = window_blocks * tpb
    return GroupSpec(
        local_group,
        CacheKind.PAGED,
        group_tag(frozenset({"swa"}), window),
        window_size=window,
        sink_blocks=sink_blocks,
    )


def state_group(local_group: int = 2) -> GroupSpec:
    return GroupSpec(local_group, CacheKind.STATE, group_tag(frozenset({"ssm"}), None))


def block_key(seed: str, ordinal: int) -> bytes:
    """Deterministic key for block ``ordinal`` of the prompt identified by ``seed``."""
    return hashlib.blake2b(f"{seed}:{ordinal}".encode(), digest_size=8).digest()


def keys_for(seed: str, num_blocks: int) -> tuple[bytes, ...]:
    return tuple(block_key(seed, o) for o in range(num_blocks))


def make_plan(
    groups: Sequence[GroupSpec],
    *,
    token_end: int,
    keys: Sequence[bytes],
    reuse_end: int = 0,
    tpb: int = TPB,
    source: str = "worker",
    no_local_fallback: bool = False,
) -> FetchPlan:
    """Build a ``FetchPlan`` the way ``Planner._build`` does, from public pieces only."""
    group_plans = tuple(
        GroupPlan(s, tuple(sorted(required_ordinals(s, token_end, reuse_end, tpb)))) for s in groups
    )
    return FetchPlan(
        token_end=token_end,
        source=source,
        hint=None,
        no_local_fallback=no_local_fallback,
        group_plans=group_plans,
        block_keys=tuple(keys),
        reuse_end=reuse_end,
        tokens_per_block=tpb,
    )


def ordinals_by_group(plan: FetchPlan) -> dict[int, tuple[int, ...]]:
    """The block ordinals each local layer group asks for, keyed by ``local_group``."""
    return {g.spec.local_group: g.ordinals for g in plan.group_plans}


def plan_unit_names(plan: FetchPlan) -> frozenset[bytes]:
    """Every unit name the plan asks for, across groups; what a full delivery serves."""
    return frozenset(unit_names(plan))


def names(spec: GroupSpec, keys: Sequence[bytes], ordinals: Iterable[int]) -> frozenset[bytes]:
    return frozenset(spec.tag + keys[o] for o in ordinals)


def extent_names(extent: CacheExtent) -> frozenset[bytes]:
    return frozenset(u.name for u in extent.units)


# ---------------------------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------------------------


@dataclass
class FakeRequest:
    """Satisfies ``RequestView``. ``seed`` identifies the prompt content: two requests with the
    same seed compute the same block keys (as two ranks holding one request do)."""

    py_request_id: int
    prompt_len: int
    is_gen_init: bool = False
    is_gen_first_context: bool = False
    route_hints: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    seed: str | None = None

    def __post_init__(self) -> None:
        if self.seed is None:
            self.seed = f"req{self.py_request_id}"


# ---------------------------------------------------------------------------------------------
# Backend fakes
# ---------------------------------------------------------------------------------------------


class FakeRoute:
    def __init__(self, hint: Mapping[str, object]) -> None:
        self.hint = hint
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


class FakeAttempt:
    """An ``Attempt`` whose outcome tests set with ``finish``."""

    def __init__(self, payload: object, outcome: Outcome | None = None) -> None:
        self.payload = payload
        self._outcome = outcome
        self.polls = 0

    @property
    def is_live(self) -> bool:
        """No outcome yet: a ``quiesce`` on it would block a real backend's caller."""
        return self._outcome is None

    def poll(self) -> Outcome | None:
        self.polls += 1
        return self._outcome

    def finish(self, outcome: Outcome) -> None:
        self._outcome = outcome

    def deliver_all(self) -> None:
        """``Delivered`` with every unit of the extent this attempt was made for."""
        self.finish(Delivered(extent_names(self.payload)))

    def deliver_all_but(self, *missing: bytes) -> None:
        self.finish(Delivered(extent_names(self.payload) - frozenset(missing)))


class FakeAuxAttempt(FakeAttempt):
    """A ``FakeAttempt`` that also satisfies ``CarriesAux``."""

    def __init__(
        self, payload: object, aux: Mapping[str, object], outcome: Outcome | None = None
    ) -> None:
        super().__init__(payload, outcome)
        self._aux = dict(aux)

    def aux(self) -> Mapping[str, object]:
        return dict(self._aux)


def check_quiesce_is_not_under_a_live_attempt(backend_name: str, attempts: tuple) -> None:
    """A real backend's ``quiesce`` waits for the attempt to end, on the engine thread; the
    coordinator must only ask once every attempt has its outcome."""
    live = [a for a in attempts if getattr(a, "is_live", False)]
    assert not live, f"{backend_name}.quiesce asked under {len(live)} live attempt(s)"


class FakeFetches:
    """Scripted ``Fetches``.

    * ``script(outcome)`` queues an outcome for the next ``fetch`` in call order; ``script_for(names,
      outcome)`` scripts by the exact unit-name set of the extent (checked first). A scripted value
      may be an ``Outcome``, ``None`` (in flight), or a callable ``extent -> Outcome``.
    * ``quiesce_answers`` is consumed one per ``quiesce`` call; ``True`` once exhausted.
    * ``strict_quiesce`` makes ``quiesce`` fail the test when asked about an attempt that has no
      outcome yet.
    * ``probe_answers`` is consumed one per ``probe`` call, else ``probe_default``. An ``Exception``
      instance is raised instead of returned.
    * ``reject_next`` makes that many upcoming ``fetch`` calls raise ``SubmissionRejected``.
    * ``aux_for_next`` makes the next attempt a ``FakeAuxAttempt`` carrying that mapping.
    * ``single_destination`` makes ``open_route`` raise ``NotImplementedError`` (a store);
      ``open_route_errors`` is a queue of exceptions the next ``open_route`` calls raise instead.
    * Every call is appended to ``calls`` as ``(method, args)``; ``quiesce`` is also appended to
      the shared ``trace`` if one is given, so its order relative to effects can be asserted.
    """

    def __init__(
        self,
        *,
        name: str = "fake",
        single_destination: bool = False,
        probe_default: frozenset[bytes] | None = None,
        trace: list | None = None,
        strict_quiesce: bool = False,
    ) -> None:
        self.name = name
        self.single_destination = single_destination
        self.probe_default = probe_default
        self.trace = trace
        self.strict_quiesce = strict_quiesce
        self.calls: list[tuple[str, tuple]] = []
        self.attempts: list[FakeAttempt] = []
        self.routes: list[FakeRoute] = []
        self._by_order: deque = deque()
        self._by_names: dict[frozenset[bytes], deque] = {}
        self.quiesce_answers: deque[bool] = deque()
        self.probe_answers: deque = deque()
        self.open_route_errors: deque[BaseException] = deque()
        self.reject_next = 0
        self.aux_for_next: Mapping[str, object] | None = None

    # -- scripting --

    def script(self, *outcomes) -> None:
        self._by_order.extend(outcomes)

    def script_for(self, unit_names: Iterable[bytes], *outcomes) -> None:
        self._by_names.setdefault(frozenset(unit_names), deque()).extend(outcomes)

    def _scripted(self, extent: CacheExtent) -> Outcome | None:
        q = self._by_names.get(extent_names(extent))
        if q:
            value = q.popleft()
        elif self._by_order:
            value = self._by_order.popleft()
        else:
            value = None
        return value(extent) if callable(value) else value

    # -- Fetches --

    def fetch(self, extent: CacheExtent, *, route=None) -> FakeAttempt:
        self.calls.append(("fetch", (extent, route)))
        if self.reject_next > 0:
            self.reject_next -= 1
            raise SubmissionRejected(f"{self.name} rejected")
        outcome = self._scripted(extent)
        if self.aux_for_next is not None:
            attempt = FakeAuxAttempt(extent, self.aux_for_next, outcome)
            self.aux_for_next = None
        else:
            attempt = FakeAttempt(extent, outcome)
        self.attempts.append(attempt)
        return attempt

    def quiesce(self, attempts: Iterable) -> bool:
        attempts = tuple(attempts)
        if self.strict_quiesce:
            check_quiesce_is_not_under_a_live_attempt(self.name, attempts)
        answer = self.quiesce_answers.popleft() if self.quiesce_answers else True
        self.calls.append(("quiesce", (attempts, answer)))
        if self.trace is not None:
            self.trace.append(("quiesce", (self.name, attempts, answer)))
        return answer

    def settle(self, attempts: Iterable) -> None:
        self.calls.append(("settle", (tuple(attempts),)))

    def probe(self, name: bytes, units: Sequence[bytes]):
        self.calls.append(("probe", (name, tuple(units))))
        answer = self.probe_answers.popleft() if self.probe_answers else self.probe_default
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def open_route(self, hint: Mapping[str, object]) -> FakeRoute:
        self.calls.append(("open_route", (hint,)))
        if self.open_route_errors:
            raise self.open_route_errors.popleft()
        if self.single_destination:
            raise NotImplementedError(f"{self.name} has a single destination")
        route = FakeRoute(hint)
        self.routes.append(route)
        return route

    # -- convenience --

    def count(self, method: str) -> int:
        return sum(1 for m, _ in self.calls if m == method)


class FakePublishes:
    """Scripted ``Publishes``; same scripting knobs as ``FakeFetches`` where they apply."""

    def __init__(
        self, *, name: str = "pub", trace: list | None = None, strict_quiesce: bool = False
    ) -> None:
        self.name = name
        self.trace = trace
        self.strict_quiesce = strict_quiesce
        self.calls: list[tuple[str, tuple]] = []
        self.attempts: list[FakeAttempt] = []
        self.quiesce_answers: deque[bool] = deque()
        self.reject_next = 0
        self.reject_methods: set[str] = set()
        """Methods (``"publish"``, ``"place"``) that always raise ``SubmissionRejected``."""

    def _submit(self, method: str, payload: object) -> FakeAttempt:
        self.calls.append((method, (payload,)))
        if self.reject_next > 0 or method in self.reject_methods:
            self.reject_next = max(0, self.reject_next - 1)
            raise SubmissionRejected(f"{self.name} rejected {method}")
        attempt = FakeAttempt(payload)
        self.attempts.append(attempt)
        return attempt

    def publish(self, extent: CacheExtent) -> FakeAttempt:
        return self._submit("publish", extent)

    def quiesce(self, attempts: Iterable) -> bool:
        attempts = tuple(attempts)
        if self.strict_quiesce:
            check_quiesce_is_not_under_a_live_attempt(self.name, attempts)
        answer = self.quiesce_answers.popleft() if self.quiesce_answers else True
        self.calls.append(("quiesce", (attempts, answer)))
        if self.trace is not None:
            self.trace.append(("quiesce", (self.name, attempts, answer)))
        return answer

    def settle(self, attempts: Iterable) -> None:
        self.calls.append(("settle", (tuple(attempts),)))

    def count(self, method: str) -> int:
        return sum(1 for m, _ in self.calls if m == method)

    def payloads(self, method: str) -> list:
        return [args[0] for m, args in self.calls if m == method]


class FakePlacingPublishes(FakePublishes):
    """A publisher that also satisfies ``PlacesPieces``."""

    def place(self, chunk) -> FakeAttempt:
        return self._submit("place", chunk)


@dataclass(frozen=True)
class FakeChunk:
    """Stands in for ``resource.page.Chunk``; the coordinator never looks inside."""

    request_id: int
    step: int


class FakeLanding:
    """A ``Landing`` whose outcome tests set with ``finish`` (or ``deliver_all``); ``place``
    hands out a ``FakeAttempt`` scripted on the owning ``FakeLandsOnHost``; ``releases`` counts
    ``release`` calls."""

    def __init__(
        self, backend: FakeLandsOnHost, name: bytes, units: Sequence[bytes], outcome=None
    ) -> None:
        self.backend = backend
        self.name = name
        self.units = tuple(units)
        self._outcome = outcome
        self.polls = 0
        self.releases = 0
        self.placements: list[FakeAttempt] = []

    def poll(self) -> Outcome | None:
        self.polls += 1
        return self._outcome

    def finish(self, outcome: Outcome) -> None:
        self._outcome = outcome

    def deliver_all(self) -> None:
        self.finish(Delivered(frozenset(self.units)))

    def deliver_all_but(self, *missing: bytes) -> None:
        self.finish(Delivered(frozenset(self.units) - frozenset(missing)))

    def place(self, extent: CacheExtent) -> FakeAttempt:
        return self.backend._place(self, extent)

    def release(self) -> None:
        self.releases += 1
        self.backend.calls.append(("release", (self,)))


class FakeLandsOnHost:
    """Scripted ``LandsOnHost``: a store that lands in its own memory first.

    * ``script(outcome)`` queues an outcome for the next ``fetch_to_host``'s landing, in call
      order; unscripted landings start in flight (``None``) and tests ``finish`` them.
    * ``script_place(outcome)`` does the same for the attempts ``Landing.place`` returns.
    * ``reject_next`` / ``reject_place_next`` make that many upcoming ``fetch_to_host`` /
      ``place`` calls raise ``SubmissionRejected``.
    * ``probe`` answers ``probe_answers`` one per call, else ``probe_default``: ``"all"`` means
      every unit asked about (the store holds the whole prompt).
    * ``quiesce_answers`` as ``FakeFetches``; every call is appended to ``calls``.
    """

    def __init__(
        self,
        *,
        name: str = "host",
        probe_default="all",
        trace: list | None = None,
        strict_quiesce: bool = False,
    ) -> None:
        self.name = name
        self.probe_default = probe_default
        self.trace = trace
        self.strict_quiesce = strict_quiesce
        self.calls: list[tuple[str, tuple]] = []
        self.landings: list[FakeLanding] = []
        self.attempts: list[FakeAttempt] = []
        self.quiesce_answers: deque[bool] = deque()
        self.probe_answers: deque = deque()
        self.reject_next = 0
        self.reject_place_next = 0
        self._landing_outcomes: deque = deque()
        self._place_outcomes: deque = deque()

    # -- scripting --

    def script(self, *outcomes) -> None:
        self._landing_outcomes.extend(outcomes)

    def script_place(self, *outcomes) -> None:
        self._place_outcomes.extend(outcomes)

    # -- LandsOnHost --

    def fetch_to_host(self, name: bytes, units: Sequence[bytes]) -> FakeLanding:
        self.calls.append(("fetch_to_host", (name, tuple(units))))
        if self.reject_next > 0:
            self.reject_next -= 1
            raise SubmissionRejected(f"{self.name} refused a landing")
        outcome = self._landing_outcomes.popleft() if self._landing_outcomes else None
        landing = FakeLanding(self, name, units, outcome)
        self.landings.append(landing)
        return landing

    def _place(self, landing: FakeLanding, extent: CacheExtent) -> FakeAttempt:
        self.calls.append(("place", (landing, extent)))
        if self.reject_place_next > 0:
            self.reject_place_next -= 1
            raise SubmissionRejected(f"{self.name} refused a placement")
        outcome = self._place_outcomes.popleft() if self._place_outcomes else None
        attempt = FakeAttempt(extent, outcome(extent) if callable(outcome) else outcome)
        landing.placements.append(attempt)
        self.attempts.append(attempt)
        return attempt

    def quiesce(self, attempts: Iterable) -> bool:
        attempts = tuple(attempts)
        if self.strict_quiesce:
            check_quiesce_is_not_under_a_live_attempt(self.name, attempts)
        answer = self.quiesce_answers.popleft() if self.quiesce_answers else True
        self.calls.append(("quiesce", (attempts, answer)))
        if self.trace is not None:
            self.trace.append(("quiesce", (self.name, attempts, answer)))
        return answer

    def settle(self, attempts: Iterable) -> None:
        self.calls.append(("settle", (tuple(attempts),)))

    def probe(self, name: bytes, units: Sequence[bytes]):
        self.calls.append(("probe", (name, tuple(units))))
        answer = self.probe_answers.popleft() if self.probe_answers else self.probe_default
        if isinstance(answer, BaseException):
            raise answer
        return frozenset(units) if answer == "all" else answer

    # -- convenience --

    def count(self, method: str) -> int:
        return sum(1 for m, _ in self.calls if m == method)

    def releases(self) -> int:
        """``release`` calls over every landing this backend handed out."""
        return sum(landing.releases for landing in self.landings)


# ---------------------------------------------------------------------------------------------
# Engine-side fakes
# ---------------------------------------------------------------------------------------------


class RecordingEffects:
    """Every ``KVTransferEffects`` method appends ``(name, args)`` to ``calls`` (and to the shared
    ``trace`` when given)."""

    def __init__(self, trace: list | None = None) -> None:
        self.calls: list[tuple[str, tuple]] = []
        self.trace = trace

    def _record(self, name: str, *args) -> None:
        self.calls.append((name, args))
        if self.trace is not None:
            self.trace.append((name, args))

    def park_for_fetch(self, requests) -> None:
        self._record("park_for_fetch", tuple(requests))

    def unpark(self, request, token_end, no_local_fallback, aux) -> None:
        self._record("unpark", request, token_end, no_local_fallback, aux)

    def give_back_fetch_pages(self, requests) -> None:
        self._record("give_back_fetch_pages", tuple(requests))

    def prepare_fetch_resources(self, requests) -> None:
        self._record("prepare_fetch_resources", tuple(requests))

    def hold_for_transfer(self, requests) -> None:
        self._record("hold_for_transfer", tuple(requests))

    def terminate_request(self, request) -> None:
        self._record("terminate_request", request)

    def fail_requests(self, requests, reason) -> None:
        self._record("fail_requests", tuple(requests), reason)

    def fail_fatal(self, exc) -> None:
        self._record("fail_fatal", exc)

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]

    def count(self, name: str) -> int:
        return self.names().count(name)

    def only(self, name: str) -> list[tuple]:
        return [args for n, args in self.calls if n == name]


class FakeEngineQueue:
    def __init__(self) -> None:
        self.pending: deque[Callable[[], None]] = deque()
        self.drains: list[int] = []
        self.ran = 0

    def post(self, fn: Callable[[], None]) -> None:
        self.pending.append(fn)

    def drain(self, budget: int) -> int:
        self.drains.append(budget)
        ran = 0
        while self.pending and ran < budget:
            self.pending.popleft()()
            ran += 1
        self.ran += ran
        return ran


class FakeDist:
    """Single rank: ``allgather`` answers with the caller's own payload; ``calls`` keeps a copy
    of every payload gathered."""

    rank = 0
    world_size = 1

    def __init__(self) -> None:
        self.calls: list[object] = []

    def allgather(self, obj) -> list:
        snapshot = copy.deepcopy(obj)
        self.calls.append(snapshot)
        return [copy.deepcopy(snapshot)]


class LockstepWorld:
    """``n`` real coordinators stepped in lockstep on one thread, each over a ``LockstepGather``
    as its ``DistLike``.

    ``run(fn)`` calls ``fn(0)``; when rank 0 reaches its one collective, its gather runs ``fn(1)``
    nested (and so on up to rank ``n-1``), so every rank's payload exists before any gather
    returns. Each rank then applies the same gathered list. ``fn`` must call ``advance`` exactly
    once per rank. Between ``run`` calls the test scripts each rank's fakes freely.

    ``authority_of(rank)`` names each rank's ``PlanAuthority`` (default: every rank ``VOTED``);
    ``Rig`` construction reads it through ``rig_kwargs``.
    """

    def __init__(self, n: int, authority_of: Callable[[int], PlanAuthority] | None = None) -> None:
        self.n = n
        self.authority_of = authority_of or (lambda rank: PlanAuthority.VOTED)
        self.gathers = [LockstepGather(self, r) for r in range(n)]
        self._fn: Callable[[int], object] | None = None
        self._payloads: dict[int, object] = {}
        self._results: list = []

    def rig_kwargs(self, rank: int) -> dict:
        """The keyword arguments that place a ``Rig`` at ``rank`` of this world."""
        return {"dist": self.gathers[rank], "plan_authority": self.authority_of(rank)}

    def run(self, fn: Callable[[int], object]) -> list:
        assert self._fn is None, "LockstepWorld.run is not reentrant"
        self._fn, self._payloads, self._results = fn, {}, [None] * self.n
        try:
            self._results[0] = fn(0)
            missing = sorted(set(range(self.n)) - set(self._payloads))
            assert not missing, f"ranks that never gathered this step: {missing}"
            return self._results
        finally:
            self._fn = None

    def _exchange(self, rank: int, payload: object) -> list:
        assert self._fn is not None, f"rank {rank} gathered outside LockstepWorld.run"
        assert rank not in self._payloads, f"rank {rank} gathered twice in one step"
        self._payloads[rank] = copy.deepcopy(payload)
        if rank + 1 < self.n:
            self._results[rank + 1] = self._fn(rank + 1)
        assert len(self._payloads) == self.n, "a peer never reached its collective"
        return [copy.deepcopy(self._payloads[r]) for r in range(self.n)]


class LockstepGather:
    """The ``DistLike`` of one rank of a ``LockstepWorld``; records every payload."""

    def __init__(self, world: LockstepWorld, rank: int) -> None:
        self._world = world
        self.rank = rank
        self.calls: list = []

    def allgather(self, payload) -> list:
        self.calls.append(copy.deepcopy(payload))
        return self._world._exchange(self.rank, payload)


class PeerGather:
    """``DistLike`` of a single real rank plus hand-written peers: ``peer(local_payload) ->
    peer_payload`` for each peer function; the gathered list is ``[local, *peers]``."""

    def __init__(self, *peers: Callable[[object], object]) -> None:
        self._peers = peers
        self.calls: list = []

    def allgather(self, payload) -> list:
        local = copy.deepcopy(payload)
        self.calls.append(local)
        return [local, *(copy.deepcopy(peer(local)) for peer in self._peers)]


# ---------------------------------------------------------------------------------------------
# Resource reader
# ---------------------------------------------------------------------------------------------


class FakeReader:
    """Table-backed ``ResourceReader`` for a synthetic model.

    * ``groups``: the layer groups this rank holds.
    * ``reuse_tokens[rid]``: tokens the local radix tree already serves (default 0).
    * ``gen_first_ready``: answer for gen-first context requests (per rid, default ``ready_default``).
    * ``fetch_extent`` names ``tag + key`` for every ordinal in every group plan, ``is_last=True``,
      minus what the reservation looks like at launch: ordinals below ``committed_blocks[rid]``
      are returned as committed names instead, ordinals at or above ``reserved_blocks[rid]`` are
      dropped (the reservation fell short).
    * ``publish_description`` pops the next scripted ``(extent, chunk)`` for the request
      (``script_publish``), or builds one covering every full block with no chunk.
    """

    def __init__(
        self,
        *,
        groups: Sequence[GroupSpec] | None = None,
        tokens_per_block: int = TPB,
        ready_default: bool = True,
    ) -> None:
        self.groups = tuple(groups) if groups is not None else (full_attention(),)
        self._tpb = tokens_per_block
        self.reuse_tokens: dict[int, int] = {}
        self.ready: dict[int, bool] = {}
        self.ready_default = ready_default
        self.committed_blocks: dict[int, int] = {}
        self.reserved_blocks: dict[int, int] = {}
        self._publish_steps: dict[int, deque] = {}
        self.calls: list[tuple[str, tuple]] = []

    @property
    def tokens_per_block(self) -> int:
        return self._tpb

    def local_reuse_tokens(self, request) -> int:
        self.calls.append(("local_reuse_tokens", (request,)))
        return self.reuse_tokens.get(request.py_request_id, 0)

    def block_keys(self, request) -> list[bytes]:
        # One key per full block that content addressing can name: the last prompt token never
        # takes part in reuse, so an aligned prompt has one key fewer than full blocks.
        return list(keys_for(request.seed, (request.prompt_len - 1) // self._tpb))

    def group_specs(self) -> Sequence[GroupSpec]:
        return self.groups

    def gen_first_ready(self, request) -> bool:
        return self.ready.get(request.py_request_id, self.ready_default)

    def fetch_extent(self, request, plan) -> tuple[CacheExtent, frozenset[bytes]]:
        self.calls.append(("fetch_extent", (request, plan)))
        rid = request.py_request_id
        committed = self.committed_blocks.get(rid, 0)
        reserved = self.reserved_blocks.get(rid, len(plan.block_keys))
        keys = plan.block_keys
        units = []
        committed_names = set()
        for g in plan.group_plans:
            for o in g.ordinals:
                if o >= len(keys):
                    continue
                if o < committed:
                    committed_names.add(g.spec.tag + keys[o])
                elif o < reserved:
                    units.append(
                        Unit(name=g.spec.tag + keys[o], local_group=g.spec.local_group, local=o)
                    )
        extent = CacheExtent(
            name=f"fetch:{request.seed}".encode(), units=tuple(units), is_last=True
        )
        return extent, frozenset(committed_names)

    def publish_description(self, request):
        self.calls.append(("publish_description", (request,)))
        steps = self._publish_steps.get(request.py_request_id)
        if steps:
            return steps.popleft()
        keys = self.block_keys(request)
        units = [
            Unit(name=s.tag + keys[o], local_group=s.local_group, local=o)
            for s in self.groups
            for o in range(len(keys))
        ]
        return CacheExtent(
            name=f"publish:{request.seed}".encode(), units=tuple(units), is_last=True
        ), None

    def forget_request(self, request_id: int) -> None:
        """Nothing to forget: this reader keeps no per-request state."""

    # -- scripting helpers --

    def script_publish(
        self, request, steps: Sequence[tuple[Sequence[int], bool, object]]
    ) -> list[CacheExtent]:
        """Script ``publish_description`` for ``request``: one ``(ordinals, is_last, chunk)`` per
        context step. Returns the extents in order so tests can refer to them."""
        keys = self.block_keys(request)
        extents = []
        q = self._publish_steps.setdefault(request.py_request_id, deque())
        for i, (ordinals, is_last, chunk) in enumerate(steps):
            units = [
                Unit(name=s.tag + keys[o], local_group=s.local_group, local=o)
                for s in self.groups
                for o in ordinals
            ]
            extent = CacheExtent(
                name=f"publish:{request.seed}:{i}".encode(), units=tuple(units), is_last=is_last
            )
            extents.append(extent)
            q.append((extent, chunk))
        return extents

    def unit_names(
        self, request, ordinals: Iterable[int], *, kinds=(CacheKind.PAGED,)
    ) -> frozenset[bytes]:
        """Names of the given blocks for every group of the given kinds (default: paged)."""
        keys = self.block_keys(request)
        ordinals = tuple(ordinals)
        return frozenset(s.tag + keys[o] for s in self.groups if s.kind in kinds for o in ordinals)


# ---------------------------------------------------------------------------------------------
# One rank's rig
# ---------------------------------------------------------------------------------------------


class Rig:
    """Everything one coordinator needs, wired with defaults; tests override by keyword.

    Sources are ``worker`` (hint key ``"ctx"``) followed by ``store`` (no hint key) unless
    ``sources`` is given; ``host`` is a ``FakeLandsOnHost`` store that holds every prompt. ``trace``
    interleaves backend ``quiesce`` calls with effects. Every backend, publishers included, is
    strict about ``quiesce``: asking under a live attempt fails the test. The collective is
    ``FakeDist`` unless a ``dist`` (``LockstepGather``, ``PeerGather``) is given; ``payloads()``
    lists what this rank sent either way. ``probe_timeout_s`` is measured on the ``now`` tests
    pass to ``advance``: with the default, a request deferred at 0.0 and 1.0 is planned without
    the store at 2.0.
    """

    def __init__(
        self,
        *,
        groups: Sequence[GroupSpec] | None = None,
        sources: Sequence[str] = ("worker", "store"),
        publishers: Sequence[FakePublishes] | None = None,
        dist=None,
        tpb: int = TPB,
        probe_timeout_s: float | None = 2.0,
        **coordinator_kwargs,
    ) -> None:
        self.plans: dict[int, FetchPlan] = {}
        """Plans as read by ``plan_and_launch`` before launching (unreadable afterwards)."""
        self.trace: list = []
        self.reader = FakeReader(groups=groups, tokens_per_block=tpb)
        self.worker = FakeFetches(name="worker", trace=self.trace, strict_quiesce=True)
        self.store = FakeFetches(
            name="store", single_destination=True, trace=self.trace, strict_quiesce=True
        )
        self.host = FakeLandsOnHost(name="host", trace=self.trace, strict_quiesce=True)
        table = {
            "worker": FetchSource("worker", self.worker, "ctx"),
            "store": FetchSource("store", self.store, None),
            "host": FetchSource("host", self.host, None),
        }
        self.sources = [table[name] for name in sources]
        self.publishers = list(publishers or [])
        for p in self.publishers:
            if getattr(p, "trace", None) is None:
                p.trace = self.trace
            p.strict_quiesce = True
        self.effects = RecordingEffects(trace=self.trace)
        self.queue = FakeEngineQueue()
        self.dist = dist if dist is not None else FakeDist()
        self.planner = Planner(self.sources, self.reader, tpb, probe_timeout_s=probe_timeout_s)
        self.coord = KVTransferCoordinator(
            self.sources,
            self.publishers,
            self.planner,
            self.reader,
            self.effects,
            self.queue,
            self.dist,
            **coordinator_kwargs,
        )

    # -- shortcuts --

    def records(self) -> list[dict]:
        return self.coord.status_dump()["records"]

    def record(self, rid: int, direction: str = "fetch") -> dict | None:
        for rec in self.records():
            if rec["request_id"] == rid and rec["direction"] == direction:
                return rec
        return None

    def fetch_record(self, rid: int) -> TransferRecord | None:
        """The fetch ``TransferRecord`` itself, for the launch bookkeeping the dump leaves out."""
        return self.coord._records.get((rid, "fetch"))

    def payloads(self) -> list:
        """Every payload this rank handed to its collective, in order."""
        return list(self.dist.calls)

    def plan_and_launch(self, req: FakeRequest, now: float = 0.0) -> FakeAttempt:
        """``advance`` with ``req`` as the only candidate, read its plan into ``plans``, then
        ``launch_fetches``; returns the attempt the worker created."""
        self.coord.advance([req], now)
        plan = self.coord.plan_fetch(req)
        assert isinstance(plan, FetchPlan), f"expected a plan before launch, got {plan!r}"
        self.plans[req.py_request_id] = plan
        before = len(self.worker.attempts)
        self.coord.launch_fetches([req], now)
        assert len(self.worker.attempts) == before + 1, "launch did not create a worker attempt"
        return self.worker.attempts[-1]

    def plan_and_land(self, req: FakeRequest, now: float = 0.0) -> FakeLanding:
        """``advance`` with ``req`` as the only candidate on the ``host`` source: the plan is
        decided and its landing started in the same round; returns that landing."""
        before = len(self.host.landings)
        self.coord.advance([req], now)
        assert len(self.host.landings) == before + 1, "deciding the plan did not start a landing"
        return self.host.landings[-1]

    def reserve_and_place(self, req: FakeRequest, now: float) -> FakeAttempt:
        """The scheduler's part for a ``STAGED`` record: read the plan (reserving is implied),
        then ``launch_fetches``; returns the placement attempt."""
        plan = self.coord.plan_fetch(req)
        assert isinstance(plan, FetchPlan), f"expected a plan to reserve for, got {plan!r}"
        self.plans[req.py_request_id] = plan
        before = len(self.host.attempts)
        self.coord.launch_fetches([req], now)
        assert len(self.host.attempts) == before + 1, "launch did not place the landing"
        return self.host.attempts[-1]


def worker_request(rid: int = 1, prompt_len: int = 29, **kw) -> FakeRequest:
    """A context request routed to the worker backend (hint key ``"ctx"``)."""
    return FakeRequest(rid, prompt_len, route_hints={"ctx": {"peer": f"peer{rid}"}}, **kw)

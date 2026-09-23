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
"""Source policy and the merge rule (design §6.3, §7.2).

The planner is the only place that reads request content in order to *decide*; building extents
and chunks belongs to ``resource/``. ``merge`` and ``retry_hint_from`` are pure functions of a plan
and a served set, so the rule that says what "arrived" means can be tested with no engine at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from typing import Iterator, Mapping, Sequence

from .base.backend import CacheKind
from .base.cache_backend import Fetches
from .base.views import GroupSpec, RequestView, ResourceReader

__all__ = [
    "DEFER",
    "Defer",
    "FetchPlan",
    "FetchSource",
    "GroupPlan",
    "Planner",
    "merge",
    "required_ordinals",
    "retry_hint_from",
]


class Defer:
    """The answer "do not schedule this request this round; ask again next round".

    A class rather than a sentinel string so that ``isinstance`` works and so that it can never be
    confused with a plan or with ``None`` (which means "compute locally, no fetch").
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "DEFER"


DEFER = Defer()
"""The single ``Defer`` instance. Compare with ``is``."""


@dataclass(frozen=True)
class FetchSource:
    """One fetch backend in the assembly table (design §7.4).

    Attributes:
        name: Stable identifier; recorded on plans and attempts, and used as the allgather key.
        backend: The backend itself.
        hint_key: Which routing hint on a request this backend reads. ``None`` for a backend whose
            destination is unique (a store), which then never gets ``open_route``.
    """

    name: str
    backend: Fetches
    hint_key: str | None


@dataclass(frozen=True)
class GroupPlan:
    """What one layer group asks for in a plan: the block ordinals it needs at ``token_end``."""

    spec: GroupSpec
    ordinals: tuple[int, ...]


@dataclass(frozen=True)
class FetchPlan:
    """A decided fetch (design §7.2). Immutable; the scheduler reads it, nothing edits it.

    Attributes:
        token_end: Where the fetch aims to bring the request; the scheduler allocates up to it. A
            block boundary, except for a gen-init plan where it is ``prompt_len``.
        source: ``FetchSource.name``.
        hint: ``open_route`` input; only for a worker source.
        no_local_fallback: True for gen-init: a failure can only fail the request.
        group_plans: The per-group asks with their specs; what ``merge`` and ``fetch_extent``
            read. Unit names are ``spec.tag + block_keys[o]`` for every ordinal ``o`` asked.
        block_keys: One key per full prompt block, by ordinal.
        reuse_end: Blocks the local radix tree already serves; nothing below it is asked for.
        tokens_per_block: ``tpb``.
    """

    token_end: int
    source: str
    hint: Mapping[str, object] | None
    no_local_fallback: bool
    group_plans: tuple[GroupPlan, ...]
    block_keys: tuple[bytes, ...]
    reuse_end: int
    tokens_per_block: int


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _stale_range(spec: GroupSpec, history: int, tpb: int) -> tuple[int, int]:
    """Blocks a group no longer reads at ``history``, as ``AttnLifeCycle.get_stale_range``."""
    num_blocks = _ceil_div(history, tpb)
    start = min(num_blocks, spec.sink_blocks)
    if spec.window_size is None:
        return start, start
    # `+ 1`: attention always runs for at least one in-flight token at position `history`, so
    # the live window is [history + 1 - window_size, history]; that block must not be dropped.
    return start, max(start, (history + 1 - spec.window_size) // tpb)


def required_ordinals(spec: GroupSpec, history: int, reuse_end: int, tpb: int) -> frozenset[int]:
    """Block ordinals ``spec`` must have fetched for the request to stand at ``history``.

    Full attention: ``[reuse_end, full)``. Windowed: the sink blocks plus the live window, minus
    what is local. State: exactly the snapshot at ``history`` when it is a block boundary past
    the local prefix; otherwise nothing, because a partial tail's state has no name and moves by
    position.
    """
    full = history // tpb
    if spec.kind is CacheKind.STATE:
        if history % tpb != 0 or full <= reuse_end:
            return frozenset()
        return frozenset({full - 1})
    beg, end = _stale_range(spec, history, tpb)
    live = chain(range(min(beg, full)), range(end, full))
    return frozenset(o for o in live if o >= reuse_end)


def _required_names(plan: FetchPlan, group: GroupPlan, history: int) -> frozenset[bytes]:
    """Unit names ``group`` needs at ``history``. An ordinal past the last key is skipped, exactly
    as ``Planner._build`` and ``naming.units_for_group`` skip it: it has no name and moves by
    position, so nothing named can be waited for."""
    ordinals = required_ordinals(group.spec, history, plan.reuse_end, plan.tokens_per_block)
    return frozenset(
        group.spec.tag + plan.block_keys[o] for o in ordinals if o < len(plan.block_keys)
    )


def _boundaries_down(plan: FetchPlan) -> Iterator[int]:
    """``token_end`` first, then every block boundary below it down to the local prefix."""
    tpb = plan.tokens_per_block
    floor = plan.reuse_end * tpb
    b = plan.token_end
    while b >= floor:
        yield b
        if b <= 0:
            return
        b = (b - 1) // tpb * tpb


def _merge(plan: FetchPlan, served: frozenset[bytes], groups: Sequence[GroupPlan]) -> int:
    for b in _boundaries_down(plan):
        ok = True
        for group in groups:
            if not _required_names(plan, group, b) <= served:
                ok = False
                break
        if ok:
            return b
    return plan.reuse_end * plan.tokens_per_block


def merge(plan: FetchPlan, served: frozenset[bytes]) -> int:
    """The largest B such that, with B as the sequence length, every group's still-needed units
    are in ``served`` (design §6.3). Never below the local prefix end."""
    return _merge(plan, served, plan.group_plans)


def retry_hint_from(plan: FetchPlan, served: frozenset[bytes]) -> int:
    """B computed over full-attention groups only: windowed and state groups need different
    units at a smaller target and are re-planned on retry. Falls back to all groups when the
    model has no full-attention group."""
    full_attention = tuple(
        g for g in plan.group_plans if g.spec.kind is CacheKind.PAGED and g.spec.window_size is None
    )
    return _merge(plan, served, full_attention or plan.group_plans)


class Planner:
    """Decides, per candidate request, whether to fetch, from where, and up to which token.

    Args:
        sources: The assembly table, in priority order.
        reader: This rank's resource view.
        tokens_per_block: ``tpb``.
        probe_timeout_s: How long, from its first deferral, a request may wait for a store's
            ``probe`` before the unanswered probe counts as "not held". Measured on the ``now``
            each ``decide`` receives, the loop clock the coordinator's deadlines also use.
            ``None`` waits indefinitely.
    """

    def __init__(
        self,
        sources: Sequence[FetchSource],
        reader: ResourceReader,
        tokens_per_block: int,
        *,
        probe_timeout_s: float | None = None,
    ) -> None:
        self._sources = tuple(sources)
        self._reader = reader
        self._tpb = tokens_per_block
        self._probe_timeout_s = probe_timeout_s
        self._first_deferred_at: dict[int, float] = {}
        """Per deferred request: the ``now`` of its first deferral."""

    def probe_query(self, req: RequestView) -> tuple[bytes, tuple[bytes, ...]] | None:
        """``(name, unit names)`` to ask a store about, or ``None`` if the request has nothing
        nameable to fetch. Rank-independent: it starts at block 0, not at the local prefix."""
        if req.is_gen_init:
            return None
        nameable = (req.prompt_len - 1) // self._tpb
        keys = self._reader.block_keys(req)
        if nameable <= 0 or nameable > len(keys):
            return None
        specs = self._reader.group_specs()
        units = tuple(s.tag + keys[o] for s in specs for o in range(nameable))
        return keys[nameable - 1], units

    def forget(self, req_id: int) -> None:
        """Drop per-request planning state once a request is decided or gone."""
        self._first_deferred_at.pop(req_id, None)

    def _may_still_wait_for_probe(self, req_id: int, now: float) -> bool:
        """Whether ``req_id`` may be deferred once more for an unanswered probe.

        The first deferral only starts the clock; the wait ends once ``probe_timeout_s`` has
        passed since then, so a zero budget allows exactly one deferral.
        """
        first = self._first_deferred_at.get(req_id)
        if first is None:
            self._first_deferred_at[req_id] = now
            return True
        return self._probe_timeout_s is None or now - first < self._probe_timeout_s

    def decide(
        self,
        req: RequestView,
        probe_answers: Mapping[str, frozenset[bytes] | None],
        *,
        now: float,
        retry_hint: int | None = None,
    ) -> FetchPlan | None | Defer:
        """The decision table of design §7.2 for one candidate.

        Args:
            req: The candidate.
            probe_answers: Per store source, what it holds; ``None`` or missing means unanswered.
            now: The loop clock this round; the probe wait is measured on it.
            retry_hint: Upper bound on ``token_end`` after a short ``served`` (fetch retry).
        """
        tpb = self._tpb
        reuse_end = self._reader.local_reuse_tokens(req) // tpb
        keys = tuple(self._reader.block_keys(req))

        if req.is_gen_init:
            source = self._gen_init_source(req)
            if source is None:
                return None
            hint = req.route_hints.get(source.hint_key) if source.hint_key is not None else None
            return self._build(req.prompt_len, source, hint, True, reuse_end, keys)

        if req.is_gen_first_context and not self._reader.gen_first_ready(req):
            return DEFER

        # The decision below reads only inputs every rank shares (prompt, hints, probe answers).
        # The local reuse depth differs per rank, so it only trims the per-group asks (possibly
        # to nothing, which is still a legal plan) and never decides whether to fetch.
        nameable = (req.prompt_len - 1) // tpb
        if nameable <= 0:
            self.forget(req.py_request_id)
            return None

        chosen: FetchSource | None = None
        hint = None
        token_end = 0
        pending = False
        for source in self._sources:
            if source.hint_key is not None:
                if source.hint_key in req.route_hints:
                    chosen, hint, token_end = (
                        source,
                        req.route_hints[source.hint_key],
                        nameable * tpb,
                    )
                    break
                continue
            answer = probe_answers.get(source.name)
            if answer is None:
                pending = True
                continue
            end = self._contiguous_prefix_end(answer, keys, nameable)
            if end > 0:
                chosen, token_end = source, end * tpb
                break

        if chosen is None:
            if pending and self._may_still_wait_for_probe(req.py_request_id, now):
                return DEFER
            self.forget(req.py_request_id)
            return None

        self.forget(req.py_request_id)
        if retry_hint is not None:
            token_end = min(token_end, retry_hint)
        if token_end <= 0:
            return None
        return self._build(token_end, chosen, hint, False, reuse_end, keys)

    def materialize(self, req: RequestView, token_end: int, source: str) -> FetchPlan:
        """The plan another rank decided, built over this rank's own layer groups and reuse depth:
        what ``decide`` would have built here for ``(token_end, source)``.

        Only a store source: a routed source's hint is per request and is not on the wire, so a
        plan for one cannot be rebuilt from ``(token_end, source)`` alone (``ValueError``).
        """
        chosen = next((s for s in self._sources if s.name == source), None)
        if chosen is None:
            raise ValueError(f"no fetch source named {source!r} on this rank")
        if chosen.hint_key is not None:
            raise ValueError(f"cannot rebuild a plan for the routed source {source!r}")
        reuse_end = self._reader.local_reuse_tokens(req) // self._tpb
        keys = tuple(self._reader.block_keys(req))
        return self._build(token_end, chosen, None, False, reuse_end, keys)

    def _gen_init_source(self, req: RequestView) -> FetchSource | None:
        """The first routed source whose hint the request carries. A gen-init fetch has one
        possible origin, the context worker named by its hint; without a matching hint there is
        nothing to route to, and a routeless fetch would read from nowhere in particular."""
        for source in self._sources:
            if source.hint_key is not None and source.hint_key in req.route_hints:
                return source
        return None

    def _contiguous_prefix_end(
        self, answer: frozenset[bytes], keys: Sequence[bytes], nameable: int
    ) -> int:
        """Blocks from 0 that the store holds for every full-attention group (or every paged
        group when there is none), stopping at the first gap."""
        specs = [s for s in self._reader.group_specs() if s.kind is CacheKind.PAGED]
        full_attention = [s for s in specs if s.window_size is None]
        check = full_attention or specs
        if not check:
            return 0
        o = 0
        while o < nameable and all(s.tag + keys[o] in answer for s in check):
            o += 1
        return o

    def _build(
        self,
        token_end: int,
        source: FetchSource,
        hint: Mapping[str, object] | None,
        no_local_fallback: bool,
        reuse_end: int,
        keys: tuple[bytes, ...],
    ) -> FetchPlan:
        # Local reuse may reach past the target; a plan never asks below its own target.
        reuse_end = min(reuse_end, token_end // self._tpb)
        # A gen-init request's recurrent state is live, not a committed snapshot; it arrives by
        # position (PlacesPieces), so state groups are not part of the named ask.
        specs = [
            s
            for s in self._reader.group_specs()
            if not (no_local_fallback and s.kind is CacheKind.STATE)
        ]
        group_plans = tuple(
            GroupPlan(s, tuple(sorted(required_ordinals(s, token_end, reuse_end, self._tpb))))
            for s in specs
        )
        return FetchPlan(
            token_end=token_end,
            source=source.name,
            hint=hint,
            no_local_fallback=no_local_fallback,
            group_plans=group_plans,
            block_keys=keys,
            reuse_end=reuse_end,
            tokens_per_block=self._tpb,
        )

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
and chunks belongs to ``resource/``. ``servable_block_end`` (how far a store answer can take a
request) and ``served_token_end`` (how far a delivery did take it) are pure functions of names
and sets, so both rules can be tested with no engine at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from typing import Iterator, Mapping, NamedTuple, Sequence

# ``backend``: Chunk/CacheKind only; ``cache_backend``: the contract.
from .base.backend import CacheKind
from .base.cache_backend import Fetches
from .base.capabilities import LandsOnHost
from .base.views import GroupSpec, RequestView, ResourceView

__all__ = [
    "DEFER",
    "Defer",
    "FetchPlan",
    "FetchSource",
    "GroupPlan",
    "Planner",
    "served_token_end",
    "required_ordinals",
    "servable_block_end",
    "unit_names",
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
        name: Stable identifier; recorded on plans (``FetchPlan.source``) and attempts
            (``AttemptRecord.backend_name``), and what the coordinator maps back to the backend for
            ``quiesce``. The ranks agree on records by ``(request_id, direction)``, not by it.
        backend: The backend itself: one that writes the caller's pages directly (``Fetches``)
            or one that lands in its own host memory first (``LandsOnHost``; a store, so
            ``hint_key`` is ``None``).
        hint_key: Which routing hint on a request this backend reads. ``None`` for a backend whose
            destination is unique (a store), which then never gets ``open_route``.
    """

    name: str
    backend: Fetches | LandsOnHost
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
        group_plans: The per-group asks with their specs; what ``served_token_end`` and
            ``fetch_extent_and_committed`` read. Unit names are ``spec.tag + block_keys[o]`` for
            every ordinal ``o`` asked.
        block_keys: One key per full prompt block, by ordinal.
        reuse_end_blocks: Blocks the local radix tree already serves; nothing below it is asked for.
        tokens_per_block: ``tpb``.
    """

    token_end: int
    source: str
    hint: Mapping[str, object] | None
    no_local_fallback: bool
    group_plans: tuple[GroupPlan, ...]
    block_keys: tuple[bytes, ...]
    reuse_end_blocks: int
    tokens_per_block: int


def unit_names(plan: FetchPlan) -> tuple[bytes, ...]:
    """Every unit name the plan asks for, group by group and ordinal by ordinal: what a
    ``LandsOnHost`` backend is handed to land, and what a full delivery serves. An ordinal past
    the last key has no name and is skipped, as ``fetch_extent_and_committed`` skips it."""
    keys = plan.block_keys
    return tuple(
        group.spec.tag + keys[o]
        for group in plan.group_plans
        for o in group.ordinals
        if o < len(keys)
    )


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


def _required_ranges(
    spec: GroupSpec, history: int, reuse_end_blocks: int, tpb: int
) -> tuple[range, ...]:
    """``required_ordinals`` as at most two non-empty ranges: the sink blocks above the local
    prefix, then the live window above it (full attention: one range,
    ``[reuse_end_blocks, full)``). ``servable_block_end`` checks each range with one prefix-sum
    lookup instead of a set walk."""
    full = history // tpb
    if spec.kind is CacheKind.STATE:
        if history % tpb != 0 or full <= reuse_end_blocks:
            return ()
        return (range(full - 1, full),)
    beg, end = _stale_range(spec, history, tpb)
    ranges = (range(reuse_end_blocks, min(beg, full)), range(max(end, reuse_end_blocks), full))
    return tuple(r for r in ranges if len(r) > 0)


def required_ordinals(
    spec: GroupSpec, history: int, reuse_end_blocks: int, tpb: int
) -> frozenset[int]:
    """Block ordinals ``spec`` must have fetched for the request to stand at ``history``.

    Full attention: ``[reuse_end_blocks, full)``. Windowed: the sink blocks plus the live window,
    minus what is local. State: exactly the snapshot at ``history`` when it is a block boundary
    past the local prefix; otherwise nothing, because a partial tail's state has no name and
    moves by position.
    """
    return frozenset(chain.from_iterable(_required_ranges(spec, history, reuse_end_blocks, tpb)))


def _required_names(plan: FetchPlan, group: GroupPlan, history: int) -> frozenset[bytes]:
    """Unit names ``group`` needs at ``history``. An ordinal past the last key is skipped, exactly
    as ``Planner._build`` and ``naming.units_for_group`` skip it: it has no name and moves by
    position, so nothing named can be waited for."""
    ordinals = required_ordinals(group.spec, history, plan.reuse_end_blocks, plan.tokens_per_block)
    return frozenset(
        group.spec.tag + plan.block_keys[o] for o in ordinals if o < len(plan.block_keys)
    )


def _boundaries_down(plan: FetchPlan) -> Iterator[int]:
    """``token_end`` first, then every block boundary below it down to the local prefix."""
    tpb = plan.tokens_per_block
    floor = plan.reuse_end_blocks * tpb
    b = plan.token_end
    while b >= floor:
        yield b
        if b <= 0:
            return
        b = (b - 1) // tpb * tpb


def _served_token_end(
    plan: FetchPlan, served: frozenset[bytes], groups: Sequence[GroupPlan]
) -> int:
    for b in _boundaries_down(plan):
        ok = True
        for group in groups:
            if not _required_names(plan, group, b) <= served:
                ok = False
                break
        if ok:
            return b
    # Reached only by a plan whose ``reuse_end_blocks`` lies above ``token_end``, which ``_build``
    # never makes: no boundary was walked, nothing was asked for, and B is the local prefix end.
    return plan.reuse_end_blocks * plan.tokens_per_block


def served_token_end(plan: FetchPlan, served: frozenset[bytes]) -> int:
    """The largest B such that, with B as the sequence length, every group's still-needed units
    are in ``served`` (design §6.3). Never below the local prefix end.

    ``served_token_end`` asks which of the units the plan *asked for* arrived: it trims below
    ``plan.reuse_end_blocks`` and walks down from ``plan.token_end``. ``servable_block_end`` asks
    the other question, which target a probe answer over every nameable block can serve. Under a
    windowed group any block missing at or above ``stale_end(token_end)`` drags B down to the
    reuse floor: the window blocks a lower boundary needs were stale at ``token_end``, so they
    were never asked for and cannot be in ``served``.
    """
    return _served_token_end(plan, served, plan.group_plans)


def servable_block_end(
    answer: frozenset[bytes],
    keys: Sequence[bytes],
    specs: Sequence[GroupSpec],
    nameable_blocks: int,
    tpb: int,
) -> int:
    """The largest ``e`` in ``[1, nameable_blocks]`` such that every paged group has each block it
    needs at ``e * tpb`` in ``answer``; 0 when there is none (or no paged group).

    Computed with ``reuse_end_blocks = 0``: the decision must read only inputs every rank shares,
    and the local reuse depth is per rank. A block the store lacks but the local tree holds
    therefore still counts as missing, which is conservative (everything in the store implies
    everything above any local prefix is in the store). State groups are ignored: the assembly
    guard refuses them, and their snapshots are never published. ``served_token_end`` answers
    the complementary question about a plan that was fetched; see its docstring for how the two
    differ on one missing block.

    ``O(groups * nameable_blocks)``: one prefix sum of held blocks per group, then each candidate
    ``e`` checks its ranges by subtraction instead of rebuilding an ordinal set per ``e``.

    A block past the last key has no name and cannot be held, so ``nameable_blocks`` is clamped
    to ``len(keys)`` rather than indexing past it.
    """
    paged = [s for s in specs if s.kind is CacheKind.PAGED]
    nameable_blocks = min(nameable_blocks, len(keys))
    if not paged or nameable_blocks <= 0:
        return 0
    held_below = []
    for spec in paged:
        counts = [0] * (nameable_blocks + 1)
        for o in range(nameable_blocks):
            counts[o + 1] = counts[o] + (spec.tag + keys[o] in answer)
        held_below.append(counts)
    for e in range(nameable_blocks, 0, -1):
        if all(
            counts[r.stop] - counts[r.start] == len(r)
            for spec, counts in zip(paged, held_below)
            for r in _required_ranges(spec, e * tpb, 0, tpb)
        ):
            return e
    return 0


class _SourceChoice(NamedTuple):
    """What ``Planner._choose_source`` settled on: the source that can serve the request (``None``
    when none can), its routing hint, the token target it serves up to, and whether a store
    passed over has yet to answer its probe."""

    source: FetchSource | None
    hint: Mapping[str, object] | None
    token_end: int
    probe_pending: bool


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
        reader: ResourceView,
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

    def probe_query(self, request: RequestView) -> tuple[bytes, tuple[bytes, ...]] | None:
        """``(name, unit names)`` to ask a store about, or ``None`` if the request has nothing
        nameable to fetch. Rank-independent: it starts at block 0, not at the local prefix.

        Every group is asked about every nameable block, not only the blocks live at the largest
        target: ``servable_block_end`` may settle on a smaller target whose window needs blocks that
        are stale at the largest one, and one probe is one RPC however many names it carries.
        """
        if request.is_disagg_generation_init:
            return None
        nameable_blocks = (request.prompt_len - 1) // self._tpb
        keys = self._reader.block_keys(request)
        if nameable_blocks <= 0 or nameable_blocks > len(keys):
            return None
        specs = self._reader.group_specs()
        units = tuple(s.tag + keys[o] for s in specs for o in range(nameable_blocks))
        return keys[nameable_blocks - 1], units

    def forget(self, request_id: int) -> None:
        """Drop per-request planning state once a request is decided or gone."""
        self._first_deferred_at.pop(request_id, None)

    def _may_still_wait_for_probe(self, request_id: int, now: float) -> bool:
        """Whether ``request_id`` may be deferred once more for an unanswered probe.

        The first deferral only starts the clock; the wait ends once ``probe_timeout_s`` has
        passed since then, so a zero budget allows exactly one deferral.
        """
        first = self._first_deferred_at.get(request_id)
        if first is None:
            self._first_deferred_at[request_id] = now
            return True
        return self._probe_timeout_s is None or now - first < self._probe_timeout_s

    def decide(
        self,
        request: RequestView,
        probe_answers: Mapping[str, frozenset[bytes] | None],
        *,
        now: float,
        retry_cap: int | None = None,
    ) -> FetchPlan | None | Defer:
        """The decision table of design §7.2 for one candidate.

        Args:
            request: The candidate.
            probe_answers: Per store source, what it holds; ``None`` or missing means unanswered.
            now: The loop clock this round; the probe wait is measured on it.
            retry_cap: Upper bound on ``token_end`` after a short ``served`` (fetch retry): the
                merged B of the failed try. It caps the candidate targets before the store's
                answer is judged, so a retry may land below it, never above.
        """
        tpb = self._tpb
        reuse_end_blocks = self._reader.local_reuse_tokens(request) // tpb
        keys = tuple(self._reader.block_keys(request))

        if request.is_disagg_generation_init:
            return self._gen_init_plan(request, reuse_end_blocks, keys)

        if request.is_generation_first_context and not self._reader.generation_first_ready(request):
            return DEFER

        # The decision below reads only inputs every rank shares (prompt, hints, probe answers).
        # The local reuse depth differs per rank, so it only trims the per-group asks (possibly
        # to nothing, which is still a legal plan) and never decides whether to fetch.
        nameable_blocks = (request.prompt_len - 1) // tpb
        cap_blocks = (
            nameable_blocks if retry_cap is None else min(nameable_blocks, retry_cap // tpb)
        )
        if cap_blocks <= 0:
            # Nothing nameable to fetch, or a retry with nothing to aim for: compute locally; no
            # probe is worth waiting on.
            self.forget(request.py_request_id)
            return None

        choice = self._choose_source(request, probe_answers, cap_blocks, keys)
        if choice.source is None:
            if choice.probe_pending and self._may_still_wait_for_probe(request.py_request_id, now):
                return DEFER
            self.forget(request.py_request_id)
            return None

        self.forget(request.py_request_id)
        return self._build(
            choice.token_end, choice.source, choice.hint, False, reuse_end_blocks, keys
        )

    def _gen_init_plan(
        self, request: RequestView, reuse_end_blocks: int, keys: tuple[bytes, ...]
    ) -> FetchPlan | None:
        """A gen-init request's plan: the whole prompt from the context worker its hint names;
        ``None`` (compute locally) when no routed source matches a hint it carries."""
        source = self._gen_init_source(request)
        if source is None:
            return None
        hint = request.route_hints.get(source.hint_key) if source.hint_key is not None else None
        return self._build(request.prompt_len, source, hint, True, reuse_end_blocks, keys)

    def _choose_source(
        self,
        request: RequestView,
        probe_answers: Mapping[str, frozenset[bytes] | None],
        cap_blocks: int,
        keys: tuple[bytes, ...],
    ) -> _SourceChoice:
        """The first source in priority order that can serve the request, up to ``cap_blocks``.
        A routed source serves when the request carries its hint; a store serves as far as its
        probe answer reaches."""
        tpb = self._tpb
        probe_pending = False
        for source in self._sources:
            if source.hint_key is not None:
                if source.hint_key in request.route_hints:
                    return _SourceChoice(
                        source,
                        request.route_hints[source.hint_key],
                        cap_blocks * tpb,
                        probe_pending,
                    )
                continue
            answer = probe_answers.get(source.name)
            if answer is None:
                probe_pending = True
                continue
            end_blocks = servable_block_end(
                answer, keys, self._reader.group_specs(), cap_blocks, tpb
            )
            if end_blocks > 0:
                return _SourceChoice(source, None, end_blocks * tpb, probe_pending)
        return _SourceChoice(None, None, 0, probe_pending)

    def plan_from_answer(self, request: RequestView, token_end: int, source: str) -> FetchPlan:
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
        reuse_end_blocks = self._reader.local_reuse_tokens(request) // self._tpb
        keys = tuple(self._reader.block_keys(request))
        return self._build(token_end, chosen, None, False, reuse_end_blocks, keys)

    def _gen_init_source(self, request: RequestView) -> FetchSource | None:
        """The first routed source whose hint the request carries. A gen-init fetch has one
        possible origin, the context worker named by its hint; without a matching hint there is
        nothing to route to, and a routeless fetch would read from nowhere in particular."""
        for source in self._sources:
            if source.hint_key is not None and source.hint_key in request.route_hints:
                return source
        return None

    def _build(
        self,
        token_end: int,
        source: FetchSource,
        hint: Mapping[str, object] | None,
        no_local_fallback: bool,
        reuse_end_blocks: int,
        keys: tuple[bytes, ...],
    ) -> FetchPlan:
        # Local reuse may reach past the target; a plan never asks below its own target.
        reuse_end_blocks = min(reuse_end_blocks, token_end // self._tpb)
        # A gen-init request's recurrent state is live, not a committed snapshot; it arrives by
        # position (PlacesPieces), so state groups are not part of the named ask.
        specs = [
            s
            for s in self._reader.group_specs()
            if not (no_local_fallback and s.kind is CacheKind.STATE)
        ]
        group_plans = tuple(
            GroupPlan(
                s, tuple(sorted(required_ordinals(s, token_end, reuse_end_blocks, self._tpb)))
            )
            for s in specs
        )
        return FetchPlan(
            token_end=token_end,
            source=source.name,
            hint=hint,
            no_local_fallback=no_local_fallback,
            group_plans=group_plans,
            block_keys=keys,
            reuse_end_blocks=reuse_end_blocks,
            tokens_per_block=self._tpb,
        )

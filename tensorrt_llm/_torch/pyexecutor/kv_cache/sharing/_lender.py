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
"""The staging and in-place lenders, their leases, and the hooks the manager calls on free, shrink,
reuse reset and shutdown."""

from __future__ import annotations

import collections
import enum
import traceback
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Deque, Literal, Mapping, NamedTuple, Sequence

import numpy as np
import torch
from cuda.bindings import driver as drv
from cuda.bindings import runtime as cudart

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.logger import logger

from . import _manager
from ._identity import Identity
from ._layout import ManagerLayout, derive_layout, layout_id
from ._slots import Runs, Slots, slot_counts
from ._types import GroupRun, Part, Readiness, RegionView, StagingOptions

if TYPE_CHECKING:
    from tensorrt_llm.mapping import Mapping as TrtllmMapping
    from tensorrt_llm.runtime.kv_cache_manager_v2 import KVCacheManager, _KVCache

    from ...llm_request import LlmRequest
    from ..kv_cache_manager_v2 import KVCacheManagerV2

# Staging memory, caches and page-index buffers retained until the process exits, by identity,
# oldest first; only a clean shutdown or a last loan's end releases an entry. Each change is one
# dict operation, atomic under the GIL, so lenders of managers on different threads need no lock.
_retained_until_exit: dict[int, object] = {}

_SHUT_DOWN = "the KV cache manager shut down"
_RESET = "the KV cache manager reset its reuse state, after which nothing is lent by name"
_SUSPENDED = "the request's cache is suspended"
_PLACEHOLDERS = "the request carries multimodal data without digests, which its names do not cover"
_ENCODER = "the request has encoder input, which its names do not cover"
_SCRATCH = (
    "the request's cache has SWA scratch reuse on: each capacity change keeps its history within "
    "the scratch rewind of the old capacity, which a fetch's grow breaks"
)
_CONTEXT_OUTPUTS = (
    "the request returns context logits or asks for additional model outputs, which the executor "
    "gives only for computed positions and keeps across a rewind and a recompute pause"
)

_LeaseKind = Literal["read", "write"]


def _retained() -> tuple[object, ...]:
    """What is retained until exit, oldest first; for tests."""
    return tuple(_retained_until_exit.values())


def _retain_until_exit(owner: object) -> None:
    """Retain ``owner`` until the process exits, or until ``_release_retained``."""
    _retained_until_exit[id(owner)] = owner


def _release_retained(owner: object) -> None:
    """Stop retaining ``owner``, compared by identity."""
    _retained_until_exit.pop(id(owner), None)


class _HostMemory:
    """Host memory of exactly ``nbytes``, page-locked where pinning pays off. Pinned memory goes
    only through ``free``, once; pageable memory goes with the object, which the exit registry
    holds, so memory retained until exit stays mapped either way."""

    def __init__(self, nbytes: int) -> None:
        self.nbytes = nbytes
        self._pageable: np.ndarray | None = None
        if prefer_pinned():
            # The driver pins the size asked; torch's pinned allocator rounds it to a power of two.
            error, address = cudart.cudaHostAlloc(nbytes, cudart.cudaHostAllocDefault)
            if error != cudart.cudaError_t.cudaSuccess:
                raise MemoryError(f"pinning {nbytes} bytes of staging memory failed: {error}")
            self.address = int(address)
        else:
            self._pageable = np.empty(nbytes, dtype=np.uint8)
            self.address = int(self._pageable.ctypes.data)

    def free(self) -> None:
        """Release the memory; later calls do nothing."""
        address, self.address = self.address, 0
        if not address or self._pageable is not None:
            self._pageable = None
            return
        (error,) = cudart.cudaFreeHost(address)
        if error != cudart.cudaError_t.cudaSuccess:
            logger.warning(f"KV cache lender: freeing the staging memory failed: {error}")


def _allocate(nbytes: int) -> _HostMemory:
    """One host allocation of ``nbytes`` for the staging parts, retained until exit."""
    # TODO: staging duplicates the manager's host tier, whose host pools move when they resize
    # (mremap), so a backend cannot register them, and hold a pool group's row in one mapping per
    # pool rather than as one contiguous slot.
    memory = _HostMemory(max(int(nbytes), 1))
    _retain_until_exit(memory)
    return memory


def _mapping(manager: KVCacheManagerV2) -> TrtllmMapping:
    """The manager's mapping; ``TypeError`` for a manager without one."""
    mapping = getattr(manager, "mapping", None)
    if mapping is None:
        raise TypeError("the KV cache lender needs a manager with a mapping")
    return mapping


def _checked_layout(manager: KVCacheManagerV2, *, pipeline: bool) -> ManagerLayout:
    """The checks both attaches share: a v2 manager, no context parallelism, no pipeline
    parallelism unless ``pipeline``, no lender yet, no recurrent state, no sparse buffer."""
    _manager.require_v2(manager)
    mapping = _mapping(manager)
    cp_size = int(mapping.cp_size)
    if cp_size > 1:
        raise ValueError(
            f"context parallelism (cp_size={cp_size}) is not supported: its ranks would give "
            "different pages the same names"
        )
    # TODO: each pipeline stage's staging slots and windows are its own, so one call can raise on
    # one stage and lend on another.
    pp_size = int(mapping.pp_size)
    if not pipeline and pp_size > 1:
        raise ValueError(
            f"pipeline parallelism (pp_size={pp_size}) is not supported by staging: each stage's "
            "slots and windows are its own, so one call could raise on one stage and lend on "
            "another"
        )
    # TODO: one lender per manager, so staging and in-place never serve one manager together.
    if _manager.attached(manager) is not None:
        raise ValueError("a lender is already attached to this KV cache manager")
    layout = derive_layout(manager)
    recurrent = [lg for lg, state in enumerate(layout.recurrent) if state]
    if recurrent:
        raise ValueError(
            f"layer groups {recurrent} hold recurrent state, which the lender does not lend"
        )
    # TODO: lending a sparse layer group needs each page's memory tier: a cache can lock such a
    # group's read-only pages in host memory.
    sparse = [lg for lg, flag in enumerate(layout.sparse) if flag]
    if sparse:
        raise ValueError(
            f"layer groups {sparse} hold sparse buffers, whose read-only pages a cache can lock "
            "in host memory"
        )
    return layout


def _check_commits(manager: KVCacheManagerV2) -> None:
    """``ValueError`` for a manager that commits no blocks: a publish lends only committed ones."""
    if not _manager.commits_blocks(manager):
        raise ValueError(
            "staging needs a manager that commits blocks: block reuse on, and joint reuse for a "
            "draft manager; with block reuse off nothing could ever be published"
        )


def _check_lookahead(manager: KVCacheManagerV2) -> None:
    """``ValueError`` for a manager whose blocks may depend on tokens past their end: a name covers
    the tokens up to the block's end, so two requests sharing them could hold different bytes. A
    read-ahead upstream has not established, as one-model DraftTarget's, counts as none."""
    # TODO: staging serves no manager built for a draft that reads ahead (Eagle, MTP).
    lookahead = _manager.prompt_lookahead(manager)
    if lookahead > 0:
        # The target manager of a draft whose layers live elsewhere is refused too: that draft's own
        # pool is, so a fetch could fill the target's blocks but never the draft's.
        raise ValueError(
            "staging needs blocks that depend on no token past their end; this manager is built "
            f"for a one-model draft that reads {lookahead} prompt tokens ahead (Eagle or MTP), so "
            "requests sharing a block's tokens could hold different bytes under one name"
        )


def _check_connector(manager: KVCacheManagerV2) -> None:
    """``ValueError`` for a manager with a KV cache connector: it serves a request's prefix at the
    first context chunk, measured from the committed tokens, so it cannot lower a history a windowed
    fetch moved, and its loads run off the manager's stream."""
    if _manager.kv_connector(manager) is not None:
        raise ValueError(
            "staging does not serve a manager with a KV cache connector: the connector serves a "
            "request's prefix at its first context chunk, measured from the committed tokens, and "
            "cannot lower a history a fetch moved"
        )


def _attach_staging(
    manager: KVCacheManagerV2, *, scope: bytes, staging: StagingOptions, cls: type | None = None
) -> Staging:
    """``attach_staging`` with the lender class as a parameter, ``Staging`` when ``None``."""
    layout = _checked_layout(manager, pipeline=False)
    _check_commits(manager)
    _check_lookahead(manager)
    _check_connector(manager)
    if not isinstance(scope, bytes):
        raise TypeError(f"scope must be bytes, got {type(scope).__name__}")
    if not isinstance(staging, StagingOptions):
        raise TypeError(f"staging must be StagingOptions, got {type(staging).__name__}")
    counts = slot_counts(layout, staging)
    identity = Identity(scope, layout_id(layout.layout), layout.layers, layout.shards)
    slots = {g: int(counts.get(g, 0)) for g in layout.pool_groups}
    sizes = {g: slots[g] * int(layout.page_bytes[g]) for g in layout.pool_groups}
    # TODO: an exception after the allocation keeps the staging memory, and the page-index buffer
    # once retained, until the process exits.
    memory = _allocate(sum(sizes.values()))
    base = memory.address
    parts = []
    offset = 0
    for g in layout.pool_groups:
        name = identity.part_name(lg for lg, pg in enumerate(layout.pool_group_of) if pg == g)
        parts.append(Part(name, base + offset, sizes[g], int(layout.page_bytes[g]), slots[g]))
        offset += sizes[g]
    lender = (cls or Staging)(
        weakref.ref(manager), layout, identity, tuple(parts), Slots(slots), weakref.ref(memory)
    )
    lender._retain_index_buffer(manager)
    logger.info(
        f"KV cache lender: namespace {identity.namespace.hex()}, staging {offset >> 20} MiB "
        f"in {len(parts)} parts"
    )
    # Installed last, so a failure above leaves nothing attached.
    _manager.install(manager, lender)
    return lender


def attach_in_place(manager: KVCacheManagerV2) -> InPlace:
    """The public ``attach_in_place``: an ``InPlace`` lender installed on ``manager``."""
    layout = _checked_layout(manager, pipeline=True)
    lender = InPlace(weakref.ref(manager), layout)
    _manager.install(manager, lender)
    return lender


class _Copy:
    """Copies queued together, complete once their event says so. ``done`` asks the event at most
    once per round of the lender's progress, and never again once it reported completion."""

    def __init__(self, event: torch.cuda.Event) -> None:
        self._event = event
        self._done = False
        self._round: int | None = None

    def done(self, round_: int | None = None) -> bool:
        """Whether the copies have completed, without waiting; ``round_=None`` always asks."""
        if not self._done and (round_ is None or round_ != self._round):
            self._round = round_
            self._done = bool(self._event.query())
        return self._done

    def wait(self) -> None:
        """Block the host until the copies have completed."""
        if not self._done:
            self._event.synchronize()
            self._done = True


class _CopyResult(NamedTuple):
    """What ``_memcpy`` queued: the copies (``None`` if no event covers them) and the first error."""

    copy: _Copy | None
    error: str | None


def _no_cache(request_id: int) -> str:
    return f"request {request_id} has no KV cache"


def _stale(
    manager: KVCacheManagerV2, layout: ManagerLayout, lg: int, history: int
) -> tuple[int, int]:
    """Block ordinals ``[beg, end)`` behind layer group ``lg``'s window at ``history``."""
    if layout.windows[lg] is None:
        return 0, 0
    return _manager.stale_blocks(manager, lg, history)


def _needed_block_ranges(
    manager: KVCacheManagerV2,
    layout: ManagerLayout,
    lg: int,
    start_block: int,
    end_block: int,
    history: int,
) -> list[tuple[int, int]]:
    """Ordinal ranges ``[beg, end)`` of ``lg`` in ``[start_block, end_block)`` that a history of
    ``history`` tokens still reads, in order and none empty: the whole range for full attention;
    the sinks and the window otherwise. Arithmetic only, so a range of any length costs nothing."""
    stale_beg, stale_end = _stale(manager, layout, lg, history)
    ranges = ((start_block, min(end_block, stale_beg)), (max(start_block, stale_end), end_block))
    return [(beg, end) for beg, end in ranges if end > beg]


def _ordinals(ranges: Sequence[tuple[int, int]]) -> np.ndarray:
    """``int64`` ordinals of ``ranges``, in order."""
    if not ranges:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate([np.arange(beg, end, dtype=np.int64) for beg, end in ranges])


def _needed_ordinals(
    manager: KVCacheManagerV2,
    layout: ManagerLayout,
    lg: int,
    start_block: int,
    end_block: int,
    history: int,
) -> np.ndarray:
    """Ordinals of ``lg`` in ``[start_block, end_block)`` that a history of ``history`` tokens
    still reads: all of them for full attention; the sinks and the window otherwise."""
    return _ordinals(_needed_block_ranges(manager, layout, lg, start_block, end_block, history))


@dataclass(eq=False)
class _GroupRows:
    """One layer group's rows, aligned: block ordinals, their device pages (-1 where a block has
    none) and, once the lease is granted, their staging slots."""

    layer_group: int
    ordinals: np.ndarray
    device_pages: np.ndarray
    staging_slots: np.ndarray | None = None


@dataclass(eq=False)
class _Rows:
    """A lease's rows: one ``_GroupRows`` per layer group, in order."""

    groups: list[_GroupRows]

    @property
    def num_rows(self) -> int:
        return sum(len(group.ordinals) for group in self.groups)

    @property
    def layer_groups(self) -> list[int]:
        return [group.layer_group for group in self.groups]

    @property
    def ordinals(self) -> list[np.ndarray]:
        return [group.ordinals for group in self.groups]

    @property
    def device_pages(self) -> list[np.ndarray]:
        return [group.device_pages for group in self.groups]


class _FetchState(enum.Enum):
    """Where one write lease's fetch into a cache stands.

    ``OPEN``: the cache grew for it and its marks have not come; readiness is ``None`` and no new
    fetch into the cache starts. ``ABANDONED``: it delivered nothing (its lease failed, was released
    before its view was returned, or the copy of its marks failed); readiness counts what earlier
    fetches into the cache delivered. ``DELIVERED``: the copy of its marked rows was queued without
    error; the fetch has settled once that copy has completed."""

    OPEN = "open"
    ABANDONED = "abandoned"
    DELIVERED = "delivered"


@dataclass(eq=False)
class _Fetch:
    """One write lease's fetch into one cache, from ``start``; ``copy`` is the copy its marks
    queued."""

    start: int
    state: _FetchState = _FetchState.OPEN
    copy: _Copy | None = None

    @property
    def delivered(self) -> bool:
        return self.state is _FetchState.DELIVERED


class _Usable(NamedTuple):
    """Readiness's answer for ``computed_tokens``: the usable end, and the floor that keeps resumes
    out of bidirectional spans below it."""

    computed_tokens: int
    usable_until: int
    span_floor: int


@dataclass(eq=False)
class _Delivered:
    """What the fetches into one cache delivered: per layer group, by block ordinal, the rows whose
    last copy from staging was queued without error and that no shrink freed since; ``origin`` is
    the lowest fetch start."""

    origin: int
    blocks: list[np.ndarray]
    usable: _Usable | None = None  # the last answer, dropped whenever ``blocks`` change


class _GrowMark(NamedTuple):
    """Left by the latest fetch whose grow moved the history past the committed tokens: what it kept
    of the tokens computed before it, and the history its grow left."""

    kept_tokens: int
    history_after: int


@dataclass(eq=False)
class _RequestRecord:
    """What the lender remembers of one request's cache ``kv``. A record of a cache that another
    replaced (a restart) is moot."""

    kv: _KVCache
    fetch: _Fetch | None = None  # the latest fetch into ``kv``
    delivered: _Delivered | None = None
    grow_mark: _GrowMark | None = None
    # A failed copy may have left the committed tail block with the fetch's bytes in some pools and
    # the bytes it held before in others; readiness is empty from then on.
    torn_tail: bool = False


# TODO: a draft pool also gives up a resume below its history whose chunk would reach it, as
# every unchunked prefill's does.
def _floor_follows_history(manager: KVCacheManagerV2) -> bool:
    """Whether a request may resume only at or past its history: a chunk resumed below it may end
    below it, where the manager's context update raises, or leave a draft pool's capacity below it,
    where the pool's context resize raises."""
    return _manager.context_moves_history(manager) or _manager.resize_ends_at_chunk(manager)


class Staging:
    """``StagingLender`` over one manager, referenced weakly. Until the manager shuts down or is
    gone, a lend past its range check, a poll and a readiness call return settled leases' slots and
    grant waiting leases in order before their own work, a mark and a release after it."""

    def __init__(
        self,
        manager: weakref.ref[KVCacheManagerV2],
        layout: ManagerLayout,
        identity: Identity,
        parts: tuple[Part, ...],
        slots: Slots,
        memory: weakref.ref[_HostMemory],
    ) -> None:
        self._manager_ref = manager
        self._layout = layout
        self._identity = identity
        self._parts = tuple(parts)
        self._slots = slots
        # Only the exit registry holds the staging memory strongly; this finds it there at shutdown.
        self._memory = memory
        self._part_of_group = {g: i for i, g in enumerate(layout.pool_groups)}
        self._any_window = any(w is not None for w in layout.windows)
        self._records: dict[int, _RequestRecord] = {}  # by request id
        self._line: Deque[_StagingLease] = collections.deque()  # waiting for slots, in order
        self._holding: list[_StagingLease] = []  # granted, slots not yet returned
        # Open leases, held strongly: an open lease keeps the staging memory at shutdown even
        # once its holder has dropped it.
        self._unreleased: set[_StagingLease] = set()
        # Open parts holds, held strongly too: a dropped hold still keeps the memory.
        self._holds: set[_PartsHold] = set()
        self._quarantined: list[Runs] = []  # slots a failed copy may still touch; never reused
        self._closed = False
        self._reset = False  # the manager reset its reuse state: nothing is lent by name again
        self._index_buffer: object | None = None  # retained until the shutdown closed every cache
        self._round = 0  # rounds of progress, so each asks a copy's event at most once

    @property
    def parts(self) -> tuple[Part, ...]:
        """See ``StagingLender.parts``."""
        return self._parts

    def hold_parts(self) -> _PartsHold:
        """See ``StagingLender.hold_parts``."""
        if self._live() is None:
            # The memory was freed or kept at the shutdown; a later hold changes neither.
            return _PartsHold(None)
        hold = _PartsHold(self)
        self._holds.add(hold)
        return hold

    def lend_read(self, request: LlmRequest, start: int, end: int) -> _StagingLease:
        """See ``StagingLender.lend_read``."""
        start, end = self._whole_blocks(start, end)
        request_id = int(request.py_request_id)
        manager = self._live()
        if manager is None:
            return _StagingLease._failed(self, "read", request_id, _SHUT_DOWN)
        self._progress()
        kv, refusal = self._cache_to_lend(manager, request, request_id)
        if refusal is not None:
            return _StagingLease._failed(self, "read", request_id, refusal)
        state = _manager.cache_state(kv)
        if end > state.committed:
            raise ValueError(
                f"the range ends at {end}, past the {state.committed} committed tokens"
            )
        if not state.active:
            return _StagingLease._failed(self, "read", request_id, _SUSPENDED)
        # TODO: a publish leaves out window blocks the request's own window has passed, although
        # the prefix tree may still hold their committed pages, so a fetch whose window still keeps
        # such a block finds it missing.
        rows = self._rows(manager, kv, start, end)
        rows = self._drop_unpaged_and_stale(manager, state.history, rows)
        counts = self._counts([len(ordinals) for ordinals in rows.ordinals])
        self._check_fits(counts)
        keys = self._keys_for(manager, request, kv, rows.ordinals)
        lease = _StagingLease(self, "read", request_id, kv, rows, keys)
        self._open(lease, counts)
        return lease

    def lend_write(self, request: LlmRequest, start: int, end: int) -> _StagingLease:
        """See ``StagingLender.lend_write``. Its checks run in phases: ``ValueError`` from the call,
        the request's prompt and the layout; what this rank holds for the request; ``ValueError``
        against the cache; this rank's refusals. Only then does the cache grow."""
        start, end = self._whole_blocks(start, end)
        request_id = int(request.py_request_id)
        manager = self._live()
        if manager is None:
            return _StagingLease._failed(self, "write", request_id, _SHUT_DOWN)
        self._progress()
        counts, ordinals = self._check_write_range(manager, request, start, end)
        # The checks above read only the call, the request's prompt and the layout, so on a
        # manager not shut down a wrong call raises whatever this rank holds for the request.
        kv, refusal = self._cache_to_lend(manager, request, request_id)
        if refusal is not None:
            return _StagingLease._failed(self, "write", request_id, refusal)
        state = _manager.cache_state(kv)
        keys = self._check_write_against_cache(manager, request, kv, state, start, end, ordinals)
        # With a window the history moves to ``end``, so windows need pages only for the blocks a
        # history of that length reads.
        new_history = end if self._any_window else state.history
        refusal = self._write_refusal(manager, request, kv, state, end, new_history)
        if refusal is not None:
            return _StagingLease._failed(self, "write", request_id, refusal)
        # Read before the grow, which moves a windowed cache's history to ``end``.
        kept_tokens = self._computed_below_start(request_id, kv, start)
        # TODO: the V2 scheduler cannot reclaim the pages a parked fetch grew.
        # TODO: route capture stops reading a request's prepopulated length once it holds the routes
        # below it, until the request finishes, so a later resume leaves positions without routes.
        # TODO: with extra KV tokens the grow locks a page past the history it leaves, which nothing
        # writes, so a later all-reusable windowed lease that leaves that block behind fails.
        # TODO: an exception once the grow resized the cache, in its fill or a step below, can leave
        # no record of a windowed cache's moved history, so readiness counts tokens never fetched as
        # computed, or a fetch record nothing settles, so readiness stays None.
        if not _manager.grow(manager, request, kv, new_history, end):
            return _StagingLease._failed(
                self, "write", request_id, f"no free pages to grow the cache to {end} tokens"
            )
        # The cache has grown, and readiness accounts for it whatever this lease's outcome.
        record = self._record_for(request_id, kv)
        if new_history > state.committed:
            record.grow_mark = _GrowMark(kept_tokens, new_history)
        fetch = _Fetch(start)
        record.fetch = fetch
        rows = self._rows(manager, kv, start, end)
        lease = _StagingLease(self, "write", request_id, kv, rows, keys, fetch)
        missing = self._missing_pages(rows)
        if missing is not None:
            # It fails at its first poll, which abandons the fetch.
            lease._fail_at_first_poll(missing)
            self._unreleased.add(lease)
            return lease
        self._open(lease, counts)
        return lease

    def readiness(self, request: LlmRequest) -> Readiness | None:
        """See ``StagingLender.readiness``."""
        request_id = int(request.py_request_id)
        manager = self._live()
        if manager is None:
            raise ValueError(f"{_no_cache(request_id)}: {_SHUT_DOWN}")
        self._progress()
        kv = _manager.kv_of(manager, request_id)
        if kv is None:
            raise ValueError(_no_cache(request_id))
        history = _manager.cache_state(kv).history
        record = self._records.get(request_id)
        if record is None or record.kv is not kv:
            # None yet, or one of a replaced cache (a restart), which is moot.
            self._records.pop(request_id, None)
            record = _RequestRecord(kv)
        if self._waits_for(record.fetch):
            return None
        if record.torn_tail:
            # Empty: the request drops its cache and computes from 0.
            return Readiness(0, max(history, 1))
        computed = self._computed_tokens(request_id, kv)
        delivered = record.delivered
        if delivered is None:
            # Nothing delivered: what was computed and kept counts, resumed no lower than the
            # history, which a windowed fetch moved to its end.
            usable_until, span_floor = self._outside_bidirectional_spans(request, computed)
            return Readiness(usable_until, max(span_floor, history))
        # A shrink the manager did not report still shows as blocks past the capacity.
        self._forget_rows_past_capacity(delivered, kv)
        if delivered.usable is None or delivered.usable.computed_tokens != computed:
            usable_until = self._usable_until(manager, delivered, computed)
            spans = self._outside_bidirectional_spans(request, usable_until)
            delivered.usable = _Usable(computed, *spans)
        usable = delivered.usable
        floor = int(self._restart_floor(manager, history, delivered.origin))
        return Readiness(usable.usable_until, max(usable.span_floor, floor))

    def _on_free(
        self, request_id: int, kv_cache: _KVCache, after_close: Callable[[], None]
    ) -> bool:
        """Manager hook, after the request's cache left the map: its waiting leases fail and its
        record goes. Returns ``False``: the manager closes the cache itself. Logs its own errors."""
        try:
            if self._live() is None:
                return False
            # Granted leases go on: a read's copy is queued already, and a write's marks copy
            # nothing into pages other than the ones lent.
            self._fail_waiting(int(request_id), "the request was freed")
            self._records.pop(int(request_id), None)
            self._progress()
        except Exception:
            # Logged, not raised into the manager's free: the cache is then closed as usual, and
            # granted leases keep their slots.
            logger.error(f"KV cache lender: freeing request {request_id}: {traceback.format_exc()}")
        return False

    def _on_shrink(self, request_id: int, kv_cache: _KVCache) -> None:
        """Manager hook, right after the request's cache may have shrunk in place: delivered rows
        past its blocks lost their pages for good. Logs its own errors."""
        try:
            record = self._record(int(request_id), kv_cache)
            if self._live() is not None and record is not None and record.delivered is not None:
                self._forget_rows_past_capacity(record.delivered, kv_cache)
        except Exception:
            # Logged, not raised into the manager's resize; readiness forgets the rows it sees past
            # the capacity.
            logger.error(
                f"KV cache lender: shrink of request {request_id}: {traceback.format_exc()}"
            )

    def _on_reset(self) -> None:
        """Manager hook after a reuse reset, with every cache closed as an in-place weight update
        ends: names stop meaning the bytes computed, so none is lent by name again; waiting leases
        and fetches went at their requests' free; granted publishes finish. Never raises."""
        # TODO: lending by name stops for good after the manager resets its reuse state, since
        # nothing names the weights it computes with from then on.
        self._reset = True

    def _on_shutdown(self, impl: KVCacheManager) -> frozenset[object]:
        """Manager hook, first in its shutdown, acting once: waits for the copies whose completion
        it recorded, fails waiting leases, and frees the staging memory unless a lease or a hold is
        open or a slot was lost to a failed copy. Returns ``frozenset()``; logs its own errors."""
        if self._closed:
            return frozenset()
        try:
            # With page-locked staging and the fresh-page fill off, the lender's only host
            # wait: no more work comes on the stream.
            for lease in self._holding:
                if lease._copy is not None:
                    lease._copy.wait()
            self._round += 1
            self._recycle()
            for lease in list(self._line):
                self._fail(lease, _SHUT_DOWN)
            failing = [lease for lease in self._unreleased if lease._first_poll_failure is not None]
            for lease in failing:
                self._fail(lease, lease._first_poll_failure)
            self._closed = True
            if not self._memory_in_use():
                # Every copy on it has completed above, and no lease or hold reaches it any more.
                memory = self._memory()
                _release_retained(memory)
                memory.free()
            else:
                logger.warning(
                    f"KV cache lender: keeping {sum(p.nbytes for p in self._parts) >> 20} MiB of "
                    f"staging memory until exit: {len(self._unreleased)} leases open, "
                    f"{len(self._holds)} parts holds open, "
                    f"{len(self._quarantined)} slot runs lost to failed copies"
                )
        except Exception:
            # Logged, not raised into the manager's shutdown; the staging memory then stays
            # retained until exit.
            self._closed = True
            logger.error(f"KV cache lender: shutting down: {traceback.format_exc()}")
        return frozenset()

    def _on_caches_closed(self) -> None:
        """Manager hook, once its shutdown has closed every cache: none writes into the page-index
        buffer any more, so the lender releases it. A close that raises skips this call, and the
        buffer stays until a retried shutdown closes every cache, or until exit."""
        # TODO: a cache whose close raised in a free stays open outside the manager's map, and a
        # lease holding it can still close it into the buffer released here.
        if self._index_buffer is None:
            return
        self._release_index_buffer()

    # Each rule is a separate method.

    def _progress(self) -> None:
        """Return the slots of settled leases, then grant waiting leases in order."""
        if self._live() is None:
            return
        self._round += 1
        self._recycle()
        self._grant_waiting()

    def _copy_done(self, copy: _Copy | None) -> bool:
        """``copy`` has completed (or there is none), its event asked at most once this round."""
        return copy is None or copy.done(self._round)

    def _read_copy_done(self, lease: _StagingLease) -> bool:
        """A read lease's copy into its slots has completed."""
        return self._copy_done(lease._copy)

    def _slots_returnable(self, lease: _StagingLease) -> bool:
        """The lease's slots may return: no backend access possible and no copy on them pending.
        The copy is asked last, so a lease still lent costs no event query."""
        # A failed lease was never ready, so no backend has seen its slots.
        if lease._state is not _LeaseState.FAILED:
            if not lease._released:
                return False
            # A write whose view was returned keeps its slots until it is marked.
            if lease._kind == "write" and lease._state is _LeaseState.VIEW_RETURNED:
                return False
        # TODO: every lender call asks each released lease's pending copy again, so N calls while P
        # copies pend cost N*P event queries.
        return self._copy_done(lease._copy)

    def _rows_on_lent_pages(self, kv: _KVCache | None, lease: _StagingLease) -> list[np.ndarray]:
        """Per layer group, the rows whose block is still in the window of the same active cache and
        still locks the GPU page lent; a page only held may sit on another tier under the same
        number."""
        groups = lease._rows.groups
        manager = self._manager_ref()
        state = _manager.cache_state(kv) if kv is not None and kv is lease._kv else None
        masks = []
        for group in groups:
            lg, ordinals = group.layer_group, group.ordinals
            same = np.zeros(len(ordinals), dtype=bool)
            if state is not None and state.active:
                pages = _manager.locked_pages(kv, lg)
                inside = ordinals < len(pages)
                same[inside] = pages[ordinals[inside]] == group.device_pages[inside]
                stale_beg, stale_end = _stale(manager, self._layout, lg, state.history)
                same &= (ordinals < stale_beg) | (ordinals >= stale_end)
            masks.append(same)
        return masks

    def _unnamed(self, request: LlmRequest) -> str | None:
        """Why the request's KV depends on more than its names cover, if it does: multimodal data
        the manager keys by placeholder tokens alone, or encoder input."""
        if _manager.keyed_by_placeholders(request):
            return _PLACEHOLDERS
        if _manager.has_encoder_input(request):
            return _ENCODER
        return None

    # TODO: the lender cannot see the model's sliding window, so it also keeps fetch ends and
    # resumes out of spans that window would cover whole.
    def _splits_bidirectional_span(self, request: LlmRequest, end: int) -> str | None:
        """Why the fetch ends strictly inside a span of multimodal tokens the scheduler keeps within
        one context chunk, if it does, whatever the span's length."""
        for b, e in _manager.bidirectional_spans(request):
            if b < end < e:
                return (
                    f"the range ends at {end}, inside the multimodal tokens [{b}, {e}), a run the "
                    "scheduler keeps within one context chunk"
                )
        return None

    def _outside_bidirectional_spans(self, request: LlmRequest, usable: int) -> tuple[int, int]:
        """The end lowered to the start of a span of multimodal tokens it falls strictly inside, and
        the lowest floor that then leaves no position strictly inside a span below that end: the
        end of the last span at or below it. Spans are those the scheduler keeps within one chunk."""
        spans = _manager.bidirectional_spans(request)
        for b, e in spans:
            if b < usable < e:
                usable = b
        return int(usable), max((e for _, e in spans if e <= usable), default=0)

    def _scratch_reuse_on(self, kv: _KVCache) -> bool:
        """A write target with SWA scratch reuse on, windowed or not: each capacity change keeps its
        history within the scratch rewind of the old capacity, which a fetch's grow breaks, and a
        window's next chunk overwrites scratch slots."""
        return _manager.scratch_reuse(kv)

    # TODO: the executor gives context outputs only for the positions a request computes, prompt
    # logprobs pair the logits with the prompt from its second token wherever they start, and a
    # rewind or a recompute pause keeps those held, so a request returning them takes no fetch.
    def _returns_context_outputs(self, request: LlmRequest) -> bool:
        """Whether the request returns context logits or asks for additional model outputs, which a
        fetch leaves without the fetched positions, or after a context step gapped or repeated."""
        return _manager.returns_context_outputs(request)

    def _history_refusal(
        self, manager: KVCacheManagerV2, history: int, new_history: int, end: int
    ) -> str | None:
        """Why the history the fetch leaves, ``new_history``, admits no resume at the positions
        readiness would count, or ``None``: in a cache without a window whose floor follows the
        history, the history stands past ``end``."""
        if self._any_window:
            return None  # the fetch moves the history to ``end``
        if not _floor_follows_history(manager):
            return None  # the floor is a delivered fetch's lowest start, or the history if lower
        if new_history > end:
            return f"the cache's history of {history} tokens stands past the fetch's end, {end}"
        return None

    def _unwritten_pages_refusal(
        self,
        manager: KVCacheManagerV2,
        request_id: int,
        kv: _KVCache,
        history: int,
        new_history: int,
    ) -> str | None:
        """Why moving the history to ``new_history`` would have the commit store bytes the request
        never wrote, or ``None``: a window leaving behind a block whose page holds tokens past the
        history keeps that page for the commit (a partial match's copy, a page grown early), and so
        does one an earlier windowed lease moved the history past without delivering it."""
        if not _manager.keeps_passed_pages(manager):
            return None
        tpb = int(self._layout.tokens_per_block)
        # Below the history the request wrote what it computed and the rows fetches delivered; only
        # an earlier windowed lease's grow leaves the history past what it computed.
        first = min(history, self._computed_tokens(request_id, kv)) // tpb
        record = self._record(request_id, kv)
        delivered = record.delivered if record is not None else None
        for lg in range(self._layout.num_layer_groups):
            stale_beg, stale_end = _stale(manager, self._layout, lg, new_history)
            beg = max(first, stale_beg)
            if beg >= stale_end:
                continue
            unwritten = _manager.locked_pages(kv, lg)[beg:stale_end] >= 0
            if delivered is not None:
                got = delivered.blocks[lg][beg : beg + len(unwritten)]
                unwritten[: len(got)] &= ~got
            blocks = (beg + np.nonzero(unwritten)[0]).tolist()
            missed = [b for b in blocks if b < history // tpb]
            # TODO: no runtime call drops one block's page in one layer group, so the fetch fails
            # where the commit could store no page for those blocks instead.
            if missed:
                return (
                    f"layer group {lg} leaves blocks {missed[:8]} behind its window at "
                    f"{new_history} tokens; an earlier lease moved the history past them to "
                    f"{history} tokens without delivering them, so their pages hold bytes the "
                    "request never wrote, which the commit would store"
                )
            if blocks:
                return (
                    f"layer group {lg} leaves blocks {blocks[:8]} behind its window at "
                    f"{new_history} tokens; past the history of {history} tokens their pages hold "
                    "bytes the request never wrote, which the commit would store"
                )
        return None

    def _unsettled(self, request_id: int, kv: _KVCache) -> str | None:
        """Why the request's earlier fetch into ``kv`` keeps a new one from starting, or ``None``:
        it has not settled on this rank, where the copy its marks queued may still be pending."""
        record = self._record(request_id, kv)
        if record is not None and self._waits_for(record.fetch):
            return f"request {request_id} already has an unsettled fetch"
        return None

    def _fetch_settled(self, fetch: _Fetch) -> bool:
        """The fetch's arrived rows are marked and their copy into the request's pages is done."""
        return fetch.delivered and self._copy_done(fetch.copy)

    def _fail_waiting(self, request_id: int, reason: str) -> None:
        """Fail the request's leases still waiting for slots."""
        for lease in [lease for lease in self._line if lease._request_id == request_id]:
            self._fail(lease, reason)

    def _abandon(self, lease: _StagingLease) -> None:
        """Abandon the write lease's fetch if it is the request's latest: it delivered nothing, and
        readiness counts what earlier fetches into the cache delivered."""
        if self._is_latest_fetch(lease):
            lease._fetch.state = _FetchState.ABANDONED

    def _computed_tokens(self, request_id: int, kv: _KVCache) -> int:
        """What readiness counts as computed besides delivered rows: the tokens up to the history,
        but only the committed ones and what the latest fetch kept while the history stays where
        that fetch's grow left it past them (a windowed fetch moves it to its end)."""
        state = _manager.cache_state(kv)
        record = self._record(request_id, kv)
        mark = record.grow_mark if record is not None else None
        # Past where that fetch left it, the history moved as the request resumed and computed on.
        if mark is None or state.history > mark.history_after:
            return max(state.committed, state.history)
        return max(state.committed, mark.kept_tokens)

    def _computed_below_start(self, request_id: int, kv: _KVCache, start: int) -> int:
        """What a fetch from ``start`` keeps of what was computed before it, since it may overwrite
        every block from ``start`` on. It never passes the history, which no shrink goes below;
        delivered rows past it keep their own record."""
        return min(self._computed_tokens(request_id, kv), start)

    def _names(self, layer_group: int, keys: np.ndarray) -> np.ndarray:
        """The rows' names: ``uint8 (n, 54)`` for ``keys`` ``uint8 (n, 32)``."""
        return self._identity.names(layer_group, keys)

    def _memory_in_use(self) -> bool:
        """A lease or a hold is unreleased, or a slot is lost to a failed copy: the memory stays."""
        return bool(self._unreleased) or bool(self._holds) or bool(self._quarantined)

    def _retain_index_buffer(self, manager: KVCacheManagerV2) -> None:
        """Retain the manager's page-index buffer until its shutdown has closed every cache: a cache
        a lease or a record holds writes its page indices there as it closes, also once the manager
        is gone."""
        self._index_buffer = _manager.index_buffer(manager)
        _retain_until_exit(self._index_buffer)

    def _release_index_buffer(self) -> None:
        """Once the manager's shutdown has closed every cache: no cache writes into the page-index
        buffer any more."""
        _release_retained(self._index_buffer)
        self._index_buffer = None

    def _end_hold(self, hold: _PartsHold) -> None:
        """The hold's holder let go; ``KeyError`` for a hold not open, which its guard prevents."""
        self._holds.remove(hold)

    def _check_fits(self, counts: Mapping[int, int]) -> None:
        """``ValueError`` if a lease needing ``counts`` rows per pool group can never be granted."""
        try:
            self._slots.check(counts)
        except ValueError as error:
            raise ValueError(
                f"the range needs more staging slots than there are ({error}); ranges of at most "
                "fetch_tokens tokens always fit"
            ) from None

    def _free_slots(self, group: int) -> int:
        """Free slots of a pool group; for tests."""
        return self._slots.free_slots(group)

    def _open_count(self) -> int:
        """Leases not yet released; for tests."""
        return len(self._unreleased)

    # The phases of a lend.

    def _cache_to_lend(
        self, manager: KVCacheManagerV2, request: LlmRequest, request_id: int
    ) -> tuple[_KVCache | None, str | None]:
        """The request's cache, or why this rank lends none of it: a reuse reset, no cache, or KV
        that depends on more than the request's names cover."""
        if self._reset:
            return None, _RESET
        kv = _manager.kv_of(manager, request_id)
        if kv is None:
            return None, _no_cache(request_id)
        unnamed = self._unnamed(request)
        if unnamed is not None:
            return None, unnamed
        return kv, None

    def _check_write_range(
        self, manager: KVCacheManagerV2, request: LlmRequest, start: int, end: int
    ) -> tuple[dict[int, int], list[np.ndarray]]:
        """``ValueError`` for a fetch range wrong by the request's prompt or the layout; else the
        rows it needs per pool group, and per layer group the ordinals a history of ``end`` reads."""
        layout = self._layout
        tpb = int(layout.tokens_per_block)
        # The request computes its last prompt token itself, for its logits, so a fetch ends at
        # the whole blocks before it.
        prompt = _manager.prompt_length(request)
        if end > (prompt - 1) // tpb * tpb:
            raise ValueError(
                f"the range ends at {end}, past the whole blocks before the request's last prompt "
                f"token ({prompt} prompt tokens)"
            )
        # Rows and their names come from the layout for a history of ``end``, so every
        # ValueError is raised before the cache changes. The rows are counted from their ranges
        # and checked to fit before any array is built.
        ranges = [
            _needed_block_ranges(manager, layout, lg, start // tpb, end // tpb, end)
            for lg in range(layout.num_layer_groups)
        ]
        counts = self._counts([sum(e - b for b, e in lg_ranges) for lg_ranges in ranges])
        self._check_fits(counts)
        return counts, [_ordinals(lg_ranges) for lg_ranges in ranges]

    def _check_write_against_cache(
        self,
        manager: KVCacheManagerV2,
        request: LlmRequest,
        kv: _KVCache,
        state: _manager.CacheState,
        start: int,
        end: int,
        ordinals: Sequence[np.ndarray],
    ) -> list[np.ndarray]:
        """``ValueError`` for a fetch range the cache rules out, raised before the cache changes;
        else the rows' reuse keys, which raise for a request with too few whole blocks."""
        tpb = int(self._layout.tokens_per_block)
        if start < (state.committed // tpb) * tpb:
            raise ValueError(
                f"the range starts at {start}, inside the committed whole blocks of "
                f"{state.committed} tokens"
            )
        keys = self._keys_for(manager, request, kv, ordinals)
        if self._any_window and end < state.history:
            raise ValueError(
                f"the range ends at {end}, below the history of {state.history} tokens its "
                "windows keep"
            )
        return keys

    def _write_refusal(
        self,
        manager: KVCacheManagerV2,
        request: LlmRequest,
        kv: _KVCache,
        state: _manager.CacheState,
        end: int,
        new_history: int,
    ) -> str | None:
        """Why this rank fails the fetch at the call, once no ``ValueError`` is due, or ``None``."""
        split = self._splits_bidirectional_span(request, end)
        if split is not None:
            return split
        if not state.active:
            return _SUSPENDED
        if self._scratch_reuse_on(kv):
            return _SCRATCH
        if self._returns_context_outputs(request):
            return _CONTEXT_OUTPUTS
        # After every ValueError: whether an earlier fetch settled is this rank's own timing.
        request_id = int(request.py_request_id)
        unsettled = self._unsettled(request_id, kv)
        if unsettled is not None:
            return unsettled
        refusal = self._history_refusal(manager, state.history, new_history, end)
        if refusal is None:
            refusal = self._unwritten_pages_refusal(
                manager, request_id, kv, state.history, new_history
            )
        return refusal

    # Request records.

    def _record(self, request_id: int, kv: _KVCache) -> _RequestRecord | None:
        """The request's record if it is of ``kv``; a record of another cache stays as it is."""
        record = self._records.get(request_id)
        return record if record is not None and record.kv is kv else None

    def _record_for(self, request_id: int, kv: _KVCache) -> _RequestRecord:
        """The request's record of ``kv``: a new one in place of none or of a replaced cache's."""
        record = self._record(request_id, kv)
        if record is None:
            record = self._records[request_id] = _RequestRecord(kv)
        return record

    def _waits_for(self, fetch: _Fetch | None) -> bool:
        """Readiness and a new fetch into the cache wait for ``fetch``: neither abandoned nor
        settled."""
        return (
            fetch is not None
            and fetch.state is not _FetchState.ABANDONED
            and not self._fetch_settled(fetch)
        )

    def _is_latest_fetch(self, lease: _StagingLease) -> bool:
        """The write lease's fetch is its request's latest, and not abandoned."""
        fetch = lease._fetch
        record = self._records.get(lease._request_id)
        return (
            fetch is not None
            and record is not None
            and record.fetch is fetch
            and fetch.state is not _FetchState.ABANDONED
        )

    # Lease records, slots and grants.

    def _live(self) -> KVCacheManagerV2 | None:
        """The manager while the lender serves it; ``None`` once it shut down or is gone."""
        if self._closed:
            return None
        return self._manager_ref()

    def _whole_blocks(self, start: int, end: int) -> tuple[int, int]:
        start, end = int(start), int(end)
        tpb = int(self._layout.tokens_per_block)
        if start < 0 or end < 0 or start > end:
            raise ValueError(f"bad token range [{start}, {end})")
        if start % tpb or end % tpb:
            raise ValueError(
                f"a staging lease covers whole blocks of {tpb} tokens, got [{start}, {end})"
            )
        return start, end

    def _counts(self, rows: Sequence[int]) -> dict[int, int]:
        """Rows per pool group, from the rows of each layer group."""
        counts: dict[int, int] = {}
        for lg, lg_rows in enumerate(rows):
            g = int(self._layout.pool_group_of[lg])
            counts[g] = counts.get(g, 0) + int(lg_rows)
        return counts

    def _open(self, lease: _StagingLease, counts: Mapping[int, int]) -> None:
        """Record a new lease and grant it now when no lease waits and its slots are free, or at
        once, out of line, when it has no rows."""
        self._unreleased.add(lease)
        if lease._rows.num_rows == 0:
            # Nothing to stage: ready at the first poll, without slots or a place in line.
            self._grant(lease, None)
            return
        lease._ticket = self._slots.ask(counts)
        self._line.append(lease)
        self._grant_waiting(fresh=lease)

    def _grant_waiting(self, fresh: _StagingLease | None = None) -> None:
        """Grant the leases in line, strictly in order; ``fresh`` was looked up in this call."""
        while self._line:
            head = self._line[0]
            # TODO: an exception while slots are taken, a MemoryError say, can leave slots taken for
            # no lease, or drop the head's ticket with the head still in line, after which every
            # round of progress raises "not waiting" until the head fails.
            runs = self._slots.take(head._ticket)
            if runs is None:
                return
            self._line.popleft()
            head._ticket = None
            try:
                self._grant(head, runs, recheck=head is not fresh)
            except Exception as error:
                # One broken grant fails only its own lease: it neither stalls the line nor raises
                # out of another lease's call. Slots a copy may have been queued on stay unused.
                if any(lease is head for lease in self._holding):
                    self._quarantine(head)
                else:
                    self._slots.give(runs)
                reason = f"granting staging slots failed: {error!r}"
                if head is fresh and head._kind == "write":
                    # The cache grew for it, so it fails at its first poll, which abandons the
                    # fetch, as a lease missing pages does.
                    head._fail_at_first_poll(reason)
                    head._view = head._runs = None
                else:
                    self._fail(head, reason)
                logger.warning(f"KV cache lender: {reason}")

    def _grant(self, lease: _StagingLease, runs: Runs | None, recheck: bool = False) -> None:
        """Give ``lease`` its slots and view; a read then queues its copy into them."""
        if recheck and lease._kind == "read":
            problem = self._source_changed(lease)
            if problem is not None:
                self._slots.give(runs)
                self._fail(lease, problem)
                return
        self._assign_staging(lease._rows, runs)
        lease._view = self._view(lease._rows, lease._keys)
        lease._state = _LeaseState.GRANTED
        if runs is None:
            return
        lease._runs = runs
        self._holding.append(lease)
        if lease._kind != "read":
            return
        copy, error = self._memcpy(self._segments(lease._rows), to_staging=True)
        if copy is None:
            self._quarantine(lease)
        else:
            lease._copy = copy
        if error is not None:
            self._fail(lease, f"the copy into staging failed: {error}")

    def _source_changed(self, lease: _StagingLease) -> str | None:
        """Why a read granted after waiting cannot copy the pages it looked up any more, if so."""
        manager = self._manager_ref()
        kv = _manager.kv_of(manager, lease._request_id)
        if kv is not lease._kv:
            return "the request's cache was freed while the lease waited"
        state = _manager.cache_state(kv)
        if not state.active:
            return "the request's cache was suspended while the lease waited"
        for group in lease._rows.groups:
            lg, ordinals = group.layer_group, group.ordinals
            pages = _manager.locked_pages(kv, lg)
            if np.any(ordinals >= len(pages)) or np.any(pages[ordinals] != group.device_pages):
                return f"layer group {lg}: pages changed while the lease waited"
            stale_beg, stale_end = _stale(manager, self._layout, lg, state.history)
            if np.any((ordinals >= stale_beg) & (ordinals < stale_end)):
                return f"layer group {lg}: blocks left the window while the lease waited"
        return None

    def _fail(self, lease: _StagingLease, reason: str) -> None:
        """Fail a lease never seen ready: it leaves the line, and a write abandons its fetch."""
        if lease._state is _LeaseState.FAILED:
            return
        lease._set_failure(reason)
        if lease._ticket is not None:
            self._slots.cancel(lease._ticket)
            lease._ticket = None
            self._line.remove(lease)
        if lease._kind == "write":
            self._abandon(lease)

    def _quarantine(self, lease: _StagingLease) -> None:
        """Keep the lease's slots from reuse for good: a copy on them may still be running."""
        self._holding = [held for held in self._holding if held is not lease]
        # TODO: quarantined slots are never reused, so failed copies erode staging capacity: a lease
        # needing a longer run than any left between lost slots waits in line for good, holding up
        # every lease behind it.
        if lease._runs is not None:
            self._quarantined.append(lease._runs)
        logger.warning(
            f"KV cache lender: a failed copy took staging slots of request {lease._request_id}"
        )

    # TODO: every lender call checks every lease holding slots, even with no copy pending, so
    # polling H such leases once each costs about H*H checks.
    def _recycle(self) -> None:
        """Return the slots of every lease ``_slots_returnable`` allows."""
        # Every lease is judged before any slot returns: an event query that raises returns none
        # and leaves every lease held for the next call.
        returnable = [self._slots_returnable(lease) for lease in self._holding]
        # TODO: an exception while slots return, a MemoryError say, can leave a lease held after
        # some or all of its slots returned, so every later round of progress raises "freed twice".
        holding = []
        for lease, done in zip(self._holding, returnable):
            if done:
                self._slots.give(lease._runs)
            else:
                holding.append(lease)
        self._holding = holding

    def _on_release(self, lease: _StagingLease) -> None:
        """The lease's holder let go. The record changes first and progress runs last, so an
        unexpected error leaves the fetch abandoned rather than unsettled."""
        self._unreleased.discard(lease)
        if self._live() is None:
            return
        if lease._ticket is not None:
            self._fail(lease, "released while waiting for staging slots")
        elif lease._kind == "write" and lease._state in _BEFORE_VIEW:
            # Released before anyone saw it ready: no backend wrote, and no marks are due.
            self._abandon(lease)
        self._progress()

    def _apply_marks(self, lease: _StagingLease, masks: list[np.ndarray]) -> None:
        """Copy the marked rows whose page is still the one lent into the request's pages. The
        fetch stays abandoned until that copy is queued without error; then it is delivered and its
        rows add to the cache's deliveries."""
        fetch = lease._fetch
        latest = self._is_latest_fetch(lease)
        if latest:
            fetch.state = _FetchState.ABANDONED
        kv = _manager.kv_of(self._manager_ref(), lease._request_id)
        lent = self._rows_on_lent_pages(kv, lease)
        copy = [mask & still for mask, still in zip(masks, lent)]
        error = None
        if any(c.any() for c in copy):
            segments = self._segments(lease._rows, copy)
            self._forget_overwritten_rows(lease, copy)
            tail = self._covers_committed_tail(kv, lease._rows, copy)
            queued = None
            try:
                queued, error = self._memcpy(segments, to_staging=False)
            finally:
                # A copy may be queued without its completion recorded, also when the call raised.
                if queued is None:
                    self._quarantine(lease)
                if tail and (queued is None or error is not None):
                    self._record_for(lease._request_id, kv).torn_tail = True
            lease._copy = queued
        if latest and error is None:
            fetch.copy = lease._copy
            fetch.state = _FetchState.DELIVERED
            self._deliver(lease._request_id, fetch, lease._rows, copy)
        self._progress()

    def _covers_committed_tail(self, kv: _KVCache, rows: _Rows, copied: list[np.ndarray]) -> bool:
        """The copy writes the block the committed tokens end inside, whose committed tokens
        readiness counts without a delivery."""
        tpb = int(self._layout.tokens_per_block)
        committed = _manager.cache_state(kv).committed
        if committed % tpb == 0:
            return False
        tail = committed // tpb
        return any(
            bool(np.any(group.ordinals[mask] == tail)) for group, mask in zip(rows.groups, copied)
        )

    def _forget_overwritten_rows(self, lease: _StagingLease, copied: list[np.ndarray]) -> None:
        """Stop counting the delivered rows ``copied`` selects, before their copy is queued: one
        that fails partway can leave a row holding each fetch's bytes in different pools.
        ``_deliver`` counts them again once the copy is queued without error."""
        record = self._record(lease._request_id, lease._kv)
        if record is None or record.delivered is None:
            return
        delivered = record.delivered
        for group, mask in zip(lease._rows.groups, copied):
            blocks = delivered.blocks[group.layer_group]
            rows = group.ordinals[mask]
            rows = rows[rows < len(blocks)]
            if blocks[rows].any():
                blocks[rows] = False
                delivered.usable = None

    def _deliver(
        self, request_id: int, fetch: _Fetch, rows: _Rows, copied: list[np.ndarray]
    ) -> None:
        """Add a fetch's copied rows to what earlier fetches into the same cache delivered, so a
        fetch split into consecutive leases counts as one."""
        record = self._records[request_id]  # the record of the cache ``fetch`` went into
        if record.delivered is None:
            empty = [np.zeros(0, dtype=bool) for _ in range(self._layout.num_layer_groups)]
            record.delivered = _Delivered(fetch.start, empty)
        delivered = record.delivered
        delivered.origin = min(delivered.origin, fetch.start)
        for group, mask in zip(rows.groups, copied):
            got = group.ordinals[mask]
            if not len(got):
                continue
            blocks = delivered.blocks[group.layer_group]
            if int(got.max()) >= len(blocks):
                blocks = np.concatenate([blocks, np.zeros(int(got.max()) + 1 - len(blocks), bool)])
            blocks[got] = True
            delivered.blocks[group.layer_group] = blocks
        delivered.usable = None

    def _forget_rows_past_capacity(self, delivered: _Delivered, kv: _KVCache) -> None:
        """Forget delivered rows at or past the cache's block count: a shrink freed their pages,
        and a regrow brings pages without their contents."""
        num_blocks = _manager.num_blocks(kv)
        for blocks in delivered.blocks:
            if blocks[num_blocks:].any():
                blocks[num_blocks:] = False
                delivered.usable = None

    def _rows(self, manager: KVCacheManagerV2, kv: _KVCache, start: int, end: int) -> _Rows:
        """The blocks of ``[start, end)`` a history of ``end`` reads, per layer group, with their
        device pages (-1 where a block has no page)."""
        layout = self._layout
        tpb = int(layout.tokens_per_block)
        groups = []
        for lg in range(layout.num_layer_groups):
            ordinals = _needed_ordinals(manager, layout, lg, start // tpb, end // tpb, end)
            pages = _manager.pages(kv, lg)
            device_pages = np.full(len(ordinals), -1, dtype=np.int64)
            inside = ordinals < len(pages)
            device_pages[inside] = pages[ordinals[inside]]
            groups.append(_GroupRows(lg, ordinals, device_pages))
        return _Rows(groups)

    def _drop_unpaged_and_stale(
        self, manager: KVCacheManagerV2, history: int, rows: _Rows
    ) -> _Rows:
        """``rows`` without blocks that have no page or that the request's own window has passed
        (their page may hold something else)."""
        groups = []
        for group in rows.groups:
            lg, ordinals, pages = group.layer_group, group.ordinals, group.device_pages
            stale_beg, stale_end = _stale(manager, self._layout, lg, history)
            ok = (pages >= 0) & ~((ordinals >= stale_beg) & (ordinals < stale_end))
            groups.append(_GroupRows(lg, ordinals[ok], pages[ok]))
        return _Rows(groups)

    def _missing_pages(self, rows: _Rows) -> str | None:
        """Why a fetch into ``rows`` cannot go on, if so: the first layer group with blocks that
        have no page."""
        for group in rows.groups:
            unpaged = group.device_pages < 0
            if np.any(unpaged):
                missing = group.ordinals[unpaged].tolist()
                return f"layer group {group.layer_group}: blocks {missing[:8]} have no page"
        return None

    def _keys_for(
        self,
        manager: KVCacheManagerV2,
        request: LlmRequest,
        kv: _KVCache,
        ordinals: Sequence[np.ndarray],
    ) -> list[np.ndarray]:
        """Per layer group, the reuse keys of its rows' blocks, ``uint8 (n, 32)``."""
        # TODO: every lease hashes the request's whole prefix again from block 0.
        blocks_hashed = max((int(o.max()) + 1 for o in ordinals if len(o)), default=0)
        keys = _manager.block_keys(manager, request, kv, blocks_hashed)
        columns = [b"".join(keys[int(o)] for o in lg_ordinals) for lg_ordinals in ordinals]
        return [np.frombuffer(column, dtype=np.uint8).reshape(-1, 32) for column in columns]

    def _assign_staging(self, rows: _Rows, runs: Runs | None) -> None:
        """Each pool group's run goes to its layer groups in order, so the rows of one pool group
        occupy consecutive slots."""
        cursor: dict[int, int] = {}
        for group in rows.groups:
            g = int(self._layout.pool_group_of[group.layer_group])
            start = runs.runs.get(g, (0, 0))[0] if runs is not None else 0
            offset = cursor.get(g, 0)
            count = len(group.ordinals)
            group.staging_slots = np.arange(start + offset, start + offset + count, dtype=np.int64)
            cursor[g] = offset + count

    def _view(self, rows: _Rows, keys: list[np.ndarray]) -> RegionView:
        runs = []
        for group, lg_keys in zip(rows.groups, keys):
            lg = group.layer_group
            index = self._part_of_group[int(self._layout.pool_group_of[lg])]
            part = self._parts[index]
            runs.append(
                GroupRun(
                    lg,
                    group.ordinals,
                    names=self._names(lg, lg_keys),
                    addresses=part.address + group.staging_slots * part.slot_bytes,
                    part=index,
                )
            )
        return RegionView(tuple(runs))

    def _segments(self, rows: _Rows, mask: list[np.ndarray] | None = None) -> list[list[int]]:
        """``[staging address, device address, bytes]`` for every pool of every row ``mask`` keeps
        (one boolean array per layer group; ``None``: every row), by device pool group. Segments
        that continue each other on both sides are merged."""
        by_group: dict[int, list[int]] = {}
        for i, group in enumerate(rows.groups):
            by_group.setdefault(int(self._layout.pool_group_of[group.layer_group]), []).append(i)
        out: list[list[int]] = []
        for g, members in by_group.items():
            dev = np.concatenate([rows.groups[i].device_pages for i in members])
            stg = np.concatenate([rows.groups[i].staging_slots for i in members])
            if mask is not None:
                keep = np.concatenate([np.asarray(mask[i], dtype=bool) for i in members])
                dev, stg = dev[keep], stg[keep]
            part = self._parts[self._part_of_group[g]]
            # A staging slot holds the device pools' slots back to back in pool order.
            offset = 0
            for pool in self._layout.device_pools[g]:
                width = int(pool.slot_bytes)
                for d, t in zip(dev.tolist(), stg.tolist()):
                    host = part.address + t * part.slot_bytes + offset
                    device = int(pool.base) + d * width
                    last = out[-1] if out else None
                    if last and last[0] + last[2] == host and last[1] + last[2] == device:
                        last[2] += width
                    else:
                        out.append([host, device, width])
                offset += width
        return out

    # TODO: copies run serially with the forward passes on the execution stream, and a copy call
    # per row and pool, which rows share only where they continue each other in a pool group of
    # one pool, keeps small rows below the host-to-device bandwidth.
    def _memcpy(self, segments: Sequence[Sequence[int]], to_staging: bool) -> _CopyResult:
        """Queue async copies on the manager's execution stream, then record an event covering
        every copy queued, even after one failed; no copy if it could record none. With page-locked
        staging the CPU does not wait for them; pageable staging may hold the call."""
        # The execution stream orders a copy after the forward passes that wrote its pages and
        # before later work on it, which a page's new owner waits for; no path that releases a page
        # waits for the copies, and a writer off that stream is not ordered after them.
        stream = _manager.stream(self._manager_ref())
        handle = drv.CUstream(stream.cuda_stream)
        error = None
        for host, device, nbytes in segments:
            dst, src = (host, device) if to_staging else (device, host)
            (result,) = drv.cuMemcpyAsync(
                drv.CUdeviceptr(dst), drv.CUdeviceptr(src), nbytes, handle
            )
            if result != drv.CUresult.CUDA_SUCCESS:
                error = f"cuMemcpyAsync of {nbytes} bytes failed: {result}"
                break
        try:
            event = torch.cuda.Event()
            event.record(stream)
        except RuntimeError as record_error:
            error = error or f"recording the copy's event failed: {record_error}"
            logger.warning(f"KV cache lender: {error}")
            return _CopyResult(None, error)
        if error is not None:
            logger.warning(f"KV cache lender: {error}")
        return _CopyResult(_Copy(event), error)

    def _restart_floor(self, manager: KVCacheManagerV2, history: int, origin: int) -> int:
        """The lowest start that needs no restart: the history where a chunk resumed below it may
        end below it and raise, or a window has released blocks at it (they have no pages), else
        the smaller of the lowest start of a delivered fetch and the history."""
        if _floor_follows_history(manager):
            return history
        for lg in range(self._layout.num_layer_groups):
            stale_beg, stale_end = _stale(manager, self._layout, lg, history)
            if stale_end > stale_beg:
                return history
        return min(origin, history)

    def _usable_until(self, manager: KVCacheManagerV2, delivered: _Delivered, computed: int) -> int:
        """The largest start ``P >= computed`` where every layer group has what it reads among the
        blocks below ``computed``, computed before the fetches, and delivered rows: full attention
        every block below ``P``, a window its sinks and in-window blocks."""
        tpb = int(self._layout.tokens_per_block)
        base = computed // tpb  # the blocks below were computed
        delivered_end = max((len(blocks) for blocks in delivered.blocks), default=0)
        if delivered_end <= base:
            return computed
        # Non-monotonic in P under windows, so each is checked. Blocks below ``computed`` behind a
        # window's history have no pages, but no start at or above the floor reads them.
        # No start past a full-attention group's first undelivered block passes that group.
        highest_end = delivered_end
        # Per layer group: undelivered[j], how many of the j blocks from ``base`` were not delivered.
        undelivered = []
        for lg, blocks in enumerate(delivered.blocks):
            have = np.zeros(delivered_end - base, dtype=bool)
            mine = blocks[base:delivered_end]
            have[: len(mine)] = mine
            undelivered.append(np.concatenate([[0], np.cumsum(~have)]))
            if self._layout.windows[lg] is None and not have.all():
                highest_end = min(highest_end, base + int(np.argmin(have)))

        def has_gap(lg: int, a: int, b: int) -> bool:
            """A block of ``[a, b)`` that layer group ``lg`` neither computed nor got delivered."""
            a, b = max(a, base), min(b, delivered_end)
            return b > a and undelivered[lg][b - base] - undelivered[lg][a - base] > 0

        # The largest start first: a fetch whose leases all landed is usable at its end at once.
        for end_block in range(highest_end, base, -1):
            for lg in range(len(delivered.blocks)):
                stale_beg, stale_end = _stale(manager, self._layout, lg, end_block * tpb)
                if stale_end > stale_beg:
                    lacking = has_gap(lg, base, min(stale_beg, end_block))
                    lacking = lacking or has_gap(lg, stale_end, end_block)
                else:
                    lacking = has_gap(lg, base, end_block)
                if lacking:
                    break
            else:
                return end_block * tpb
        return computed


# TODO: an in-place lease gets none of the staging guarantees: nothing guards its pages against
# suspend, shrink, window advance, a rebasing commit or a pool rebalance while lent, it is ready
# before the stream is done with them, and mark_arrived feeds no readiness.
class InPlace:
    """``InPlaceLender`` over one manager, which it references weakly. A lease holds a loan on the
    request's cache; the request's free keeps a lent cache open until its last loan ends."""

    def __init__(self, manager: weakref.ref[KVCacheManagerV2], layout: ManagerLayout) -> None:
        self._manager_ref = manager
        self._layout = layout
        # Open loans per cache, holding the cache strongly so that dropping a lease never lets a
        # collector close it on another thread.
        self._loans: dict[_KVCache, int] = {}
        self._freed: dict[_KVCache, Callable[[], None]] = {}  # lent caches the manager freed
        self._kept_at_shutdown: frozenset[object] | None = None  # set by the manager's shutdown
        # Retained while a loan is open, and until exit once the manager is gone.
        self._index_buffer: object | None = None

    def lend_read(self, request: LlmRequest, start: int, end: int) -> _InPlaceLease:
        """See ``InPlaceLender.lend_read``."""
        return self._lend(request, start, end, "read")

    def lend_write(self, request: LlmRequest, start: int, end: int) -> _InPlaceLease:
        """See ``InPlaceLender.lend_write``."""
        return self._lend(request, start, end, "write")

    def _on_free(
        self, request_id: int, kv_cache: _KVCache, after_close: Callable[[], None]
    ) -> bool:
        """Manager hook: ``True`` if ``kv_cache`` is on loan, which the release ending its last loan
        then closes, running ``after_close`` in that call. Logs its own errors."""
        try:
            if kv_cache not in self._loans:
                return False
            # TODO: a lent cache's late close and its after_close clear its request id's stats
            # records (dirty mark, exclusion), which a new cache under that id may hold by then.
            self._freed[kv_cache] = after_close
            return True
        except Exception:
            # Logged, not raised into the manager's free. It keeps the cache open while any loan
            # is, since closing a lent cache would hand its pages to another request.
            logger.error(f"KV cache lender: freeing request {request_id}: {traceback.format_exc()}")
            return bool(self._loans)

    def _on_shrink(self, request_id: int, kv_cache: _KVCache) -> None:
        """Manager hook after an in-place shrink: nothing to do, since the caller keeps a lent
        cache from shrinking."""

    def _on_reset(self) -> None:
        """Manager hook after it reset its reuse state: nothing to do, since in-place views carry
        no names."""

    def _on_shutdown(self, impl: KVCacheManager) -> frozenset[object]:
        """Manager hook: the caches still on loan, retained with ``impl`` until the process exits;
        the same set on every later call."""
        if self._kept_at_shutdown is not None:
            return self._kept_at_shutdown
        # Outside the try: a hook that cannot list the caches on loan raises, so the manager's
        # shutdown stops before it closes or frees anything.
        self._kept_at_shutdown = frozenset(self._loans)
        if self._kept_at_shutdown:
            try:
                # A device pool cannot be freed in part, so the pools stay with the lent caches.
                for owner in (impl, *self._kept_at_shutdown):
                    _retain_until_exit(owner)
                logger.warning(
                    f"KV cache lender: keeping {len(self._kept_at_shutdown)} lent caches and their "
                    "pools until exit"
                )
            except Exception:
                # Logged, not raised into the manager's shutdown, which still leaves the lent
                # caches open and the pools unfreed.
                logger.error(f"KV cache lender: shutting down: {traceback.format_exc()}")
        return self._kept_at_shutdown

    def _on_caches_closed(self) -> None:
        """Manager hook after its shutdown closed every cache: nothing to do, since a loan retains
        the page-index buffer itself and a release drops its cache."""

    def _end_loan(self, kv_cache: _KVCache) -> None:
        """End one loan on ``kv_cache``; the last loan on a freed cache closes it in this call."""
        if self._kept_at_shutdown is not None:
            return  # every cache on loan at the manager's shutdown is retained until exit
        left = self._loans.get(kv_cache, 0) - 1
        if left > 0:
            self._loans[kv_cache] = left
            return
        # TODO: the last loan ends before its cache closes, so a close that raises is left to the
        # cache's destructor, and its after_close never runs.
        self._loans.pop(kv_cache, None)
        after_close = self._freed.pop(kv_cache, None)
        if after_close is not None:
            _manager.close_cache(kv_cache)
            after_close()
        if not self._loans:
            self._release_index_buffer()

    def _retain_index_buffer(self, manager: KVCacheManagerV2) -> None:
        """Retain the manager's page-index buffer while a loan is open: a lent cache the manager
        does not detach writes its page indices there as it closes, even after the manager is
        gone."""
        # TODO: without a shutdown nothing detaches a lent cache from the buffer, so a lease still
        # held when the exit registry goes at process exit closes its cache into the freed buffer.
        self._index_buffer = _manager.index_buffer(manager)
        _retain_until_exit(self._index_buffer)

    def _release_index_buffer(self) -> None:
        """The last loan ended: release the buffer to its manager. With the manager gone, a cache
        this lender held may still close after this call, so the buffer stays until exit."""
        if self._manager_ref() is None:
            return
        _release_retained(self._index_buffer)
        self._index_buffer = None

    def _lend(self, request: LlmRequest, start: int, end: int, kind: _LeaseKind) -> _InPlaceLease:
        start, end = int(start), int(end)
        if start < 0 or end < 0 or start > end:
            raise ValueError(f"bad token range [{start}, {end})")
        request_id = int(request.py_request_id)
        manager = self._manager_ref()
        if self._kept_at_shutdown is not None or manager is None:
            return _InPlaceLease._failed(self, kind, _SHUT_DOWN)
        kv = _manager.kv_of(manager, request_id)
        if kv is None:
            return _InPlaceLease._failed(self, kind, _no_cache(request_id))
        state = _manager.cache_state(kv)
        if not state.active:
            return _InPlaceLease._failed(self, kind, _SUSPENDED)
        runs = []
        for lg in range(self._layout.num_layer_groups):
            ordinals, pages, beyond = self._device_pages(manager, kv, lg, start, end)
            paged = pages >= 0
            if kind == "write" and (beyond or not paged.all()):
                missing = ordinals[~paged].tolist() + beyond
                return _InPlaceLease._failed(
                    self, kind, f"layer group {lg}: blocks {missing[:8]} have no page"
                )
            # A read leaves blocks without a page out.
            runs.append(GroupRun(lg, ordinals[paged]))
        # TODO: the V2 scheduler cannot reclaim the pages a cache on loan locks, also after its
        # request's free.
        # The loan opens here: from now on the request's free keeps the cache open.
        if not self._loans:
            self._retain_index_buffer(manager)
        self._loans[kv] = self._loans.get(kv, 0) + 1
        return _InPlaceLease(self, kind, RegionView(tuple(runs)), kv)

    def _device_pages(
        self, manager: KVCacheManagerV2, kv: _KVCache, lg: int, start: int, end: int
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """The blocks of ``lg`` that ``[start, end)`` touches, a partial last one included and none
        for an empty range, that a history of ``end`` reads: those the cache has, with their device
        pages (-1 where a block has no locked page), and up to eight past them as Python ints."""
        tpb = int(self._layout.tokens_per_block)
        # Only pages the cache locks: a window block behind its history keeps at most a held page,
        # which a lower cache tier may take at any time.
        pages = _manager.locked_pages(kv, lg)
        held = len(pages)
        end_block = -(-end // tpb) if end > start else start // tpb
        ranges = _needed_block_ranges(manager, self._layout, lg, start // tpb, end_block, end)
        inside = _ordinals([(beg, min(stop, held)) for beg, stop in ranges if held > beg])
        # Past the cache's blocks, which have no page, only the first eight are listed, as many as a
        # failed write names, and as Python ints: a range of any length and start stays small.
        beyond: list[int] = []
        for beg, stop in ranges:
            first_beyond = max(beg, held)
            beyond.extend(range(first_beyond, min(stop, first_beyond + 8 - len(beyond))))
        return inside, pages[inside], beyond


class _LeaseState(enum.Enum):
    """Where a lease is in its life; ``_StagingLease`` lists what moves a staging lease between
    them. An in-place lease starts ``GRANTED``, or ``FAILED`` at its call."""

    WAITING = "waiting"
    GRANTED = "granted"
    VIEW_RETURNED = "view returned"
    MARKED = "marked"
    FAILS_AT_FIRST_POLL = "fails at first poll"
    FAILED = "failed"


# A write released in one of these abandons its fetch: no backend saw its view.
_BEFORE_VIEW = (_LeaseState.WAITING, _LeaseState.GRANTED, _LeaseState.FAILS_AT_FIRST_POLL)


class _LeaseBase:
    """What both leases share: their state and failure, and the checks of a write's one mark."""

    @property
    def failure(self) -> str | None:
        """See ``Lease.failure``."""
        return self._reason if self._state is _LeaseState.FAILED else None

    def _return_view(self) -> RegionView | None:
        """The view ``poll`` returns; a granted lease is then ``VIEW_RETURNED``."""
        if self._state is _LeaseState.GRANTED:
            self._state = _LeaseState.VIEW_RETURNED
        return self._view

    def _take_marks(self, masks: Sequence[np.ndarray]) -> list[np.ndarray]:
        """Copies of ``masks`` for the one mark of a write whose view ``poll()`` returned;
        ``ValueError`` unless one bool mask of shape ``(len(run),)`` per run."""
        if self._kind != "write":
            raise RuntimeError("mark_arrived is for write leases")
        if self._state not in (_LeaseState.VIEW_RETURNED, _LeaseState.MARKED):
            raise RuntimeError("mark_arrived before poll() returned the view")
        if self._state is _LeaseState.MARKED:
            raise RuntimeError("mark_arrived called twice")
        masks, runs = list(masks), self._view.runs
        if len(masks) != len(runs):
            raise ValueError(f"{len(masks)} masks for {len(runs)} runs")
        out = []
        for run, mask in zip(runs, masks):
            mask = np.asarray(mask)
            if mask.dtype != np.bool_ or mask.shape != (len(run),):
                raise ValueError(
                    f"layer group {run.layer_group}: the mask must be bool of shape "
                    f"({len(run)},), got {mask.dtype} {mask.shape}"
                )
            out.append(mask.copy())
        self._state = _LeaseState.MARKED
        return out


class _StagingLease(_LeaseBase):
    """A staging lease; holds its lender weakly and has no finalizer. Backends may read its view's
    arrays on their own threads until release, and never call it.

    Its state, and what moves it on (``_reason`` says why it failed or fails at its first poll):

    - ``WAITING``: not granted yet; in line for slots while it holds ``_ticket``. ``_grant`` moves
      it to ``GRANTED``; ``_fail`` to ``FAILED``.
    - ``GRANTED``: it has its view, and its slots if it has rows; a read's copy into staging may
      still run. ``poll`` returning the view moves it to ``VIEW_RETURNED``; a read whose copy into
      staging failed is ``FAILED``.
    - ``VIEW_RETURNED``: a backend may use its slots. ``mark_arrived`` moves a write to ``MARKED``.
    - ``MARKED``: ``mark_arrived`` took a write's masks; the marked rows still lent are copied
      into the request's pages.
    - ``FAILS_AT_FIRST_POLL``: a write whose cache grew but that cannot go on: a block without a
      page, or a grant within its call that raised. ``poll`` and the shutdown move it to
      ``FAILED``.
    - ``FAILED``: ``failure`` says why; it never returns its view.

    Release is separate, legal in every state: a released lease is never polled again, and a write
    released before its view was returned abandons its fetch."""

    def __init__(
        self,
        lender: Staging,
        kind: _LeaseKind,
        request_id: int,
        kv: _KVCache | None = None,
        rows: _Rows | None = None,
        keys: list[np.ndarray] | None = None,
        fetch: _Fetch | None = None,
    ) -> None:
        self._lender = weakref.ref(lender)
        self._kind = kind
        self._request_id = request_id
        self._kv = kv
        self._rows = rows
        self._keys = keys
        self._fetch = fetch
        self._state = _LeaseState.WAITING
        self._reason: str | None = None
        self._ticket: int | None = None  # its place in line while it waits for slots
        self._runs: Runs | None = None
        self._view: RegionView | None = None
        self._copy: _Copy | None = None
        self._released = False

    @classmethod
    def _failed(
        cls, lender: Staging, kind: _LeaseKind, request_id: int, reason: str
    ) -> _StagingLease:
        """A lease failed at the call; open until released, like any other."""
        lease = cls(lender, kind, request_id)
        lease._set_failure(reason)
        if not lender._closed:
            lender._unreleased.add(lease)
        return lease

    @property
    def _first_poll_failure(self) -> str | None:
        """Why it fails at its first poll, while it is ``FAILS_AT_FIRST_POLL``."""
        return self._reason if self._state is _LeaseState.FAILS_AT_FIRST_POLL else None

    def _set_failure(self, reason: str) -> None:
        self._state = _LeaseState.FAILED
        self._reason = reason

    def _fail_at_first_poll(self, reason: str) -> None:
        self._state = _LeaseState.FAILS_AT_FIRST_POLL
        self._reason = reason

    def poll(self) -> RegionView | None:
        """See ``Lease.poll``."""
        if self._released:
            raise RuntimeError("poll after release")
        lender = self._lender()
        if lender is None or lender._live() is None:
            return self._ended_poll()
        lender._progress()
        if self._state is _LeaseState.FAILED:
            return None
        if self._state is _LeaseState.FAILS_AT_FIRST_POLL:
            lender._fail(self, self._reason)
            return None
        if self._state is _LeaseState.WAITING:
            return None
        if self._kind == "read" and not lender._read_copy_done(self):
            return None
        return self._return_view()

    def mark_arrived(self, masks: Sequence[np.ndarray]) -> None:
        """See ``Lease.mark_arrived``."""
        masks = self._take_marks(masks)
        lender = self._lender()
        if lender is not None and lender._live() is not None:
            lender._apply_marks(self, masks)

    def release(self) -> None:
        """See ``Lease.release``."""
        if self._released:
            return
        self._released = True
        lender = self._lender()
        if lender is not None:
            lender._on_release(self)

    def _ended_poll(self) -> RegionView | None:
        """``poll`` after the lender stopped serving: no grant or copy, just what already landed."""
        if self._state not in (
            _LeaseState.GRANTED,
            _LeaseState.VIEW_RETURNED,
            _LeaseState.MARKED,
        ):
            return None
        if self._kind == "read" and self._copy is not None and not self._copy.done():
            return None
        return self._return_view()


class _InPlaceLease(_LeaseBase):
    """An in-place lease: the loan on the request's cache, held until release. Holds its lender
    weakly and has no finalizer."""

    def __init__(
        self,
        lender: InPlace,
        kind: _LeaseKind,
        view: RegionView | None,
        kv_cache: _KVCache | None = None,
        failure: str | None = None,
    ) -> None:
        self._lender = weakref.ref(lender)
        self._kind = kind
        self._view = view
        self._cache = kv_cache  # the loan, until release
        self._state = _LeaseState.GRANTED if failure is None else _LeaseState.FAILED
        self._reason = failure
        self._released = False

    @classmethod
    def _failed(cls, lender: InPlace, kind: _LeaseKind, reason: str) -> _InPlaceLease:
        """A lease failed at the call; it holds no loan."""
        return cls(lender, kind, None, failure=reason)

    def poll(self) -> RegionView | None:
        """See ``Lease.poll``."""
        if self._released:
            raise RuntimeError("poll after release")
        if self._state is _LeaseState.FAILED:
            return None
        # The request's own pages: ready at once; the caller has let the stream's work on them end.
        return self._return_view()

    def mark_arrived(self, masks: Sequence[np.ndarray]) -> None:
        """See ``Lease.mark_arrived``."""
        # The rows are in the request's pages already: only the masks are checked.
        self._take_marks(masks)

    def release(self) -> None:
        """See ``Lease.release``."""
        if self._released:
            return
        self._released = True
        cache, self._cache = self._cache, None
        lender = self._lender()
        if cache is not None and lender is not None:
            lender._end_loan(cache)


class _PartsHold:
    """A hold on the staging memory; holds its lender weakly and has no finalizer. Without a lender
    it is inert."""

    def __init__(self, lender: Staging | None) -> None:
        self._lender = weakref.ref(lender) if lender is not None else None
        self._released = False

    def release(self) -> None:
        """See ``PartsHold.release``."""
        if self._released:
            return
        self._released = True
        lender = self._lender() if self._lender is not None else None
        if lender is not None:
            lender._end_hold(self)

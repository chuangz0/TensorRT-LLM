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
"""The lender's public types and protocols.

Numpy and the standard library only, so any side can import them without a live KV cache manager.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from ...llm_request import LlmRequest

# A row's name: a fixed-length opaque key.
_NAME_BYTES = 54


def _freeze(array: np.ndarray) -> np.ndarray:
    """``array`` marked read-only (a view when it is not already)."""
    if array.flags.writeable:
        array = array.view()
        array.flags.writeable = False
    return array


def _integer(name: str, value: object, least: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < least:
        raise ValueError(f"{name} must be at least {least}, got {value}")
    return int(value)


@dataclass(frozen=True)
class StagingOptions:
    """The staging size, in whole fetches.

    The lender sizes one fetch from the layer groups and windows; any range of at most
    ``fetch_tokens`` tokens fits. A capacity budget, not concurrency: a lease takes one contiguous
    run of slots per pool group, granted strictly first come, first served, so holes released leases
    leave can make it wait although enough slots are free. A lease with no rows is ready at its
    first poll, with no place in line.

    Attributes:
        fetch_tokens: The tokens of one fetch.
        max_fetches: How many fetches staging holds.
        max_bytes: A cap on the staging bytes, or ``None``; ``attach_staging`` raises ``ValueError``
            if it is below one fetch.

    Raises:
        TypeError: A field is not an integer (``bool`` included).
        ValueError: A field is not positive.
    """

    fetch_tokens: int
    max_fetches: int = 1
    max_bytes: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "fetch_tokens", _integer("fetch_tokens", self.fetch_tokens))
        object.__setattr__(self, "max_fetches", _integer("max_fetches", self.max_fetches))
        if self.max_bytes is not None:
            object.__setattr__(self, "max_bytes", _integer("max_bytes", self.max_bytes))


@dataclass(frozen=True)
class Part:
    """One device pool group's staging host region, to register as ``(address, nbytes)``.

    ``StagingLender.parts`` states what a backend may rely on and when it may register;
    ``PartsHold`` states when it deregisters.

    Attributes:
        name: Equal on instances laid out alike.
        address: The region's fixed host address.
        nbytes: The region's size in bytes.
        slot_bytes: The bytes of one slot, which holds one row.
        slots: The slots, back to back from ``address``.
    """

    name: str
    address: int
    nbytes: int
    slot_bytes: int
    slots: int


@dataclass(frozen=True, eq=False)
class GroupRun:
    """One layer group's rows, with read-only arrays: row ``i`` is block ``ordinals[i]``.

    Staging rows carry names, slot addresses and a part; in-place rows carry none, as the caller
    addresses them with its own page-table code. Equal names mean interchangeable bytes within the
    limits ``attach_staging`` states, so a name is usable as a store key. Their format is not API,
    but their width, 54 bytes, is: consumers store names and compare them for equality, and nothing
    else. A change of format or layout makes objects stored under the old one miss, never hit
    wrongly.

    Attributes:
        layer_group: The manager's local index of the layer group.
        ordinals: ``int64 (n,)``, each row's block index.
        names: Staging only: ``uint8 (n, 54)``, each row's opaque name (``names[i].tobytes()``).
        addresses: Staging only: ``int64 (n,)``; row ``i``'s slot is the part's ``slot_bytes`` bytes
            at ``addresses[i]``.
        part: Staging only: the index of the rows' part in ``StagingLender.parts``.
    """

    layer_group: int
    ordinals: np.ndarray
    names: Optional[np.ndarray] = None
    addresses: Optional[np.ndarray] = None
    part: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer_group", _integer("layer_group", self.layer_group, 0))
        ordinals = _freeze(np.ascontiguousarray(self.ordinals, dtype=np.int64))
        if ordinals.ndim != 1:
            raise ValueError(f"ordinals must be one-dimensional, got shape {ordinals.shape}")
        object.__setattr__(self, "ordinals", ordinals)
        placed = (self.names is not None, self.addresses is not None, self.part is not None)
        if any(placed) and not all(placed):
            raise ValueError("names, addresses and part are all given or all None")
        if not all(placed):
            return
        names = np.asarray(self.names)
        if names.dtype != np.uint8 or names.shape != (len(ordinals), _NAME_BYTES):
            raise ValueError(
                f"names must be uint8 ({len(ordinals)}, {_NAME_BYTES}), "
                f"got {names.dtype} {names.shape}"
            )
        addresses = np.asarray(self.addresses)
        if not np.issubdtype(addresses.dtype, np.integer) or addresses.shape != ordinals.shape:
            raise ValueError(
                f"addresses must be integers of shape {ordinals.shape}, "
                f"got {addresses.dtype} {addresses.shape}"
            )
        object.__setattr__(self, "names", _freeze(np.ascontiguousarray(names)))
        object.__setattr__(
            self, "addresses", _freeze(np.ascontiguousarray(addresses, dtype=np.int64))
        )
        object.__setattr__(self, "part", _integer("part", self.part, 0))

    def __len__(self) -> int:
        return int(self.ordinals.shape[0])

    def select(self, mask: np.ndarray) -> GroupRun:
        """The rows where the boolean ``mask`` (one entry per row) is true, in order.

        Args:
            mask: One boolean per row, ``True`` for each row to keep.

        Returns:
            A run of the same layer group and part that holds the kept rows, in order.

        Raises:
            ValueError: ``mask`` is not boolean with one entry per row.
        """
        mask = np.asarray(mask)
        if mask.dtype != np.bool_ or mask.shape != self.ordinals.shape:
            raise ValueError(
                f"mask must be bool of shape {self.ordinals.shape}, got {mask.dtype} {mask.shape}"
            )
        return GroupRun(
            self.layer_group,
            self.ordinals[mask],
            None if self.names is None else self.names[mask],
            None if self.addresses is None else self.addresses[mask],
            self.part,
        )


@dataclass(frozen=True, eq=False)
class RegionView:
    """What a ready lease lends: at most one run per layer group.

    Backends may read its arrays on their own threads until the lease is released, and touch the
    memory they point to only until then.

    Attributes:
        runs: The runs, at most one per layer group.

    Raises:
        TypeError: A run is not a ``GroupRun``.
        ValueError: A layer group appears in two runs.
    """

    runs: Tuple[GroupRun, ...]

    def __post_init__(self) -> None:
        runs = tuple(self.runs)
        seen = set()
        for run in runs:
            if not isinstance(run, GroupRun):
                raise TypeError(f"runs hold GroupRun, got {type(run).__name__}")
            if run.layer_group in seen:
                raise ValueError(f"layer group {run.layer_group} appears twice")
            seen.add(run.layer_group)
        object.__setattr__(self, "runs", runs)

    @property
    def num_rows(self) -> int:
        """Rows over all runs."""
        return sum(len(run) for run in self.runs)

    def row_masks(self, value: bool = False) -> Tuple[np.ndarray, ...]:
        """Make one writable boolean mask per run, filled with ``value``.

        Args:
            value: What every row's entry starts as.

        Returns:
            One mask per run, in run order: the shape ``mark_arrived`` takes.
        """
        return tuple(np.full(len(run), bool(value), dtype=bool) for run in self.runs)


class Readiness(NamedTuple):
    """Where a request may resume after a fetch.

    It may resume at a ``p`` with ``restart_floor <= p <= usable_until``, no lower than its context
    position; below the floor it may read blocks a sliding window released. ``readiness`` does not
    read the context position, so the interval is empty if
    ``max(restart_floor, context position) > usable_until``. The interval is this rank's own and
    covers only this manager's blocks.

    Bounds:
        - ``usable_until`` need not be a multiple of ``tokens_per_block``. It reaches the request's
          prompt length only where the request computed its whole prompt itself, as a fetch ends
          before the last prompt token.
        - Under a block reuse policy other than all-reusable the floor is also at least the
          request's history, since the manager's context update never moves a history back.
        - The same holds in a joint-reuse draft pool, whose context resize sets the capacity from
          the chunk it runs and raises where that capacity is below the request's history.
        - With ``mm_bidirectional_blocks``, no position in the interval lies strictly inside a run
          of multimodal tokens (the scheduler keeps a run within one context chunk).
          ``usable_until`` stops at the start of a run it would fall inside and ``restart_floor``
          rises to the end of the last run below it, possibly emptying the interval. A context
          position strictly inside a run can leave it ending below that position, which the caller
          treats as empty too.
        - Across ranks the waiter takes the largest floor and the smallest end. With a joint-reuse
          draft pool sharing the request's cursor (``StagingLender``), the request resumes where
          both intervals allow: the smaller ``usable_until``, the larger ``restart_floor``, so never
          below the draft pool's history.

    Caller must:
        - Park the request from ``lend_write`` until it resumes: keep it among the executor's active
          requests, as the disaggregated transfer-in-progress state does, but unscheduled. Pool
          rebalance suspends only active requests' caches (and its CUDA-graph padding dummies), so a
          request taken out of them with an active cache stops the executor loop.
        - Resume only at a ``p`` in the interval no lower than its context position; lower, the next
          scheduling pass can make a chunk negative and raise.
        - Resume as a KV cache connector skips a served prefix, on any context chunk. These steps
          form a set; only the last two keep their order:

          - Set ``py_connector_served_position`` to ``p``.
          - Raise the cache's history to ``p`` where below.
          - Once the request is back in its context state, set its context chunk to span to the
            prompt's end.
          - Then move the prepopulated length and context position to ``p`` together by
            ``set_prepopulated_prompt_len(p, tokens_per_block)``, with the manager's
            ``tokens_per_block``.

          ``set_prepopulated_prompt_len(0, tokens_per_block)`` leaves the context position as is: a
          resume at 0 needs no step from position 0, and elsewhere means dropping the cache in every
          manager and computing from 0. Without the call, a request past its first chunk stays at
          its context position; a position moved alone leaves the prepopulated length behind, which
          context logits and a pipelined cache transfer's first chunk read.
        - On an empty interval, drop the request's cache in every manager it fetched into (a target
          and its joint-reuse draft pool alike), compute from 0 and do not fetch again.
        - With a sliding window, a non-empty interval does not by itself permit another lease: see
          the split rule in ``StagingLender.lend_write``.

    Notes:
        - The next chunk is a first context chunk, which the manager places at ``p``. A pass
          failing to admit it drops its caches in every manager and rewinds it to 0, as for any
          first chunk.
        - ``cached_tokens`` reports where the first context step started, until a recompute pause,
          so a fetch resumed from later adds nothing.
        - With ``enable_return_routed_experts``, a resume can leave skipped positions without routes
          (``StagingLender``).

    Attributes:
        usable_until: The last position the request may resume at.
        restart_floor: The first position the request may resume at, unless its context position is
            higher.
    """

    usable_until: int
    restart_floor: int


@runtime_checkable
class Lease(Protocol):
    """One lent range: poll until the view or ``failure``, mark a write once, release.

    Its methods run only on the manager's thread (see the package docstring). Backend threads tell
    the holder through their own channel when done; without that signal the lease stays open. After
    the manager's shutdown, ``poll``, ``mark_arrived`` and ``release`` only end records. Outcomes
    are this rank's own (slots, copies, free pages and a pipeline stage's windows are per rank), so
    ranks lending alike can end differently; the lender runs no collective. Failure is final.

    Caller must:
        - Poll every open lease each iteration of the executor loop, also when no new request
          arrives: all progress happens inside lender calls. Once no request is live or waiting, the
          executor waits for one, with no timeout under MPI and up to 1200 s under Ray. Keep that
          wait from blocking while a lease is open or a backend has work for the holder, as the
          executor does while a KV cache connector's transfers pend.
        - Combine every rank's outcome and decide for all ranks. A lease already failed when
          ``lend_read`` or ``lend_write`` returns changed nothing in the request's cache: compute
          locally or try later. A write lease that fails after the call has grown the cache: then
          ``readiness`` decides, and on an empty interval the request does not fetch again.
        - Release every lease, failed ones too, once the backend let go of the memory; an unreleased
          staging lease keeps the staging memory past shutdown until the process exits.
        - A lender, lease or hold call that raises for a reason its docstring does not list (a
          ``MemoryError``, say) is not recovered from. Then only release that lender's leases and
          holds, and treat requests parked through it as on an empty interval. Its records may
          disagree with caches and slots, a staging lender's for other requests too, and may keep
          caches, device pools and staging memory until exit.
    """

    def poll(self) -> Optional[RegionView]:
        """Does pending work and returns the view once the lease is ready.

        Ready: a staging read once its copy into the slots completed, a staging write at the first
        poll after its slots were granted, an in-place lease at its first poll, with no stream wait.

        Returns:
            The view once ready, the same object every time; ``None`` while pending, and for good
            once failed.

        Raises:
            RuntimeError: The lease was released.
        """
        ...

    @property
    def failure(self) -> Optional[str]:
        """Why the lease will never be ready, once that is known; for logs only."""
        ...

    def mark_arrived(self, masks: Sequence[np.ndarray]) -> None:
        """Marks the rows of a write lease that arrived whole.

        Write leases only, once, after ``poll()`` returned the view, before or after release.
        Required for staging writes: only marked rows reach the request's pages, in one copy batch
        that every later-queued forward waits for, about the fetch's bytes over the host-to-device
        bandwidth. So callers split long fetches, but only by the split rule in
        ``StagingLender.lend_write`` and its extra-KV-token bound: the next lease once ``readiness``
        is not None on every rank and, with a sliding window, its ``usable_until`` reaches this
        lease's end.

        Rows are copied only where the active cache still locks the lent GPU page inside its window:
        after the request exits the holder still marks and releases, and freed pages get nothing. A
        staging write's slots return only after release, ``mark_arrived`` and the copy's completion.
        A write released before its view needs no mark. Optional in place: it checks only the masks
        and feeds no readiness.

        Args:
            masks: One boolean array per run of the view, ``True`` where the row arrived whole;
                ``RegionView.row_masks()`` makes all-``False`` ones.

        Raises:
            RuntimeError: A read lease, a call before ``poll()`` returned the view, or a second
                call.
            ValueError: A wrong number, shape or dtype of masks; nothing is recorded.
        """
        ...

    def release(self) -> None:
        """The backend has stopped touching the lent memory.

        Required for every lease, failed ones too; legal in every state, and later calls do nothing.
        The release that ends the last in-place loan on a freed request's cache closes that cache.
        """
        ...


@runtime_checkable
class PartsHold(Protocol):
    """A staging backend's hold on the staging memory.

    Take it with ``StagingLender.hold_parts()`` on the manager's thread before registering
    ``StagingLender.parts``; deregister the parts before the manager shuts down, and release it on
    the manager's thread once deregistration is confirmed. A hold still open at the manager's
    shutdown keeps the staging memory until the process exits, so a backend that cannot confirm its
    deregistration keeps its hold. The lender holds it, so dropping it unreleased keeps the memory.
    """

    def release(self) -> None:
        """The backend has deregistered the parts and cannot reach them.

        Runs only on the manager's thread, like every lender and lease method; legal in every state,
        and later calls do nothing.
        """
        ...


# TODO: names exist only in ready views, so a flow that looks up remote hits before it reserves
# pages for them cannot name blocks without lending them.
@runtime_checkable
class StagingLender(Protocol):
    """Relays whole blocks between a request's device pages and host staging slots.

    ``lend_read`` publishes, ``lend_write`` fetches, ``readiness`` says where to resume. Its methods
    run only on the manager's thread (see the package docstring); backends never lend and never call
    a lease. Leases needing slots wait in line (``StagingOptions``).

    Copies queue on the manager's stream, serially with the forward passes on the GPU, so a step
    queuing copies takes about their time longer. With page-locked staging and the fresh-page fill
    off, no lend, poll, mark or readiness call waits for them on the CPU; only the manager's
    shutdown does. Under confidential computing staging is pageable, and a copy can hold its call
    until the stream reaches it. The executor's pool rebalance suspends active caches, parked ones
    too, and moves their pages, lent ones too. Queued copies finish first, and a read waiting for
    slots fails at its grant where pages moved. A write marked afterwards copies only the rows whose
    pages stayed, and ``readiness`` counts only those.

    Caller must:
        - Hold and register the parts before a backend touches a lease, and deregister them before
          shutdown (see ``PartsHold``).
        - Each executor iteration, also with no new request, the holder polls every open lease and
          the waiter asks ``readiness`` for every parked request.
        - Split a fetch only by the rule in ``lend_write``: the next lease once ``readiness`` is not
          None on every rank and, with a sliding window, its ``usable_until`` reaches the previous
          lease's end. The lender does not check this yet.
        - For a one-model draft with its own joint-reuse pool, which shares the request's context
          cursor, fetch the same range into both managers through their own lenders, resume within
          both intervals and publish both. A fetch into one alone leaves the other without the
          prefix as the cursor moves past it.
        - Shut down in order: the executor loop stops; staging backends stop, deregister and release
          their holds; the holder marks and releases what they give back; the manager shuts down
          last.

    First-version limits:
        - ``attach_staging`` lists the refused managers. Lending stops for good once the manager
          resets its reuse state.
        - Window gap: a publish leaves out window blocks its request's window has passed, though the
          prefix tree may still hold their committed pages, so a fetch whose window still keeps such
          a block finds its row missing.
        - DeepSeek-V4 keeps every window the draft length wider under any speculative decoding, and
          a fetch asks for the rows of that margin too. At 128 tokens per block and a draft length
          of 2 or more, a publisher whose history stands at least the draft length less one token
          past the fetch's end leaves out that row, and the fetch misses it. Its default SWA scratch
          reuse fails fetches (``lend_write``): set
          ``kv_cache_config.enable_swa_scratch_reuse=False``.
        - Under the all-reusable policy, a fetch fails at the call where a window leaves behind a
          block whose page holds tokens past the cache's history (``lend_write``). Rows an earlier
          lease missed are not checked: the split rule covers those.
        - Names exist only in ready views: a caller cannot ask a source how far it holds a prefix
          before a fetch grows the cache.
        - A publish lends a sliding-window layer group's rows only for the window at its ``end``,
          and a windowed fetch needs the window at the history it leaves. So two cases find rows
          missing and compute from 0. A fork from a published prompt at an earlier block does so
          once the window at the publisher's history has released blocks. A fetch past the published
          end (a longer next turn) does so once the window at the fetch's end has released blocks.
          The exception is a prefix whose publishes together lend the window at the fetch's end, as
          one ending there does if its publisher's window had passed none of that window's blocks.
          With no block released at the publisher's history, a fork finds every row. A fetch past
          the published end, with none released at its own end either, misses only the rows past
          the published end. It resumes at the first of them where the request may resume below its
          history (all-reusable policy, outside a joint-reuse draft pool), else computes from 0.
        - The pages a fetch grows are outside the V2 scheduler's reach: it neither evicts, pauses
          nor preempts a parked request. Its deadlock check counts no pass while any request is in
          a disaggregated transfer state, such as transfer-in-progress. With requests parked in
          another state, keep room in each pool group (package docstring), or a request that cannot
          be admitted, resume or grow can deadlock. A share of each pool group for the parked
          fetches alone does not ensure it, with or without a cache tier below the GPU.
        - Under attention data parallelism without a cache transceiver, parked requests count as
          schedulable: a rank with all its active requests parked, at its cap or without pages for a
          padding dummy, schedules nothing, and its empty batch holds every rank's forward until a
          fetch settles. Keep a rank from parking all its active requests, or accept the stall.
        - With ``enable_return_routed_experts``, the caller fetches into a request that asks for
          routed experts only before its first context step runs, so never after a recompute pause.
          Route capture stops reading the prepopulated length once it holds the routes below it
          (from the first context step when that length is 0), also across a recompute pause, so a
          later resume can leave skipped positions without routes and the request's completion
          raises out of the executor loop.
        - Copies issue at most one copy call per row and pool, merging rows contiguous in both pages
          and slots only in a single-pool pool group, so small scattered rows and rows over several
          pools run below the host-to-device bandwidth.
        - A call asks each copy's event at most once, but every call asks again the pending copy of
          each released lease (N calls over P such copies: about N * P queries). Every call also
          checks each lease holding slots, even with no copy pending (polling H such leases: about H
          * H checks).
        - A writer of recycled pages off the manager's stream is not ordered after staging copies,
          which precede only later work on that stream; an integrator adding one makes it wait on
          that stream first, as the disaggregated receive does. With connectors refused, the only
          such writer is the fresh-page fill (``TRTLLM_KV_FRESH_PAGE_FILL``, a diagnostic off by
          default): it synchronizes the device before it fills pages and after, so it overwrites no
          page a queued copy reads, and a growing ``lend_write`` then waits on the CPU for queued
          copies.
        - Every lease hashes the request's whole prefix again from block 0, so a lease late in a
          long prompt costs time in proportion to the prompt.
        - A slot lost to a failed copy is never reused and keeps the staging memory until the
          process exits. A slot is lost when a copy into or out of it may have been queued without
          its completion recorded: the completion could not be recorded, or a read's grant failed
          while queuing. A lease needing, in some pool group, a longer run of slots than the longest
          without a lost slot waits in line for good without failing, holding up every lease behind
          it until released or its request is freed.
    """

    @property
    def parts(self) -> Tuple[Part, ...]:
        """The staging host regions, one per device pool group.

        Each stays at a fixed address for the lender's life and can be registered once, after the
        attach and before the first lease is used. Nothing more is promised: not one allocation, not
        an order in memory, not that parts are back to back. A backend able to register only one
        region sorts them by address, checks each ends where the next begins and registers their
        span, or raises at construction and does not start. The manager's shutdown frees them, once,
        unless an unreleased lease (failed ones included), an unreleased hold (dropped ones
        included) or a slot lost to a failed copy keeps them until the process exits.
        """
        ...

    def hold_parts(self) -> PartsHold:
        """Returns a new hold on the staging memory, as ``PartsHold`` describes.

        After the manager's shutdown the hold is inert: the memory was freed or kept then.

        Returns:
            The hold, open until its ``release``.
        """
        ...

    def lend_read(self, request: LlmRequest, start: int, end: int) -> Lease:
        """Publishes the committed blocks ``[start, end)``: a copy of them into staging slots.

        Once its slots are granted, at once or after waiting in line, the lease queues the copy;
        then the request may exit, and a queued copy completes. After ``poll()`` returns the view,
        the backend reads those slots and no others, until release. A sliding-window layer group
        lends only the blocks a history of ``end`` reads that its window still keeps (window gap:
        ``StagingLender``); blocks without a page are left out.

        Its outcome is this rank's own; the caller combines every rank's outcome. It fails at the
        call on no or a suspended cache, multimodal data without digests or with encoder input
        (which names do not cover), a manager shut down or one that reset its reuse state, or a copy
        that could not be queued. A read that waited for slots also fails if its cache was freed,
        suspended or changed meanwhile.

        Args:
            request: The request whose blocks are read.
            start: The first token, a multiple of ``tokens_per_block``.
            end: The end token, a multiple of ``tokens_per_block``, at most the committed tokens.

        Returns:
            The lease, possibly failed already.

        Raises:
            ValueError: Bounds negative, reversed or not whole blocks, an end past the committed
                tokens, or a range needing more slots than a part has.
        """
        ...

    def lend_write(self, request: LlmRequest, start: int, end: int) -> Lease:
        """Fetches into blocks ``[start, end)``: grows the cache to ``end`` and lends empty slots.

        Slots are granted as for ``lend_read``; the backend fills them, the holder marks the rows
        that arrived whole and releases. A sliding-window layer group lends only the blocks a
        history of ``end`` reads. Grown pages stay until the request is freed.

        Args:
            request: The request fetched into.
            start: A multiple of ``tokens_per_block``, at least the committed tokens rounded down.
            end: A multiple of ``tokens_per_block``, at most the whole blocks before the request's
                last prompt token (which it computes for its logits), not below a windowed cache's
                history.

        Returns:
            The lease, possibly failed already. One failing after the call leaves the cache grown;
            ``readiness`` accounts for it.

        Raises:
            ValueError: Before the cache changes: bounds negative, reversed, not whole blocks or
                outside the limits above, or a range needing more slots than a part has.

        Caller must:
            - Split a fetch only this way: lend the next segment only once ``readiness`` is not None
              on every rank (settling is this rank's own).
            - With a sliding window, also wait until ``usable_until`` reaches the previous lease's
              end. If it falls short, first drop the request's cache in every manager it fetched
              into, or compute from ``usable_until`` through that end. Not checked yet: a later
              lease would move the history past rows nothing computed, which under the all-reusable
              policy the commit stores.
            - Fetch only into a request whose pages hold what its history covers, since the lender
              takes the tokens below the cache's history as computed: not one whose history runs
              ahead of its data, such as a disaggregated generation request before its transfer
              lands.
            - With ``mm_bidirectional_blocks``, end a fetch on a whole block outside every
              multimodal run, as the scheduler ends a chunk.

        Readiness accounting:
            - Consecutive leases count as one fetch until the request commits past them; an
              abandoned lease drops only its own rows. If all were abandoned, the request may resume
              only at its history, and only if what was kept reaches it.
            - The committed tokens still count, and so do the other tokens the request had computed
              below ``start``; past ``start`` the fetch may overwrite them. Delivered rows add to
              what is kept.
            - With a sliding window, each lease moves the history to its end at the call. The
              request may still resume below it, up to the delivered rows, if three things hold: no
              window has released blocks at that end, the policy is all-reusable, and the manager is
              no joint-reuse draft pool. Otherwise ``restart_floor`` rises to that end, usable if
              the delivered rows reach it; else, as after a later lease that is abandoned or misses
              rows, the interval is empty and the request computes from 0.
            - Without a sliding window the history stays where it was. ``restart_floor`` is that
              history under a policy other than all-reusable or in a joint-reuse draft pool.
              Otherwise it is the lower of that history and the lowest start of a fetch whose marked
              rows were copied without error; with no such fetch it is that history.
            - A failed copy over the block the committed tokens end inside empties the interval for
              good (``readiness``).

        Fails as ``lend_read``, and also at the call:
            - on no free pages, or while an earlier fetch into the request has not settled on this
              rank;
            - with SWA scratch reuse on (``enable_swa_scratch_reuse``) in a target cache: a grow
              breaks the old capacity's scratch rewind, and a window's next chunk overwrites scratch
              slots;
            - where the history already stands past the positions ``readiness`` would count, as
              after computing past ``end`` under another policy;
            - under the all-reusable policy, where a window leaves behind at ``end`` a block whose
              page holds tokens past the cache's history (see Unwritten pages below);
            - where the request's multimodal data sets ``mm_bidirectional_blocks`` and ``end`` falls
              strictly inside a run of multimodal tokens, whatever the run's length: a chunk resumed
              there sees the run only within the model's sliding window, which the lender cannot
              see;
            - on a request that returns context logits (prompt logprobs do) or additional model
              outputs, whether or not it holds any yet. The executor gives these only for the
              positions the request computes, and neither a rewind nor a recompute pause clears the
              rows held, so they would miss the fetched positions. Prompt logprobs pair the rows
              with the prompt's tokens from the second on, wherever the rows start, so they would
              pair rows with the wrong tokens. After a context step a fetch could also leave gaps
              or repeated rows and overflow the prompt-sized context logits, failing every active
              request.

            It fails at its first poll on a block left without a page, or on an error granting its
            slots within the call.

        Unwritten pages:
            Under the all-reusable policy a block a window leaves behind at ``end`` is neither
            fetched nor computed, yet keeps any page it had until the commit stores it whole; the
            manager cannot drop one block's page in one layer group. Pages holding tokens the
            request never wrote:

            - the block the committed tokens end inside, holding past them what the local match
              copied from another request's page (common with partial reuse: on a sliding-window
              model such a request computes from its local match);
            - pages grown before the fetch;
            - with speculative decoding's extra KV tokens (``num_extra_kv_tokens``, the draft length
              less one under one-model speculative decoding), their block, which each lease's grow
              pages past the history it leaves. Under a ``W``-token window, a later consecutive
              lease spanning at least ``W + tokens_per_block - 1`` tokens leaves it behind and
              fails: keep later leases shorter, or compute the rest.
        """
        ...

    def readiness(self, request: LlmRequest) -> Optional[Readiness]:
        """Where the request may resume once every fetch into it has settled.

        No fetched token counts as computed until this returns a ``Readiness``: when the copy
        ``mark_arrived`` queued is done or the fetch is abandoned. How each fetch counts: see
        ``lend_write``. The interval's bounds, such as the floor at the history, and how to combine
        ranks and pools: see ``Readiness``.

        A failed copy from ``mark_arrived`` counts none of its rows; where it covered the block the
        committed tokens end inside, the interval is empty from then on, as that block may mix
        fetched and older bytes across pools. Like ``lend_write``, it takes the tokens below the
        cache's history as computed, apart from those a windowed fetch moved the history over; the
        caller asks only while the request's pages hold the rest, so not while its history runs
        ahead of its data. A shrink in place, such as a context rollback of a fetch's growth, voids
        delivered rows past the new capacity for good: regrowth brings pages, not contents. Copies
        run serial with the forward on the GPU (``StagingLender``).

        Args:
            request: The request a fetch went into.

        Returns:
            ``None`` while a fetch into the request is unsettled, else where it may resume.

        Raises:
            ValueError: The request has no KV cache, which includes after the manager's shutdown.
        """
        ...


@runtime_checkable
class InPlaceLender(Protocol):
    """Lends a request's own device pages in place.

    The caller addresses lent blocks with its own page-table code (rows carry only layer groups and
    ordinals); ``lend_read`` states which blocks a lease covers. The lender never grows the cache.
    Its methods run only on the manager's thread (see the package docstring); backend threads access
    the lent memory. Lent pages stay until the last release, even after the request's free. Copy the
    lent blocks' page indices while the request still has its cache: once its page-index slot is
    freed (at the free, or after a context-only request's forward), an array viewing that slot can
    show another request's pages. Neither lender protocol derives from the other, but ``isinstance``
    checks only members: a staging lender passes ``isinstance(lender, InPlaceLender)`` too, so the
    attach that made a lender tells its mode.

    In-place lending is a transitional capability with explicit preconditions, not a general
    address-stable loan: a lent page keeps its address only while the caller keeps them.

    Caller must, while any loan is open, also after the request is freed:
        - Keep the request unscheduled, unsuspended, unshrunk and its window still: its sliding
          windows do not advance.
        - Commit none of the request's blocks (``try_commit_blocks``): commit before the first loan
          or after the last release. A commit can rebase the request onto blocks another request
          committed and return the request's own pages, lent ones included, to the pool.
        - Neither rebalance the pools nor reset the prefix cache:
          ``kv_cache_config.enable_kv_pool_rebalance`` stays off, since the executor's pool
          rebalance moves lent pages and, while a cache stays on loan after its request's free,
          raises out of the executor loop.
        - Synchronize as ``lend_read`` and ``lend_write`` say, for instance after
          ``prepare_resources`` queued the work: the lender waits on no stream. A KV cache
          connector's asynchronous loads and saves run off the manager's stream; wait for those
          touching lent pages too.
        - Own completion and the validity of what the backend reads and writes.
        - With several tensor- or pipeline-parallel ranks, release the last loan on a freed
          request's cache in the same executor iteration on every rank (say after combining
          completions): ranks schedule on their own, so a cache still open on one rank leaves its
          pools fewer free pages.
        - Where generation requests run on the manager, keep room for them in each pool group
          (package docstring). A cache on loan keeps every page it locks, not only the lent blocks',
          until its last release, even after its request's free, and the V2 scheduler cannot reclaim
          those pages. Without that room a generation request that cannot resume or grow can
          deadlock: a share of each pool group for the loans alone does not prevent it, with or
          without a cache tier below the GPU.
        - Where requests wait to be admitted on the manager, keep enough free pages to admit them
          beside the pages on loan, or keep each request whose cache is on loan among the
          executor's requests, in a disaggregated transfer state, until its last release. A cache
          on loan after its request's free belongs to no request the scheduler sees, so with
          nothing else running its deadlock check raises.

    First-version limits:
        - The lender guards lent pages only against the request's free and the manager's shutdown;
          nothing checks the preconditions above.
    """

    def lend_read(self, request: LlmRequest, start: int, end: int) -> Lease:
        """Lends for reading the pages of the blocks ``[start, end)`` touches, ready at once.

        Any token range, a partial last block included. A sliding-window layer group lends only its
        sinks and the blocks a history of ``end`` reads. In every layer group, only pages the
        request's active cache locks are lent, and every block without one is left out. Window
        blocks behind the history, which keep at most a held page, are one such case. Its outcome
        is this rank's own; the caller combines every rank's outcome. It fails at the call on no
        cache, a suspended cache or a manager shut down.

        Caller must:
            - Before the backend reads, ensure the work the manager's stream queued that still
              writes those pages completed.
            - Keep every precondition ``InPlaceLender`` lists while any loan is open; in particular,
              commit before lending or after the last release.

        Args:
            request: The request whose pages are lent.
            start: The first token.
            end: The end token.

        Returns:
            The lease, possibly failed already.

        Raises:
            ValueError: A negative or reversed range.
        """
        ...

    def lend_write(self, request: LlmRequest, start: int, end: int) -> Lease:
        """As ``lend_read``, for writing.

        The lease covers no block behind the window; a block it would lend that has no page its
        cache locks fails it at the call.

        Caller must:
            - Before the backend writes, ensure all work the manager's stream queued for those pages
              has completed.
            - Keep every precondition ``InPlaceLender`` lists while any loan is open; in particular,
              commit before lending or after the last release.

        Args:
            request: The request whose pages are lent.
            start: The first token.
            end: The end token.

        Returns:
            The lease, possibly failed already.

        Raises:
            ValueError: A negative or reversed range.
        """
        ...

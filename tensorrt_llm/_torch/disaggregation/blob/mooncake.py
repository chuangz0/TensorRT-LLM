# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A store driver over Mooncake's distributed store, pinned to SDK 0.3.13.

Each row is one object holding exactly the row's payload, its spans back to back, written from
and read into registered memory without a copy: the lender's host staging slots, or, for a driver
given ``allocation_range``, the KV cache's device pages too. The driver is synchronous: ``put``
and ``get`` return only after every SDK call they make has returned. Every SDK return code is
translated here and nowhere else.

Put: ``batch_put_session_start`` per key (the master reserves the object), one
``batch_put_from_multi_buffer_ranges`` scatter for all rows, then ``batch_put_session_end`` for
the rows written whole (their objects become readable) and ``batch_put_session_revoke`` for the
rest, except the sessions ``MooncakeDriver`` leaves open. Put uses sessions because this scatter has
no timeout, so a failed row is revoked only once nothing writes into its space; ``batch_put_from``
revokes a timed-out transfer at once, while its RDMA writes may still land in space the master has
given to another object.

Get: one ``batch_get_into_multi_buffers``. A get session would add two SDK calls, and every call
waits for the GIL on return, which a busy executor thread makes cost milliseconds.

Device memory is registered one piece per CUDA allocation (Mooncake refuses a GPUDirect
registration that crosses two), and a span crossing pieces goes to the SDK as one buffer per
piece: the transfer engine places each of its slices inside one registration.

When access ends. The put scatter waits, without a timeout, until every fragment it posted has
physically finished, so its return ends access whatever it answers, TRANSFER_FAIL included. The
get waits on one transfer future per key, and a future can give up with TRANSFER_FAIL (after 60 s,
on a status error, after a partly posted RDMA batch) while its transfer still runs; every other
answer comes after all futures finished or from a step that posted nothing. So a get ends access
only when no answer is TRANSFER_FAIL. In both, a raise or a wrong answer count leaves access
unproven. Session, existence and end calls touch only master and local metadata, never row
memory. This holds for a wheel built without ``USE_TENT`` (the newer transfer engine), the default
replicate config (one memory replica, no NoF SSD replica) and RDMA; the TCP transport's scatter
completion was not checked line by line. Over TCP a put row answered TRANSFER_FAIL may also mean the
initiator gave up on a slow target (no status within ``MC_TCP_STATUS_TIMEOUT_SEC``, or no progress
within ``MC_TCP_PROGRESS_TIMEOUT_SEC``, 30 s each by default), which may still write what it
received into the object's space; a revoke would free that space at once.

An operation whose access has not ended when its call returns never will by anything the driver
can wait on: it is ``access_unprovable``, and it keeps its regions from being unregistered and the
driver from draining.
"""

from __future__ import annotations

import bisect
import dataclasses
import itertools
import operator
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Optional

from tensorrt_llm.logger import logger

from ..base.memory import MemoryKind
from .store import (
    DriverCapabilities,
    DriverRejected,
    Hint,
    ProbeStatus,
    RegionLayout,
    RegistrationError,
    RowIO,
    RowResult,
    RowStatus,
    Span,
    StoreError,
    faulted,
)

__all__ = ["AllocationRange", "MooncakeDriver"]

AllocationRange = Callable[[int], Optional[tuple[int, int]]]
"""The ``(base, size)`` of the CUDA allocation holding an address, or None where the driver does
not answer: ``pyexecutor.connectors.mooncake_store.gpudirect.allocation_range``, which this
package may not import."""


class _Code(IntEnum):
    """The SDK return codes the driver names (``mooncake-store/include/types.h``)."""

    OK = 0
    INTERNAL_ERROR = -1
    NO_AVAILABLE_HANDLE = -200
    INVALID_PARAMS = -600
    ILLEGAL_CLIENT = -601
    INVALID_WRITE = -700
    INVALID_REPLICA = -702
    REPLICA_IS_NOT_READY = -703
    OBJECT_NOT_FOUND = -704
    OBJECT_ALREADY_EXISTS = -705
    LEASE_EXPIRED = -707
    TRANSFER_FAIL = -800
    RPC_FAIL = -900
    RPC_TIMEOUT = -901
    # Not an SDK code: a call raised or miscounted; storeWorker.h kCallFailed has the same value.
    CALL_FAILED = -100_000
    # Not an SDK code: another put of this client held the key, so the row was not put;
    # storeWorker.h kPutInFlightHere has the same value.
    PUT_IN_FLIGHT_HERE = -100_001


OBJECT_EXISTS, OBJECT_ABSENT = 1, 0  # batch_is_exist answers
# TODO(mooncake): the put scatter and the get answer each row's bytes as a 32-bit int.
MAX_ROW_BYTES = (1 << 31) - 1
# Keys in one batch_is_exist call: bounds the RPC's message and the master's lock time. A
# 32k-token prompt of 32-token blocks is 1,024 keys.
PROBE_KEYS_PER_CALL = 8192
_CAPABILITIES = DriverCapabilities(
    get=True,
    put=True,
    cancel=False,
    probe=True,
    max_rows_per_op=None,
    # The master answers OBJECT_NOT_FOUND for a key it does not hold; failures keep their own
    # codes and stay FAULT.
    absent_is_miss=True,
    device_memory=False,  # set per driver: only one given ``allocation_range`` registers it
    multi_span=True,
)
# The results that carry no fault text, one shared object each.
_PENDING = RowResult(RowStatus.PENDING)
_READY = RowResult(RowStatus.READY)
_MISS = RowResult(RowStatus.MISS)
_NOT_READY = RowResult(RowStatus.NOT_READY)
_DECLINED = RowResult(RowStatus.DECLINED)

# get -703: no complete replica, as while the object is written or once its replica's endpoint
# went stale.
_BEING_WRITTEN = "being_written"
# put start -600: the master refused the row (a cachelib master caps an object at 16 MiB - 16 B),
# or the SDK's own record of this client's put sessions holds the key; the master's log says which.
_OWN_SESSION_OPEN = "own_session_open"
# A put session the driver left open holds the key: the driver never puts that key again, and the
# master shows no complete object. Warned of once, when the session was left open.
_SESSION_LEFT_OPEN = "session_left_open"
# Another writer is still putting the object: put start -705 with the object uncommitted on the
# master, or another put of this client holds the key (skipped without a store call: the SDK keys
# put sessions per client). Normal traffic (two requests computing one prompt prefix at once), so
# no failure to warn about. That writer normally completes the object. If its put fails, nobody
# retries: the object stays missing until a server computes the prefix again without local reuse.
_ANOTHER_WRITER = "another_writer"
# put start -200: no space for the object. A full store is a cache's normal state: the master
# evicts within milliseconds to make room (Mooncake 0.3.13), so it is warned about once per
# process, not per operation.
_POOL_FULL = "pool_full"
# The result each skip reason gives (nothing moved and nothing went wrong); ``stats()`` counts
# skipped rows by reason and result.
_SKIPS = {
    _BEING_WRITTEN: _NOT_READY,
    _OWN_SESSION_OPEN: _NOT_READY,
    _SESSION_LEFT_OPEN: _NOT_READY,
    _ANOTHER_WRITER: _NOT_READY,
    _POOL_FULL: _DECLINED,
}
_QUIET_SKIPS = (_ANOTHER_WRITER, _POOL_FULL, _SESSION_LEFT_OPEN)
_PUT_START_SKIPS = {
    _Code.NO_AVAILABLE_HANDLE: _POOL_FULL,
    _Code.INVALID_PARAMS: _OWN_SESSION_OPEN,
    _Code.PUT_IN_FLIGHT_HERE: _ANOTHER_WRITER,
}
# Session-end answers after which the end may still reach the master, even when its retry is
# refused. Such a session is left open and its key never put again: the master matches an end by
# key and client, so a late end would mark complete this client's next object of the key.
_END_OUTCOME_UNKNOWN = (_Code.RPC_FAIL, _Code.RPC_TIMEOUT, _Code.CALL_FAILED)


class _LeftOpen(Enum):
    """Who ends a put session the driver left open; the value is its log text."""

    FOR_DRAIN = "for the shutdown drain to revoke"  # once no put runs
    TO_MASTER = "for the master to free"  # a target may still write into its object


@dataclass(eq=False)
class _Registration:
    """One registered region: its SDK registrations still in place, in address order (see
    ``MooncakeDriver._pieces``), and whether it admits rows: from its last piece's registration
    until its unregistration starts."""

    span: Span
    pieces: list[Span]
    live: bool = False

    def buffers(self, span: Span) -> tuple[list[int], list[int]]:
        """The addresses and lengths of ``span``'s parts, one per piece it covers, in order."""
        if len(self.pieces) == 1:
            return [span.address], [span.nbytes]
        addresses, lengths = [], []
        address = span.address
        # The first piece ending past the span's start holds it.
        index = bisect.bisect_right(self.pieces, address, key=_END)
        while address < span.end:
            stop = min(self.pieces[index].end, span.end)
            addresses.append(address)
            lengths.append(stop - address)
            address, index = stop, index + 1
        return addresses, lengths


_END = operator.attrgetter("end")


@dataclass(eq=False)
class _Op:
    """One settled put or get: its results and access end are known when the call returns."""

    results: tuple[RowResult, ...]
    access_ended: bool
    owner: MooncakeDriver


@dataclass(eq=False)
class _Batch:
    """One ``put`` or ``get`` call's rows in the SDK's terms, and their results so far."""

    keys: list[str]
    # Each row's spans in order, cut where a registration's pieces meet.
    buffers: list[list[int]]
    lengths: list[list[int]]
    sizes: list[int]  # each row's bytes: its object's size
    regions: list[_Registration]  # the registrations the rows lie in
    results: list[RowResult]
    skip_reasons: dict[int, str]  # row -> why it was skipped, a key of _SKIPS

    def skip(self, row: int, reason: str) -> None:
        """Concludes ``row`` as NOT_READY or DECLINED: nothing moved and nothing went wrong."""
        self.results[row] = _SKIPS[reason]
        self.skip_reasons[row] = reason

    def fault(self, indices: Sequence[int], reason: str) -> None:
        for row in indices:
            self.results[row] = faulted(reason)

    def pick(
        self, indices: Sequence[int]
    ) -> tuple[list[str], list[list[int]], list[list[int]], list[int]]:
        """The keys, buffers, lengths and sizes of the rows at ``indices``."""
        return (
            [self.keys[i] for i in indices],
            [self.buffers[i] for i in indices],
            [self.lengths[i] for i in indices],
            [self.sizes[i] for i in indices],
        )


class _WrongAnswerCount(Exception):
    """The SDK answered for a different number of keys than it was given; its state is unknown."""


class MooncakeDriver:
    """A ``StoreDriver`` over one set-up ``mooncake.store.MooncakeDistributedStore``.

    The SDK names a put session by its key, per client, not per caller, so the driver ends or
    revokes only the put sessions it started. It leaves a session open, and never puts its key
    again (a later put of it only checks whether the object exists):

    - for ``drain`` to revoke once no put runs: an end whose outcome is unknown
      (``_END_OUTCOME_UNKNOWN``), or a revoke that failed;
    - to the master, which frees it no earlier than ``put_start_release_timeout_sec`` (600 s by
      default) after the put started, and on a store with room up to about twice that (it looks
      every 600 s): a row the target may still write into, answered TRANSFER_FAIL or in a scatter
      that raised or miscounted.

    Args:
        store: An SDK store whose ``setup`` returned 0.
        allocation_range: Where given, the driver registers device regions too.
    """

    def __init__(self, store: Any, *, allocation_range: AllocationRange | None = None) -> None:
        self._store = store
        self._allocation_range = allocation_range
        self._capabilities = dataclasses.replace(
            _CAPABILITIES, device_memory=allocation_range is not None
        )
        # Guards the fields below and wakes ``drain``; never held across an SDK call.
        self._lock = threading.Condition()
        self._registrations: list[_Registration] = []
        self._draining = False
        self._closed = False
        # By key, never put again; drain drops those it revokes (no put runs after a drain).
        self._sessions_left_open: dict[str, _LeftOpen] = {}
        self._left_open_counts: Counter[_LeftOpen] = Counter()  # sessions ever left open
        self._keys_being_put: set[str] = set()  # by puts that have not returned
        self._in_flight = 0  # put/get calls past admission and not yet returned
        self._unprovable = 0  # operations whose access never ended
        self._busy: Counter[_Registration] = Counter()  # operations whose access may touch it
        self._skipped_rows: Counter[str] = Counter()  # by reason

    @property
    def capabilities(self) -> DriverCapabilities:
        return self._capabilities

    def register(self, address: int, nbytes: int, layout: RegionLayout) -> _Registration:
        try:
            span = Span(address, nbytes)
        except ValueError as error:
            raise DriverRejected(str(error)) from None
        if layout.nbytes > nbytes:
            raise DriverRejected(f"layout needs {layout.nbytes} bytes, region has {nbytes}")
        pieces = self._pieces(span, layout.memory_kind)
        with self._lock:
            self._require_open(DriverRejected)
            overlapping = [
                r.span for r in self._registrations if r.pieces and r.span.overlaps(span)
            ]
        if overlapping:
            raise DriverRejected(f"region {span} overlaps a live registration {overlapping[0]}")
        started = time.monotonic()
        registration = _Registration(span, [])
        for piece in pieces:
            try:
                code = self._store.register_buffer(piece.address, piece.nbytes)
            except Exception as error:
                registration.pieces.append(piece)  # the SDK may have registered it
                self._add(registration)
                raise RegistrationError(
                    f"register_buffer raised {error!r}", registration
                ) from error
            if code != _Code.OK:  # the SDK rolled this piece back; the earlier ones are ours
                raise self._rolled_back(registration, f"register_buffer returned {_describe(code)}")
            registration.pieces.append(piece)
        registration.live = True
        self._add(registration)
        if layout.memory_kind is MemoryKind.DEVICE:
            logger.info(
                f"mooncake: registered device region {layout.label} of {nbytes / (1 << 30):.2f} "
                f"GiB in {len(pieces)} pieces in {time.monotonic() - started:.1f} s"
            )
        return registration

    def unregister(self, handle: _Registration) -> None:
        with self._lock:
            registration = self._registration(handle)
            if not registration.pieces:
                return
            if self._busy[registration]:
                raise DriverRejected(
                    f"{self._busy[registration]} operation(s) may still access the region"
                )
            registration.live = False  # admits no new rows from now on
        failure = self._unregister_pieces(registration)
        if failure is not None:  # the pieces left stay registered: the call may be retried
            raise StoreError(failure)

    def put(self, rows: Sequence[RowIO]) -> _Op:
        batch, started = self._admit(rows), time.monotonic()  # the only raise is in _admit
        own, left_open_rows = self._hold_keys_to_put(batch)
        try:
            if left_open_rows:  # never put again, but the object may be complete
                self._confirm_existing(batch, left_open_rows, _SESSION_LEFT_OPEN)
            ended = self._put_rows(batch, own) if own else True
        except Exception as error:  # a bug, not an SDK answer: a transfer may still be running
            batch.fault(own + left_open_rows, f"put failed: {error!r}")
            ended = False
        finally:
            with self._lock:
                self._keys_being_put.difference_update(batch.keys[i] for i in own)
        return self._conclude("put", batch, ended, time.monotonic() - started)

    def get(self, rows: Sequence[RowIO], hint: Hint = None) -> _Op:
        if hint is not None:
            raise DriverRejected("this driver reads no hints")
        batch, started = self._admit(rows), time.monotonic()  # the only raise is in _admit
        try:
            ended = self._get_rows(batch, list(range(len(batch.keys))), may_reread=True)
        except Exception as error:  # a bug, not an SDK answer: a transfer may still be running
            batch.fault(range(len(batch.keys)), f"get failed: {error!r}")
            ended = False
        return self._conclude("get", batch, ended, time.monotonic() - started)

    def poll(self, op: _Op) -> tuple[RowResult, ...]:
        return self._op(op).results

    def access_ended(self, op: _Op) -> bool:
        return self._op(op).access_ended

    def access_unprovable(self, op: _Op) -> bool:
        return not self._op(op).access_ended  # it settled when the call returned

    def cancel(self, op: _Op) -> None:
        """Never called: the SDK has no cancel, so ``capabilities.cancel`` is false."""
        self._op(op)

    def probe(
        self, keys: Sequence[str], budget_s: float, hint: Hint = None
    ) -> tuple[ProbeStatus, ...]:
        """Asks the master which keys hold a readable object, ``PROBE_KEYS_PER_CALL`` keys per
        call. ``batch_is_exist`` answers 0 both for an absent key and for one whose put is still
        open, so both are MISS. It takes no deadline: the client's RPC timeout
        (``rpc_timeout_ms`` of the store config), not ``budget_s``, bounds each call; a key it
        timed out on is UNKNOWN. Any other error code raises ``StoreError``.

        It also grants every existing key a read lease of the master's ``default_kv_lease_ttl``,
        so probed objects outlive eviction that long.

        TODO(mooncake): a per-call deadline for batch_is_exist, and an answer that tells an open
        put from an absent key."""
        if hint is not None:
            raise DriverRejected("this driver reads no hints")
        self._require_open()
        statuses: list[ProbeStatus] = []
        for start in range(0, len(keys), PROBE_KEYS_PER_CALL):
            chunk = list(keys[start : start + PROBE_KEYS_PER_CALL])
            statuses += [
                _probe_status(answer)
                for answer in _answers(self._store.batch_is_exist(chunk), len(chunk))
            ]
        return tuple(statuses)

    def drain(self, budget_s: float) -> bool:
        """Refuses new work for good and waits up to ``budget_s`` for the calls in progress;
        False at once while an operation is unprovable. A True return also revokes the put
        sessions left open for drain (``_end_drain``)."""
        with self._lock:
            self._require_open()
            self._draining = True
            self._lock.wait_for(lambda: self._unprovable or not self._in_flight, budget_s)
        return self._end_drain()

    def close(self) -> None:
        """Closes the SDK client once a drain returned True and every region is unregistered.
        Never called in production: it holds the GIL through unmount RPCs; process exit frees it."""
        if self._mark_closed():
            self._store.close()

    def stats(self) -> dict[str, int]:
        """A snapshot for logs and tests; not part of the driver protocol."""
        with self._lock:
            return {
                **{
                    f"rows_{r.status.value}_{reason}": self._skipped_rows[reason]
                    for reason, r in _SKIPS.items()
                },
                "ops_unprovable": self._unprovable,
                "put_sessions_left_for_drain": self._left_open_counts[_LeftOpen.FOR_DRAIN],
                "put_sessions_left_to_master": self._left_open_counts[_LeftOpen.TO_MASTER],
            }

    # Put: start, one scatter, end what was written whole, revoke the rest.

    def _hold_keys_to_put(self, batch: _Batch) -> tuple[list[int], list[int]]:
        """Splits the batch's rows: those to put, whose keys are now held until this put returns,
        and those whose key's session the driver left open, which are only checked for existence.
        A row whose key another put of this driver holds is skipped as another writer's."""
        own, left_open_rows = [], []
        with self._lock:
            for row, row_key in enumerate(batch.keys):
                if row_key in self._sessions_left_open:
                    left_open_rows.append(row)
                elif row_key in self._keys_being_put:
                    batch.skip(row, _ANOTHER_WRITER)
                else:
                    self._keys_being_put.add(row_key)
                    own.append(row)
        return own, left_open_rows

    def _put_rows(self, batch: _Batch, rows: list[int]) -> bool:
        """Puts ``rows`` of the batch; returns whether access ended."""
        keys, _, _, sizes = batch.pick(rows)
        starts = self._metadata_call(self._store.batch_put_session_start, keys, sizes)
        started, existing = [], []
        for row, code in zip(rows, starts):
            if code == _Code.OK:
                started.append(row)
            elif code == _Code.OBJECT_ALREADY_EXISTS:
                existing.append(row)
            elif code in _PUT_START_SKIPS:
                batch.skip(row, _PUT_START_SKIPS[code])
            else:
                batch.fault([row], _put_start_fault(code))
        ended = self._write_and_finish_sessions(batch, started) if started else True
        if existing:
            self._confirm_existing(batch, existing, _ANOTHER_WRITER)
        return ended

    def _write_and_finish_sessions(self, batch: _Batch, started: list[int]) -> bool:
        """Writes the rows whose put session started in one scatter, then ends the session of each
        row written whole and revokes the rest; leaves open the sessions the class docstring
        names. Returns whether access ended."""
        keys, buffers, lengths, sizes = batch.pick(started)
        # A row's buffers fill its object back to back from offset 0.
        dst_offsets = [list(itertools.accumulate(row[:-1], initial=0)) for row in lengths]
        try:
            scatter = self._store.batch_put_from_multi_buffer_ranges
            answers = _answers(scatter(keys, buffers, lengths, dst_offsets), len(keys))
        except Exception as error:  # the scatter may still be reading the slots
            batch.fault(started, f"batch_put_from_multi_buffer_ranges raised {error!r}")
            # Neither end (the object may be torn) nor revoke (the master would give its space
            # to another object while the scatter may still write into it).
            self._leave_open(keys, _LeftOpen.TO_MASTER)
            return False
        written, to_revoke, to_master = started, [], []
        if answers != sizes:  # rare: some row was not written whole
            written = []
            for i, answer, size in zip(started, answers, sizes):
                if answer == size:
                    written.append(i)
                    continue
                batch.fault([i], _put_ranges_fault(answer, size))  # never to be marked complete
                # Over TCP the target may still write into the object's space (module docstring).
                (to_master if answer == _Code.TRANSFER_FAIL else to_revoke).append(i)
        to_revoke += self._end_put_sessions(batch, written)
        self._revoke_put_sessions([batch.keys[i] for i in to_revoke])
        self._leave_open([batch.keys[i] for i in to_master], _LeftOpen.TO_MASTER)
        return True

    def _end_put_sessions(self, batch: _Batch, written: list[int]) -> list[int]:
        """Ends the sessions of rows written whole; returns the rows whose end was refused, to
        revoke. A session whose end may still reach the master is left open for drain."""
        if not written:
            return []
        end = self._store.batch_put_session_end
        keys = [batch.keys[i] for i in written]
        first = self._metadata_call(end, keys)
        final = dict(zip(keys, first))
        retry = [key for key, code in final.items() if code in (_Code.RPC_FAIL, _Code.RPC_TIMEOUT)]
        if retry:  # ending is idempotent on the master
            final.update(zip(retry, self._metadata_call(end, retry)))
        refused, outcome_unknown = [], []
        for i, row_key, first_code in zip(written, keys, first):
            code = final[row_key]
            if code == _Code.OK:
                batch.results[i] = _READY
                continue
            batch.results[i] = faulted(_put_end_fault(code))
            (outcome_unknown if first_code in _END_OUTCOME_UNKNOWN else refused).append(i)
        self._leave_open([batch.keys[i] for i in outcome_unknown], _LeftOpen.FOR_DRAIN)
        return refused

    def _revoke_put_sessions(self, keys: list[str]) -> None:
        if keys:
            codes = self._metadata_call(self._store.batch_put_session_revoke, keys)
            failed = [key for key, code in zip(keys, codes) if not _gone(code)]
            self._leave_open(failed, _LeftOpen.FOR_DRAIN)

    def _leave_open(self, keys: list[str], kind: _LeftOpen) -> None:
        if keys:
            _warn_left_open(keys, kind)
            with self._lock:
                self._record_left_open(keys, kind)

    def _record_left_open(self, keys: Sequence[str], kind: _LeftOpen) -> None:
        """Under the lock."""
        self._sessions_left_open.update(dict.fromkeys(keys, kind))
        self._left_open_counts[kind] += len(keys)

    def _confirm_existing(self, batch: _Batch, rows: list[int], reason_if_absent: str) -> None:
        """Marks READY the rows whose object the master shows complete; skips the others for
        ``reason_if_absent``, or faults them when the master could not tell."""
        keys = [batch.keys[i] for i in rows]
        for i, answer in zip(rows, self._metadata_call(self._store.batch_is_exist, keys)):
            if answer == OBJECT_EXISTS:
                batch.results[i] = _READY
            elif answer == OBJECT_ABSENT:
                batch.skip(i, reason_if_absent)
            else:
                batch.fault([i], _existence_fault(answer))

    # Get: one batch_get_into_multi_buffers; rows whose read lease expired mid-read are read once
    # more.

    def _get_rows(self, batch: _Batch, indices: list[int], may_reread: bool) -> bool:
        """Returns whether access ended."""
        keys, buffers, lengths, sizes = batch.pick(indices)
        # TODO(mooncake): a get into buffers longer than the object answers TRANSFER_FAIL without
        # posting a transfer, so its access stays unproven; an INVALID_PARAMS answer would end it.
        try:
            get = self._store.batch_get_into_multi_buffers
            answers = _answers(get(keys, buffers, lengths), len(indices))
        except Exception as error:  # the transfers may still be writing the slots
            batch.fault(indices, f"batch_get_into_multi_buffers raised {error!r}")
            return False
        if answers == sizes:  # usually every row was read whole
            for i in indices:
                batch.results[i] = _READY
            return True
        for i, answer in zip(indices, answers):
            if not may_reread and answer in (_Code.OBJECT_NOT_FOUND, _Code.REPLICA_IS_NOT_READY):
                batch.results[i] = faulted(_reread_fault(answer))
            elif answer == _Code.OBJECT_NOT_FOUND:
                batch.results[i] = _MISS
            elif answer == _Code.REPLICA_IS_NOT_READY:
                batch.skip(i, _BEING_WRITTEN)
            else:
                batch.results[i] = _get_result(answer, batch.sizes[i])
        # A TRANSFER_FAIL future may give up while its RDMA still runs (module docstring).
        # TODO(mooncake): a failure that says every transfer it posted ended (a peer that died
        # mid-get, say) would let the reader count a miss instead of ending.
        ended = _Code.TRANSFER_FAIL not in answers
        expired = [i for i, answer in zip(indices, answers) if answer == _Code.LEASE_EXPIRED]
        if expired and may_reread:
            # The read lease expired before the transfer finished, so the master may have
            # reused the object's space mid-read: read once more. The reread is READY only if it
            # reads the whole object again; anything else, a second expiry included, is a FAULT.
            reread_ended = self._get_rows(batch, expired, may_reread=False)
            ended = ended and reread_ended  # a TRANSFER_FAIL of the first read stays
        return ended

    # Bookkeeping, shared with the native driver.

    def _admit(self, rows: Sequence[RowIO]) -> _Batch:
        """Checks ``rows`` and records the operation, or raises ``DriverRejected`` having done
        neither."""
        if not rows:
            raise DriverRejected("an operation needs at least one row")
        keys = [row.key for row in rows]
        # The SDK keys put sessions and get targets by name: a repeat would overwrite the other.
        if len(set(keys)) != len(keys):
            raise DriverRejected("a batch repeats a key")
        sizes = [row.nbytes for row in rows]
        if max(sizes) > MAX_ROW_BYTES:
            row = next(row for row in rows if row.nbytes > MAX_ROW_BYTES)
            raise DriverRejected(f"a row holds at most {MAX_ROW_BYTES} bytes, got {row}")
        spans = [span for row in rows for span in row.spans]
        holders = self._reserve(spans)
        # Busy, the registrations keep their pieces.
        buffers, lengths = _buffers(rows, holders)
        regions = list(dict.fromkeys(holders))
        return _Batch(keys, buffers, lengths, sizes, regions, [_PENDING] * len(rows), {})

    def _reserve(self, spans: Sequence[Span]) -> list[_Registration]:
        """The live registration each span lies in, all of them marked busy for one more
        operation; raises ``DriverRejected`` having done nothing."""
        low, high = min(span.address for span in spans), max(span.end for span in spans)
        with self._lock:
            self._require_open(DriverRejected)
            if self._draining:
                raise DriverRejected("the driver is draining")
            live = [r for r in self._registrations if r.live]
            # Usually one registration holds every span: one check instead of one per span.
            covering = next((r for r in live if r.span.address <= low and high <= r.span.end), None)
            holders = [covering] * len(spans) if covering is not None else _holders(spans, live)
            self._in_flight += 1
            self._busy.update(set(holders))
        return holders

    def _release(
        self, regions: Sequence[_Registration], ended: bool, skip_reasons: Iterable[str]
    ) -> None:
        """Ends an operation ``_reserve`` admitted: its regions go free if its access ended, and
        are held for good otherwise."""
        with self._lock:
            self._skipped_rows.update(skip_reasons)
            self._in_flight -= 1
            if ended:
                self._busy.subtract(regions)
            else:
                self._unprovable += 1
            self._lock.notify_all()

    def _end_drain(self) -> bool:
        """Whether nothing runs and nothing is unprovable. When so, it also revokes the put
        sessions left open for drain: no put runs any more, so none is in use. They hold no lender
        memory, so they never change the answer, and one whose revocation fails stays open until
        the master discards it. Sessions left to the master are never revoked here."""
        with self._lock:
            quiet = not (self._in_flight or self._unprovable)
            for_drain = []
            if quiet:
                left = self._sessions_left_open.items()
                for_drain = sorted(k for k, kind in left if kind is _LeftOpen.FOR_DRAIN)
        if for_drain:
            codes = self._metadata_call(self._store.batch_put_session_revoke, for_drain)
            with self._lock:
                for k, code in zip(for_drain, codes):
                    if _gone(code):
                        self._sessions_left_open.pop(k, None)
        return quiet

    def _mark_closed(self) -> bool:
        """True the first time, once a drain returned True and every region is unregistered;
        raises ``DriverRejected`` otherwise."""
        with self._lock:
            if self._closed:
                return False
            if not self._draining or self._in_flight or self._unprovable:
                raise DriverRejected("close needs a drain that returned True")
            if any(r.pieces for r in self._registrations):
                raise DriverRejected("close needs every registration unregistered")
            self._closed = True
            return True

    def _metadata_call(self, call: Callable, keys: list[str], *args) -> list[int]:
        """A session or existence call. It touches no row memory, so a raise or a wrong answer
        count fails only its keys."""
        try:
            return _answers(call(keys, *args), len(keys))
        except Exception as error:
            logger.warning(f"mooncake: {getattr(call, '__name__', call)} failed: {error!r}")
            return [_Code.CALL_FAILED] * len(keys)

    def _conclude(self, direction: str, batch: _Batch, ended: bool, seconds: float) -> _Op:
        self._release(batch.regions, ended, batch.skip_reasons.values())
        if _POOL_FULL in batch.skip_reasons.values():
            _warn_store_full()
        expected = (RowStatus.READY, RowStatus.MISS)
        failed = [
            i
            for i, r in enumerate(batch.results)
            if r.status not in expected and batch.skip_reasons.get(i) not in _QUIET_SKIPS
        ]
        first = None
        if failed:
            result = batch.results[failed[0]]
            why = result.fault or f"{result.status.value} ({batch.skip_reasons[failed[0]]})"
            first = f"{batch.keys[failed[0]]}: {why}"
        _report(
            direction,
            rows=len(batch.keys),
            failed=len(failed),
            first=first,
            ended=ended,
            seconds=seconds,
        )
        return _Op(tuple(batch.results), ended, self)

    def _add(self, registration: _Registration) -> None:
        with self._lock:
            self._registrations.append(registration)

    def _pieces(self, span: Span, memory_kind: MemoryKind) -> list[Span]:
        """What a region is registered as: host memory whole, device memory one piece per CUDA
        allocation it covers."""
        if memory_kind is MemoryKind.HOST:
            return [span]
        if self._allocation_range is None:
            raise DriverRejected("this driver registers host memory only")
        pieces, address = [], span.address
        while address < span.end:
            allocation = self._allocation_range(address)
            if allocation is None:
                raise DriverRejected(f"no CUDA allocation is known at {address:#x}")
            base, size = allocation
            if not base <= address < base + size:
                raise DriverRejected(f"the CUDA allocation at {address:#x} does not hold it")
            stop = min(base + size, span.end)
            pieces.append(Span(address, stop - address))
            address = stop
        return pieces

    def _rolled_back(self, registration: _Registration, reason: str) -> RegistrationError:
        """Unregisters the pieces a failed registration made; its RegistrationError, without a
        handle when nothing stays registered."""
        failure = self._unregister_pieces(registration)
        if failure is None:
            return RegistrationError(reason)
        self._add(registration)
        return RegistrationError(f"{reason}; then rolling back: {failure}", registration)

    def _unregister_pieces(self, registration: _Registration) -> str | None:
        """Unregisters the registration's pieces, the latest first, up to the first failure: its
        reason, or None when none is left."""
        while registration.pieces:
            try:
                code = self._store.unregister_buffer(registration.pieces[-1].address)
            except Exception as error:
                return f"unregister_buffer raised {error!r}"
            if code != _Code.OK:
                return f"unregister_buffer returned {_describe(code)}"
            registration.pieces.pop()
        return None

    def _registration(self, handle: _Registration) -> _Registration:
        self._require_open()
        if not isinstance(handle, _Registration) or handle not in self._registrations:
            raise ValueError("handle does not belong to this driver")
        return handle

    def _op(self, op: _Op) -> _Op:
        self._require_open()
        if not isinstance(op, _Op) or op.owner is not self:
            raise ValueError("operation does not belong to this driver")
        return op

    def _require_open(self, error: type[Exception] = RuntimeError) -> None:
        """Raises ``error`` once closed: DriverRejected refuses new work, RuntimeError the rest."""
        if self._closed:
            raise error("the driver is closed")


def _holders(spans: Sequence[Span], live: list[_Registration]) -> list[_Registration]:
    """The registration each span lies in."""
    holders = []
    for span in spans:
        holder = next((r for r in live if span.within(r.span)), None)
        if holder is None:
            raise DriverRejected(f"{span} is outside every live registration")
        holders.append(holder)
    return holders


def _buffers(
    rows: Sequence[RowIO], holders: list[_Registration]
) -> tuple[list[list[int]], list[list[int]]]:
    """Each row's buffers and their lengths: its spans in order, each cut at the pieces of its
    holder, the registration it lies in."""
    holder = iter(holders)
    buffers, lengths = [], []
    for row in rows:
        row_buffers, row_lengths = [], []
        for span in row.spans:
            addresses, nbytes = next(holder).buffers(span)
            row_buffers += addresses
            row_lengths += nbytes
        buffers.append(row_buffers)
        lengths.append(row_lengths)
    return buffers, lengths


def _get_result(answer: int, size: int) -> RowResult:
    """A read answers the object's size, which must be the row's."""
    return _READY if answer == size else faulted(_get_fault(answer, size))


# The fault texts of both Mooncake drivers, one per store call that can fault a row.


def _get_fault(answer: int, size: int) -> str:
    if answer >= 0:
        return f"object is {answer} bytes, expected {size}"
    return f"batch_get_into_multi_buffers: {_describe(answer)}"


def _reread_fault(answer: int) -> str:
    """A reread that found no object or an unfinished one: the first read already wrote the slot,
    so MISS or NOT_READY would claim that nothing landed there."""
    return f"reread after LEASE_EXPIRED: {_describe(answer)}"


def _put_start_fault(code: int) -> str:
    return f"put session start: {_describe(code)}"


def _put_ranges_fault(answer: int, size: int) -> str:
    return f"put ranges: {_moved(answer, size)}"


def _put_end_fault(code: int) -> str:
    return f"put session end: {_describe(code)}"


def _existence_fault(code: int) -> str:
    return f"existence check: {_describe(code)}"


def _gone(revoke_answer: int) -> bool:
    """Whether a revocation left no session: revoked, or none was left (INVALID_PARAMS)."""
    return revoke_answer in (_Code.OK, _Code.INVALID_PARAMS)


def _moved(answer: int, size: int) -> str:
    return f"moved {answer} of {size} bytes" if answer >= 0 else _describe(answer)


def _probe_status(answer: int) -> ProbeStatus:
    if answer == OBJECT_EXISTS:
        return ProbeStatus.HIT
    if answer == OBJECT_ABSENT:
        return ProbeStatus.MISS
    if answer == _Code.RPC_TIMEOUT:
        return ProbeStatus.UNKNOWN
    raise StoreError(f"batch_is_exist answered {_describe(answer)}")


def _answers(answers: Sequence[int], count: int) -> list[int]:
    answers = list(answers)
    if len(answers) != count:
        raise _WrongAnswerCount(f"{len(answers)} answers for {count} keys")
    return answers


def _describe(code: int) -> str:
    try:
        return f"{_Code(code).name}({code})"
    except ValueError:
        return f"code({code})"


def _warn_left_open(keys: Sequence[str], kind: _LeftOpen) -> None:
    logger.warning(f"mooncake: {len(keys)} put sessions left open {kind.value}, first {keys[0]}")


def _warn_store_full() -> None:
    logger.warning_once(
        "mooncake: the store declined objects for lack of space; its master evicts to make room "
        "(driver stats rows_declined_pool_full)",
        key="mooncake: the store declined objects for lack of space",
    )


def _report(
    direction: str, *, rows: int, failed: int, first: str | None, ended: bool, seconds: float
) -> None:
    """One warning per operation that failed or skipped a row (``first`` names the first, with
    why) or left access open; none otherwise. Cancelled rows count as neither, nor do puts of
    objects another writer is still putting, the full store declined, or whose session this
    driver left open (warned of when it was left)."""
    if ended and not failed:
        return
    detail = f"; first {first}" if first else ""
    access = "ended" if ended else "may still be in progress"
    logger.warning(
        f"mooncake {direction}: {failed} of {rows} rows failed or skipped in {seconds:.3f} s"
        f"{detail}; access {access}"
    )

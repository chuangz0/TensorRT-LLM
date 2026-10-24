# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A Mooncake driver whose store calls run on the C++ store worker
(``tensorrt_llm.mooncake_store_worker``), on threads that never hold the GIL.

Under a busy executor thread, every SDK call that returns into Python waits for the GIL, about one
switch interval (5 ms) each; that halves the bandwidth of the Python driver's calls. Here ``put``
and ``get`` only queue an operation. The worker runs gets before puts, cuts each operation into store
calls and follows the protocol of ``mooncake.py``: it records for each row a state, the SDK's answer
and the call that gave it, from which this module builds the fault texts and skip reasons with
``mooncake.py``'s helpers. Registration, admission, the probe and draining's session cleanup stay
``MooncakeDriver``'s, on the Python client; the worker calls the same client, so it sees the
registrations.
"""

from __future__ import annotations

import dataclasses
import time
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np

from tensorrt_llm.logger import logger

from .mooncake import (
    _ANOTHER_WRITER,
    _POOL_FULL,
    _PUT_START_SKIPS,
    _QUIET_SKIPS,
    AllocationRange,
    MooncakeDriver,
    _Code,
    _existence_fault,
    _get_fault,
    _put_end_fault,
    _put_ranges_fault,
    _put_start_fault,
    _Registration,
    _report,
    _reread_fault,
    _warn_store_full,
)
from .store import (
    DriverRejected,
    FinishedOp,
    Hint,
    RowArrays,
    RowIO,
    RowResult,
    RowStatus,
    Span,
    faulted,
)

__all__ = ["NativeMooncakeDriver"]

# storeWorker.h RowState, by value.
_ROW_STATES = (
    RowStatus.PENDING,
    RowStatus.READY,
    RowStatus.MISS,
    RowStatus.NOT_READY,
    RowStatus.DECLINED,
    RowStatus.CANCELLED,
    RowStatus.FAULT,
)
_READY, _MISS = RowResult(RowStatus.READY), RowResult(RowStatus.MISS)
_CANCELLED = RowResult(RowStatus.CANCELLED)


class _StoreCall(IntEnum):
    """storeWorker.h StoreCall: the store call whose answer gave a row its code."""

    NONE = 0
    GET = 1
    REREAD = 2
    PUT_START = 3
    PUT_RANGES = 4
    PUT_END = 5
    EXISTENCE = 6
    REVOKE = 7


# The worker's errors of these calls fail only their keys: logged, as MooncakeDriver logs them.
_METADATA_CALLS = (
    _StoreCall.PUT_START,
    _StoreCall.PUT_END,
    _StoreCall.EXISTENCE,
    _StoreCall.REVOKE,
)
# The worker's errors of these calls become their rows' fault text, as MooncakeDriver's do.
_DATA_CALLS = (_StoreCall.GET, _StoreCall.REREAD, _StoreCall.PUT_RANGES)
# How long the process's exit waits for running store calls, before Mooncake tears its clients down.
_NATIVE_EXIT_WAIT_S = 5.0


@dataclass(eq=False)
class _NativeOp:
    """One queued put or get; ``finished`` and what follows are set once, when its rows all
    concluded."""

    owner: NativeMooncakeDriver
    worker_op: int
    is_put: bool
    stem: str
    names: np.ndarray  # uint8 [rows, width]
    hex_names: bool
    row_bytes: np.ndarray  # int64 [rows], a broadcast view when every row is alike
    regions: list[_Registration]
    submitted: float
    finished: FinishedOp | None = None
    other_statuses: dict[int, RowStatus] = field(default_factory=dict)  # neither READY, MISS
    # other_statuses rows another writer is still putting or the full store declined: not warned
    quiet_skips: int = 0
    store_full: bool = False
    cancelled: np.ndarray | None = None  # bool [rows]; None when no row was cancelled
    first_failure: str | None = None  # the first row in other_statuses, with why, for the log
    call_errors: list[str] = field(default_factory=list)  # metadata calls' raises, for the log
    sessions_left_open: list[str] = field(default_factory=list)
    results: tuple[RowResult, ...] | None = None  # built for the first poll that wants them

    def key(self, row: int) -> str:
        name = self.names[row].tobytes()
        return self.stem + (name.hex() if self.hex_names else name.rstrip(b"\0").decode())


class NativeMooncakeDriver(MooncakeDriver):
    """A ``QueuedStoreDriver`` over one set-up ``MooncakeDistributedStore`` and a store worker
    that calls the same client.

    Args:
        store: An SDK store whose ``setup`` returned 0.
        worker: A ``mooncake_store_worker.StoreWorker`` over ``store``'s client.
        allocation_range: Where given, the driver registers device regions too.
    """

    def __init__(
        self, store: Any, worker: Any, *, allocation_range: AllocationRange | None = None
    ) -> None:
        super().__init__(store, allocation_range=allocation_range)
        self._worker = worker
        self._capabilities = dataclasses.replace(self._capabilities, cancel=True, queued=True)
        self._open: dict[int, _NativeOp] = {}  # by worker op, until finished
        self._untaken: list[_NativeOp] = []  # finished, not yet handed out by take_finished
        # Keyed by live registrations: their pieces never change once live, and one whose
        # unregistration started is never live again (a re-registration is a new _Registration).
        self._cuts_by_regions: dict[tuple[_Registration, ...], np.ndarray] = {}
        # Python's exit hooks run before Mooncake's C exit handler tears its clients down: the worker's
        # threads stop first, waiting a few seconds for running calls.
        weakref.finalize(self, worker.stop, _NATIVE_EXIT_WAIT_S)

    # The backend's path: rows as arrays.

    def put_rows(self, rows: RowArrays) -> _NativeOp:
        return self._queue_rows(True, rows)

    def get_rows(self, rows: RowArrays) -> _NativeOp:
        return self._queue_rows(False, rows)

    def take_finished(self) -> list[FinishedOp]:
        self._take()
        if not self._untaken:  # an executor's poll usually finds nothing new
            return []
        with self._lock:
            taken, self._untaken = self._untaken, []
        return [op.finished for op in taken]

    def finished_waiting_s(self) -> float:
        # A poll or a drain already took these from the worker; their age is not kept, so they
        # count as overdue.
        return float("inf") if self._untaken else self._worker.finished_waiting_s()

    # The StoreDriver protocol's path: rows as RowIO.

    def put(self, rows: Sequence[RowIO]) -> _NativeOp:
        return self._queue_row_ios(True, rows)

    def get(self, rows: Sequence[RowIO], hint: Hint = None) -> _NativeOp:
        if hint is not None:
            raise DriverRejected("this driver reads no hints")
        return self._queue_row_ios(False, rows)

    def poll(self, op: _NativeOp) -> tuple[RowResult, ...]:
        op = self._op(op)
        self._take()
        if op.finished is None:
            return (RowResult(RowStatus.PENDING),) * len(op.names)
        if op.results is None:
            results = [_READY if ready else _MISS for ready in op.finished.ready.tolist()]
            if op.cancelled is not None:
                for row in np.flatnonzero(op.cancelled).tolist():
                    results[row] = _CANCELLED
            for row, status in op.other_statuses.items():
                fault = op.finished.faults.get(row)
                results[row] = RowResult(status) if fault is None else faulted(fault)
            op.results = tuple(results)
        return op.results

    def access_ended(self, op: _NativeOp) -> bool:
        op = self._op(op)
        self._take()
        return op.finished is not None and op.finished.access_ended

    def access_unprovable(self, op: _NativeOp) -> bool:
        op = self._op(op)
        self._take()
        return op.finished is not None and not op.finished.access_ended

    def cancel(self, op: _NativeOp) -> None:
        """Calls of ``op`` not started yet conclude their rows CANCELLED; running ones go on."""
        self._worker.cancel(self._op(op).worker_op)

    def drain(self, budget_s: float) -> bool:
        """Refuses new work for good, concludes the queued calls CANCELLED and waits up to
        ``budget_s`` for the running ones without the GIL; at once while an operation is
        unprovable. When it returns True it also revokes the put sessions left open."""
        with self._lock:
            self._require_open()
            self._draining = True
        self._take()
        self._worker.drain(0.0 if self._unprovable else budget_s)
        self._take()
        return self._end_drain()

    def close(self) -> None:
        """Stops the worker's threads, then closes the SDK client, as ``MooncakeDriver.close``:
        a call running beside the client's teardown would use freed memory."""
        if self._mark_closed():
            self._worker.stop(0.0)
            self._store.close()

    # Queueing and collecting.

    def _queue_rows(self, is_put: bool, rows: RowArrays) -> _NativeOp:
        # One bounding span per column: a column is one pool or one staging region.
        span_bytes = np.array(rows.span_bytes, dtype=np.int64)
        lows, highs = rows.starts.min(0), rows.starts.max(0) + span_bytes
        holders = self._reserve([Span(int(low), int(high - low)) for low, high in zip(lows, highs)])
        regions = list(dict.fromkeys(holders))
        return self._queue(
            is_put,
            stem=rows.stem,
            names=rows.names,
            hex_names=True,
            starts=rows.starts,
            lengths=span_bytes[None],  # one row of lengths: every row alike
            cuts=self._cuts(regions),
            row_bytes=np.broadcast_to(span_bytes.sum(), len(rows.names)),
            regions=regions,
        )

    def _queue_row_ios(self, is_put: bool, rows: Sequence[RowIO]) -> _NativeOp:
        """The RowIO path: rows admitted as ``MooncakeDriver`` admits them and named by their raw
        keys; the worker cuts their spans at registration pieces, as on the array path."""
        batch = self._admit(rows)
        names = np.array([key.encode() for key in batch.keys], dtype=bytes).view(np.uint8)
        starts = np.zeros((len(rows), max(len(row.spans) for row in rows)), dtype=np.int64)
        lengths = np.zeros_like(starts)
        for i, row in enumerate(rows):
            starts[i, : len(row.spans)] = [span.address for span in row.spans]
            lengths[i, : len(row.spans)] = [span.nbytes for span in row.spans]
        return self._queue(
            is_put,
            stem="",
            names=names.reshape(len(rows), -1),
            hex_names=False,
            starts=starts,
            lengths=lengths,
            cuts=self._cuts(batch.regions),
            row_bytes=np.array(batch.sizes, dtype=np.int64),
            regions=batch.regions,
        )

    def _queue(
        self,
        is_put: bool,
        *,
        stem: str,
        names: np.ndarray,
        hex_names: bool,
        starts: np.ndarray,
        lengths: np.ndarray,
        cuts: np.ndarray,
        row_bytes: np.ndarray,
        regions: list[_Registration],
    ) -> _NativeOp:
        submit = self._worker.put if is_put else self._worker.get
        with self._lock:  # the op is known before the worker can finish it
            try:
                worker_op = submit(stem, names, hex_names, starts, lengths, cuts)
            except (ValueError, RuntimeError) as error:  # nothing was queued
                self._release(regions, ended=True, skip_reasons=())
                raise DriverRejected(str(error)) from None
            op = _NativeOp(
                owner=self,
                worker_op=worker_op,
                is_put=is_put,
                stem=stem,
                names=names,
                hex_names=hex_names,
                row_bytes=row_bytes,
                regions=regions,
                submitted=time.monotonic(),
            )
            self._open[worker_op] = op
        return op

    def _take(self) -> None:
        """Records the operations the worker finished since the last take; logs as
        ``MooncakeDriver`` does, outside the lock."""
        with self._lock:
            finished = self._worker.take_finished()
            if not finished:
                return
            ops = [self._conclude_finished(done) for done in finished]
        for op in ops:
            _log_finished(op)

    def _conclude_finished(self, done: Any) -> _NativeOp:
        """Under the lock: gives a finished worker op's handle its results and releases its
        regions. Only rows neither READY, MISS nor CANCELLED cost Python per row."""
        op = self._open.pop(done.op)
        errors, left_open = done.errors, done.sessions_left_open  # each read converts anew
        faults, skip_reasons = {}, []
        for row, state, code, call in done.other_rows:
            status, call = _ROW_STATES[state], _StoreCall(call)
            op.other_statuses[row] = status
            if status is RowStatus.FAULT:
                faults[row] = why = _fault_text(call, code, int(op.row_bytes[row]), errors)
            else:  # NOT_READY or DECLINED
                reason = _skip_reason(call, code)
                skip_reasons.append(reason)
                if reason in _QUIET_SKIPS:
                    op.quiet_skips += 1
                    op.store_full = op.store_full or reason == _POOL_FULL
                    continue
                why = f"{status.value} ({reason})"
            op.first_failure = op.first_failure or f"{op.key(row)}: {why}"
        op.cancelled = done.cancelled
        op.call_errors = [error for call, error in errors.items() if call in _METADATA_CALLS]
        op.sessions_left_open = left_open
        op.finished = FinishedOp(op, done.ready, faults, done.access_ended)
        self._release(op.regions, done.access_ended, skip_reasons)
        self._sessions_left_open.update(left_open)
        self._untaken.append(op)
        return op

    def _cuts(self, regions: list[_Registration]) -> np.ndarray:
        """Where the registration pieces of ``regions`` begin inside them, sorted: the worker cuts
        a span there, since each transfer slice must lie in one registration."""
        key = tuple(regions)
        cuts = self._cuts_by_regions.get(key)
        if cuts is None:
            inner = sorted(piece.address for region in regions for piece in region.pieces[1:])
            cuts = self._cuts_by_regions[key] = np.array(inner, dtype=np.int64)
        return cuts

    def _op(self, op: _NativeOp) -> _NativeOp:
        self._require_open()
        if not isinstance(op, _NativeOp) or op.owner is not self:
            raise ValueError("operation does not belong to this driver")
        return op


def _log_finished(op: _NativeOp) -> None:
    for error in op.call_errors:
        logger.warning(f"mooncake: {error}")
    if op.sessions_left_open:
        left_open = op.sessions_left_open
        logger.warning(f"mooncake: {len(left_open)} put sessions left open, first {left_open[0]}")
    if op.store_full:
        _warn_store_full()
    _report(
        "put" if op.is_put else "get",
        rows=len(op.names),
        failed=len(op.other_statuses) - op.quiet_skips,
        first=op.first_failure,
        ended=op.finished.access_ended,
        seconds=time.monotonic() - op.submitted,
    )


def _fault_text(call: _StoreCall, code: int, size: int, errors: Mapping[int, str]) -> str:
    """The text ``MooncakeDriver`` gives a row that ``call`` answered ``code``."""
    if code == _Code.CALL_FAILED and call in _DATA_CALLS:  # the call raised or miscounted
        return errors[call]
    if call is _StoreCall.REREAD and code in (_Code.OBJECT_NOT_FOUND, _Code.REPLICA_IS_NOT_READY):
        return _reread_fault(code)
    if call in (_StoreCall.GET, _StoreCall.REREAD):
        return _get_fault(code, size)
    if call is _StoreCall.PUT_START:
        return _put_start_fault(code)
    if call is _StoreCall.PUT_RANGES:
        return _put_ranges_fault(code, size)
    if call is _StoreCall.PUT_END:
        return _put_end_fault(code)
    if call is _StoreCall.EXISTENCE:
        return _existence_fault(code)
    raise AssertionError(f"no row faults on {call!r}")  # NONE and REVOKE never fault a row


def _skip_reason(call: _StoreCall, code: int) -> str:
    """Why a NOT_READY or DECLINED row was skipped, as ``MooncakeDriver`` names it."""
    if call is _StoreCall.PUT_START:
        return _PUT_START_SKIPS[code]
    if call is _StoreCall.EXISTENCE:
        return _ANOTHER_WRITER
    if call is _StoreCall.GET:  # REPLICA_IS_NOT_READY
        return "being_written"
    raise AssertionError(f"no row skips on {call!r}")

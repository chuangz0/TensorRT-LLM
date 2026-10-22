# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A store driver over Mooncake's distributed store, pinned to SDK 0.3.13.

Each row is one object holding exactly the row's payload, written from and read into the lender's
registered host slots without a copy. The driver is synchronous: ``put`` and ``get`` return only
after every SDK call they make has returned. Every SDK return code is translated here and nowhere
else.

Put: ``batch_put_session_start`` per key (the master reserves the object), one
``batch_put_from_multi_buffer_ranges`` scatter for all rows, then ``batch_put_session_end`` for
the rows written whole (their objects become readable) and ``batch_put_session_revoke`` for the
rest. Put uses sessions because this scatter has no timeout, so a failed row is revoked only once
nothing writes into its space; ``batch_put_from`` revokes a timed-out transfer at once, while its
RDMA writes may still land in space the master has given to another object.

Get: one ``batch_get_into``. A get session would add two SDK calls, and every call waits for the
GIL on return, which a busy executor thread makes cost milliseconds.

When access ends. The put scatter waits, without a timeout, until every fragment it posted has
physically finished, so its return ends access whatever it answers, TRANSFER_FAIL included.
``batch_get_into`` waits on one transfer future per key, and a future can give up with
TRANSFER_FAIL (after 60 s, on a status error, after a partly posted RDMA batch) while its
transfer still runs; every other answer comes after all futures finished or from a step that
posted nothing. So a get ends access only when no answer is TRANSFER_FAIL. In both, a raise or a
wrong answer count leaves access unproven. Session, existence and end calls touch only master and
local metadata, never row memory. This holds for a wheel built without ``USE_TENT`` (the newer
transfer engine), the default replicate config (one memory replica, no NoF SSD replica) and RDMA;
the TCP transport's scatter completion was not checked line by line.

An operation whose access has not ended when its call returns never will by anything the driver
can wait on: it is ``access_unprovable``, and it keeps its regions from being unregistered and the
driver from draining.
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

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

__all__ = ["MooncakeDriver"]


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
    CALL_FAILED = -100_000  # not an SDK code: a metadata call raised or miscounted (logged)


OBJECT_EXISTS, OBJECT_ABSENT = 1, 0  # batch_is_exist answers
MAX_ROW_BYTES = (1 << 31) - 1  # the put scatter answers each row's bytes as a 32-bit int
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
    device_memory=False,  # registers host memory only: the lender stages rows there
    # One span per row: batch_get_into reads each object into one buffer; a get of several
    # spans would need batch_get_into_multi_buffers.
    multi_span=False,
)
# The results that carry no fault text, one shared object each.
_PENDING = RowResult(RowStatus.PENDING)
_READY = RowResult(RowStatus.READY)
_MISS = RowResult(RowStatus.MISS)
_NOT_READY = RowResult(RowStatus.NOT_READY)
_DECLINED = RowResult(RowStatus.DECLINED)

# Why a row was skipped (nothing moved and nothing went wrong), and the result it gets; ``stats()``
# counts skipped rows by both.
_SKIPS = {
    # get -703: no complete replica, as while the object is written or once its replica's
    # endpoint went stale
    "being_written": _NOT_READY,
    # put start -600: this client already holds a put session for the key, from a concurrent
    # put of it or one left open (which keeps the key NOT_READY here until ``drain``), or the
    # master refused the row with the same -600 (a cachelib master caps an object at
    # 16 MiB - 16 B); the master's log says which
    "own_session_open": _NOT_READY,
    "uncommitted": _NOT_READY,  # put start -705, and the master shows the object uncommitted
    "pool_full": _DECLINED,  # put start -200: no space for the object
}
_PUT_START_SKIPS = {
    _Code.NO_AVAILABLE_HANDLE: "pool_full",
    _Code.INVALID_PARAMS: "own_session_open",
}


@dataclass(eq=False)
class _Registration:
    span: Span
    live: bool = True


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
    addresses: list[int]
    sizes: list[int]
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

    def pick(self, indices: Sequence[int]) -> tuple[list[str], list[int], list[int]]:
        """The keys, addresses and sizes of the rows at ``indices``."""
        return (
            [self.keys[i] for i in indices],
            [self.addresses[i] for i in indices],
            [self.sizes[i] for i in indices],
        )


class _WrongAnswerCount(Exception):
    """The SDK answered for a different number of keys than it was given; its state is unknown."""


class MooncakeDriver:
    """A ``StoreDriver`` over one set-up ``mooncake.store.MooncakeDistributedStore``.

    The SDK names a put session by its key, per client, not per caller, so the driver ends or
    revokes only the put sessions it started. One it cannot end or revoke is left open, and only
    ``drain`` retries it, once no put runs.

    Args:
        store: An SDK store whose ``setup`` returned 0.
    """

    def __init__(self, store: Any) -> None:
        self._store = store
        # Guards the fields below and wakes ``drain``; never held across an SDK call.
        self._lock = threading.Condition()
        self._registrations: list[_Registration] = []
        self._draining = False
        self._closed = False
        self._sessions_left_open: set[str] = set()
        self._in_flight = 0  # put/get calls past admission and not yet returned
        self._unprovable = 0  # operations whose access never ended
        self._busy: Counter[_Registration] = Counter()  # operations whose access may touch it
        self._skipped_rows: Counter[str] = Counter()  # by reason

    @property
    def capabilities(self) -> DriverCapabilities:
        return _CAPABILITIES

    def register(self, address: int, nbytes: int, layout: RegionLayout) -> _Registration:
        try:
            span = Span(address, nbytes)
        except ValueError as error:
            raise DriverRejected(str(error)) from None
        if layout.nbytes > nbytes:
            raise DriverRejected(f"layout needs {layout.nbytes} bytes, region has {nbytes}")
        if layout.memory_kind is not MemoryKind.HOST:
            raise DriverRejected("this driver registers host memory only")
        with self._lock:
            self._require_open(DriverRejected)
            overlapping = [r.span for r in self._registrations if r.live and r.span.overlaps(span)]
        if overlapping:
            raise DriverRejected(f"region {span} overlaps a live registration {overlapping[0]}")
        registration = _Registration(span)
        try:
            code = self._store.register_buffer(address, nbytes)
        except Exception as error:
            self._add(registration)
            raise RegistrationError(f"register_buffer raised {error!r}", registration) from error
        if code != _Code.OK:  # the SDK rolled its partial registration back
            raise RegistrationError(f"register_buffer returned {_describe(code)}")
        self._add(registration)
        return registration

    def unregister(self, handle: _Registration) -> None:
        with self._lock:
            registration = self._registration(handle)
            if not registration.live:
                return
            if self._busy[registration]:
                raise DriverRejected(
                    f"{self._busy[registration]} operation(s) may still access the region"
                )
            registration.live = False  # admits no new rows while the SDK unregisters it
        try:
            code = self._store.unregister_buffer(registration.span.address)
        except Exception as error:
            failure = StoreError(f"unregister_buffer raised {error!r}")
        else:
            if code == _Code.OK:
                return
            failure = StoreError(f"unregister_buffer returned {_describe(code)}")
        with self._lock:
            registration.live = True  # still registered: the call may be retried
        raise failure

    def put(self, rows: Sequence[RowIO]) -> _Op:
        batch, started = self._admit(rows), time.monotonic()  # the only raise is in _admit
        try:
            ended = self._put_rows(batch)
        except Exception as error:  # a bug, not an SDK answer: a transfer may still be running
            batch.fault(range(len(batch.keys)), f"put failed: {error!r}")
            ended = False
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
        False at once while an operation is unprovable. When it returns True it also revokes the
        put sessions left open, once: no put runs any more, so none is in use. They hold no lender
        memory, so they never change the answer, and one whose revocation fails stays open until
        the master discards it."""
        with self._lock:
            self._require_open()
            self._draining = True
            self._lock.wait_for(lambda: self._unprovable or not self._in_flight, budget_s)
            quiet = not (self._in_flight or self._unprovable)
            left_open = sorted(self._sessions_left_open) if quiet else []
        if left_open:
            codes = self._metadata_call(self._store.batch_put_session_revoke, left_open)
            with self._lock:
                gone = (k for k, code in zip(left_open, codes) if _gone(code))
                self._sessions_left_open.difference_update(gone)
        return quiet

    def close(self) -> None:
        """Closes the SDK client once a drain returned True and every region is unregistered.
        Never called in production: it holds the GIL through unmount RPCs; process exit frees it."""
        with self._lock:
            if self._closed:
                return
            if not self._draining or self._in_flight or self._unprovable:
                raise DriverRejected("close needs a drain that returned True")
            if any(r.live for r in self._registrations):
                raise DriverRejected("close needs every registration unregistered")
            self._closed = True
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
                "put_sessions_left_open": len(self._sessions_left_open),
            }

    # Put: start, one scatter, end what was written whole, revoke the rest.

    def _put_rows(self, batch: _Batch) -> bool:
        """Returns whether access ended."""
        starts = self._metadata_call(self._store.batch_put_session_start, batch.keys, batch.sizes)
        started, existing = [], []
        for row, code in enumerate(starts):
            if code == _Code.OK:
                started.append(row)
            elif code == _Code.OBJECT_ALREADY_EXISTS:
                existing.append(row)
            elif code in _PUT_START_SKIPS:
                batch.skip(row, _PUT_START_SKIPS[code])
            else:
                batch.fault([row], f"put session start: {_describe(code)}")
        ended = self._write_and_finish_sessions(batch, started) if started else True
        if existing:
            self._confirm_existing(batch, existing)
        return ended

    def _write_and_finish_sessions(self, batch: _Batch, started: list[int]) -> bool:
        """Writes the rows whose put session started in one scatter, then ends the session of each
        row written whole and revokes the rest; a session it can neither end nor revoke is left
        open. Returns whether access ended."""
        keys, addresses, sizes = batch.pick(started)
        # One range per row: the whole payload, written at offset 0 of its object.
        buffers, lengths = [[a] for a in addresses], [[n] for n in sizes]
        dst_offsets = [[0]] * len(keys)
        try:
            scatter = self._store.batch_put_from_multi_buffer_ranges
            answers = _answers(scatter(keys, buffers, lengths, dst_offsets), len(keys))
        except Exception as error:  # the scatter may still be reading the slots
            batch.fault(started, f"batch_put_from_multi_buffer_ranges raised {error!r}")
            # Neither end (the object may be torn) nor revoke (the master would give its space
            # to another object while the scatter may still write into it).
            self._leave_open(keys)
            return False
        written, unwritten = started, []
        if answers != sizes:  # rare: some row was not written whole
            written = []
            for i, answer, size in zip(started, answers, sizes):
                if answer == size:
                    written.append(i)
                else:  # its object must never be marked complete
                    unwritten.append(i)
                    batch.fault([i], f"put ranges: {_moved(answer, size)}")
        not_ended = self._end_put_sessions(batch, written)
        self._revoke_put_sessions([batch.keys[i] for i in unwritten + not_ended])
        return True

    def _end_put_sessions(self, batch: _Batch, written: list[int]) -> list[int]:
        """Ends the sessions of rows written whole; returns the rows whose session did not end."""
        if not written:
            return []
        end = self._store.batch_put_session_end
        keys = [batch.keys[i] for i in written]
        codes = self._metadata_call(end, keys)
        retry = [
            key for key, code in zip(keys, codes) if code in (_Code.RPC_FAIL, _Code.RPC_TIMEOUT)
        ]
        if retry:  # ending is idempotent on the master
            again = dict(zip(retry, self._metadata_call(end, retry)))
            codes = [again.get(key, code) for key, code in zip(keys, codes)]
        for i, code in zip(written, codes):
            batch.results[i] = (
                _READY if code == _Code.OK else faulted(f"put session end: {_describe(code)}")
            )
        return [i for i, code in zip(written, codes) if code != _Code.OK]

    def _revoke_put_sessions(self, keys: list[str]) -> None:
        if keys:
            codes = self._metadata_call(self._store.batch_put_session_revoke, keys)
            self._leave_open([key for key, code in zip(keys, codes) if not _gone(code)])

    def _leave_open(self, keys: list[str]) -> None:
        if keys:
            logger.warning(f"mooncake: {len(keys)} put sessions left open, first {keys[0]}")
            with self._lock:
                self._sessions_left_open.update(keys)

    def _confirm_existing(self, batch: _Batch, existing: list[int]) -> None:
        """A key that already exists is readable only once the master shows it committed."""
        keys = [batch.keys[i] for i in existing]
        for i, answer in zip(existing, self._metadata_call(self._store.batch_is_exist, keys)):
            if answer == OBJECT_EXISTS:
                batch.results[i] = _READY
            elif answer == OBJECT_ABSENT:
                batch.skip(i, "uncommitted")
            else:
                batch.fault([i], f"existence check: {_describe(answer)}")

    # Get: one batch_get_into; rows whose read lease expired mid-read are read once more.

    def _get_rows(self, batch: _Batch, indices: list[int], may_reread: bool) -> bool:
        """Returns whether access ended."""
        keys, addresses, sizes = batch.pick(indices)
        try:
            answers = _answers(self._store.batch_get_into(keys, addresses, sizes), len(indices))
        except Exception as error:  # the transfers may still be writing the slots
            batch.fault(indices, f"batch_get_into raised {error!r}")
            return False
        if answers == sizes:  # usually every row was read whole
            for i in indices:
                batch.results[i] = _READY
            return True
        for i, answer in zip(indices, answers):
            if not may_reread and answer in (_Code.OBJECT_NOT_FOUND, _Code.REPLICA_IS_NOT_READY):
                # A reread: the first read already wrote this slot, and MISS or NOT_READY would
                # claim that nothing landed there.
                batch.results[i] = faulted(f"reread after LEASE_EXPIRED: {_describe(answer)}")
            elif answer == _Code.OBJECT_NOT_FOUND:
                batch.results[i] = _MISS
            elif answer == _Code.REPLICA_IS_NOT_READY:
                batch.skip(i, "being_written")
            else:
                batch.results[i] = _get_result(answer, batch.sizes[i])
        # A TRANSFER_FAIL future may give up while its RDMA still runs (module docstring).
        ended = _Code.TRANSFER_FAIL not in answers
        expired = [i for i, answer in zip(indices, answers) if answer == _Code.LEASE_EXPIRED]
        if expired and may_reread:
            # The read lease expired before the transfer finished, so the master may have
            # reused the object's space mid-read: read once more. The reread is READY only if it
            # reads the whole object again; anything else, a second expiry included, is a FAULT.
            reread_ended = self._get_rows(batch, expired, may_reread=False)
            ended = ended and reread_ended  # a TRANSFER_FAIL of the first read stays
        return ended

    # Bookkeeping.

    def _admit(self, rows: Sequence[RowIO]) -> _Batch:
        """Checks ``rows`` and records the operation, or raises ``DriverRejected`` having done
        neither."""
        if not rows:
            raise DriverRejected("an operation needs at least one row")
        keys = [row.key for row in rows]
        # The SDK keys put sessions and get targets by name: a repeat would overwrite the other.
        if len(set(keys)) != len(keys):
            raise DriverRejected("a batch repeats a key")
        if any(len(row.spans) != 1 for row in rows):
            raise DriverRejected("this driver takes rows of one span")
        payloads = [row.spans[0] for row in rows]
        addresses = [payload.address for payload in payloads]
        sizes = [payload.nbytes for payload in payloads]
        if max(sizes) > MAX_ROW_BYTES:
            payload = next(payload for payload in payloads if payload.nbytes > MAX_ROW_BYTES)
            raise DriverRejected(f"a row holds at most {MAX_ROW_BYTES} bytes, got {payload}")
        low, high = min(addresses), max(a + n for a, n in zip(addresses, sizes))
        with self._lock:
            self._require_open(DriverRejected)
            if self._draining:
                raise DriverRejected("the driver is draining")
            live = [r for r in self._registrations if r.live]
            # Usually one registration holds every row: one check instead of one per row.
            covering = next((r for r in live if r.span.address <= low and high <= r.span.end), None)
            regions = [covering] if covering is not None else _regions_of(payloads, live)
            self._in_flight += 1
            self._busy.update(regions)
        return _Batch(keys, addresses, sizes, regions, [_PENDING] * len(rows), {})

    def _metadata_call(self, call: Callable, keys: list[str], *args) -> list[int]:
        """A session or existence call. It touches no row memory, so a raise or a wrong answer
        count fails only its keys."""
        try:
            return _answers(call(keys, *args), len(keys))
        except Exception as error:
            logger.warning(f"mooncake: {getattr(call, '__name__', call)} failed: {error!r}")
            return [_Code.CALL_FAILED] * len(keys)

    def _conclude(self, direction: str, batch: _Batch, ended: bool, seconds: float) -> _Op:
        with self._lock:
            self._skipped_rows.update(batch.skip_reasons.values())
            self._in_flight -= 1
            if ended:
                self._busy.subtract(batch.regions)
            else:  # holds its regions for good
                self._unprovable += 1
            self._lock.notify_all()
        _report(direction, batch, ended, seconds)
        return _Op(tuple(batch.results), ended, self)

    def _add(self, registration: _Registration) -> None:
        with self._lock:
            self._registrations.append(registration)

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


def _regions_of(payloads: Sequence[Span], live: list[_Registration]) -> list[_Registration]:
    """The registrations ``payloads`` lie in, found row by row."""
    regions: list[_Registration] = []
    for payload in payloads:
        region = next((r for r in live if payload.within(r.span)), None)
        if region is None:
            raise DriverRejected(f"{payload} is outside every live registration")
        if region not in regions:
            regions.append(region)
    return regions


def _get_result(answer: int, size: int) -> RowResult:
    """A read answers the object's size, which must be the row's."""
    if answer == size:
        return _READY
    if answer >= 0:
        return faulted(f"object is {answer} bytes, expected {size}")
    return faulted(f"batch_get_into: {_describe(answer)}")


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


def _report(direction: str, batch: _Batch, ended: bool, seconds: float) -> None:
    """One warning per call that failed or skipped a row or left access open; none otherwise."""
    expected = (RowStatus.READY, RowStatus.MISS)
    failed_or_skipped = [i for i, r in enumerate(batch.results) if r.status not in expected]
    if ended and not failed_or_skipped:
        return
    first = ""
    if failed_or_skipped:
        row = failed_or_skipped[0]
        result = batch.results[row]
        why = result.fault or f"{result.status.value} ({batch.skip_reasons[row]})"
        first = f"; first {batch.keys[row]}: {why}"
    access = "ended" if ended else "may still be in progress"
    logger.warning(
        f"mooncake {direction}: {len(failed_or_skipped)} of {len(batch.keys)} rows failed or "
        f"skipped in {seconds:.3f} s{first}; access {access}"
    )

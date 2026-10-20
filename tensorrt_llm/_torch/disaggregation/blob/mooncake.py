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

When access ends, as read in the Mooncake sources at tag v0.3.13 (paths below are in that tree), for
a wheel built without ``USE_TENT``, as the release wheels are (``mooncake-common/common.cmake:174``,
``.github/workflows/_build-wheel.yaml:67-107``). They also assume settings that whoever opens the
SDK must refuse: ``MC_USE_TENT`` and ``MC_USE_TEV1``, which switch to the newer engine at run time
(``mooncake-transfer-engine/src/transfer_engine.cpp:393-410``); the SDK's local hot cache, which
could serve a torn read later; and TCP's legacy framing, which reports a write before it lands and
is also used toward a process of the pool that runs an SDK older than 0.3.12. The default
replicate config is assumed: one memory replica, no NoF or SSD replica.

- The put scatter waits, without a timeout, until every task it published is terminal, also when
  submitting failed part way (``transfer_engine.cpp:987-993, 1236-1265, 1315-1319``). An RDMA task
  is FAILED only once all its slices ended (``mooncake-transfer-engine/src/transport/
  rdma_transport/rdma_transport.cpp:1023-1048``), and a TCP v2 write ends, either way, only after
  the body write released the caller's buffer (``mooncake-transfer-engine/src/transport/
  tcp_transport/tcp_transport_session_impl.h:818-905``). So the
  scatter's return ends access to the slots whatever it answers, TRANSFER_FAIL included.
- Over TCP a TRANSFER_FAIL may also mean the initiator gave up on a slow target: no
  acknowledgment within ``MC_TCP_STATUS_TIMEOUT_SEC`` (30 s by default;
  ``tcp_transport_session_impl.h:394-401, 447-470``) or no progress within
  ``MC_TCP_PROGRESS_TIMEOUT_SEC`` (``:404-414, 496-526``), while the target may still apply data
  it received into the object's space (``:838-840``). A revoke frees that space at once
  (``mooncake-store/src/master_service.cpp:4776, 4869``), so such a row's session is never revoked
  by the driver; the master frees a put left unfinished only after
  ``put_start_release_timeout_sec`` (600 s by default; ``master_service.cpp:4441-4450``).
- ``batch_get_into`` waits on one transfer future per key, and a future can give up with
  TRANSFER_FAIL (after 60 s, on a status error, after a partly posted RDMA batch;
  ``mooncake-store/src/transfer_task.cpp:746-905, 1239-1258``) while its transfer still runs;
  every other answer comes after all futures finished or from a step that posted nothing. So a get
  ends access only when no answer is TRANSFER_FAIL.
- In both, a raise or a wrong answer count leaves access unproven. Session, existence and end
  calls touch only master and local metadata, never row memory.

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
    CHECKSUM_MISMATCH = -801  # with MOONCAKE_STORE_CHECKSUM: the bytes of a get landed, then failed
    RPC_FAIL = -900
    RPC_TIMEOUT = -901
    CALL_FAILED = -100_000  # not an SDK code: a metadata call raised or miscounted (logged)


OBJECT_EXISTS, OBJECT_ABSENT = 1, 0  # batch_is_exist answers
MAX_ROW_BYTES = (1 << 31) - 1  # the put scatter answers each row's bytes as a 32-bit int
_CAPABILITIES = DriverCapabilities(
    get=True,
    put=True,
    cancel=False,
    probe=False,
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
    # put: this driver holds the key's put session, from another put of it in progress, or one it
    # left open while the object does not exist complete; no session is started
    "own_session_open": _NOT_READY,
    "uncommitted": _NOT_READY,  # put start -705, and the master shows the object uncommitted
    "pool_full": _DECLINED,  # put start -200: no space for the object
}
_PUT_START_SKIPS = {_Code.NO_AVAILABLE_HANDLE: "pool_full"}
# Answers to a session end after which the end may still reach the master, even when a retry is then
# refused. A late end would mark complete this client's next object of the key, since the master
# matches only key and client, so the session stays open and the driver never puts the key again.
_END_OUTCOME_UNKNOWN = (_Code.RPC_FAIL, _Code.RPC_TIMEOUT, _Code.CALL_FAILED)
# Retried once: the master ends a session idempotently. A call that raised or miscounted is not.
_END_RETRIED = (_Code.RPC_FAIL, _Code.RPC_TIMEOUT)


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
class _CallRows:
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
    """A ``StoreDriver`` over one set-up ``mooncake.store.MooncakeDistributedStore``. It must be the
    only driver over its SDK client: the SDK names a put session by its key, per client.

    A put of a batch of rows:

    1. Reserves each key that no put of this driver is writing and no session it left open holds.
       A key another put is writing is NOT_READY at once. A key whose session it left open is
       READY when the master shows its object complete and NOT_READY otherwise; it is never put
       again.
    2. Starts a session per reserved key, writes every started row in one scatter, ends the
       session of each row written whole (its object becomes readable) and revokes the rest.
    3. Leaves a session open when that went wrong:

       - for ``drain`` to revoke once no call runs: a start that raised or miscounted, an end whose
         outcome is unknown (``_END_OUTCOME_UNKNOWN``), a revoke that failed, a put that raised;
       - to the master, never revoked here, since the target may still write into the object: a
         scatter that raised or miscounted, a row answered TRANSFER_FAIL. The master frees them
         after ``put_start_release_timeout_sec``.

    Args:
        store: An SDK store whose ``setup`` returned 0.
    """

    def __init__(self, store: Any) -> None:
        self._store = store
        # Guards the fields below and wakes ``drain``; never held across an SDK call.
        self._lock = threading.Condition()
        self._registrations: list[_Registration] = []
        self._draining = False  # set by the first drain, whatever it returns
        self._drained = False  # a drain returned True: no call runs and none is unprovable
        self._closed = False
        self._keys_being_put: set[str] = set()  # keys of the puts in progress
        # The keys of the put sessions left open (class docstring, step 3); the two are disjoint.
        self._sessions_for_drain: set[str] = set()
        self._sessions_for_master: set[str] = set()
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
        if code != _Code.OK:  # the SDK rolled back what it registered, as best it could
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
        batch, start_s = self._admit(rows), time.monotonic()  # the only raise is in _admit
        try:
            ended = self._put_rows(batch)
        except Exception as error:  # a bug, not an SDK answer: a transfer may still be running
            batch.fault(range(len(batch.keys)), f"put failed: {error!r}")
            ended = False
        return self._conclude("put", batch, ended, time.monotonic() - start_s)

    def get(self, rows: Sequence[RowIO], hint: Hint = None) -> _Op:
        if hint is not None:
            raise DriverRejected("this driver reads no hints")
        batch, start_s = self._admit(rows), time.monotonic()  # the only raise is in _admit
        try:
            ended = self._get_rows(batch, list(range(len(batch.keys))), is_reread=False)
        except Exception as error:  # a bug, not an SDK answer: a transfer may still be running
            batch.fault(range(len(batch.keys)), f"get failed: {error!r}")
            ended = False
        return self._conclude("get", batch, ended, time.monotonic() - start_s)

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
        """Never called: ``capabilities.probe`` is false."""
        self._require_open()
        return tuple(ProbeStatus.UNKNOWN for _ in keys)

    def drain(self, budget_s: float) -> bool:
        """Refuses new work for good and waits up to ``budget_s`` for the calls in progress;
        False at once while an operation is unprovable. Each call that returns True also tries
        once to revoke the sessions left for drain: no put runs any more, so none is in use. They
        hold no lender memory, so they never change the answer; one whose revocation fails stays
        for the next drain, or until the master discards it. Sessions left to the master are never
        revoked here."""
        with self._lock:
            self._require_open()
            self._draining = True
            self._lock.wait_for(lambda: self._unprovable or not self._in_flight, budget_s)
            quiet = not (self._in_flight or self._unprovable)
            if quiet:
                self._drained = True
            to_revoke = sorted(self._sessions_for_drain) if quiet else []
        if to_revoke:
            codes = self._metadata_call(self._store.batch_put_session_revoke, to_revoke)
            with self._lock:
                gone = (key for key, code in zip(to_revoke, codes) if _gone(code))
                self._sessions_for_drain.difference_update(gone)
        return quiet

    def close(self) -> None:
        """Closes the SDK client once a drain returned True and every region is unregistered.
        Never called in production: it holds the GIL through unmount RPCs; process exit frees it."""
        with self._lock:
            if self._closed:
                return
            if not self._drained:
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
                "put_sessions_left_for_drain": len(self._sessions_for_drain),
                "put_sessions_left_to_master": len(self._sessions_for_master),
            }

    # Put: start, one scatter, end what was written whole, revoke the rest.

    def _put_rows(self, batch: _CallRows) -> bool:
        """Returns whether access ended."""
        reserved, held_by_left_session = self._reserve_keys(batch)
        try:
            if held_by_left_session:  # never put again, but another client may have written it
                self._confirm_existing(batch, held_by_left_session, absent="own_session_open")
            return self._put_reserved_rows(batch, reserved) if reserved else True
        except Exception:  # any of the sessions this put started may be open
            self._leave_for_drain([batch.keys[i] for i in reserved])
            raise
        finally:
            with self._lock:
                self._keys_being_put.difference_update(batch.keys[i] for i in reserved)

    def _reserve_keys(self, batch: _CallRows) -> tuple[list[int], list[int]]:
        """Reserves every key that no put of this driver is writing and no session it left open
        holds, and returns their rows, then the rows whose key a left-open session holds. Skips as
        NOT_READY the rows whose key another put of this driver is writing."""
        reserved, held_by_left_session, being_put = [], [], []
        with self._lock:
            for row, key in enumerate(batch.keys):
                if key in self._keys_being_put:
                    being_put.append(row)
                elif key in self._sessions_for_drain or key in self._sessions_for_master:
                    held_by_left_session.append(row)
                else:
                    reserved.append(row)
            self._keys_being_put.update(batch.keys[row] for row in reserved)
        for row in being_put:
            batch.skip(row, "own_session_open")
        return reserved, held_by_left_session

    def _put_reserved_rows(self, batch: _CallRows, rows: list[int]) -> bool:
        keys, _, sizes = batch.pick(rows)
        starts = self._metadata_call(self._store.batch_put_session_start, keys, sizes)
        if _Code.CALL_FAILED in starts:  # the SDK may have opened any of these sessions
            self._leave_for_drain(keys)
        started, existing = [], []
        for row, code in zip(rows, starts):
            if code == _Code.OK:
                started.append(row)
            elif code == _Code.OBJECT_ALREADY_EXISTS:
                existing.append(row)
            elif code in _PUT_START_SKIPS:
                batch.skip(row, _PUT_START_SKIPS[code])
            else:
                # Any other code faults the row. That includes -600: this driver holds no session
                # of the key, so it is something else, such as a cachelib master's object cap
                # (16 MiB - 16 B) or a replicate config the session path refuses (the driver
                # passes none); the master's log says which.
                batch.fault([row], f"put session start: {_describe(code)}")
        ended = self._write_and_finish_sessions(batch, started) if started else True
        if existing:  # readable only once the master shows it committed
            self._confirm_existing(batch, existing, absent="uncommitted")
        return ended

    def _write_and_finish_sessions(self, batch: _CallRows, started: list[int]) -> bool:
        """Writes the rows whose put session started in one scatter, then ends the session of each
        row written whole and revokes the rest, leaving open those the class docstring lists (step
        3). Returns whether access ended."""
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
            self._leave_to_master(keys)
            return False
        written, revoke_now, still_written_to = started, [], []
        if answers != sizes:  # rare: some row was not written whole
            written = []
            for i, answer, size in zip(started, answers, sizes):
                if answer == size:
                    written.append(i)
                    continue
                # Its object must never be marked complete. After a TRANSFER_FAIL a TCP target
                # may still write into the object's space (module docstring).
                (still_written_to if answer == _Code.TRANSFER_FAIL else revoke_now).append(i)
                batch.fault([i], f"put ranges: {_moved(answer, size)}")
        end_refused, end_outcome_unknown = self._end_put_sessions(batch, written)
        self._revoke_put_sessions([batch.keys[i] for i in revoke_now + end_refused])
        self._leave_for_drain([batch.keys[i] for i in end_outcome_unknown])
        self._leave_to_master([batch.keys[i] for i in still_written_to])
        return True

    def _end_put_sessions(
        self, batch: _CallRows, written: list[int]
    ) -> tuple[list[int], list[int]]:
        """Ends the sessions of rows written whole. Returns the rows whose end was refused, then
        those whose end may still reach the master: answered ``_END_OUTCOME_UNKNOWN`` first, even
        when the retry was refused."""
        if not written:
            return [], []
        end = self._store.batch_put_session_end
        keys = [batch.keys[i] for i in written]
        codes = self._metadata_call(end, keys)
        outcome_unknown = {key for key, code in zip(keys, codes) if code in _END_OUTCOME_UNKNOWN}
        retry = [key for key, code in zip(keys, codes) if code in _END_RETRIED]
        if retry:
            again = dict(zip(retry, self._metadata_call(end, retry)))
            codes = [again.get(key, code) for key, code in zip(keys, codes)]
        end_refused, end_outcome_unknown = [], []
        for i, key, code in zip(written, keys, codes):
            if code == _Code.OK:
                batch.results[i] = _READY
                continue
            batch.results[i] = faulted(f"put session end: {_describe(code)}")
            (end_outcome_unknown if key in outcome_unknown else end_refused).append(i)
        return end_refused, end_outcome_unknown

    def _revoke_put_sessions(self, keys: list[str]) -> None:
        if keys:
            codes = self._metadata_call(self._store.batch_put_session_revoke, keys)
            self._leave_for_drain([key for key, code in zip(keys, codes) if not _gone(code)])

    def _leave_for_drain(self, keys: list[str]) -> None:
        """Leaves ``keys``' sessions open for ``drain`` to revoke once no call runs. A key already
        left to the master stays there."""
        if keys:
            logger.warning(f"mooncake: {len(keys)} put sessions left for drain, first {keys[0]}")
            with self._lock:
                self._sessions_for_drain.update(set(keys) - self._sessions_for_master)

    def _leave_to_master(self, keys: list[str]) -> None:
        """Leaves ``keys``' sessions open for good: the target may still write into their objects,
        so only the master frees them, after its own release timeout."""
        if keys:
            first = keys[0]
            logger.warning(f"mooncake: {len(keys)} put sessions left to the master, first {first}")
            with self._lock:
                self._sessions_for_drain.difference_update(keys)
                self._sessions_for_master.update(keys)

    def _confirm_existing(self, batch: _CallRows, rows: list[int], absent: str) -> None:
        """Concludes ``rows`` READY when the master shows their objects complete; skips the others
        as ``absent``, a reason in ``_SKIPS``."""
        keys = [batch.keys[i] for i in rows]
        for i, answer in zip(rows, self._metadata_call(self._store.batch_is_exist, keys)):
            if answer == OBJECT_EXISTS:
                batch.results[i] = _READY
            elif answer == OBJECT_ABSENT:
                batch.skip(i, absent)
            else:
                batch.fault([i], f"existence check: {_describe(answer)}")

    # Get: one batch_get_into; rows whose read lease expired mid-read are read once more.

    def _get_rows(self, batch: _CallRows, indices: list[int], is_reread: bool) -> bool:
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
            if is_reread and answer in (_Code.OBJECT_NOT_FOUND, _Code.REPLICA_IS_NOT_READY):
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
        if expired and not is_reread:
            # The read lease expired before the transfer finished, so the master may have
            # reused the object's space mid-read: read once more. The reread is READY only if it
            # reads the whole object again; anything else, a second expiry included, is a FAULT.
            reread_ended = self._get_rows(batch, expired, is_reread=True)
            ended = ended and reread_ended  # a TRANSFER_FAIL of the first read stays
        return ended

    # Bookkeeping.

    def _admit(self, rows: Sequence[RowIO]) -> _CallRows:
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
        return _CallRows(keys, addresses, sizes, regions, [_PENDING] * len(rows), {})

    def _metadata_call(self, call: Callable, keys: list[str], *args) -> list[int]:
        """A session or existence call. It touches no row memory, so a raise or a wrong answer
        count fails only its keys."""
        try:
            return _answers(call(keys, *args), len(keys))
        except Exception as error:
            logger.warning(f"mooncake: {getattr(call, '__name__', call)} failed: {error!r}")
            return [_Code.CALL_FAILED] * len(keys)

    def _conclude(self, direction: str, batch: _CallRows, ended: bool, seconds: float) -> _Op:
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


def _report(direction: str, batch: _CallRows, ended: bool, seconds: float) -> None:
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

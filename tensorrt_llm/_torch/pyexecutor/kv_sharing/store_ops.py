# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The table of open store operations and the one rule that closes them.

Every rank of a tensor-parallel group keeps the same ordered table: entries are appended only by
code that runs alike on every rank with the same inputs, whatever this rank's own lend or submit
did, and only a round removes them. Rank 0 announces how many entries a round covers; each rank
reports on its first open ones, every report padded to that length, and the round reduces the
reports (the minimum over ranks; the identity on one rank). Every rank turns the reduced report
into the same decisions with ``decide``, which reads nothing but its arguments.
"""

from __future__ import annotations

import enum
import hashlib
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Optional, Sequence

import numpy as np

from ..kv_cache.sharing import MemoryKind

if TYPE_CHECKING:
    from ..kv_cache.sharing import Lease
    from ..llm_request import LlmRequest

# The report header: ok, the table hash h and -h. Under the minimum over ranks, min(h) equals
# -min(-h) only where every rank sent the same h. Then three values per entry.
HEADER = 3
COLUMNS = 3
# A table hash fits a signed 64-bit integer with room to negate it.
_HASH_MASK = (1 << 62) - 1


class OpKind(enum.Enum):
    FETCH = 0
    PUBLISH = 1


class OpState(enum.Enum):
    """``PARKED``: a fetch whose request waits, its store access open. ``SETTLING``: a fetch
    whose rows went back to the lender, the copy into the KV cache queued. ``OPEN``: a lease
    still out that no request waits for: a publication, or a fetch whose request was dropped or
    cancelled."""

    PARKED = 0
    SETTLING = 1
    OPEN = 2


@dataclass(eq=False)
class StoreOp:
    """One open store operation. The fields up to ``end`` are alike on every rank; the rest
    are this rank's own."""

    seq: int
    kind: OpKind
    request: LlmRequest
    memory_kind: MemoryKind
    state: OpState
    # The prompt range [start, end): a fetch's local prefix and stored end, a publication's range.
    start: int
    end: int
    # The transfer manager's operation, or None where it took no lease.
    operation_id: Optional[int] = None
    # A lease the transfer manager did not take: refused by the lender or by the submit.
    lease: Optional[Lease] = None
    # A fetch is late on this rank from this monotonic time on.
    deadline: float = math.inf
    closed: bool = False

    @property
    def request_id(self) -> int:
        return self.request.py_request_id


class Action(enum.Enum):
    """``RETURN_ALL``: return the lease with every row arrived. ``RETURN_NONE``: return it with
    none. ``SETTLE``: wait for the rows' copy into the KV cache. ``RESUME``: the request computes
    on from the fetch's end. ``DROP``: the request computes its prompt from its local prefix.
    ``ABANDON``: stop waiting for the store; the lease goes back once its access ends.
    ``STOP_SHARING``: admit no new operation. ``FATAL``: end the process."""

    RETURN_ALL = "return_all"
    RETURN_NONE = "return_none"
    SETTLE = "settle"
    RESUME = "resume"
    DROP = "drop"
    ABANDON = "abandon"
    STOP_SHARING = "stop_sharing"
    FATAL = "fatal"


class Decision(NamedTuple):
    """An action on one entry, or on all. ``reason`` is a FATAL's message, or a DROP's cause:
    "late" (some rank's deadline passed), "short" (some rank lacks a row) or "stopped" (some
    rank's lender stopped)."""

    action: Action
    op: Optional[StoreOp] = None
    reason: str = ""


def table_hash(entries: Sequence[StoreOp]) -> int:
    """A hash of the fields every rank must agree on, stable across processes: the range too, as
    it sets the pages each entry pins."""
    fields = np.array(
        [
            (op.seq, op.kind.value, op.request_id, op.state.value, op.start, op.end)
            for op in entries
        ],
        dtype=np.int64,
    )
    digest = hashlib.blake2b(fields.tobytes(), digest_size=8).digest()
    return int.from_bytes(digest, "little") & _HASH_MASK


def report(entries: Sequence[StoreOp], announced: int, ok: bool, columns: np.ndarray) -> np.ndarray:
    """This rank's report for a round covering ``announced`` entries: ``[ok, h, -h]``, ``h`` the
    hash of ``entries`` (the first open ones, at most ``announced``: entries every rank closed
    alike since the announcement are gone), then ``columns`` (``[done, ready, on_time]`` per entry)
    padded with ones, which change no minimum, to ``announced`` rows. Every rank's report has the
    same length; a rank whose table differs reports another hash."""
    rows = np.ones((announced, COLUMNS), dtype=np.int64)
    rows[: len(entries)] = columns.reshape(-1, COLUMNS)
    digest = table_hash(entries)
    header = np.array([int(ok), digest, -digest], dtype=np.int64)
    return np.concatenate([header, rows.reshape(-1)])


def decide(entries: Sequence[StoreOp], reduced: np.ndarray) -> list[Decision]:
    """The decisions for ``entries`` from the reduced report. Every rank gets the same ones.

    A fetch resumes at its end only if every rank got every row on time; otherwise its request
    is dropped and computes from its local prefix. No lease goes back but by a decision.
    """
    ok, digest, negated = (int(value) for value in reduced[:HEADER])
    if digest != -negated:  # some rank's table differs
        return [Decision(Action.FATAL, reason="store operation tables differ across ranks")]
    if ok == 0 and any(op.memory_kind is MemoryKind.DEVICE for op in entries):
        return [
            Decision(
                Action.FATAL, reason="a lender or store binding stopped with GPU Direct leases out"
            )
        ]
    decisions = [] if ok else [Decision(Action.STOP_SHARING)]
    # Every rank stops waiting for a late fetch: some rank's lender stopped or deadline passed.
    late_reason = "late" if ok else "stopped"
    for op, (done, ready, on_time) in zip(entries, reduced[HEADER:].reshape(-1, COLUMNS)):
        late = not ok or not on_time
        if op.kind is OpKind.FETCH and op.state is OpState.PARKED:
            if done and ready and not late:
                after_return = (
                    Action.RESUME if op.memory_kind is MemoryKind.DEVICE else Action.SETTLE
                )
                decisions += [Decision(Action.RETURN_ALL, op), Decision(after_return, op)]
            elif done:
                drop = Decision(Action.DROP, op, late_reason if late else "short")
                decisions += [Decision(Action.RETURN_NONE, op), drop]
            elif late:
                decisions += [Decision(Action.ABANDON, op), Decision(Action.DROP, op, late_reason)]
        elif op.state is OpState.SETTLING and done:
            # A settling entry reports on_time = 1; a stopped lender vouches for no copy.
            decisions.append(
                Decision(Action.RESUME, op) if ok else Decision(Action.DROP, op, "stopped")
            )
        elif op.state is OpState.OPEN and done:
            decisions.append(Decision(Action.RETURN_NONE, op))
    return decisions

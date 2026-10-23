# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The blob backend: the public ``base.backend.Backend`` over a private store driver.

A submission is validated and cut into batches on the submitting thread, in the backend's call
sequence; worker threads hand the batches to the driver, and one progress thread watches the calls
whose results or access end were still open when they returned, and after shutdown drains the
driver and unregisters the regions. An attempt quiesces only once every batch it started has proven
its access ended.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple, NoReturn, Protocol

import numpy as np

from tensorrt_llm.logger import logger

from ..base.backend import (
    Capabilities,
    Hint,
    InvalidSubmission,
    LayerGroupRows,
    Outcome,
    SlotRegion,
    SubmissionRejected,
    View,
    WakeFn,
)
from ..base.memory import MemoryKind
from .keys import key_stem
from .store import (
    DriverRejected,
    OpHandle,
    RegionHandle,
    RegionLayout,
    RegistrationError,
    RowIO,
    RowResult,
    RowStatus,
    Span,
    StoreDriver,
    concluded,
    faulted,
)

__all__ = [
    "BlobAttempt",
    "BlobBackend",
    "BlobConfig",
    "ConstructionRejected",
    "RegionBinding",
    "RegionHold",
]

# Fetches sit on the time-to-first-token path and go first. After this many fetch batches in a row a
# waiting publish batch goes, so publishes, whose leases pin the lender's slots, never starve.
FETCH_BATCHES_PER_PUBLISH_BATCH = 4
PROGRESS_MIN_DELAY_S = 0.001
PROGRESS_MAX_DELAY_S = 0.1
# A store call may block long: a Mooncake put waits, without a timeout, until every transfer ends
# physically, and a get up to about 60 s for the transfers it submitted. A call past this is logged
# once; its submission keeps holding its slots meanwhile.
STUCK_CALL_WARNING_S = 60.0
SHUTDOWN_FAULT = "blob backend shut down"

_instance_ids = itertools.count()  # names each backend and binding in thread names and logs
_Results = Sequence[RowResult] | None  # None: not concluded yet, or no new evidence


@dataclass(frozen=True)
class BlobConfig:
    """How the backend cuts submissions into driver calls; the ``blob`` section of a store config.

    Attributes:
        key_prefix: The deployment's key namespace; matches ``[A-Za-z0-9_-]{1,64}``.
        workers: Driver calls at once, each on a backend thread of its own. Publishes take at most
            ``max(1, workers - 1)`` of them, so unless there is one worker, one is always left for
            fetches. One Mooncake put does not fill the links, and each running get busy-polls a
            core.
        rows_per_batch: Most rows in one driver call. A call prepares and maps its rows in Python
            for a few microseconds each, holding the GIL, so this bounds that stretch too.
        bytes_per_batch: Most payload bytes in one driver call. Larger calls return into Python
            fewer times per byte, and a busy executor thread makes each return wait for the GIL;
            but one call must end well inside the store's read lease (Mooncake: 10 s), and its
            submission holds every slot until the call returns, or for good when a get gives up
            on a transfer (TRANSFER_FAIL).
    """

    key_prefix: str = "trtllm-kv"
    workers: int = 4
    rows_per_batch: int = 256
    bytes_per_batch: int = 128 << 20

    def __post_init__(self) -> None:
        if not isinstance(self.key_prefix, str):
            raise ValueError(f"key_prefix must be a str, got {self.key_prefix!r}")
        key_stem(self.key_prefix)
        for name in ("workers", "rows_per_batch", "bytes_per_batch"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an int of at least 1, got {value!r}")


class ConstructionRejected(RuntimeError):
    """``RegionBinding.register`` failed and left nothing registered, held or accessible: no cleanup
    is needed, and a backend factory maps it to the runtime's certified rejection.

    Any other exception from ``register`` may leave a registration live; the binding then keeps the
    hold, and ``release_hold`` refuses until every registration is revoked.
    """


class RegionHold(Protocol):
    """What this package uses of ``pyexecutor.kv_cache.sharing.RegionHold``, restated because blob
    modules may not import ``pyexecutor``."""

    def release(self) -> None: ...


class RegionBinding:
    """The lender's region hold and the registrations made under it.

    Registers, once each and on the builder thread, the regions the driver can access; revokes the
    registrations only when nothing can access them, on the backend's progress thread; releases the
    hold only after every registration is confirmed revoked, on the manager's thread. The driver is
    never called under the binding's lock, so reading the live registrations never waits on a
    revocation.

    Args:
        regions: The lender's regions, in its order; region ``i`` is labelled ``region:<i>``.
        driver: The store driver that registers them.
        hold: The lender's hold, taken before registering.
    """

    def __init__(
        self, regions: Sequence[SlotRegion], driver: StoreDriver, hold: RegionHold
    ) -> None:
        self.regions = tuple(regions)
        self.driver = driver
        device_memory = driver.capabilities.device_memory
        # The regions register() registers, by index: those with slots in memory the driver can
        # access. A row in any other region is one the backend does not recognize.
        self.registered = frozenset(
            index
            for index, region in enumerate(self.regions)
            if region.slot_count > 0 and (region.memory_kind is MemoryKind.HOST or device_memory)
        )
        self._hold = hold
        self._name = f"region-binding-{next(_instance_ids)}"
        self._lock = threading.Lock()
        # Region index -> its handle, or None for a registration that may exist without one.
        self._live: dict[int, RegionHandle | None] = {}

    def register(self) -> None:
        """Registers the regions in ``registered``.

        Raises:
            ConstructionRejected: Nothing is registered and the hold is released: there is no region
                to register, or a registration failed with nothing escaping and the earlier ones
                were revoked.
            RuntimeError: A registration may be live; the hold is kept.
        """
        if not self.registered:  # a backend that could serve no row
            self._hold.release()
            raise ConstructionRejected("no region has slots in memory the store driver can access")
        for index in sorted(self.registered):
            region = self.regions[index]
            layout = RegionLayout(
                region.slot_bytes, region.slot_count, region.layout_label, region.memory_kind
            )
            try:
                handle = self.driver.register(region.address, region.nbytes, layout)
            except Exception as error:
                self._raise_for_failed_registration(index, error)
            self._keep(index, handle)

    def unregister(self) -> bool:
        """Revokes every live registration it can. True when none is left; a revocation that failed
        stays live for the next call."""
        with self._lock:
            revocable = [(i, handle) for i, handle in self._live.items() if handle is not None]
        for index, handle in revocable:
            try:
                self.driver.unregister(handle)
            except Exception as error:
                logger.warning_once(
                    f"blob backend: revoking region:{index} failed, will retry: {error!r}",
                    key=(self._name, "unregister", index),
                )
                continue
            with self._lock:
                del self._live[index]
        with self._lock:
            return not self._live

    def release_hold(self) -> None:
        """Releases the lender's hold, on the manager's thread.

        Raises:
            RuntimeError: A registration may still be live; the hold is kept.
        """
        live = self.live_labels()
        if live:
            raise RuntimeError(f"the region hold is kept: {sorted(live)} may still be registered")
        self._hold.release()

    def live_labels(self) -> frozenset[str]:
        """``region:<i>`` for every region that may still be registered."""
        with self._lock:
            return frozenset(f"region:{index}" for index in self._live)

    def _raise_for_failed_registration(self, index: int, error: Exception) -> NoReturn:
        handle = error.handle if isinstance(error, RegistrationError) else None
        nothing_escaped = isinstance(error, DriverRejected) or (
            isinstance(error, RegistrationError) and handle is None
        )
        if not nothing_escaped:
            self._keep(index, handle)
            raise RuntimeError(f"region:{index} may be registered after {error!r}") from error
        if not self.unregister():
            raise RuntimeError(
                f"registering region:{index} failed and an earlier region stays registered: {error}"
            ) from error
        # Nothing is registered or accessible, and construction precedes the executor loop, so the
        # builder thread may still give the hold back.
        self._hold.release()
        raise ConstructionRejected(f"registering region:{index} failed: {error}") from error

    def _keep(self, index: int, handle: RegionHandle | None) -> None:
        with self._lock:
            self._live[index] = handle


class _GroupRows(NamedTuple):
    """One group's arrays, taken from the view at submission: row ``i`` is the spans starting at
    ``starts[i, j]``, ``span_bytes[j]`` bytes each, back to back."""

    object_keys: np.ndarray  # uint8 [rows, width]
    starts: np.ndarray  # int64 [rows, spans]
    span_bytes: tuple[int, ...]


@dataclass(eq=False)
class _Batch:
    """Rows ``[start, stop)`` of one group of an attempt; the unit of one driver call."""

    attempt: BlobAttempt
    group: int  # position in the view's groups
    start: int
    stop: int
    op: OpHandle | None = None
    results_recorded: bool = False  # set once, under the backend's lock
    access_recorded: bool = False  # set once, under the backend's lock
    unprovable: bool = False  # the driver can never prove its access ended

    def __len__(self) -> int:
        return self.stop - self.start

    @property
    def settled(self) -> bool:
        """Nothing more to learn from the driver: results recorded, and access ended or never
        provable. An unprovable batch keeps its access open, so its attempt never quiesces."""
        return self.results_recorded and (self.access_recorded or self.unprovable)


class BlobAttempt:
    """The handle of one submission: a ``base.backend.Attempt``.

    Its state changes only under the backend's lock; ``poll`` and ``quiesced`` read one reference
    without it, so they never block.
    """

    def __init__(
        self, backend: BlobBackend, label: str, is_fetch: bool, groups: list[_GroupRows]
    ) -> None:
        self.label = label
        """The attempt's label among ``BlobBackend.shutdown``'s, ``attempt:<n>``."""
        self.is_fetch = is_fetch
        self._backend = backend
        self._groups: list[_GroupRows] | None = groups
        self._masks = [np.zeros(len(rows.object_keys), dtype=bool) for rows in groups]
        self._queued_batches = 0  # zeroed by a stop; the deque keeps the batches, _take skips them
        self._results_pending = 0
        self._access_open = 0  # an unprovable batch stays in it for good
        self._fault: str | None = None  # the first fault, which also stops the attempt
        self._stopped = False  # set by a cancel, the first fault or a shutdown
        self._outcome: Outcome | None = None
        self._quiesced = False

    def poll(self) -> Outcome | None:
        return self._outcome

    def quiesced(self) -> bool:
        return self._quiesced

    def cancel(self) -> None:
        """Stops dispatching the attempt's queued batches; started ones run on.

        Not forwarded to the driver: the contract's cancel is advisory, and the store this backend
        serves has no cancel to forward it to.
        """
        self._backend._cancel(self)

    def _apply(self, batch: _Batch, results: Sequence[RowResult]) -> None:
        self._masks[batch.group][batch.start : batch.stop] = [
            result.status is RowStatus.READY for result in results
        ]
        for row, result in enumerate(results, batch.start):
            if result.fault is not None:
                self._fault = self._fault or f"group {batch.group} row {row}: {result.fault}"
                self._stop()
                return

    def _stop(self) -> None:
        self._stopped = True
        self._queued_batches = 0

    def _conclude_or_quiesce(self) -> bool:
        """Lands the outcome and turns quiesced once their conditions hold; True when either did."""
        changed = False
        # After a fault the outcome still waits for the results of batches already started: with
        # several workers a later batch can fail first, and freezing then would drop the earlier
        # rows that a prefix reuse needs.
        if self._outcome is None and self._queued_batches == 0 and self._results_pending == 0:
            for mask in self._masks:  # final, so the outcome keeps them without a copy
                mask.flags.writeable = False
            self._outcome = Outcome(tuple(self._masks), self._fault)
            changed = True
        if not self._quiesced and self._queued_batches == 0 and self._access_open == 0:
            self._quiesced = True
            self._groups = None  # nothing reads the view's arrays once the attempt quiesced
            changed = True
        return changed

    def _force_outcome(self, reason: str) -> None:
        self._stop()
        fault = reason if self._fault is None else f"{reason}; first fault: {self._fault}"
        # Copied: batches already started still write their results into these masks.
        self._outcome = Outcome(tuple(self._masks), fault)


class BlobBackend:
    """Moves a view's rows between the lender's registered regions and a byte store: a
    ``base.backend.Backend``. A process may hold several, one per lender, each with its own driver.

    Args:
        binding: The lender's regions, already registered with ``binding.driver``. The backend
            revokes the registrations once nothing can access them after shutdown.
        wake: Called outside every lock whenever an attempt's outcome lands or it quiesces, once
            the change shows through ``poll`` or ``quiesced``.
        config: How submissions are cut into driver calls.
    """

    def __init__(self, binding: RegionBinding, wake: WakeFn, config: BlobConfig) -> None:
        self._binding = binding
        self._driver = binding.driver
        self._regions = binding.regions
        self._wake = wake
        self._config = config
        self._key_stem = key_stem(config.key_prefix)
        self._lock = threading.Lock()
        self._work_ready = threading.Condition(self._lock)
        self._progress_due = threading.Condition(self._lock)
        self._progress_poked = False
        self._fetch_queue: deque[_Batch] = deque()
        self._publish_queue: deque[_Batch] = deque()
        self._fetches_in_a_row = 0
        self._watched: set[_Batch] = set()
        self._attempts: set[BlobAttempt] = set()  # until concluded and quiesced
        self._attempt_ids = itertools.count()
        self._closing = False
        self._drained = False
        self._in_driver: dict[_Batch, float] = {}  # batch -> when its driver call started
        self._reported_labels: frozenset[str] | None = None
        self._name = name = f"blob-backend-{next(_instance_ids)}"
        capabilities = self._driver.capabilities
        self._max_rows_per_op = capabilities.max_rows_per_op
        self._capabilities = Capabilities(
            fetch=capabilities.get,
            publish=capabilities.put,
            cancel=True,
            # A row placed by slot is one span per region it lists.
            slot_rows=capabilities.multi_span,
        )
        self._unprovable_batches = 0
        self._workers_alive = config.workers
        self._publish_limit = max(1, config.workers - 1)
        self._publishes_running = 0
        self._workers = [
            threading.Thread(target=self._worker_loop, name=f"{name}-worker-{i}", daemon=True)
            for i in range(config.workers)
        ]
        self._progress = threading.Thread(
            target=self._progress_loop, name=f"{name}-progress", daemon=True
        )
        for thread in (*self._workers, self._progress):
            thread.start()

    @property
    def capabilities(self) -> Capabilities:
        return self._capabilities

    def fetch(self, view: View, source: Hint) -> BlobAttempt:
        return self._submit(view, source, is_fetch=True)

    def publish(self, view: View, destination: Hint) -> BlobAttempt:
        return self._submit(view, destination, is_fetch=False)

    def shutdown(self, timeout: float) -> frozenset[str]:
        """Forces outcomes, then waits up to ``timeout`` for the progress thread to finish cleanly.

        Never calls the driver: the progress thread drains and revokes the registrations once every
        worker has left and every attempt quiesced, since revocation can be slow (RDMA deregisters
        on every NIC) and a call with no budget left must still return at once.
        """
        deadline = time.monotonic() + max(0.0, timeout)
        if self._close_and_force_outcomes():
            self._call_wake()
        self._progress.join(max(0.0, deadline - time.monotonic()))
        labels = self._labels()
        self._report(labels)
        return labels

    def stats(self) -> dict[str, int]:
        """A snapshot of the work in flight, for logs and tests."""
        now = time.monotonic()
        with self._lock:
            return {
                "attempts_active": len(self._attempts),
                "batches_queued": sum(a._queued_batches for a in self._attempts),
                "batches_access_open": sum(a._access_open for a in self._attempts),
                "batches_unprovable": self._unprovable_batches,
                "oldest_driver_call_ms": int(
                    1e3 * (now - min(self._in_driver.values(), default=now))
                ),
            }

    # Submission, in the call sequence.

    def _submit(self, view: View, hint: Hint, is_fetch: bool) -> BlobAttempt:
        if hint is not None:
            raise InvalidSubmission(f"the blob backend takes only a None hint, got {hint!r}")
        if not (self._capabilities.fetch if is_fetch else self._capabilities.publish):
            raise InvalidSubmission(f"the store driver cannot {'get' if is_fetch else 'put'}")
        groups = [self._group_rows(group) for group in view.groups]
        if is_fetch:
            _refuse_shared_destinations(groups)
        attempt = BlobAttempt(self, f"attempt:{next(self._attempt_ids)}", is_fetch, groups)
        batches = self._cut(attempt, groups)
        with self._lock:
            if self._closing:
                raise SubmissionRejected("the blob backend is shut down")
            self._attempts.add(attempt)
            attempt._queued_batches = len(batches)
            (self._fetch_queue if is_fetch else self._publish_queue).extend(batches)
            self._work_ready.notify(len(batches))
            wake_due = self._conclude_or_quiesce(attempt)
        if wake_due:
            self._call_wake()
        return attempt

    def _group_rows(self, group: LayerGroupRows) -> _GroupRows:
        """The group's keys and the spans of each row, checked; rows placed by address or, where
        the driver takes rows of several spans, by slot."""
        where = f"layer group {group.layer_group_index}"
        rows = len(group)
        if group.object_keys is None:
            raise InvalidSubmission(f"{where}: rows need object keys")
        keys = np.asarray(group.object_keys)
        if keys.dtype != np.uint8 or keys.ndim != 2 or keys.shape[0] != rows or keys.shape[1] == 0:
            raise InvalidSubmission(
                f"{where}: object keys must be uint8 [{rows}, width > 0], got {keys.dtype} "
                f"{keys.shape}"
            )
        if group.addresses is not None and group.region_index is not None:
            starts, span_bytes = self._address_spans(where, rows, group)
        else:
            # Optional members of the contract: only a backend declaring slot_rows reads them.
            slots, indices = getattr(group, "slots", None), getattr(group, "region_indices", None)
            if not (self._capabilities.slot_rows and slots is not None and indices is not None):
                raise InvalidSubmission(f"{where}: rows need a placement this backend reads")
            starts, span_bytes = self._slot_spans(where, rows, np.asarray(slots), indices)
        return _GroupRows(keys, starts, span_bytes)

    def _address_spans(
        self, where: str, rows: int, group: LayerGroupRows
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        addresses = np.asarray(group.addresses)
        if addresses.dtype != np.int64 or addresses.shape != (rows,):
            raise InvalidSubmission(
                f"{where}: addresses must be int64 [{rows}], got {addresses.dtype} "
                f"{addresses.shape}"
            )
        region = self._registered_region(where, group.region_index)
        slot, misalignment = np.divmod(addresses - region.address, region.slot_bytes)
        if np.any((slot < 0) | (slot >= region.slot_count) | (misalignment != 0)):
            raise InvalidSubmission(f"{where}: an address is not a slot of its region")
        return addresses[:, None], (region.slot_bytes,)

    def _slot_spans(
        self, where: str, rows: int, slots: np.ndarray, region_indices: Sequence[int]
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        if slots.dtype != np.int64 or slots.shape != (rows,):
            raise InvalidSubmission(
                f"{where}: slots must be int64 [{rows}], got {slots.dtype} {slots.shape}"
            )
        if not region_indices:
            raise InvalidSubmission(f"{where}: rows placed by slot name no region")
        regions = [self._registered_region(where, index) for index in region_indices]
        if np.any((slots < 0) | (slots >= min(region.slot_count for region in regions))):
            raise InvalidSubmission(f"{where}: a slot is outside its regions")
        starts = np.stack([region.address + slots * region.slot_bytes for region in regions], 1)
        return starts, tuple(region.slot_bytes for region in regions)

    def _registered_region(self, where: str, index: int) -> SlotRegion:
        if index not in self._binding.registered:  # refused, never skipped
            raise InvalidSubmission(f"{where}: region {index} is not registered")
        return self._regions[index]

    def _cut(self, attempt: BlobAttempt, groups: list[_GroupRows]) -> list[_Batch]:
        batches = []
        for group, group_rows in enumerate(groups):
            step = self._rows_per_batch(sum(group_rows.span_bytes))
            count = len(group_rows.object_keys)
            batches += [
                _Batch(attempt, group, start, min(start + step, count))
                for start in range(0, count, step)
            ]
        return batches

    def _rows_per_batch(self, row_bytes: int) -> int:
        limits = [self._config.rows_per_batch, self._config.bytes_per_batch // row_bytes]
        if self._max_rows_per_op is not None:
            limits.append(self._max_rows_per_op)
        return max(1, min(limits))

    def _cancel(self, attempt: BlobAttempt) -> None:
        with self._lock:
            if attempt._outcome is not None:
                return
            attempt._stop()
            wake_due = self._conclude_or_quiesce(attempt)
        if wake_due:
            self._call_wake()

    # Workers: up to ``config.workers`` driver calls at once, each on a thread of its own.

    def _worker_loop(self) -> None:
        try:
            while (batch := self._take()) is not None:
                self._run(batch)
        finally:
            with self._lock:
                self._workers_alive -= 1
                self._poke_progress()

    def _take(self) -> _Batch | None:
        with self._lock:
            while True:
                batch = self._pop_next()
                if batch is None:
                    if self._closing:
                        return None
                    self._work_ready.wait()
                    continue
                attempt = batch.attempt
                if attempt._stopped:  # its queued count was cleared when it stopped
                    continue
                if not attempt.is_fetch:
                    self._publishes_running += 1
                self._fetches_in_a_row = self._fetches_in_a_row + 1 if attempt.is_fetch else 0
                # Counted as started before the driver call: while the call runs, this count alone
                # keeps the attempt from quiescing once a cancel or a fault has cleared its queue.
                attempt._queued_batches -= 1
                attempt._results_pending += 1
                attempt._access_open += 1
                self._in_driver[batch] = time.monotonic()
                return batch

    def _pop_next(self) -> _Batch | None:
        publish_due = (
            self._publish_queue
            and self._publishes_running < self._publish_limit
            and (not self._fetch_queue or self._fetches_in_a_row >= FETCH_BATCHES_PER_PUBLISH_BATCH)
        )
        if publish_due:
            return self._publish_queue.popleft()
        return self._fetch_queue.popleft() if self._fetch_queue else None

    def _run(self, batch: _Batch) -> None:
        driver_called = False
        try:
            row_ios = self._row_ios(batch)
            driver_called = True
            op = self._driver.get(row_ios) if batch.attempt.is_fetch else self._driver.put(row_ios)
        except Exception as error:
            # DriverRejected certifies nothing escaped; any other raise from the driver is outside
            # its protocol: the rows may be accessed for good.
            refused = not driver_called or isinstance(error, DriverRejected)
            reason = f"refused: {error}" if refused else f"the store driver raised {error!r}"
            self._finish(batch, _faults(batch, reason), ended=refused, unprovable=not refused)
            return
        batch.op = op
        self._finish(batch, *self._observe(batch))

    def _row_ios(self, batch: _Batch) -> list[RowIO]:
        group_rows = batch.attempt._groups[batch.group]
        span_bytes, stem = group_rows.span_bytes, self._key_stem
        names = group_rows.object_keys[batch.start : batch.stop]
        hexed, width = names.tobytes().hex(), 2 * names.shape[1]  # one pass for the batch
        starts = group_rows.starts[batch.start : batch.stop].tolist()
        return [
            RowIO(
                stem + hexed[i * width : (i + 1) * width],
                tuple(Span(start, nbytes) for start, nbytes in zip(row, span_bytes)),
            )
            for i, row in enumerate(starts)
        ]

    def _observe(self, batch: _Batch) -> tuple[_Results, bool, bool]:
        """The driver's results (when concluded), access end and unprovability for ``batch``."""
        try:
            results = self._driver.poll(batch.op)
            ended = self._driver.access_ended(batch.op)
            unprovable = not ended and self._driver.access_unprovable(batch.op)
            count, done = len(results), concluded(results)
        except Exception as error:  # no new evidence this time
            logger.warning_once(
                f"blob backend: observing a store operation raised {error!r}",
                key=(self._name, "observe", type(error)),
            )
            return None, False, False
        if count != len(batch):  # outside the driver protocol: no later answer proves an access end
            reason = f"the store driver gave {count} results for {len(batch)} rows"
            return _faults(batch, reason), False, not batch.access_recorded
        return (results if done else None), ended, unprovable

    def _finish(self, batch: _Batch, results: _Results, ended: bool, unprovable: bool) -> None:
        with self._lock:
            del self._in_driver[batch]
            # No notify: this worker takes its next batch right after, so the publish room it frees
            # is its own to use; an idle worker waits only while nothing it may take is queued.
            if not batch.attempt.is_fetch:
                self._publishes_running -= 1
            wake_due = self._record(batch, results, ended, unprovable)
            if batch.op is not None and not batch.settled:
                self._watched.add(batch)
                self._poke_progress()
        if wake_due:
            self._call_wake()

    # Bookkeeping, under the lock.

    def _record(self, batch: _Batch, results: _Results, ended: bool, unprovable: bool) -> bool:
        attempt = batch.attempt
        if results is not None and not batch.results_recorded:
            batch.results_recorded = True
            attempt._results_pending -= 1
            attempt._apply(batch, results)
        if ended and not batch.access_recorded:
            batch.access_recorded = True
            attempt._access_open -= 1
        if unprovable and not batch.unprovable:  # counted, held for good, no longer polled
            batch.unprovable = True
            self._unprovable_batches += 1
            logger.warning_once(
                f"blob backend: a store call of {attempt.label} can never prove its access "
                "ended; its slots stay held",
                key=(self._name, "unprovable"),
            )
        if batch.settled:
            self._watched.discard(batch)
        return self._conclude_or_quiesce(attempt)

    def _conclude_or_quiesce(self, attempt: BlobAttempt) -> bool:
        wake_due = attempt._conclude_or_quiesce()
        if attempt._outcome is not None and attempt._quiesced:
            self._attempts.discard(attempt)
        return wake_due

    def _poke_progress(self) -> None:
        self._progress_poked = True
        self._progress_due.notify()

    # The progress thread: the only one that observes an operation after its worker let go.

    def _progress_loop(self) -> None:
        delay = PROGRESS_MIN_DELAY_S
        while True:
            try:
                self._observe_watched()
                self._report_stuck_calls()
                if self._closing and self._finish_shutdown():
                    return
            except Exception as error:  # the thread must live on, or the labels never clear
                logger.warning_once(
                    f"blob backend: a progress round raised {error!r}",
                    key=(self._name, "progress", type(error)),
                )
            delay = self._sleep(delay)

    def _observe_watched(self) -> None:
        with self._lock:
            batches = list(self._watched)
        if not batches:
            return
        observations = [(batch, *self._observe(batch)) for batch in batches]
        with self._lock:
            wake_due = [self._record(*observation) for observation in observations]
        if any(wake_due):
            self._call_wake()

    def _report_stuck_calls(self) -> None:
        now = time.monotonic()
        with self._lock:
            stuck = [
                b for b, started in self._in_driver.items() if now - started > STUCK_CALL_WARNING_S
            ]
        for batch in stuck:
            logger.warning_once(
                f"blob backend: a store call of {batch.attempt.label} has run for over "
                f"{STUCK_CALL_WARNING_S:g} s; its slots stay held until it returns",
                key=(self._name, "stuck", batch.attempt.label, batch.group, batch.start),
            )

    def _sleep(self, delay: float) -> float:
        with self._lock:
            if not self._progress_poked:
                # Idle, the thread wakes only as often as stuck driver calls need reporting.
                busy = self._watched or self._closing
                self._progress_due.wait(delay if busy else STUCK_CALL_WARNING_S)
            poked, self._progress_poked = self._progress_poked, False
        return PROGRESS_MIN_DELAY_S if poked else min(2 * delay, PROGRESS_MAX_DELAY_S)

    def _finish_shutdown(self) -> bool:
        """True once the driver is drained and the registrations revoked; a step that is not done
        yet is retried next round. The driver is never closed: Mooncake's close holds the GIL and
        sends RPCs, so the client is left to process exit."""
        with self._lock:
            if self._workers_alive or any(not attempt._quiesced for attempt in self._attempts):
                return False
        for worker in self._workers:
            worker.join()  # each has left its loop already
        if not self._drained:
            self._drained = self._driver.drain(0.0)
        return self._drained and self._binding.unregister()

    # Shutdown, on the shutdown thread.

    def _close_and_force_outcomes(self) -> bool:
        with self._lock:
            self._closing = True
            forced = [attempt for attempt in self._attempts if attempt._outcome is None]
            for attempt in forced:
                attempt._force_outcome(SHUTDOWN_FAULT)
                self._conclude_or_quiesce(attempt)
            self._work_ready.notify_all()
            self._poke_progress()
        return bool(forced)

    def _labels(self) -> frozenset[str]:
        with self._lock:
            attempts = {attempt.label for attempt in self._attempts if not attempt._quiesced}
        return frozenset(attempts) | self._binding.live_labels()

    def _report(self, labels: frozenset[str]) -> None:
        if labels == self._reported_labels:
            return
        self._reported_labels = labels
        if labels:
            logger.warning(f"blob backend: shutting down, still accessible: {sorted(labels)}")
        else:
            logger.info("blob backend: shut down; nothing is accessible any more")

    def _call_wake(self) -> None:
        try:
            self._wake()
        except Exception as error:  # a raising wake must not kill a backend thread
            logger.warning_once(
                f"blob backend: wake raised {error!r}", key=(self._name, "wake", type(error))
            )


def _refuse_shared_destinations(groups: list[_GroupRows]) -> None:
    """Two fetched rows in one slot would let one report READY over the other's bytes. A publish
    only reads its slots, so only a fetch is checked. Rows on one slot share their first span,
    and the spans of rows in different slots never meet."""
    firsts = np.concatenate([rows.starts[:, 0] for rows in groups]) if groups else np.empty(0)
    if np.unique(firsts).size != firsts.size:
        raise InvalidSubmission("two rows of a fetch write the same slot")


def _faults(batch: _Batch, reason: str) -> tuple[RowResult, ...]:
    return (faulted(reason),) * len(batch)

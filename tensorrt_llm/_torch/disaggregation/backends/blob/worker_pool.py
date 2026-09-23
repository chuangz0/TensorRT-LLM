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
"""A fixed pool of daemon worker threads for the store backend's blocking store calls.

The standard ``ThreadPoolExecutor`` joins its threads at interpreter exit, so a worker stuck in a
store call that never returns would keep the process alive after the engine gave the backend up.
Daemon threads cannot; whoever waits for them does so with the engine's own timeout.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Callable

__all__ = ["DaemonWorkerPool"]

logger = logging.getLogger(__name__)


class DaemonWorkerPool:
    """``submit`` runs a callable on one of ``num_workers`` daemon threads; ``shutdown`` stops them.

    Threads start on the first ``submit``, as the standard executor's do. A submission after
    ``shutdown`` raises ``RuntimeError``, as the standard executor does, so callers written
    against that contract need no change.
    """

    def __init__(self, num_workers: int, thread_name_prefix: str) -> None:
        if num_workers <= 0:
            raise ValueError(f"num_workers must be > 0, got {num_workers}")
        self._work: queue.SimpleQueue = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._is_shut_down = False
        self._threads = [
            threading.Thread(target=self._serve, name=f"{thread_name_prefix}-{i}", daemon=True)
            for i in range(num_workers)
        ]

    def submit(self, fn: Callable[..., object], *args: object) -> None:
        with self._lock:
            if self._is_shut_down:
                raise RuntimeError("cannot schedule new work after shutdown")
            self._start_threads_once()
            self._work.put((fn, args))

    def _start_threads_once(self) -> None:
        """Caller holds the lock."""
        for thread in self._threads:
            if not thread.is_alive() and thread.ident is None:
                thread.start()

    def shutdown(self, wait: bool = True) -> None:
        """Refuse further work; let queued work finish; with ``wait``, join the threads.

        Idempotent, and a later ``wait=True`` still joins after an earlier ``wait=False``, as the
        standard executor's does; the stop sentinels are queued once.
        """
        with self._lock:
            first_call = not self._is_shut_down
            self._is_shut_down = True
            started = [thread for thread in self._threads if thread.ident is not None]
        if first_call:
            for _ in started:
                self._work.put(None)
        if wait:
            for thread in started:
                thread.join()

    def _serve(self) -> None:
        while True:
            item = self._work.get()
            if item is None:
                return
            fn, args = item
            # Broad on purpose, against CODING_GUIDELINES: this is a thread boundary, and the
            # callables submitted here report their own failures through their attempt; anything
            # that still escapes must not kill the worker for the deliveries queued behind it.
            try:
                fn(*args)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "store worker %s: unhandled error", threading.current_thread().name
                )

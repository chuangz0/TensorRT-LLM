# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A pause of one kind of store work after it failed several times in a row."""

from __future__ import annotations

from dataclasses import dataclass

from tensorrt_llm.logger import logger

# A kind of store work pauses after this many failures in a row, for this long; then it serves
# again, and the count starts over.
FAILURES_TO_PAUSE = 8
PAUSE_S = 30.0


@dataclass
class Breaker:
    """Pauses one kind of store work ("probes", "fetches", "publications") for PAUSE_S after
    FAILURES_TO_PAUSE failures in a row; a success resets the count."""

    what: str  # the kind of store work, for the log
    failures_in_a_row: int = 0
    paused_until: float = 0.0  # monotonic; 0 while no pause is pending

    def record(self, failed: bool, now: float, why: str = "") -> bool:
        """Counts one outcome; the FAILURES_TO_PAUSE-th failure in a row starts a pause. Outcomes
        during a pause, of work started before it, count nothing. Returns whether a pause
        started."""
        if now < self.paused_until:
            return False
        if not failed:
            self.failures_in_a_row = 0
            return False
        self.failures_in_a_row += 1
        if self.failures_in_a_row < FAILURES_TO_PAUSE:
            return False
        self.failures_in_a_row = 0
        self.paused_until = now + PAUSE_S
        logger.warning(
            f"kv_sharing: {self.what} pause for {PAUSE_S:.0f} s after {FAILURES_TO_PAUSE} store "
            f"failures in a row (last: {why})"
        )
        return True

    def allows(self, now: float) -> bool:
        """Whether the work may run now."""
        if now < self.paused_until:
            return False
        if self.paused_until:
            self.paused_until = 0.0
            logger.info(f"kv_sharing: {self.what} resume after a pause")
        return True

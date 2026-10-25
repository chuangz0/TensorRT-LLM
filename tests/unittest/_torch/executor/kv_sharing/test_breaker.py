# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A kind of store work pauses after failures in a row, for a fixed time, and warns once per
pause."""

import pytest

from tensorrt_llm._torch.pyexecutor.kv_sharing import breaker as breaker_module
from tensorrt_llm._torch.pyexecutor.kv_sharing.breaker import FAILURES_TO_PAUSE, PAUSE_S, Breaker

pytestmark = pytest.mark.cpu_only


def test_failures_in_a_row_pause_for_a_fixed_time_and_a_success_resets_the_count(monkeypatch):
    warnings = []
    monkeypatch.setattr(breaker_module.logger, "warning", warnings.append)
    breaker = Breaker("fetches")
    for _ in range(FAILURES_TO_PAUSE - 1):
        assert not breaker.record(0.0, failed=True)
    assert not breaker.record(0.0, failed=False)  # a success starts the count over
    started = [breaker.record(1.0, failed=True, why="late") for _ in range(FAILURES_TO_PAUSE)]
    assert started == [False] * (FAILURES_TO_PAUSE - 1) + [True]
    assert not breaker.allows(1.0) and not breaker.allows(1.0 + PAUSE_S / 2)
    # Outcomes of work started before the pause count nothing.
    assert not any(breaker.record(2.0, failed=True) for _ in range(FAILURES_TO_PAUSE))
    assert breaker.allows(1.0 + PAUSE_S)
    assert len(warnings) == 1 and "fetches pause" in warnings[0] and "late" in warnings[0]
    assert breaker.failures_in_a_row == 0

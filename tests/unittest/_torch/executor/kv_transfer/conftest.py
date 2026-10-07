# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Engine-side tests of the KV transfer layer (``pyexecutor/kv_transfer/``, the KV v2 wrapper
changes, the scheduler seam).

These import the coordination layer *through* ``tensorrt_llm`` (the modules under test do), which
is why they live under ``executor/`` and not next to ``disaggregation/orchestration/kv_transfer``:
that suite asserts, process-wide, that the layer was never imported through the package.
``test_kv_v2_view_and_wrapper.py`` is the deliberate exception to the tests-mirror-source rule: it
tests ``disaggregation/resource/`` but needs a GPU and imports through ``tensorrt_llm``, so it
lives here too.

Run from ``tests/unittest`` with ``PYTHONPATH=<repo root>`` so the working tree (with its in-tree
bindings) is imported rather than an installed wheel; the fakes in ``engine_fakes.py`` are found
through pytest's rootdir import mode, so ``--noconftest`` only loses the fixtures below.
``test_kv_v2_view_and_wrapper.py`` allocates device pools and skips without a GPU.
"""

import time
from unittest.mock import Mock

import pytest
from engine_fakes import EngineRig

from tensorrt_llm._torch.pyexecutor.kv_transfer import effects, hooks


@pytest.fixture
def rig():
    return EngineRig()


@pytest.fixture
def clock(monkeypatch):
    """Freeze ``time.monotonic`` for the hooks and the coordinator; tests advance ``clock["t"]``.
    The fake collective's barriers keep their own wall clock, so a frozen clock never stalls
    them."""
    now = {"t": 1000.0}
    monkeypatch.setattr(time, "monotonic", lambda: now["t"])
    return now


@pytest.fixture
def effects_logger(monkeypatch):
    fake = Mock()
    monkeypatch.setattr(effects, "logger", fake)
    return fake


@pytest.fixture
def hooks_logger(monkeypatch):
    fake = Mock()
    monkeypatch.setattr(hooks, "logger", fake)
    return fake

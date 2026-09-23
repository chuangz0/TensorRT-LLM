# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Engine-side tests of the KV transfer layer (``pyexecutor/kv_transfer_*``, the KV v2 wrapper
changes, the scheduler seam).

These import the coordination layer *through* ``tensorrt_llm`` (the modules under test do), which
is why they live under ``executor/`` and not next to ``disaggregation/kv_transfer``: that suite
asserts, process-wide, that the layer was never imported through the package.

Run from ``tests/unittest`` with ``PYTHONPATH=<repo root>`` so the working tree (with its in-tree
bindings) is imported rather than an installed wheel; the fakes in ``engine_fakes.py`` are found
through pytest's rootdir import mode, so ``--noconftest`` needs nothing further.
``test_kv_v2_reader_layout.py`` allocates device pools and skips without a GPU.
"""

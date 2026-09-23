# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests of the store layer on TinyLlama: the ``mooncake_cluster`` fixture (a
``mooncake_master`` and a segment provider on free loopback ports) and the shared helpers of
``mooncake_cluster.py``.

Requirements, each of which otherwise skips the tests: a GPU, the ``mooncake.store`` bindings,
``mooncake_master`` on ``PATH`` or in ``~/.local/bin``, and the TinyLlama weights under
``$LLM_MODELS_ROOT`` (``TinyLlama-1.1B-Chat-v1.0`` or ``llama-models-v2/TinyLlama-1.1B-Chat-v1.0``),
``$TINYLLAMA_MODEL_PATH``, or the repo's ``.models/``. Run from ``tests/unittest`` with
``PYTHONPATH=<repo root>``: the engines run in spawned worker processes that must import the
working tree, not an installed wheel. ``mooncake_cluster.py`` is found through pytest's rootdir
import mode, so ``--noconftest`` only loses the fixture registration below.
"""

from mooncake_cluster import mooncake_cluster  # noqa: F401

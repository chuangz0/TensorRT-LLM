# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Import the coordination layer as ``disaggregation.*`` straight from ``tensorrt_llm/_torch``.

The modules under test depend on nothing outside their own package, so they are tested without
importing ``tensorrt_llm`` (whose compiled bindings need not be present). The declaration below
is file-scoped (see ``tests/test_common/magic_import.py``) and governs this directory; every file
here repeats it so the suite also runs under ``--noconftest``.
"""

__extra_import_path__ = ["~/tensorrt_llm/_torch"]

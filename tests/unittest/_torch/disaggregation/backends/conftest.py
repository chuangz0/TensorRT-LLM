# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Import-light tests of ``disaggregation/backends/``: the KV transfer config file
(``test_config.py``) and the backend registry (``test_registry.py``) here, one sub-directory per
backend KIND below (``blob/``, which also holds the built-in ``mooncake`` entry's factory tests).

Like ``../orchestration/kv_transfer``, the modules under test are imported as ``disaggregation.*``
straight from ``tensorrt_llm/_torch`` (file-scoped ``__extra_import_path__``, see
``tests/test_common/magic_import.py``), so ``tensorrt_llm`` itself is not needed and no
``PYTHONPATH`` is required. The declaration is repeated in every file so the suite also runs under
``--noconftest``.
"""

__extra_import_path__ = ["~/tensorrt_llm/_torch"]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coordination layer over the ``base/cache_backend.py`` contract.

``coordinator.py`` is the only writer of the record table; ``interfaces.py`` holds its Protocols
towards the engine side and the page-table side; ``records.py`` keeps one record per request per
direction. The policy input ``Planner`` lives in ``disaggregation/remote_cache.py``.
"""

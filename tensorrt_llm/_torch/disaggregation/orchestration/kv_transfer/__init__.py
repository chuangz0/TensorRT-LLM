# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coordination layer over the ``base/cache_backend.py`` contract.

``coordinator.py`` is the only writer of the record table; ``consensus.py`` is what the ranks say
to each other each round and how it is reduced; ``engine_protocols.py`` holds the coordinator's
Protocols towards the engine side; ``records.py`` keeps one record per request per direction;
``build.py`` assembles a coordinator from the config and the built backends. The read-only views
it consumes live in ``base/views.py``, the optional backend capabilities it recognises in
``base/capabilities.py``, and the policy input ``Planner`` in ``disaggregation/remote_cache.py``.
"""

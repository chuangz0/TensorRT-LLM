# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The status dumps the KV transfer layer writes at ``close``: where an engine is told to write
them, how they are read back per rank, and what a store round trip's counters must say.

Every engine of a test writes ``kvt-<tag>-<pid>.json`` into the test's directory, one file per
rank (``dump_template``); the readers find them by tag.
"""

from __future__ import annotations

import glob
import json
import os

# Literal copy of ``assembly.KV_TRANSFER_STATUS_DUMP_ENV``: this module imports ``tensorrt_llm``
# nowhere, so the e2e tests collect (and skip) without it.
KV_TRANSFER_STATUS_DUMP_ENV = "TRTLLM_KV_TRANSFER_STATUS_DUMP"

DUMP_KEYS = frozenset({"started_at", "pid", "rank", "coordinator", "backends"})
"""The top-level keys of one rank's dump, as ``KVTransferHooks.status_dump`` writes them."""


def dump_template(directory, tag: str) -> str:
    """The ``KV_TRANSFER_STATUS_DUMP_ENV`` value for the engine tagged ``tag``: one file per rank,
    the rank's process id filled in by the layer."""
    return os.path.join(str(directory), f"kvt-{tag}-{{pid}}.json")


def check_dump_schema(dump: dict) -> None:
    assert set(dump) == DUMP_KEYS, sorted(dump)


def read_rank_dumps(directory, tag: str, world_size: int) -> list[dict]:
    """One dump per rank of the engine tagged ``tag``, ordered by rank; exactly ``world_size`` of
    them, each with the dump's schema."""
    dumps = []
    for path in glob.glob(os.path.join(str(directory), f"kvt-{tag}-*.json")):
        with open(path, encoding="utf-8") as f:
            dump = json.load(f)
        check_dump_schema(dump)
        dumps.append(dump)
    dumps.sort(key=lambda d: d["rank"])
    assert [d["rank"] for d in dumps] == list(range(world_size)), [d["rank"] for d in dumps]
    return dumps


def read_single_rank_dump(directory, tag: str) -> dict:
    """The one dump a single-rank engine tagged ``tag`` wrote, which plans for every rank."""
    (dump,) = read_rank_dumps(directory, tag, 1)
    assert dump["coordinator"]["plan_authority"] == "ALL_RANKS"
    return dump


def counters(dump: dict) -> dict:
    """The counters of the one ``shared-store`` mooncake backend every e2e config names."""
    (backend,) = dump["backends"]
    assert backend["type"] == "mooncake" and backend["name"] == "shared-store"
    return backend["counters"]


def landing_of(dump: dict) -> str:
    """The landing the factory resolved for the one backend, as the status dump reports it."""
    (backend,) = dump["backends"]
    return backend["landing"]


def assert_no_leftover_records(dump: dict) -> None:
    coordinator = dump["coordinator"]
    assert coordinator["records"] == [], coordinator
    assert coordinator["finished_pending"] == [], coordinator
    assert coordinator["decided_plans"] == 0, coordinator


def assert_round_trip_counters(
    dump_a: dict, dump_b: dict, *, blocks: int, probe_hits_by_b: int | None = None
) -> None:
    """Engine A published one prompt's ``blocks`` and fetched nothing; engine B's probes found
    exactly ``probe_hits_by_b`` (one lookup of the prompt's blocks per request B saw, default one),
    it fetched the prompt's blocks once, missed nothing, failed nothing and offered nothing the
    store lacked; neither left a record behind."""
    counters_a = counters(dump_a)
    assert counters_a["publish_stored"] == blocks, counters_a
    assert counters_a["fetch_hits"] == 0 and counters_a["fetch_misses"] == 0, counters_a
    assert counters_a["failed_attempts"] == 0, counters_a
    assert_no_leftover_records(dump_a)

    counters_b = counters(dump_b)
    assert counters_b["fetch_hits"] == blocks, counters_b
    assert counters_b["fetch_misses"] == 0, counters_b
    assert counters_b["failed_attempts"] == 0, counters_b
    assert counters_b["probe_hits"] == (blocks if probe_hits_by_b is None else probe_hits_by_b), (
        counters_b
    )
    # B recomputed nothing the store had: what it offers is already present.
    assert counters_b["publish_stored"] == 0, counters_b
    assert_no_leftover_records(dump_b)

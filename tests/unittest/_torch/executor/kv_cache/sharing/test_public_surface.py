# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The lender's public surface and boundaries: the eleven names, their members, what importing the
package loads, who may import its private modules, what its public docstrings promise callers, and
which beams the manager detaches a lent cache from. Every scan has a positive control that plants
the fault it looks for. No GPU."""

import ast
import dataclasses
import inspect
import json
import os
import re
import subprocess
import sys
import weakref
from pathlib import Path
from typing import List, Protocol, Set

import numpy as np
import pytest

import tensorrt_llm
from tensorrt_llm._torch.pyexecutor.kv_cache import sharing
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import (
    GroupRun,
    InPlaceLender,
    Lease,
    Part,
    PartsHold,
    Readiness,
    RegionView,
    StagingLender,
    StagingOptions,
    attach_in_place,
    attach_staging,
)

SH = "tensorrt_llm._torch.pyexecutor.kv_cache.sharing"
MGR = "tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2"
PUBLIC = sorted(
    "GroupRun InPlaceLender Lease Part PartsHold Readiness RegionView StagingLender StagingOptions "
    "attach_in_place attach_staging".split()
)
MODULES = ["__init__.py", "_identity.py", "_layout.py", "_lender.py", "_manager.py", "_slots.py"]
MODULES = sorted(MODULES + ["_types.py"])
PKG_DIR = Path(sharing.__file__).resolve().parent
ROOT = Path(tensorrt_llm.__file__).resolve().parent
TESTS_DIR = Path(__file__).resolve().parent
# Manager and runtime members only the manager facade may touch.
INTERNALS = {
    "_reuse_token_source",
    "_augment_tokens_for_block_reuse",
    "_stale_block_range",
    "_resize_for_connector_prefix",
    "_fill_fresh_kv_pages",
    "_fresh_page_fill",
    "_fresh_pages_filled",
    "_can_publish_block_reuse",
    "block_reuse_policy",
    "_draft_prompt_lookahead",
    "kv_connector_manager",
    "commit_min_snapshot",
    "reuse_match_backoff",
    "_stream",
    "_layer_attn_to_layer_id",
    "_sharing",
    "host_kv_cache_block_offsets",
    "kv_cache_map",
    "py_multimodal_data",
    "multimodal_hashes",
    "multimodal_positions",
    "multimodal_lengths",
    "try_get_encoder_output_len",
    "py_return_context_logits",
    "py_additional_outputs",
    "py_result",
    "additional_context_outputs",
    "prompt_len",
    "is_draft",
}
BACKEND_WORDS = re.compile(r"\b(kvcr|mooncake|blob|native)\b", re.IGNORECASE)


def fact(text: str) -> re.Pattern:
    """A fact a docstring states: ``text`` matched case-insensitively, any run of whitespace for a
    space, so a fact wrapped in prose or indented under a Google-style section reads the same."""
    return re.compile(r"\s+".join(re.escape(word) for word in text.split()), re.IGNORECASE)


# What callers rely on, per documented name: the thread rule, rank-local outcomes, copies serial
# with the forward, what keeps memory past the manager's shutdown, the in-place preconditions, what
# scope covers and the first-version limits. The guide does not ship, so these are the only place a
# caller reads them. Each fact is stated once, at the name it belongs to; a name that relies on a
# fact stated elsewhere points there, and its entry pins that pointer.
THREADS = (
    fact("only on the manager's thread"),
    fact("call no lease or lender method"),
    fact("their own channel"),
)
THREADS_SEE = fact("only on the manager's thread (see the package docstring)")
RANK_LOCAL = (fact("this rank's own"), fact("the caller combines every rank's outcome"))
HOLD_AT_SHUTDOWN = fact(
    "A hold still open at the manager's shutdown keeps the staging memory until the process exits"
)
FIRST_VERSION = fact("First-version limits")
ATTACH_LIMITS = (
    FIRST_VERSION,
    fact("One lender per manager, for the manager's life"),
    fact("Helix"),
    fact("recurrent state"),
    fact("whose read-only pages a cache can lock in host memory"),
)
ATTACH_ORDER = (
    fact("thread that builds the executor"),
    fact("Shut the manager down last"),
    fact("drop the last reference on a thread that has used the device's CUDA context"),
)
KV_ARRANGEMENT = fact("the attention backend's K/V arrangement")
SCOPE = (
    fact("every configuration that changes a block's bytes beyond what the manager declares"),
    KV_ARRANGEMENT,
    fact("The framework's assembly builds it"),
)
NO_REBALANCE = fact("Neither rebalance the pools nor reset the prefix cache")
REBALANCE_OFF = fact("``kv_cache_config.enable_kv_pool_rebalance`` stays off")
LOCKSTEP_RELEASE = fact(
    "release the last loan on a freed request's cache in the same executor iteration on every rank"
)
PREFIX_HASH = fact("Every lease hashes the request's whole prefix again")
FLOOR_AT_HISTORY = fact("the floor is also at least the request's history")
# StagingLender.readiness points to Readiness for the floor at the history.
FLOOR_SEE = fact(
    "such as the floor at the history, and how to combine ranks and pools: see ``Readiness``"
)
DRAFT_POOL_FLOOR = fact(
    "a joint-reuse draft pool, whose context resize sets the capacity from the chunk it runs"
)
FILL_OFF = fact("with page-locked staging and the fresh-page fill off")
PARK_AND_RESUME = fact("Set ``py_connector_served_position`` to ``p``")
RESUME_STEP = fact(
    "Move the prepopulated length and context position to ``p`` together by "
    "``set_prepopulated_prompt_len(p, tokens_per_block)``"
)
RESUME_FORWARD = fact("Resume only at a ``p`` in the interval no lower than its context position")
PARKED_ACTIVE = fact("keep it among the executor's active requests")
DROP_BOTH = fact("the request's cache in every manager it fetched into")
EMPTY_DROPS_BOTH = fact("On an empty interval, drop the request's cache in every manager")
TAKES_THE_HISTORY = fact("takes the tokens below the cache's history as computed")
WRITE_AHEAD_OF_DATA = fact("not one whose history runs ahead of its data")
READ_AHEAD_OF_DATA = fact("not while its history runs ahead of its data")
# StagingLender points to attach_staging for the managers staging refuses.
STAGING_REFUSALS = fact("``attach_staging`` lists the refused managers")
WINDOW_GAP = fact("a fetch whose window still keeps such a block finds its row missing")
UNWRITTEN_PAGES = fact("a block whose page holds tokens past the cache's history")
SCHEDULER_REACH = fact("The pages a fetch grows are outside the V2 scheduler's reach")
ROOM_TO_RESUME = fact("which a resume locks all at once, and the pages its next step adds")
# The package docstring defines room and the deadlock check; the lenders point there.
STAGING_ROOM = fact("keep room in each pool group (package docstring)")
IN_PLACE_ROOM = fact("keep room for them in each pool group (package docstring)")
# The split rule: stated in StagingLender.lend_write, repeated where a caller splits a fetch.
SPLIT_RULE = fact(
    "lend the next segment only once ``readiness`` is not None on every rank (settling is this "
    "rank's own)"
)
SPLIT_WINDOW = fact("also wait until ``usable_until`` reaches the previous lease's end")
SPLIT_UNCHECKED = fact("Not checked yet")
WINDOWED_FETCH_END = fact(
    "lends a sliding-window layer group's rows only for the window at its ``end``"
)
ROUTED_EXPERTS = fact(
    "the caller fetches into a request that asks for routed experts only before its first context "
    "step runs, so never after a recompute pause"
)
DSV4_MARGIN = fact(
    "DeepSeek-V4 keeps every window the draft length wider under any speculative decoding, and a "
    "fetch asks for the rows of that margin too"
)
STAGING_LIMITS = (
    FIRST_VERSION,
    STAGING_REFUSALS,
    WINDOW_GAP,
    DSV4_MARGIN,
    UNWRITTEN_PAGES,
    fact("Rows an earlier lease missed are not checked: the split rule covers those"),
    SCHEDULER_REACH,
    STAGING_ROOM,
    fact("A share of each pool group for the parked fetches alone does not ensure it"),
    WINDOWED_FETCH_END,
    ROUTED_EXPERTS,
    fact("at most one copy call per row and pool"),
    fact("is not ordered after staging copies"),
    fact("synchronizes the device before it fills pages"),
    PREFIX_HASH,
    fact("is never reused"),
    fact("waits in line for good without failing"),
)
READS_AHEAD = fact(
    "built for a one-model draft that reads prompt tokens past a position is refused"
)
NOT_IS_DRAFT = fact("The check reads the manager, not ``is_draft``")
UNESTABLISHED_READ_AHEAD = fact(
    "A draft whose read-ahead upstream has not established counts as reading none"
)
KV_CONNECTOR = fact("A manager with a KV cache connector is refused")
IN_PLACE_READS_AHEAD = fact("a one-model draft that reads prompt tokens ahead is accepted")
COMMIT_BEFORE_OR_AFTER = fact("commit before lending or after the last release")
NO_COMMIT = fact("Commit none of the request's blocks (``try_commit_blocks``)")
COMMIT_WHEN = fact("commit before the first loan or after the last release")
REBASE = fact("A commit can rebase the request onto blocks another request committed")
LENT_PAGES_TO_POOL = fact("return the request's own pages, lent ones included, to the pool")
# The in-place lend methods point to InPlaceLender for the preconditions.
PRECONDITIONS_SEE = fact("Keep every precondition ``InPlaceLender`` lists while any loan is open")
TRANSITIONAL = fact(
    "In-place lending is a transitional capability with explicit preconditions, not a general "
    "address-stable loan"
)
IN_PLACE_PRECONDITIONS = (
    FIRST_VERSION,
    fact("guards lent pages only against the request's free and the manager's shutdown"),
    fact("While any loan is open, also after the request is freed"),
    fact("unscheduled, unsuspended, unshrunk and its window still"),
    NO_REBALANCE,
    REBALANCE_OFF,
    LOCKSTEP_RELEASE,
    fact("The lender waits on no stream"),
    fact("Own completion and the validity"),
    IN_PLACE_ROOM,
    fact("a share of each pool group for the loans alone does not prevent it"),
    fact("keeps every page it locks, not only the lent blocks'"),
    fact("Where requests wait to be admitted on the manager"),
    fact("A KV cache connector's asynchronous loads and saves run off the manager's stream"),
)
IN_PLACE_WINDOW = fact(
    "a sliding-window layer group lends only its sinks and the blocks a history of ``end`` reads"
)
# The glossary's terms, as lend_write and readiness use them.
ABANDONED = fact(
    "its write lease failed, was released before ``poll()`` returned its view, or the copy of its "
    "marked rows failed"
)
HISTORY_AFTER_GROW = fact(
    "Until the history passes where that grow left it, only the committed tokens and what was "
    "computed below the fetch's start count"
)
EMPTY_INTERVAL = fact("empty if ``max(restart_floor, context position) > usable_until``")
DOC_FACTS = {
    SH: THREADS
    + (
        fact("one call at a time"),
        fact("read views and access the memory they point to"),
        fact("they write slots during a fetch, and pages when lent in place"),
        ABANDONED,
        HISTORY_AFTER_GROW,
        EMPTY_INTERVAL,
        fact("which ``readiness`` does not read"),
        fact("Release every lease, failed ones too"),
        fact("Keep the executor's idle wait from blocking while a lease is open"),
        fact("or a KV cache connector's load is pending"),
        ROOM_TO_RESUME,
        fact("Split a fetch only by the rule in ``StagingLender.lend_write``"),
        fact("reaches the previous lease's end (not checked yet)"),
    ),
    "attach_staging": ATTACH_LIMITS
    + ATTACH_ORDER
    + SCOPE
    + (
        fact("block reuse off, or a draft manager without joint reuse"),
        fact("commits no blocks"),
        fact("Staging needs block reuse"),
        fact("changes names silently"),
        fact("keeps the staging memory and its own page-index host buffer"),
        fact("Its leases, released ones too"),
        fact("until the leases and the lender are dropped"),
        READS_AHEAD,
        NOT_IS_DRAFT,
        fact("a block's name covers only the tokens up to the block's end"),
        UNESTABLISHED_READ_AHEAD,
        fact("the draft pool's under a ``scope`` of its own"),
        fact("Pipeline-parallel managers (``mapping.pp_size > 1``) are refused"),
        KV_CONNECTOR,
        fact("lending by name stops for good once the manager resets its reuse state"),
        fact("not other inputs a request carries"),
        fact("A name trusts the multimodal digests the input pipeline computes"),
        fact("keeps them apart with per-item ``multi_modal_uuids``"),
        fact("alone keeps them apart only with that cache off"),
        fact("until a retry closes them all, or else until the process exits"),
        fact("A fetch fills only this manager's blocks"),
        # The window gap, the DeepSeek-V4 margin and unwritten pages: StagingLender states them.
        fact("Publishes and fetches have limits of their own: see ``StagingLender``"),
        fact("An adapter enters a name only through the request's ``lora_task_id``"),
        fact("with a ``mapping`` that the limits below accept"),
        fact("the limits below refuse the manager"),
    ),
    "attach_in_place": ATTACH_LIMITS
    + ATTACH_ORDER
    + (
        fact("on loan at the manager's shutdown"),
        fact("stay until the process exits"),
        fact("collected without a shutdown keeps the manager's page-index host buffer"),
        fact("until the lease is released or both it and the lender are dropped"),
        IN_PLACE_READS_AHEAD,
    ),
    "Lease": (
        THREADS_SEE,
        fact("their own channel"),
        fact("Without that signal the lease stays open"),
        fact("Poll every open lease each iteration"),
        fact("from blocking while a lease is open or a backend has work for the holder"),
        fact("also when no new request arrives"),
        fact("Release every lease, failed ones too"),
        fact("the lender runs no collective"),
        fact("Failure is final"),
        fact(
            "A lease already failed when ``lend_read`` or ``lend_write`` returns changed nothing "
            "in the request's cache"
        ),
        fact("A write lease that fails after the call has grown the cache"),
        fact("on an empty interval the request does not fetch again"),
        fact("is not recovered from"),
    ),
    "Lease.mark_arrived": (
        fact("Required for staging writes"),
        fact("split long fetches, but only by the split rule in ``StagingLender.lend_write``"),
        fact("over the host-to-device bandwidth"),
        fact("Optional in place"),
    ),
    "Lease.release": (fact("Required for every lease, failed ones too"),),
    "StagingLender": (
        THREADS_SEE,
        fact("Hold and register the parts before a backend touches a lease"),
        fact("deregister them before shutdown (see ``PartsHold``)"),
        fact("polls every open lease"),
        fact("the manager shuts down last"),
        fact("Under confidential computing staging is pageable"),
        FILL_OFF,
        fact("no lend, poll, mark or readiness call waits for them on the CPU"),
        fact("counts no pass while any request is in a disaggregated transfer state"),
        fact("at least the draft length less one token past the fetch's end"),
        fact("and ``readiness`` counts only those"),
        fact("Lending stops for good once the manager resets its reuse state"),
        fact("resume within both intervals and publish both"),
        fact("copies only the rows whose pages stayed"),
        fact("Split a fetch only by the rule in ``lend_write``"),
        fact("The lender does not check this yet"),
    )
    + STAGING_LIMITS,
    "InPlaceLender": (THREADS_SEE, fact("access the lent memory"), fact("own page-table code"))
    + IN_PLACE_PRECONDITIONS
    + (TRANSITIONAL, NO_COMMIT, COMMIT_WHEN, REBASE, LENT_PAGES_TO_POOL)
    + (fact("``lend_read`` states which blocks a lease covers"),)
    + (fact("The lender never grows the cache"), fact("its sliding windows do not advance"))
    + (fact("Copy the lent blocks' page indices while the request still has its cache"),)
    + (fact("a staging lender passes ``isinstance(lender, InPlaceLender)`` too"),),
    "PartsHold": (
        HOLD_AT_SHUTDOWN,
        fact("deregister the parts before the manager shuts down"),
        fact("release it on the manager's thread once deregistration is confirmed"),
        fact("dropping it unreleased keeps the memory"),
    ),
    "PartsHold.release": (fact("only on the manager's thread"),),
    "StagingLender.lend_read": RANK_LOCAL
    + (
        fact("one that reset its reuse state"),
        fact("multimodal data without digests or with encoder input"),
    ),
    "StagingLender.lend_write": (
        fact("Fails as ``lend_read``"),
        fact("no free pages"),
        fact("SWA scratch reuse"),
        fact("an abandoned lease drops only its own rows"),
        fact("each lease moves the history to its end at the call"),
        fact("the request may still resume below it"),
        fact("Without a sliding window the history stays where it was"),
        fact("as after a later lease that is abandoned or misses rows, the interval is empty"),
        fact("so do the other tokens the request had computed below ``start``"),
        TAKES_THE_HISTORY,
        WRITE_AHEAD_OF_DATA,
        fact("at most the whole blocks before the request's last prompt token"),
        fact("history already stands past the positions ``readiness`` would count"),
        fact("while an earlier fetch into the request has not settled on this rank"),
        SPLIT_RULE,
        SPLIT_WINDOW,
        SPLIT_UNCHECKED,
        fact("leaves behind at ``end`` a block whose page holds tokens past the cache's history"),
        fact("what the local match copied from another request's page"),
        fact("pages grown before the fetch"),
        fact("or compute from ``usable_until`` through that end"),
        DROP_BOTH,
        fact("the request may resume only at its history"),
        fact("sets ``mm_bidirectional_blocks``"),
        fact("falls strictly inside a run of multimodal tokens, whatever the run's length"),
        fact("on a request that returns context logits"),
        fact("whether or not it holds any yet"),
        fact("neither a rewind nor a recompute pause clears the rows held"),
        fact("Prompt logprobs pair the rows with the prompt's tokens from the second on"),
        fact("the lowest start of a fetch whose marked rows were copied without error"),
        fact("with no such fetch it is that history"),
    ),
    # How a fetch counts, the interval's bounds and copy timing: lend_write, Readiness and
    # StagingLender state them, and readiness points there.
    "StagingLender.readiness": (
        fact("No fetched token counts as computed until this returns"),
        fact(
            "where it covered the block the committed tokens end inside, the interval is empty "
            "from then on"
        ),
        fact("How each fetch counts: see ``lend_write``"),
        FLOOR_SEE,
        TAKES_THE_HISTORY,
        READ_AHEAD_OF_DATA,
        fact("serial with the forward"),
    ),
    # The in-place preconditions: InPlaceLender states them, and its lend methods point there.
    "InPlaceLender.lend_read": RANK_LOCAL
    + (
        IN_PLACE_WINDOW,
        fact("every block without one is left out"),
        fact("work the manager's stream queued that still writes those pages completed"),
        PRECONDITIONS_SEE,
        COMMIT_BEFORE_OR_AFTER,
    ),
    "InPlaceLender.lend_write": (
        fact("As ``lend_read``, for writing"),
        fact("The lease covers no block behind the window"),
        fact("a block it would lend that has no page its cache locks fails it at the call"),
        fact("all work the manager's stream queued for those pages has completed"),
        PRECONDITIONS_SEE,
        COMMIT_BEFORE_OR_AFTER,
    ),
    "StagingOptions": (fact("A capacity budget, not concurrency"), fact("holes")),
    "StagingLender.parts": (
        fact("registered once, after the attach and before the first lease is used"),
        fact("not that parts are back to back"),
        fact("an unreleased lease"),
        fact("an unreleased hold"),
        fact("a slot lost to a failed copy"),
    ),
    "Readiness": (
        PARK_AND_RESUME,
        RESUME_STEP,
        RESUME_FORWARD,
        fact("a resume can leave skipped positions without routes (``StagingLender``)"),
        PARKED_ACTIVE,
        fact("as the disaggregated transfer-in-progress state does, but unscheduled"),
        EMPTY_DROPS_BOTH,
        fact("a non-empty interval does not by itself permit another lease"),
        fact("the largest floor"),
        fact("do not fetch again"),
        FLOOR_AT_HISTORY,
        DRAFT_POOL_FLOOR,
        fact(
            "reaches the request's prompt length only where the request computed its whole prompt"
        ),
        fact("covers only this manager's blocks"),
        fact("the smaller ``usable_until``, the larger ``restart_floor``"),
        fact("need not be a multiple of ``tokens_per_block``"),
        fact("no position in the interval lies strictly inside a run of multimodal tokens"),
        fact("which the caller treats as empty too"),
        EMPTY_INTERVAL,
        fact("only the last two keep their order"),
    ),
    "Part": (fact("``PartsHold`` states when it deregisters"),),
    "GroupRun": (
        fact("Their format is not API"),
        fact("but their width, 54 bytes, is"),
        fact("for equality, and nothing else"),
    ),
}
FIRST_YEAR = 2026
COPYRIGHT = re.compile(
    r"# SPDX-FileCopyrightText: Copyright \(c\) (?:(20\d\d)-)?(20\d\d) NVIDIA CORPORATION & "
    r"AFFILIATES\. All rights reserved\.$"
)
LICENSE = [
    "# SPDX-License-Identifier: Apache-2.0",
    "#",
    '# Licensed under the Apache License, Version 2.0 (the "License");',
    "# you may not use this file except in compliance with the License.",
    "# You may obtain a copy of the License at",
    "#",
    "# http://www.apache.org/licenses/LICENSE-2.0",
    "#",
    "# Unless required by applicable law or agreed to in writing, software",
    '# distributed under the License is distributed on an "AS IS" BASIS,',
    "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.",
    "# See the License for the specific language governing permissions and",
    "# limitations under the License.",
]


def has_header(source: str) -> bool:
    """The NVIDIA header: a copyright year, or a range of years, ending in 2026 or later, then the
    Apache license text."""
    lines = source.splitlines()
    match = COPYRIGHT.match(lines[0]) if lines else None
    if match is None or lines[1 : 1 + len(LICENSE)] != LICENSE:
        return False
    first, last = int(match.group(1) or match.group(2)), int(match.group(2))
    return first <= last and last >= FIRST_YEAR


def package_files() -> List[Path]:
    return sorted(PKG_DIR.glob("*.py"))


def module_name(path: Path, root: Path = ROOT) -> str:
    relative = path.resolve().relative_to(root.parent).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def package_of(path: Path, root: Path = ROOT) -> str:
    """The package a file's relative imports resolve against."""
    name = module_name(path, root)
    return name if path.name == "__init__.py" else name.rpartition(".")[0]


def imported_modules(source: str, package: str) -> Set[str]:
    """Every module (and ``module.name``) a source imports, lazy imports and
    ``importlib.import_module`` with a literal included; relative imports resolved."""
    found: Set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")
                base = base[: len(base) - (node.level - 1)]
                module = ".".join(base + ([node.module] if node.module else []))
            else:
                module = node.module or ""
            found.add(module)
            found.update(f"{module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            first = node.args[0] if node.args else None
            if name in ("import_module", "__import__") and isinstance(first, ast.Constant):
                found.add(str(first.value))
    return found


PRIVATE_IMPORT = re.compile(r"^" + re.escape(SH) + r"\._")


def private_imports(source: str, package: str) -> List[str]:
    """What a source outside the package imports of the package's private modules."""
    hits = sorted(m for m in imported_modules(source, package) if PRIVATE_IMPORT.match(m))
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "kv_cache.sharing._" in node.value:
                hits.append(node.value)
    return hits


def private_importers(root: Path, pkg_dir: Path) -> dict:
    """Modules under ``root`` outside ``pkg_dir`` that import the package's private modules."""
    offenders = {}
    for path in root.rglob("*.py"):
        if pkg_dir in path.resolve().parents:
            continue
        source = path.read_text(encoding="utf-8", errors="replace")
        if "sharing" not in source:
            continue
        try:
            hits = private_imports(source, package_of(path, root))
        except SyntaxError:
            continue  # not importable, so it imports nothing
        if hits:
            offenders[path.relative_to(root).as_posix()] = hits
    return offenders


def disaggregation_imports(source: str, package: str) -> List[str]:
    prefix = "tensorrt_llm._torch.disaggregation"
    return sorted(m for m in imported_modules(source, package) if m.startswith(prefix))


def internal_reads(source: str) -> List[str]:
    """Attribute names (or ``getattr`` literals) of manager internals a source uses."""
    hits = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and node.attr in INTERNALS:
            hits.append(node.attr)
        elif isinstance(node, ast.Constant) and node.value in INTERNALS:
            hits.append(str(node.value))
    return sorted(hits)


def threads_or_finalizers(source: str) -> List[str]:
    hits = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            hits += [a.name for a in node.names if a.name.split(".")[0] in _THREADING]
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in _THREADING:
            hits.append(node.module)
        elif isinstance(node, ast.ImportFrom) and any(a.name == "finalize" for a in node.names):
            hits.append("finalize")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__del__":
            hits.append("__del__")
        elif isinstance(node, ast.Attribute) and node.attr == "finalize":
            hits.append("finalize")
    return hits


_THREADING = {"threading", "_thread", "concurrent", "multiprocessing"}


def identifiers(source: str) -> Set[str]:
    """Every name the code defines or uses: variables, attributes, functions, classes, arguments
    and imported names; comments and docstrings aside."""
    found: Set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.add(node.name)
        elif isinstance(node, ast.arg):
            found.add(node.arg)
        elif isinstance(node, ast.alias):
            found.add((node.asname or node.name).split(".")[-1])
    return found


def compute_identities(source: str) -> List[str]:
    return sorted(n for n in identifiers(source) if "compute_id" in n)


# What no file of the package holds. Which instance computed a block is the caller's: the lender
# names blocks by layout, scope and reuse key alone.
PACKAGE_SCANS = {
    "disaggregation_imports": lambda path: disaggregation_imports(
        path.read_text(), package_of(path)
    ),
    "threads_locks_or_finalizers": lambda path: threads_or_finalizers(path.read_text()),
    "compute_identities": lambda path: compute_identities(path.read_text()),
}


def run_python(code: str) -> dict:
    """Run ``code`` in a new interpreter importing this tree's ``tensorrt_llm``; it prints JSON."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT.parent), env.get("PYTHONPATH")]))
    done = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT.parent),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode == 0, done.stderr[-4000:]
    result = json.loads(done.stdout.strip().splitlines()[-1])
    # Proof the child imported the package under test, not another installed copy.
    assert Path(result["file"]).resolve() == Path(sharing.__file__).resolve()
    return result


# -- the names --------------------------------------------------------------------------------


@pytest.mark.cpu_only
def test_importing_the_package_loads_its_types_alone_and_exposes_nothing_else():
    result = run_python(
        "import json, sys\n"
        f"import {SH} as s\n"
        "print(json.dumps({'file': s.__file__, 'all': s.__all__,\n"
        "    'public': sorted(n for n in vars(s) if not n.startswith('_')),\n"
        f"    'loaded': sorted(m for m in sys.modules if m.startswith('{SH}'))}}))\n"
    )
    assert result["public"] == sorted(result["all"]) == PUBLIC
    assert result["loaded"] == [SH, f"{SH}._types"]


def subpackages(pkg_dir: Path) -> List[str]:
    """The directories under ``pkg_dir`` that could hold modules."""
    return [p.name for p in pkg_dir.iterdir() if p.is_dir() and p.name != "__pycache__"]


@pytest.mark.cpu_only
def test_the_package_is_its_init_and_six_private_modules(tmp_path):
    assert [p.name for p in package_files()] == MODULES
    assert subpackages(PKG_DIR) == []
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "planted").mkdir()
    assert subpackages(tmp_path) == ["planted"], "the listing sees no subpackage"


@pytest.mark.cpu_only
def test_the_protocols_keep_their_members_and_stay_unrelated():
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    def members(cls):
        return {n for n in vars(cls) if not n.startswith("_")}

    assert members(Lease) == {"poll", "failure", "mark_arrived", "release"}
    assert members(StagingLender) == {"parts", "hold_parts", "lend_read", "lend_write", "readiness"}
    assert members(InPlaceLender) == {"lend_read", "lend_write"}
    assert members(PartsHold) == {"release"}
    assert StagingLender not in InPlaceLender.__mro__
    assert InPlaceLender not in StagingLender.__mro__
    assert Lease not in PartsHold.__mro__ and PartsHold not in Lease.__mro__
    for protocol in (Lease, StagingLender, InPlaceLender, PartsHold):
        assert Protocol in protocol.__mro__
        assert not isinstance(object(), protocol), "checkable at run time"
    assert isinstance(Lease.failure, property) and isinstance(StagingLender.parts, property)
    # The checks are structural: a staging lender passes as an in-place one, a lease as a hold.
    lender, lease = object.__new__(_lender.Staging), object.__new__(_lender._StagingLease)
    assert isinstance(lender, InPlaceLender) and isinstance(lease, PartsHold)

    def params(func):
        return list(inspect.signature(func).parameters)

    assert params(Lease.poll) == ["self"]
    assert params(Lease.mark_arrived) == ["self", "masks"]
    assert params(Lease.release) == ["self"]
    assert params(PartsHold.release) == ["self"]
    assert params(StagingLender.hold_parts) == ["self"]
    for protocol in (StagingLender, InPlaceLender):
        assert params(protocol.lend_read) == ["self", "request", "start", "end"]
        assert params(protocol.lend_write) == ["self", "request", "start", "end"]
    assert params(StagingLender.readiness) == ["self", "request"]


def public_names(obj) -> Set[str]:
    return {n for n in dir(obj) if not n.startswith("_")}


LEASE_MEMBERS = {"poll", "failure", "mark_arrived", "release"}
STAGING_MEMBERS = {"parts", "hold_parts", "lend_read", "lend_write", "readiness"}


@pytest.mark.cpu_only
def test_lenders_leases_and_holds_show_only_their_protocol():
    from types import SimpleNamespace

    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    class Owner:  # anything a weak reference can point to
        pass

    owner = Owner()
    ref = weakref.ref(owner)
    layout = SimpleNamespace(pool_groups=(), windows=())  # all the constructors read
    shown = {
        "staging lender": (_lender.Staging(ref, layout, None, (), None, ref), STAGING_MEMBERS),
        "in-place lender": (_lender.InPlace(ref, layout), {"lend_read", "lend_write"}),
        "staging lease": (_lender._StagingLease(owner, "read", 1), LEASE_MEMBERS),
        "in-place lease": (_lender._InPlaceLease(owner, "write", None), LEASE_MEMBERS),
        "parts hold": (_lender._PartsHold(owner), {"release"}),
    }
    assert {what: public_names(obj) for what, (obj, _) in shown.items()} == {
        what: members for what, (_, members) in shown.items()
    }
    # A lease showing its view as an attribute, and a lender showing a manager hook, are found.
    leaky = _lender._StagingLease(owner, "read", 1)
    leaky.view = None
    assert public_names(leaky) - LEASE_MEMBERS == {"view"}
    hooked = type("Hooked", (_lender.InPlace,), {"on_free": _lender.InPlace._on_free})
    assert public_names(hooked(ref, layout)) - {"lend_read", "lend_write"} == {"on_free"}


@pytest.mark.cpu_only
def test_the_types_keep_their_fields():
    def fields(cls):
        """``name`` or ``name=default`` per field, in order."""
        missing = dataclasses.MISSING
        return " ".join(
            f.name if f.default is missing else f"{f.name}={f.default!r}"
            for f in dataclasses.fields(cls)
        )

    assert fields(StagingOptions) == "fetch_tokens max_fetches=1 max_bytes=None"
    assert fields(Part) == "name address nbytes slot_bytes slots"
    assert fields(GroupRun) == "layer_group ordinals names=None addresses=None part=None"
    assert fields(RegionView) == "runs"
    for cls in (StagingOptions, Part, GroupRun, RegionView):
        assert cls.__dataclass_params__.frozen, cls
    assert Part("p", 4096, 8192, 4096, 2) == Part("p", 4096, 8192, 4096, 2)
    assert Readiness._fields == ("usable_until", "restart_floor")
    assert issubclass(Readiness, tuple)

    def beyond_fields(cls):
        named = {f.name for f in dataclasses.fields(cls)}
        return {n for n in dir(cls) if not n.startswith("_")} - named

    assert beyond_fields(GroupRun) == {"select"}
    assert beyond_fields(RegionView) == {"num_rows", "row_masks"}
    assert beyond_fields(Part) == beyond_fields(StagingOptions) == set()
    assert {n for n in dir(Readiness) if not n.startswith("_")} == {
        "usable_until",
        "restart_floor",
        "count",
        "index",
    }


@pytest.mark.cpu_only
def test_the_attach_functions_keep_their_signatures():
    kind = inspect.Parameter

    def shape(func):
        return [(p.name, p.kind, p.default) for p in inspect.signature(func).parameters.values()]

    assert shape(attach_staging) == [
        ("manager", kind.POSITIONAL_OR_KEYWORD, kind.empty),
        ("scope", kind.KEYWORD_ONLY, kind.empty),
        ("staging", kind.KEYWORD_ONLY, kind.empty),
    ]
    assert shape(attach_in_place) == [("manager", kind.POSITIONAL_OR_KEYWORD, kind.empty)]


@pytest.mark.cpu_only
def test_a_name_is_as_long_as_the_identity_makes_it():
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _identity, _types

    assert _types._NAME_BYTES == _identity.NAME_BYTES == 54


def public_lending_names(cls) -> List[str]:
    """The public names of ``cls`` about lending."""
    words = re.compile(r"lend|lent|loan|sharing|staging|retain")
    return [n for n in dir(cls) if not n.startswith("_") and words.search(n)]


@pytest.mark.cpu_only
def test_the_manager_has_no_public_name_about_lending():
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2

    assert "_sharing" in vars(KVCacheManagerV2) and KVCacheManagerV2._sharing is None
    assert public_lending_names(KVCacheManagerV2) == []
    planted = type("Planted", (KVCacheManagerV2,), {"lend_read": None, "staging_part": None})
    assert public_lending_names(planted) == ["lend_read", "staging_part"], "the scan sees nothing"


class OwnBeamsOnly:
    """A runtime cache stand-in whose page-index table has entries for its own beams only, as
    the native one does; ``set_base_page_index_buf`` past them raises instead of writing out of
    bounds."""

    def __init__(self, beam_width):
        self.beam_width = beam_width
        self.detached = []

    def set_base_page_index_buf(self, beam, pool, buf):
        if not 0 <= int(beam) < self.beam_width:
            raise IndexError(f"beam {int(beam)} past the cache's width {self.beam_width}")
        self.detached.append((int(beam), pool, buf))


@pytest.mark.cpu_only
@pytest.mark.parametrize("beam_width", [1, 2])
def test_a_lent_cache_is_detached_beam_by_beam_up_to_its_own_width(beam_width):
    """Under beam search a context request's cache has one beam while the manager allows more."""
    from types import SimpleNamespace

    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2

    cache = OwnBeamsOnly(beam_width)
    KVCacheManagerV2._detach_index_buffer(SimpleNamespace(max_beam_width=2, num_pools=2), cache)
    assert cache.detached == [(b, p, None) for b in range(beam_width) for p in range(2)]


# -- boundaries -------------------------------------------------------------------------------


@pytest.mark.cpu_only
def test_the_scans_catch_every_import_form_they_look_for():
    package = "tensorrt_llm._torch.pyexecutor.kv_cache"
    caught = [
        f"from {SH}._lender import attach_staging",
        f"from {SH} import _types",
        f"import {SH}._slots",
        "from .sharing._identity import Identity",
        "from .sharing import _layout",
        f"import importlib\nimportlib.import_module('{SH}._manager')",
        "def later():\n    from .sharing._lender import Staging\n",
    ]
    for source in caught:
        assert private_imports(source, package), source
    for source in (f"from {SH} import GroupRun", "from .sharing import attach_staging"):
        assert private_imports(source, package) == [], source
    assert disaggregation_imports("from ....disaggregation.resource.page import MapperKind", SH)
    assert disaggregation_imports("def f():\n    import tensorrt_llm._torch.disaggregation\n", SH)
    assert disaggregation_imports("from .._layout import x", SH) == []
    assert internal_reads("def f(m):\n    return m._stream, getattr(m, 'kv_cache_map')\n") == [
        "_stream",
        "kv_cache_map",
    ]
    assert threads_or_finalizers("import threading\n") == ["threading"]
    assert threads_or_finalizers("class A:\n    def __del__(self):\n        pass\n") == ["__del__"]
    assert threads_or_finalizers("import weakref\nweakref.finalize(o, f)\n") == ["finalize"]
    planted = '"""compute_id in prose is fine."""\ndef name(key, compute_id):\n    return key\n'
    assert compute_identities(planted) == ["compute_id"]


@pytest.fixture
def tree_with_a_private_importer(tmp_path):
    """A ``tensorrt_llm`` tree whose package imports its own private module, as it may, and one
    module outside it that imports a private module too, as none may."""
    root = tmp_path / "tensorrt_llm"
    pkg_dir = root / "_torch/pyexecutor/kv_cache/sharing"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text("from ._types import Part\n")
    (pkg_dir / "_lender.py").write_text("from ._types import Part\n")
    (root / "_torch/pyexecutor/public_user.py").write_text(
        "from .kv_cache.sharing import attach_staging\n"
    )
    (root / "_torch/pyexecutor/private_user.py").write_text(
        "def build():\n    from .kv_cache.sharing._lender import Staging\n"
    )
    return root.resolve(), pkg_dir.resolve()


@pytest.mark.cpu_only
def test_the_boundary_scan_catches_a_module_importing_a_private_one(tree_with_a_private_importer):
    root, pkg_dir = tree_with_a_private_importer
    assert private_importers(root, pkg_dir) == {
        "_torch/pyexecutor/private_user.py": [f"{SH}._lender", f"{SH}._lender.Staging"]
    }


@pytest.mark.cpu_only
def test_nothing_outside_the_package_imports_its_private_modules():
    assert private_importers(ROOT, PKG_DIR) == {}


@pytest.mark.cpu_only
@pytest.mark.parametrize("scan", list(PACKAGE_SCANS))
def test_the_package_holds_nothing_a_scan_looks_for(scan):
    hits = {path.name: PACKAGE_SCANS[scan](path) for path in package_files()}
    assert {name: found for name, found in hits.items() if found} == {}


@pytest.mark.cpu_only
def test_the_manager_never_imports_the_package():
    path = ROOT / "_torch/pyexecutor/kv_cache/kv_cache_manager_v2.py"
    source = path.read_text()
    loaded = imported_modules(source, package_of(path))
    assert [m for m in loaded if m.startswith(SH)] == []
    planted = source + "\nfrom .sharing import attach_in_place\n"
    assert [m for m in imported_modules(planted, package_of(path)) if m.startswith(SH)]


@pytest.mark.cpu_only
def test_the_native_path_loads_nothing_of_the_package():
    result = run_python(
        "import json, sys\n"
        "import tensorrt_llm._torch.disaggregation.transceiver\n"
        f"import {MGR}\n"
        f"loaded = sorted(m for m in sys.modules if m.startswith('{SH}'))\n"
        f"import {SH} as s\n"
        "print(json.dumps({'file': s.__file__, 'loaded': loaded}))\n"
    )
    assert result["loaded"] == []


@pytest.mark.cpu_only
def test_only_the_manager_facade_touches_manager_internals():
    offenders = {}
    for path in package_files():
        if path.name == "_manager.py":
            continue
        hits = internal_reads(path.read_text())
        if hits:
            offenders[path.name] = hits
    assert offenders == {}


@pytest.mark.cpu_only
def test_the_context_output_check_reads_the_request_alone():
    """The fetch's context-output check reads two request attributes and never the executor's
    result, whose additional outputs concatenate their chunks at each read."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _manager

    class Unread:
        def __getattr__(self, name):
            raise AssertionError(f"read py_result.{name}")

    class Request:
        py_result = Unread()

        def __init__(self, logits, outputs):
            self.py_return_context_logits, self.py_additional_outputs = logits, outputs

    cases = [(False, None, False), (False, [], False), (True, None, True), (False, ["x"], True)]
    for logits, outputs, returns in cases:
        assert _manager.returns_context_outputs(Request(logits, outputs)) is returns


# -- public docstrings and headers ------------------------------------------------------------


def public_docstrings():
    """(where, docstring) for the package, each exported name and each of their public members."""
    yield SH, sharing.__doc__ or ""
    for name in PUBLIC:
        obj = getattr(sharing, name)
        yield name, obj.__doc__ or ""
        if not inspect.isclass(obj):
            continue
        for member, value in vars(obj).items():
            if member.startswith("_"):
                continue
            if isinstance(value, (staticmethod, classmethod)):
                value = value.__func__
            doc = getattr(value, "__doc__", None)
            if doc and (inspect.isfunction(value) or isinstance(value, property)):
                yield f"{name}.{member}", doc


def backend_words(docstrings) -> List[str]:
    return [where for where, doc in docstrings if BACKEND_WORDS.search(doc)]


@pytest.mark.cpu_only
def test_public_docstrings_name_no_backend(monkeypatch):
    documented = list(public_docstrings())
    assert {where for where, _ in documented} >= {"attach_staging", "Lease.poll", "Part"}
    assert backend_words(documented) == []
    # A planted word in one member's docstring is found.
    doc = StagingLender.lend_read.__doc__
    monkeypatch.setattr(StagingLender.lend_read, "__doc__", doc + " Suits a Mooncake store.")
    assert backend_words(public_docstrings()) == ["StagingLender.lend_read"]


def missing_facts(docstrings) -> dict:
    """Per documented name of ``DOC_FACTS``, the facts its docstring does not state."""
    docs = dict(docstrings)
    missing = {}
    for where, facts in DOC_FACTS.items():
        lacking = [f.pattern for f in facts if not f.search(docs.get(where, ""))]
        if lacking:
            missing[where] = lacking
    return missing


GOOGLE_STYLE_READINESS = """Where the request may resume once its fetch settled.

    No fetched token
    counts as computed until this returns a ``Readiness``. How each fetch counts: see
    ``lend_write``. A failed copy counts none of its rows; where it covered the block the committed
    tokens end inside, the interval is empty from then on. It takes the tokens below the cache's
    history as computed, so not while its history runs ahead of its data.

    Args:
        request: The request a fetch went into.

    Returns:
        ``None`` while a fetch into the request is unsettled. Copies queue on the manager's stream,
        serial with the forward
        on the GPU. The interval's bounds, such
        as the floor at the history, and how to combine ranks and pools: see
        ``Readiness``.
    """


@pytest.mark.cpu_only
def test_public_docstrings_state_what_callers_rely_on(monkeypatch):
    assert missing_facts(public_docstrings()) == {}
    # A Google-style docstring stating the same facts across its sections passes.
    monkeypatch.setattr(StagingLender.readiness, "__doc__", GOOGLE_STYLE_READINESS)
    assert missing_facts(public_docstrings()) == {}
    # Docstrings that drop a fact are found, and a hold whose release reads as freeing.
    monkeypatch.setattr(InPlaceLender, "__doc__", "Lends a request's own device pages.")
    monkeypatch.setattr(StagingLender.lend_read, "__doc__", "A copy of the committed blocks.")
    monkeypatch.setattr(
        StagingLender.readiness, "__doc__", GOOGLE_STYLE_READINESS.replace("serial", "parallel")
    )
    hold_doc = " ".join(PartsHold.__doc__.split())
    frees = HOLD_AT_SHUTDOWN.sub("Releasing a hold frees the staging memory", hold_doc)
    monkeypatch.setattr(PartsHold, "__doc__", frees)
    missing = missing_facts(public_docstrings())
    assert sorted(missing) == [
        "InPlaceLender",
        "PartsHold",
        "StagingLender.lend_read",
        "StagingLender.readiness",
    ]
    assert missing["PartsHold"] == [HOLD_AT_SHUTDOWN.pattern]
    assert missing["StagingLender.readiness"] == [fact("serial with the forward").pattern]


def without(doc: str, clause: str) -> str:
    """``doc`` with whitespace collapsed and ``clause``, which it states once, taken out."""
    text = " ".join(doc.split())
    assert text.count(clause) == 1, clause
    return text.replace(clause, "")


KEPT_ON_WRITE = (
    "The committed tokens still count, and so do the other tokens the request had computed below "
    "``start``; past ``start`` the fetch may overwrite them. "
)
# readiness points to lend_write for what a fetch keeps.
KEPT_ON_READINESS = "How each fetch counts: see ``lend_write``. "


@pytest.mark.cpu_only
def test_a_docstring_dropping_what_a_fetch_keeps_is_found(monkeypatch):
    # Each dropped alone from otherwise unchanged docstrings is found, and nothing else: the
    # statement in lend_write and the pointer to it in readiness.
    no_write = without(StagingLender.lend_write.__doc__, KEPT_ON_WRITE)
    assert "an abandoned lease drops only its own rows" in no_write
    monkeypatch.setattr(StagingLender.lend_write, "__doc__", no_write)
    no_readiness = without(StagingLender.readiness.__doc__, KEPT_ON_READINESS)
    assert "the fetch is abandoned" in no_readiness
    monkeypatch.setattr(StagingLender.readiness, "__doc__", no_readiness)
    assert missing_facts(public_docstrings()) == {
        "StagingLender.lend_write": [
            fact("so do the other tokens the request had computed below ``start``").pattern
        ],
        "StagingLender.readiness": [fact("How each fetch counts: see ``lend_write``").pattern],
    }


FLOOR_ON_TYPE = (
    "Under a block reuse policy other than all-reusable the floor is also at least the request's "
    "history, since the manager's context update never moves a history back. "
)
# readiness points to Readiness for the floor.
FLOOR_ON_READINESS = (
    "The interval's bounds, such as the floor at the history, and how to combine ranks and pools: "
    "see ``Readiness``. "
)


@pytest.mark.cpu_only
def test_a_docstring_dropping_the_history_floor_is_found(monkeypatch):
    # Each dropped alone from otherwise unchanged docstrings is found, and nothing else.
    no_type_floor = without(Readiness.__doc__, FLOOR_ON_TYPE)
    assert "the largest floor" in no_type_floor
    monkeypatch.setattr(Readiness, "__doc__", no_type_floor)
    no_readiness_floor = without(StagingLender.readiness.__doc__, FLOOR_ON_READINESS)
    assert "How each fetch counts" in no_readiness_floor
    monkeypatch.setattr(StagingLender.readiness, "__doc__", no_readiness_floor)
    assert missing_facts(public_docstrings()) == {
        "Readiness": [FLOOR_AT_HISTORY.pattern],
        "StagingLender.readiness": [FLOOR_SEE.pattern],
    }


HISTORY_ON_WRITE = (
    "Fetch only into a request whose pages hold what its history covers, since the lender takes "
    "the tokens below the cache's history as computed: not one whose history runs ahead of its "
    "data, such as a disaggregated generation request before its transfer lands. "
)
HISTORY_ON_READINESS = (
    "Like ``lend_write``, it takes the tokens below the cache's history as computed, apart from "
    "those a windowed fetch moved the history over; the caller asks only while the request's "
    "pages hold the rest, so not while its history runs ahead of its data. "
)


@pytest.mark.cpu_only
def test_a_docstring_dropping_the_history_taken_as_computed_is_found(monkeypatch):
    # Each dropped alone from otherwise unchanged docstrings is found, and nothing else.
    no_write = without(StagingLender.lend_write.__doc__, HISTORY_ON_WRITE)
    assert "an abandoned lease drops only its own rows" in no_write
    monkeypatch.setattr(StagingLender.lend_write, "__doc__", no_write)
    no_readiness = without(StagingLender.readiness.__doc__, HISTORY_ON_READINESS)
    assert "No fetched token counts as computed" in no_readiness
    monkeypatch.setattr(StagingLender.readiness, "__doc__", no_readiness)
    assert missing_facts(public_docstrings()) == {
        "StagingLender.lend_write": [TAKES_THE_HISTORY.pattern, WRITE_AHEAD_OF_DATA.pattern],
        "StagingLender.readiness": [TAKES_THE_HISTORY.pattern, READ_AHEAD_OF_DATA.pattern],
    }


@pytest.mark.cpu_only
def test_a_docstring_dropping_one_obligation_is_found(monkeypatch):
    # The in-place caller's pool and prefix-cache precondition, the K/V arrangement scope covers,
    # and the staging lender's prefix-hashing limit, each dropped alone from otherwise unchanged
    # docstrings, are found and nothing else.
    no_rebalance = without(
        InPlaceLender.__doc__,
        "- Neither rebalance the pools nor reset the prefix cache: "
        "``kv_cache_config.enable_kv_pool_rebalance`` stays off, since the executor's pool "
        "rebalance moves lent pages and, while a cache stays on loan after its request's free, "
        "raises out of the executor loop. ",
    )
    assert "rebalance" not in no_rebalance and "own page-table code" in no_rebalance
    monkeypatch.setattr(InPlaceLender, "__doc__", no_rebalance)
    arrangement = attach_staging.__doc__[attach_staging.__doc__.index("the attention backend's") :]
    arrangement = " ".join(arrangement[: arrangement.index("),") + 3].split())
    no_arrangement = without(attach_staging.__doc__, arrangement)
    assert "head-major" not in no_arrangement and "LoRA adapters" in no_arrangement
    monkeypatch.setattr(attach_staging, "__doc__", no_arrangement)
    no_hashing = without(
        StagingLender.__doc__,
        "- Every lease hashes the request's whole prefix again from block 0, so a lease late in a "
        "long prompt costs time in proportion to the prompt. ",
    )
    assert "hashes" not in no_hashing and "is never reused" in no_hashing
    monkeypatch.setattr(StagingLender, "__doc__", no_hashing)
    assert missing_facts(public_docstrings()) == {
        "InPlaceLender": [NO_REBALANCE.pattern, REBALANCE_OFF.pattern],
        "StagingLender": [PREFIX_HASH.pattern],
        "attach_staging": [KV_ARRANGEMENT.pattern],
    }


IN_PLACE_COMMIT = (
    "- Commit none of the request's blocks (``try_commit_blocks``): commit before the first loan "
    "or after the last release. A commit can rebase the request onto blocks another request "
    "committed and return the request's own pages, lent ones included, to the pool. "
)
IN_PLACE_TRANSITIONAL = (
    "In-place lending is a transitional capability with explicit preconditions, not a general "
    "address-stable loan: a lent page keeps its address only while the caller keeps them. "
)
# StagingLender points to attach_staging, which states the read-ahead limit among the refusals.
STAGING_LIMIT = "``attach_staging`` lists the refused managers. "
IN_PLACE_ACCEPTS = (
    ", and a manager built for a one-model draft that reads prompt tokens ahead is accepted, "
    "since in-place views carry no names"
)
# The lend methods point to InPlaceLender's preconditions and name the commit rule among them.
READ_COMMIT = WRITE_COMMIT = "; in particular, commit before lending or after the last release"


def staging_lookahead_limit() -> str:
    """The bullet of ``attach_staging`` refusing a draft that reads ahead, whitespace collapsed."""
    text = " ".join(attach_staging.__doc__.split())
    start = text.index("- A manager built for a one-model draft")
    return text[start : text.index("- Helix", start)]


@pytest.mark.cpu_only
def test_a_docstring_dropping_the_read_ahead_limit_is_found(monkeypatch):
    # Each dropped alone from otherwise unchanged docstrings is found, and nothing else.
    no_limit = without(attach_staging.__doc__, staging_lookahead_limit())
    assert "Helix" in no_limit and "is_draft" not in no_limit
    monkeypatch.setattr(attach_staging, "__doc__", no_limit)
    no_staging_limit = without(StagingLender.__doc__, STAGING_LIMIT)
    assert "Lending stops for good" in no_staging_limit
    monkeypatch.setattr(StagingLender, "__doc__", no_staging_limit)
    no_acceptance = without(attach_in_place.__doc__, IN_PLACE_ACCEPTS)
    assert "Block reuse may be on or off" in no_acceptance
    monkeypatch.setattr(attach_in_place, "__doc__", no_acceptance)
    assert missing_facts(public_docstrings()) == {
        "StagingLender": [STAGING_REFUSALS.pattern],
        "attach_in_place": [IN_PLACE_READS_AHEAD.pattern],
        "attach_staging": [
            READS_AHEAD.pattern,
            NOT_IS_DRAFT.pattern,
            fact("a block's name covers only the tokens up to the block's end").pattern,
            UNESTABLISHED_READ_AHEAD.pattern,
        ],
    }


@pytest.mark.cpu_only
def test_a_docstring_dropping_the_in_place_commit_rule_is_found(monkeypatch):
    # Each dropped alone from otherwise unchanged docstrings is found, and nothing else.
    no_commit = without(InPlaceLender.__doc__, IN_PLACE_COMMIT)
    assert "try_commit_blocks" not in no_commit and "own page-table code" in no_commit
    monkeypatch.setattr(InPlaceLender, "__doc__", without(no_commit, IN_PLACE_TRANSITIONAL))
    no_read_commit = without(InPlaceLender.lend_read.__doc__, READ_COMMIT)
    assert "Keep every precondition" in no_read_commit
    monkeypatch.setattr(InPlaceLender.lend_read, "__doc__", no_read_commit)
    no_write_commit = without(InPlaceLender.lend_write.__doc__, WRITE_COMMIT)
    assert "covers no block behind the window" in no_write_commit
    monkeypatch.setattr(InPlaceLender.lend_write, "__doc__", no_write_commit)
    assert missing_facts(public_docstrings()) == {
        "InPlaceLender": [
            TRANSITIONAL.pattern,
            NO_COMMIT.pattern,
            COMMIT_WHEN.pattern,
            REBASE.pattern,
            LENT_PAGES_TO_POOL.pattern,
        ],
        "InPlaceLender.lend_read": [COMMIT_BEFORE_OR_AFTER.pattern],
        "InPlaceLender.lend_write": [COMMIT_BEFORE_OR_AFTER.pattern],
    }


def written_files() -> List[Path]:
    return package_files() + sorted(TESTS_DIR.glob("*.py"))


@pytest.mark.cpu_only
def test_every_new_file_carries_the_license_header():
    assert [p.name for p in written_files() if not has_header(p.read_text())] == []
    body = "\n".join(LICENSE)

    def header(years: str) -> str:
        owner = "NVIDIA CORPORATION & AFFILIATES. All rights reserved."
        return f"# SPDX-FileCopyrightText: Copyright (c) {years} {owner}\n{body}"

    for years in ("2026", "2027", "2025-2026", "2026-2027", "2031"):
        assert has_header(header(years)), years
    for years in ("2025", "2024-2025", "2027-2026"):
        assert not has_header(header(years)), years
    assert not has_header("# SPDX-FileCopyrightText: Copyright (c) 2026 Someone else.\n" + body)
    assert not has_header(header("2026").splitlines()[0] + "\n# SPDX-License-Identifier: MIT\n")


# -- the public types' own rules --------------------------------------------------------------


def staging_run(n=3, layer_group=0, part=0):
    return GroupRun(
        layer_group,
        np.arange(n),
        np.zeros((n, 54), np.uint8),
        np.arange(n, dtype=np.int64) * 4096,
        part,
    )


@pytest.mark.cpu_only
def test_a_group_run_carries_all_three_placements_or_none_in_their_shapes():
    ordinals = np.arange(3)
    names = np.zeros((3, 54), np.uint8)
    addresses = np.arange(3, dtype=np.int64)
    refused = [
        (ValueError, (0, ordinals, names, None, None)),
        (ValueError, (0, ordinals, None, addresses, 0)),
        (ValueError, (0, ordinals, names, addresses, None)),
        (ValueError, (0, np.zeros((3, 1)))),
        (ValueError, (0, ordinals, np.zeros((3, 53), np.uint8), addresses, 0)),
        (ValueError, (0, ordinals, np.zeros((3, 54), np.int8), addresses, 0)),
        (ValueError, (0, ordinals, names, addresses[:2], 0)),
        (ValueError, (-1, ordinals)),
        (ValueError, (0, ordinals, names, addresses, -1)),
        (TypeError, (True, ordinals)),
        (TypeError, (0, ordinals, names, addresses, 1.0)),
    ]
    for error, args in refused:
        with pytest.raises(error):
            GroupRun(*args)
    in_place = GroupRun(1, ordinals)
    assert (in_place.names, in_place.addresses, in_place.part) == (None, None, None)
    assert len(in_place) == 3 and in_place.ordinals.dtype == np.int64


@pytest.mark.cpu_only
def test_a_group_run_s_arrays_are_read_only_views():
    ordinals = np.arange(3, dtype=np.int64)
    run = GroupRun(0, ordinals, np.zeros((3, 54), np.uint8), np.arange(3, dtype=np.int64), 0)
    for array in (run.ordinals, run.names, run.addresses):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array[0] = 1
    assert ordinals.flags.writeable, "the caller's own array is left as it was"


@pytest.mark.cpu_only
def test_select_keeps_the_marked_rows_in_order():
    run = staging_run(4, layer_group=2, part=1)
    picked = run.select(np.array([True, False, True, True]))
    assert picked.ordinals.tolist() == [0, 2, 3]
    assert picked.addresses.tolist() == [0, 8192, 12288]
    assert picked.names.shape == (3, 54) and (picked.layer_group, picked.part) == (2, 1)
    assert GroupRun(0, np.arange(2)).select(np.array([False, True])).names is None
    with pytest.raises(ValueError):
        run.select(np.array([True, False]))
    with pytest.raises(ValueError):
        run.select(np.array([1, 0, 1, 1]))


@pytest.mark.cpu_only
def test_a_region_view_holds_one_run_per_layer_group():
    view = RegionView([staging_run(3, 0), staging_run(2, 1, part=1)])
    assert isinstance(view.runs, tuple) and view.num_rows == 5
    masks = view.row_masks()
    assert [m.shape for m in masks] == [(3,), (2,)]
    assert all(m.dtype == np.bool_ and not m.any() and m.flags.writeable for m in masks)
    assert all(m.all() for m in view.row_masks(True))
    assert RegionView(()).num_rows == 0 and RegionView(()).row_masks() == ()
    with pytest.raises(ValueError):
        RegionView((staging_run(3, 0), staging_run(1, 0)))
    with pytest.raises(TypeError):
        RegionView((object(),))


@pytest.mark.cpu_only
def test_staging_options_take_positive_integers():
    options = StagingOptions(256)
    assert (options.fetch_tokens, options.max_fetches, options.max_bytes) == (256, 1, None)
    assert StagingOptions(256, max_fetches=4, max_bytes=1 << 30).max_bytes == 1 << 30
    for bad in (dict(fetch_tokens=0), dict(fetch_tokens=8, max_fetches=0)):
        with pytest.raises(ValueError):
            StagingOptions(**bad)
    with pytest.raises(ValueError):
        StagingOptions(8, max_bytes=0)
    for bad in (
        dict(fetch_tokens=True),
        dict(fetch_tokens=8.0),
        dict(fetch_tokens=8, max_bytes=1.5),
    ):
        with pytest.raises(TypeError):
            StagingOptions(**bad)
    with pytest.raises(dataclasses.FrozenInstanceError):
        options.max_fetches = 2

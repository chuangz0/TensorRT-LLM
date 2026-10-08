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
"""Lending a KV cache manager v2's blocks to transfer backends.

Callers import only these names; the other modules are private, and the attach functions load the
implementation. Each obligation and limit is stated once, where it belongs. Positions are tokens.

Glossary:
    block: ``tokens_per_block`` tokens of a layer group; block ``b`` covers
        ``[b * tokens_per_block, (b + 1) * tokens_per_block)``.
    row: A block of one layer group in a lent view (``GroupRun``).
    page: The memory of one block of a request's cache in one pool group. It is device memory,
        except that a held page may sit on another tier.
    slot: Host staging memory for one row; a part is one pool group's slots.
    locked page: A page the request's active cache locks.
    held page: A page a block keeps without that lock, maybe on another tier; a window block behind
        the history keeps at most one.
    committed tokens: Prompt tokens in the prefix-reuse tree, reused then computed: committed after
        each context chunk (only the last one under a reuse policy other than all-reusable), never
        past the context's end. Generated tokens never commit.
    history: The cache's history length. Tokens below it count as computed, except after a fetch
        whose grow left the history past the committed tokens. Until the history passes where that
        grow left it, only the committed tokens and what was computed below the fetch's start
        count (``StagingLender.lend_write``). A sliding window keeps only the blocks such a history
        reads.
    publish: ``lend_read``: committed blocks copied into slots, to store.
    fetch: ``lend_write``: the cache grown and empty slots lent, to fill; rows marked arrived are
        then copied into the cache.
    lease: One lent range (``Lease``); in place, a loan.
    abandoned: Said of a fetch that delivered nothing: its write lease failed, was released before
        ``poll()`` returned its view, or the copy of its marked rows failed.
    delivered rows: Marked rows whose copy into the request's pages completed.
    settled: Said of a fetch once ``readiness`` returns a ``Readiness``.
    parked request: Active but unscheduled, from its ``lend_write`` until it resumes
        (``Readiness``).
    readiness interval: ``[restart_floor, usable_until]`` (``Readiness``), and never below the
        request's context position, which ``readiness`` does not read. It is empty if
        ``max(restart_floor, context position) > usable_until``.
    deadlock check: Unless a request is in a disaggregated transfer state or a KV cache connector's
        load is pending, the V2 scheduler raises "V2 scheduler deadlock" out of the executor loop
        after 1000 passes in a row that schedule and reclaim nothing while a context or generation
        request waits.
    room: A pool group has room when these fit within ``max_util_for_resume`` of its GPU pages:

        - the pages lent or parked caches lock,
        - the other pages the V2 scheduler cannot reclaim,
        - for the neediest generation request, its cache's pages within its windows, which a resume
          locks all at once, and the pages its next step adds.

        While the request still has its cache, ``get_page_indices_by_layer_group`` lists its
        pages per layer group for one beam; each further beam locks its own page per block not
        wholly inside the prompt. ``impl.pool_group_descs`` give each pool group's GPU pages and
        layer groups.

Roles:
    integrator: Attaches one lender to the manager the executor serves.
    holder: Lends, polls, marks arrivals and releases.
    backend: Moves the bytes.
    waiter: Asks ``readiness`` and resumes parked requests.

Threads:
    Lender, lease and hold methods run only on the manager's thread, one call at a time: the
    builder, then the executor loop, then the shutdown thread. The lender takes no locks, binds no
    owner thread and starts no thread, so all progress happens inside lender calls. Backends' own
    threads read views and access the memory they point to: they write slots during a fetch, and
    pages when lent in place. They call no lease or lender method, and tell the holder through
    their own channel when done.

Caller checklist for staging:
    1. Attach (``attach_staging``) on the thread that builds the executor.
    2. Each backend takes ``StagingLender.hold_parts()``, then registers ``StagingLender.parts``.
    3. Publish: ``lend_read``, poll, store the rows by name, release.
    4. Fetch: ``lend_write``, park the request, poll, fill, ``mark_arrived``, release. Split a fetch
       only by the rule in ``StagingLender.lend_write``: the next lease once ``readiness`` is not
       None on every rank and, with a sliding window, its ``usable_until`` reaches the previous
       lease's end. A windowed lease that breaks it can fail at the call.
    5. Each executor iteration, poll every open lease and ask ``readiness`` for every parked
       request; combine ranks (``Readiness``).
    6. Resume within the interval; on an empty one, drop the cache in every manager and compute from
       0 (``Readiness``).
    7. Release every lease, failed ones too, once its backend let go of the memory (``Lease``).
    8. Keep the executor's idle wait from blocking while a lease is open or a backend has work for
       the holder (``Lease``).
    9. Shut down in the order ``StagingLender`` gives, the manager last.
"""

import typing as _typing

from ._types import (
    GroupRun,
    InPlaceLender,
    Lease,
    Part,
    PartsHold,
    Readiness,
    RegionView,
    StagingLender,
    StagingOptions,
)

if _typing.TYPE_CHECKING:
    from ..kv_cache_manager_v2 import KVCacheManagerV2

__all__ = [
    "GroupRun",
    "InPlaceLender",
    "Lease",
    "Part",
    "PartsHold",
    "Readiness",
    "RegionView",
    "StagingLender",
    "StagingOptions",
    "attach_in_place",
    "attach_staging",
]


def attach_staging(
    manager: "KVCacheManagerV2", *, scope: bytes, staging: StagingOptions
) -> StagingLender:
    """Attach the manager's one lender, which relays whole blocks through host staging.

    It copies blocks between device pages and host slots on the manager's stream, for backends that
    reach only host memory. ``StagingLender`` holds the rules of lending.

    Args:
        manager: A ``KVCacheManagerV2`` with a ``mapping`` that the limits below accept. Subclasses
            are accepted, but one whose ``try_commit_blocks`` keys blocks by other tokens or reuse
            scope changes names silently, at worst putting wrong bytes under a correct name.
        scope: Equal exactly where KV bytes mean the same; at most 65535 bytes. The lender only puts
            it into every name's namespace, so different scopes share no name. The framework's
            assembly builds it from its configuration, so no backend guesses which settings it
            needs. It covers every configuration that changes a block's bytes beyond what the
            manager declares:

            - the model, its weights and LoRA adapters (the adapter an id maps to),
            - the one-model draft whose pool it is (weights, mode, target layers read),
            - element types and quantization,
            - the attention (windows such as ``max_attention_window``, sinks, sparse attention),
            - the attention backend's K/V arrangement (head-major or token-major; names follow the
              declared one, and the base manager declares head-major whichever backend writes),
            - a release that changes what KV bytes mean.

            An adapter enters a name only through the request's ``lora_task_id``: instances sharing
            a ``scope`` map each id to the same adapter, lend no LoRA request by name, or take
            different scopes.
        staging: The staging size, in whole fetches.

    Returns:
        The lender, installed on ``manager`` for the manager's life.

    Raises:
        TypeError: ``manager`` is not a ``KVCacheManagerV2`` or has no mapping; ``scope`` is not
            ``bytes``; ``staging`` is not ``StagingOptions``.
        ValueError: A lender is attached already; the limits below refuse the manager; its layer
            groups, layer configs and declarations disagree; it has no pages to stage; ``scope``
            exceeds 65535 bytes; ``staging.max_bytes`` is below one fetch.
        MemoryError: Page-locking the staging memory failed.

    Caller must:
        - Attach on the thread that builds the executor, to the manager the executor serves.
        - Shut the manager down last, after the steps ``StagingLender`` lists. Once leases and
          lender are unused, drop the last reference on a thread that has used the device's CUDA
          context.
        - For a one-model draft with its own joint-reuse pool, attach a lender to each manager, the
          draft pool's under a ``scope`` of its own (alike layouts name blocks alike). A fetch fills
          only this manager's blocks; ``StagingLender`` says how to use the pair.

    Memory:
        - Shutdown frees the staging memory except as ``StagingLender.parts`` lists; a lender on a
          manager built only to estimate the KV cache size stops serving at that early shutdown.
        - A manager that never shuts down keeps the staging memory and its own page-index host
          buffer until the process exits. A shutdown raising while closing caches keeps that buffer
          until a retry closes them all, or else until the process exits.
        - Its leases, released ones too, and the lender's records keep the requests' caches and
          device pools until the leases and the lender are dropped.
        - An attach raising for an unlisted reason may keep its staging memory until exit.

    First-version limits:
        - One lender per manager, for the manager's life: staging and in-place cannot serve one
          manager together.
        - Staging needs block reuse: a publish lends only committed blocks, so a manager that
          commits no blocks (block reuse off, or a draft manager without joint reuse) is refused.
        - A manager built for a one-model draft that reads prompt tokens past a position is refused
          (one-model Eagle and MTP-Eagle read 1, vanilla MTP its draft length): a block's name
          covers only the tokens up to the block's end, so draft bytes could differ under one name.
          The check reads the manager, not ``is_draft``: it refuses draft layers in the target's
          manager, a joint-reuse draft pool, and the target's manager of a draft living elsewhere,
          and accepts drafts reading nothing ahead (PARD, DFlash, DSpark) and managers without
          speculative decoding. A draft whose read-ahead upstream has not established counts as
          reading none: for one-model DraftTarget the target's manager is accepted and its unpaired
          draft pool, committing nothing, refused, so a fetch fills only the target, as prefix reuse
          does.
        - Helix and every other context-parallel manager (``mapping.cp_size > 1``) are refused: the
          layout derives only tensor-parallel shards, so ranks would name different pages alike.
        - Pipeline-parallel managers (``mapping.pp_size > 1``) are refused: with per-stage slots and
          windows, one call could raise on one stage and lend on another.
        - Layer groups holding recurrent state are refused, as are sparse buffers (``is_sparse``),
          whose read-only pages a cache can lock in host memory; sparse attention without buffers
          marked sparse is not.
        - A manager with a KV cache connector is refused: serving a prefix from the committed tokens
          at the first chunk, it cannot lower a history a windowed fetch moved, and its loads run
          off the manager's stream, unordered with staging copies.
        - A name covers the committed tokens, their multimodal digests and the reuse scope, not
          other inputs a request carries: requests with multimodal data but no digests, or with
          encoder input (an encoder-decoder model's decoder self-attention pool), are not lent by
          name.
        - A name trusts the multimodal digests the input pipeline computes, as prefix reuse does. A
          digest covers an item's bytes, not its processing: an equal NumPy array and
          ``torch.Tensor`` share one though some processors rescale only one, and
          ``mm_processor_kwargs`` are not digested, so different KV bytes can share a name. For
          requests the processor may treat differently, the caller keeps them apart with per-item
          ``multi_modal_uuids``, which names and the multimodal encoder cache both cover. A
          ``cache_salt`` covers names but not that cache, which some models keep by default and
          which keys embeddings by digest and ``mm_processor_kwargs``: a ``cache_salt`` alone keeps
          them apart only with that cache off (``multimodal_config.encoder_cache_max_bytes=0``).
        - ``scope`` is fixed at the attach, and lending by name stops for good once the manager
          resets its reuse state (``reset_reuse_state``, every cache closed), as an in-place weight
          update does. Later lends raising no ``ValueError`` fail at the call; a publish granted
          before finishes.
        - Publishes and fetches have limits of their own: see ``StagingLender``.
    """
    from ._lender import _attach_staging as _attach

    return _attach(manager, scope=scope, staging=staging)


def attach_in_place(manager: "KVCacheManagerV2") -> InPlaceLender:
    """Attach the manager's one lender, which lends requests' own device pages in place.

    The caller addresses lent blocks with its own page-table code and owns their ordering and
    validity; ``InPlaceLender`` lists its preconditions while a loan is open. Block reuse may be on
    or off, and a manager built for a one-model draft that reads prompt tokens ahead is accepted,
    since in-place views carry no names.

    Args:
        manager: A ``KVCacheManagerV2`` with a ``mapping``.

    Returns:
        The lender, installed on ``manager`` for the manager's life.

    Raises:
        TypeError: ``manager`` is not a ``KVCacheManagerV2`` or has no mapping.
        ValueError: A lender is attached already; the limits below refuse the manager; its layer
            groups, layer configs and declarations disagree.

    Caller must:
        - Attach on the thread that builds the executor, to the manager the executor serves, not one
          built only to estimate the KV cache size.
        - Shut the manager down last, after the executor loop stopped and every loan was released.
          Once leases and lender are unused, drop the last reference on a thread that has used the
          device's CUDA context.

    Memory:
        - Caches on loan at the manager's shutdown, with its device pools, stay until the process
          exits.
        - A loan open when the manager is collected without a shutdown keeps the manager's
          page-index host buffer until exit, and its cache and the device pools until the lease is
          released or both it and the lender are dropped.

    First-version limits:
        - One lender per manager, for the manager's life: staging and in-place cannot serve one
          manager together.
        - Helix and every other context-parallel manager (``mapping.cp_size > 1``) are refused.
        - Layer groups holding recurrent state are refused, as are sparse buffers (``is_sparse``),
          whose read-only pages a cache can lock in host memory; sparse attention without buffers
          marked sparse is not.
    """
    from ._lender import attach_in_place as _attach

    return _attach(manager)

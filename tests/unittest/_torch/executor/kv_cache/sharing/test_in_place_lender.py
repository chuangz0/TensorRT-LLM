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
"""The in-place lender over real KV cache managers, through the public API: views of a request's
own pages, the two explicit failures, and loans kept across the request's free and the manager's
shutdown. Oracles read pages and the allocator directly; each has a negative control."""

import gc
import sys
import threading
import weakref
from contextlib import contextmanager

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import (
    InPlaceLender,
    Lease,
    StagingLender,
    StagingOptions,
    attach_in_place,
    attach_staging,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="allocates KV cache pools")

TPB = 32
WINDOW = 64
PROMPT = list(range(1000, 1097))  # three whole blocks and a one-token tail: four pages
END = len(PROMPT)
WINDOWED_PROMPT = list(range(2000, 2161))  # five whole blocks and a one-token tail
WINDOWED_END = len(WINDOWED_PROMPT)
SOURCE, TARGET = 1, 2


def ceil_blocks(tokens: int) -> int:
    return -(-tokens // TPB)


def check_view(kit, view, kv, expected):
    """``expected``: layer group -> ordinals, every layer group listed. Rows carry no names,
    addresses or part, and each is a block with a page of its own."""
    assert {run.layer_group: run.ordinals.tolist() for run in view.runs} == {
        lg: list(ordinals) for lg, ordinals in expected.items()
    }
    for run in view.runs:
        assert run.names is None and run.addresses is None and run.part is None
        own = kit.pages(kv, run.layer_group)
        assert all(own[o] >= 0 for o in run.ordinals.tolist())


@contextmanager
def lent_and_freed(kit, mgr, lender, write=False):
    """``SOURCE`` computed ``PROMPT``, a lease lends all of it, and ``SOURCE`` is freed. Yields the
    lease, the freed cache and its pages."""
    request = kit.published(mgr, SOURCE, PROMPT)
    kv = kit.kv(mgr, request)
    lent = set(kit.pages(kv, 0))
    assert len(lent) == ceil_blocks(END) and kit.pool_pages(mgr) - len(lent) >= 1
    lease = (lender.lend_write if write else lender.lend_read)(request, 0, END)
    assert lease.poll() is not None
    mgr.free_resources(request)
    assert kit.kv(mgr, request) is None
    yield lease, kv, lent


# -- what it lends ----------------------------------------------------------------------------


def test_an_in_place_lender_lends_and_promises_nothing_more(real_manager):
    with real_manager() as mgr:
        lender = attach_in_place(mgr)
        assert isinstance(lender, InPlaceLender) and not isinstance(lender, StagingLender)
        assert not hasattr(lender, "readiness") and not hasattr(lender, "parts")
        with pytest.raises(ValueError, match="already attached"):
            attach_in_place(mgr)
        with pytest.raises(ValueError, match="already attached"):
            attach_staging(mgr, scope=b"scope", staging=StagingOptions(TPB))


def test_a_read_is_ready_at_its_first_poll_without_waiting_for_the_stream(kit, real_manager):
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        kv = kit.kv(mgr, source)
        lender = attach_in_place(mgr)
        with kit.held_stream(mgr._stream) as gate:
            lease = lender.lend_read(source, 0, END)
            assert isinstance(lease, Lease)
            view = lease.poll()
            assert view is not None, "an in-place lease waited for the stream"
            assert lease.poll() is view
            gate.open()
        check_view(kit, view, kv, {0: range(ceil_blocks(END))})
        assert END % TPB and view.runs[0].ordinals[-1] == END // TPB, "the partial last block"
        pieces = [
            (lender.lend_read(source, TPB, 2 * TPB), [1]),
            (lender.lend_read(source, 2 * TPB, END), [2, 3]),
            (lender.lend_read(source, 5, 40), [0, 1]),  # any token range
        ]
        for piece, ordinals in pieces:
            check_view(kit, piece.poll(), kv, {0: ordinals})
        for held in [lease] + [piece for piece, _ in pieces]:
            held.release()
        with pytest.raises(RuntimeError):
            lease.poll()


def test_a_windowed_view_keeps_only_the_blocks_the_window_reads(kit, real_manager):
    with real_manager(windows=[WINDOW, 256]) as mgr:
        source = kit.published(mgr, SOURCE, WINDOWED_PROMPT)
        kv = kit.kv(mgr, source)
        lender = attach_in_place(mgr)
        windows = kit.windows(mgr)
        sliding, full = windows.index(WINDOW), windows.index(None)
        beg, end = kit.stale_blocks(mgr, sliding, WINDOWED_END)
        assert end > beg == 0, "the window must have left some blocks behind"
        blocks = range(ceil_blocks(WINDOWED_END))
        lease = lender.lend_read(source, 0, WINDOWED_END)
        in_window = [o for o in blocks if not beg <= o < end]
        check_view(kit, lease.poll(), kv, {full: blocks, sliding: in_window})
        lease.release()


def test_a_generation_view_leaves_out_paged_blocks_the_window_no_longer_reads(kit, real_manager):
    with real_manager(windows=[WINDOW, 256]) as mgr:
        request = kit.make_request(TARGET, WINDOWED_PROMPT)
        assert mgr.prepare_context(request)
        assert mgr.resize_context(request, request.context_remaining_length)
        kv = kit.kv(mgr, request)
        lender = attach_in_place(mgr)
        sliding = kit.windows(mgr).index(WINDOW)
        beg, end = kit.stale_blocks(mgr, sliding, WINDOWED_END)
        own = kit.pages(kv, sliding)
        assert end > beg and all(own[o] >= 0 for o in range(beg, end)), (
            "the blocks below the window must still hold pages here"
        )
        blocks = range(ceil_blocks(WINDOWED_END))
        lease = lender.lend_write(request, 0, WINDOWED_END)
        in_window = [o for o in blocks if not beg <= o < end]
        check_view(kit, lease.poll(), kv, {1 - sliding: blocks, sliding: in_window})
        lease.release()


def test_a_k_only_view_lists_every_block_including_the_partial_last(kit, real_manager):
    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.bindings.internal.batch_manager import CacheType

    mla = dict(kv_cache_type=CacheType.SELFKONLY, num_kv_heads=1, head_dim=128, dtype=DataType.BF16)
    with real_manager(**mla) as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        lender = attach_in_place(mgr)
        lease = lender.lend_read(source, 0, END)
        check_view(kit, lease.poll(), kit.kv(mgr, source), {0: range(ceil_blocks(END))})
        lease.release()


def test_a_write_changes_neither_the_cache_nor_its_pages(kit, real_manager):
    with real_manager() as mgr:
        kit.published(mgr, SOURCE, PROMPT)
        request = kit.make_request(TARGET, PROMPT)
        assert mgr.prepare_context(request)
        kv = kit.kv(mgr, request)
        assert kv.num_committed_tokens >= TPB, "the prefix must be reused"
        assert mgr.resize_context(request, request.context_remaining_length)
        lender = attach_in_place(mgr)
        shape = (kv.capacity, kv.history_length, kv.num_committed_tokens)
        lease = lender.lend_write(request, 0, END)  # starts inside the reused prefix
        view = lease.poll()
        check_view(kit, view, kv, {0: range(ceil_blocks(END))})
        assert (kv.capacity, kv.history_length, kv.num_committed_tokens) == shape, "grown"
        dev = kit.DevicePages(mgr)
        slots = [kit.pages(kv, 0)[o] for o in view.runs[0].ordinals.tolist()]
        content = [dev.read(0, slot) for slot in slots]
        lease.mark_arrived(view.row_masks(True))
        mgr._stream.synchronize()
        now = kit.digest([dev.read(0, slot) for slot in slots])
        assert now == kit.digest(content), "marks copied something"
        assert (kv.capacity, kv.history_length, kv.num_committed_tokens) == shape
        lease.release()


def test_a_view_may_end_past_the_committed_tokens_with_reuse_off(kit, real_manager):
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

    with real_manager(enable_block_reuse=False) as mgr:
        request = kit.make_request(SOURCE, PROMPT)
        assert mgr.prepare_context(request)
        assert mgr.resize_context(request, request.context_remaining_length)
        request.move_to_next_context_chunk()
        batch = ScheduledRequests()
        batch.append_context_request(request)
        mgr.update_context_resources(batch)
        kv = kit.kv(mgr, request)
        assert kv.num_committed_tokens < END
        lender = attach_in_place(mgr)
        lease = lender.lend_read(request, 0, END)
        check_view(kit, lease.poll(), kv, {0: range(ceil_blocks(END))})
        lease.release()


# -- failures ---------------------------------------------------------------------------------


def test_blocks_without_a_page_are_left_out_of_a_read_and_fail_a_write(kit, real_manager):
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        kv = kit.kv(mgr, source)
        blocks = ceil_blocks(END)
        assert len(kit.pages(kv, 0)) == blocks
        lender = attach_in_place(mgr)
        beyond = (blocks + 1) * TPB
        read = lender.lend_read(source, 0, beyond)
        check_view(kit, read.poll(), kv, {0: range(blocks)})
        empty = lender.lend_read(source, blocks * TPB, beyond)
        assert empty.poll().num_rows == 0
        write = lender.lend_write(source, 0, beyond)
        assert write.failure is not None and write.poll() is None
        whole = lender.lend_write(source, 0, blocks * TPB)  # every block it touches has a page
        assert whole.poll() is not None
        for lease in (read, empty, whole):
            lease.release()
        mgr.free_resources(source)
        assert kit.closed(kv), "a failed lease holds no loan"
        write.release()


@pytest.mark.parametrize("windowed", [False, True], ids=["full", "windowed"])
def test_a_range_of_any_length_lends_only_what_the_cache_has(kit, real_manager, windowed):
    prompt = WINDOWED_PROMPT if windowed else PROMPT
    with real_manager(windows=[WINDOW, 256] if windowed else None) as mgr:
        source = kit.published(mgr, SOURCE, prompt)
        kv = kit.kv(mgr, source)
        lender = attach_in_place(mgr)
        blocks = ceil_blocks(len(prompt))
        # A window that far on reads none of the cache's blocks; full attention reads them all.
        lent = {lg: [] if w else range(blocks) for lg, w in enumerate(kit.windows(mgr))}
        # Past the manager's 32-bit token counts, and ordinals alone would take 256 TiB.
        for huge in (1 << 31, 1 << 50):
            read = lender.lend_read(source, 0, huge)
            check_view(kit, read.poll(), kv, lent)
            write = lender.lend_write(source, 0, huge)
            assert write.poll() is None and "have no page" in str(write.failure)
            if not windowed:
                assert f"blocks [{blocks}, " in write.failure, "not the blocks past the cache"
            for lease in (read, write):
                lease.release()


@pytest.mark.parametrize("windowed", [False, True], ids=["full", "windowed"])
def test_a_range_starting_past_every_block_lends_no_rows_and_fails_a_write(
    kit, real_manager, windowed
):
    """Any token range, also one starting past every 64-bit block ordinal: a read lends no rows and
    a write fails at the call, naming the first blocks past the cache, and neither call raises."""
    prompt = WINDOWED_PROMPT if windowed else PROMPT
    with real_manager(windows=[WINDOW, 256] if windowed else None) as mgr:
        source = kit.published(mgr, SOURCE, prompt)
        lender = attach_in_place(mgr)
        first = 1 << 63  # the first block ordinal an int64 cannot hold
        start = first * TPB
        for end in (start + 1, start + 2 * TPB):
            try:
                read = lender.lend_read(source, start, end)
                write = lender.lend_write(source, start, end)
            except OverflowError as error:
                raise AssertionError(f"a lend of [{start}, {end}) raised {error!r}") from error
            view = read.poll()
            assert view is not None and view.num_rows == 0, f"[{start}, {end}): {read.failure}"
            assert write.poll() is None and f"blocks [{first}" in str(write.failure), write.failure
            for lease in (read, write):
                lease.release()


def test_the_check_catches_an_in_place_lender_listing_blocks_past_the_cache_as_int64(
    kit, real_manager, monkeypatch
):
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    real = _lender.InPlace._device_slots

    def as_int64(self, manager, kv, lg, start, end):  # the blocks past the cache as int64 too
        inside, slots, beyond = real(self, manager, kv, lg, start, end)
        return inside, slots, np.array(beyond, dtype=np.int64).tolist()

    monkeypatch.setattr(_lender.InPlace, "_device_slots", as_int64)
    with pytest.raises(AssertionError, match="raised OverflowError"):
        test_a_range_starting_past_every_block_lends_no_rows_and_fails_a_write(
            kit, real_manager, False
        )


def keep_sinks(monkeypatch, tokens):
    """Managers built from now on keep ``tokens`` sink tokens in their ``WINDOW`` layers, which no
    manager configures on its own: the runtime then locks those blocks however far the window
    moves."""
    from dataclasses import replace

    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionLayerConfig

    build = KVCacheManagerV2._build_cache_config

    def with_sinks(self, config):
        layers = [
            AttentionLayerConfig(
                layer_id=layer.layer_id,
                buffers=layer.buffers,
                sliding_window_size=layer.sliding_window_size,
                num_sink_tokens=tokens,
            )
            if isinstance(layer, AttentionLayerConfig) and layer.sliding_window_size == WINDOW
            else layer
            for layer in config.layers
        ]
        return build(self, replace(config, layers=layers))

    monkeypatch.setattr(KVCacheManagerV2, "_build_cache_config", with_sinks)


def test_a_windowed_group_lends_its_sinks_at_any_history(kit, real_manager, monkeypatch):
    """A range of any length lends a windowed group's sink blocks, which its window reads at every
    history, and nothing else of that group: a history past the manager's 32-bit token counts is
    read as one whose range still starts after the sinks."""
    keep_sinks(monkeypatch, TPB)
    with real_manager(windows=[WINDOW, 256]) as mgr:
        source = kit.published(mgr, SOURCE, WINDOWED_PROMPT)
        kv = kit.kv(mgr, source)
        windows = kit.windows(mgr)
        sliding = windows.index(WINDOW)
        assert kit.stale_blocks(mgr, sliding, WINDOWED_END) == (1, 3), "no sink block below"
        locked = [int(p) for p in kv.get_base_page_indices(sliding)[: kv.num_blocks]]
        assert locked[0] >= 0 and locked[1] < 0, f"the runtime keeps no sink block: {locked}"
        lender = attach_in_place(mgr)
        blocks = ceil_blocks(len(WINDOWED_PROMPT))
        lent = {lg: [0] if w else range(blocks) for lg, w in enumerate(windows)}
        for huge in (1 << 31, 1 << 50):
            read = lender.lend_read(source, 0, huge)
            check_view(kit, read.poll(), kv, lent)
            read.release()


def test_an_empty_range_lends_no_rows_wherever_it_sits(kit, real_manager):
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        lender = attach_in_place(mgr)
        # On a block boundary, inside a block, at the partial last block, past the cache's blocks.
        for position in (2 * TPB, 2 * TPB + 1, END, (ceil_blocks(END) + 1) * TPB + 1):
            for lend in (lender.lend_read, lender.lend_write):
                lease = lend(source, position, position)
                view = lease.poll()
                assert view is not None, f"[{position}, {position}) failed: {lease.failure}"
                lent = {run.layer_group: run.ordinals.tolist() for run in view.runs}
                assert view.num_rows == 0, f"[{position}, {position}) lent {lent}"
                lease.release()


def test_a_suspended_cache_fails_both_lends_at_the_call(kit, real_manager):
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        lender = attach_in_place(mgr)
        mgr.suspend_request(source)
        assert not kit.kv(mgr, source).is_active
        for lend in (lender.lend_read, lender.lend_write):
            lease = lend(source, 0, END)
            assert lease.failure is not None and lease.poll() is None
            lease.release()
        assert mgr.resume_request(source)  # active again, both go through
        for lend in (lender.lend_read, lender.lend_write):
            lease = lend(source, 0, END)
            assert lease.failure is None and lease.poll() is not None
            lease.release()


def test_only_a_negative_or_reversed_range_is_an_argument_error(kit, real_manager):
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        lender = attach_in_place(mgr)
        for lend in (lender.lend_read, lender.lend_write):
            for start, end in ((-1, TPB), (TPB, 0), (0, -1)):
                with pytest.raises(ValueError):
                    lend(source, start, end)
            nobody = lend(kit.make_request(9, PROMPT), 0, TPB)
            assert nobody.failure is not None and nobody.poll() is None
            nobody.release()
        mgr.shutdown()
        for lend in (lender.lend_read, lender.lend_write):
            lease = lend(source, 0, END)
            assert lease.failure is not None and lease.poll() is None
            lease.release()


def test_a_reuse_reset_leaves_in_place_lending_as_it_was(kit, real_manager):
    with real_manager() as mgr:
        lender = attach_in_place(mgr)
        mgr.free_resources(kit.published(mgr, SOURCE, PROMPT))
        mgr.reset_reuse_state()  # with every cache closed; in-place views carry no names
        source = kit.published(mgr, SOURCE, PROMPT)
        kv = kit.kv(mgr, source)
        for lend in (lender.lend_read, lender.lend_write):
            lease = lend(source, 0, END)
            view = lease.poll()
            assert view is not None, f"in-place lending stopped at the reset: {lease.failure}"
            check_view(kit, view, kv, {0: range(ceil_blocks(END))})
            lease.release()


def test_the_check_catches_an_in_place_lender_stopping_at_a_reuse_reset(
    kit, real_manager, monkeypatch
):
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    monkeypatch.setattr(_lender.InPlace, "_on_reset", lambda self: setattr(self, "_kept", ()))
    with pytest.raises(AssertionError, match="stopped at the reset"):
        test_a_reuse_reset_leaves_in_place_lending_as_it_was(kit, real_manager)


def test_mark_arrived_checks_the_masks_and_nothing_else(kit, real_manager):
    with real_manager() as mgr:
        source = kit.published(mgr, SOURCE, PROMPT)
        lender = attach_in_place(mgr)
        read = lender.lend_read(source, 0, END)
        read_view = read.poll()
        with pytest.raises(RuntimeError):
            read.mark_arrived(read_view.row_masks())
        write = lender.lend_write(source, 0, END)
        with pytest.raises(RuntimeError):
            write.mark_arrived(())  # before poll() gave the view
        view = write.poll()
        rows = len(view.runs[0])
        for masks in ((), (np.ones(rows + 1, bool),), (np.ones(rows, np.int64),)):
            with pytest.raises(ValueError):
                write.mark_arrived(masks)
        write.release()
        write.mark_arrived(view.row_masks(True))  # after the release too
        with pytest.raises(RuntimeError):
            write.mark_arrived(view.row_masks(True))
        read.release()


# -- managers built for a draft that reads past a block's end ---------------------------------


def eagle3():
    """One-model Eagle 3: its draft at position ``i`` reads prompt token ``i + 1``."""
    from tensorrt_llm.llmapi.llm_args import Eagle3DecodingConfig

    return Eagle3DecodingConfig(max_draft_len=1, speculative_model="draft-model")


READS_AHEAD = {
    "eagle3_layers_in_the_target": lambda: dict(spec_config=eagle3()),
    "eagle3_joint_reuse_draft_pool": lambda: dict(
        spec_config=eagle3(), is_draft=True, joint_kv_cache_reuse=True
    ),
}
REFUSED = "the attach refused it"


@pytest.mark.parametrize("case", list(READS_AHEAD))
def test_in_place_lends_from_a_manager_whose_draft_reads_ahead(kit, real_manager, case):
    """In-place views carry no names, so a draft reading past a block's end changes nothing they
    say: the attach accepts the manager and a target's pages are lent."""
    with real_manager(**READS_AHEAD[case]()) as mgr:
        assert mgr.reuse_match_backoff == 1, "the draft does not read ahead: proves nothing"
        try:
            lender = attach_in_place(mgr)
        except ValueError as error:
            raise AssertionError(f"{REFUSED}: {error}") from error
        assert isinstance(lender, InPlaceLender)
        if mgr.is_draft:
            return
        request = kit.published(mgr, SOURCE, PROMPT)
        lease = lender.lend_read(request, 0, END)
        view = lease.poll()
        check_view(kit, view, kit.kv(mgr, request), {0: range(ceil_blocks(END))})
        lease.release()


@pytest.mark.parametrize("case", list(READS_AHEAD))
def test_the_check_catches_an_in_place_attach_refusing_a_draft_reading_ahead(
    kit, real_manager, monkeypatch, case
):
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    real_attach = _lender.attach_in_place

    def refusing(manager):
        _lender._check_lookahead(manager)
        return real_attach(manager)

    monkeypatch.setattr(_lender, "attach_in_place", refusing)
    with pytest.raises(AssertionError, match=REFUSED):
        test_in_place_lends_from_a_manager_whose_draft_reads_ahead(kit, real_manager, case)


# -- a commit while lent ----------------------------------------------------------------------


def test_a_commit_while_lent_returns_the_lent_pages_the_caller_must_not_commit(kit, real_manager):
    """What ``InPlaceLender`` forbids while a loan is open: a commit rebases the request onto blocks
    another request committed, and its own lent pages go back to the pool for others to take. The
    lender keeps the cache open, not the pages a commit replaces. This pins the manager behaviour
    the rule rests on, so a lender that guards lent pages against a commit fails it."""
    blocks = END // TPB
    with real_manager(max_tokens=kit.POOL_TOKENS, max_batch_size=16) as mgr:
        lender = attach_in_place(mgr)
        target = kit.admitted(mgr, TARGET, PROMPT)
        assert mgr.resize_context(target, target.context_remaining_length)
        kv = kit.kv(mgr, target)
        source = kit.published(mgr, SOURCE, PROMPT)
        lease = lender.lend_read(target, 0, blocks * TPB)
        view = lease.poll()
        lent = {kit.pages(kv, 0)[o] for o in view.runs[0].ordinals.tolist()}
        assert len(lent) == blocks
        target.context_chunk_size = blocks * TPB
        target.move_to_next_context_chunk()
        mgr.try_commit_blocks(target)
        assert kit.pages(kv, 0)[:blocks] == kit.pages(kit.kv(mgr, source), 0)[:blocks]
        others = kit.Requests(mgr)
        try:
            while others.allocate(1, chunk=1):
                pass
            assert lent <= others.pages(), "a commit kept the lent pages from the pool"
        finally:
            lease.release()
            others.free()


# -- the request's free -----------------------------------------------------------------------


def test_a_freed_request_keeps_its_lent_pages_until_the_release(kit, real_manager):
    with real_manager(max_tokens=kit.POOL_TOKENS) as mgr:
        dev = kit.DevicePages(mgr)
        lender = attach_in_place(mgr)
        with lent_and_freed(kit, mgr, lender) as (lease, kv, lent):
            content = {slot: dev.read(0, slot) for slot in lent}
            assert not kit.closed(kv), "a freed cache stays open while lent"
            assert kit.taken_by_others(mgr, lent) == set(), "another request got a lent page"
            now = kit.digest({slot: dev.read(0, slot) for slot in lent})
            assert now == kit.digest(content), "a lent page changed"
            lease.release()
            assert kit.closed(kv), "the last release closes the cache in its call"
            assert kit.whole_pool_goes_to_others(mgr, lent), "the freed pages are reused"


def test_the_check_catches_a_lender_letting_the_free_close_a_lent_cache(
    kit, real_manager, monkeypatch
):
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    monkeypatch.setattr(_lender.InPlace, "_on_free", lambda self, rid, kv, after: False)
    with pytest.raises(AssertionError, match="a freed cache stays open while lent"):
        test_a_freed_request_keeps_its_lent_pages_until_the_release(kit, real_manager)


def test_the_page_oracle_sees_a_page_the_free_returned(kit, real_manager, monkeypatch):
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    monkeypatch.setattr(_lender.InPlace, "_on_free", lambda self, rid, kv, after: False)
    with real_manager(max_tokens=kit.POOL_TOKENS) as mgr:
        lender = attach_in_place(mgr)
        with lent_and_freed(kit, mgr, lender) as (lease, kv, lent):
            assert kit.closed(kv)
            assert kit.taken_by_others(mgr, lent), "the oracle cannot see a returned page"
            lease.release()


@pytest.mark.parametrize("mark_first", [True, False], ids=["mark_first", "release_first"])
def test_marks_and_the_release_may_come_in_either_order(kit, real_manager, mark_first):
    with real_manager(max_tokens=kit.POOL_TOKENS) as mgr:
        lender = attach_in_place(mgr)
        with lent_and_freed(kit, mgr, lender, write=True) as (lease, kv, lent):
            view = lease.poll()
            if mark_first:
                lease.mark_arrived(view.row_masks(True))
                assert not kit.closed(kv), "marks do not end the loan"
                assert kit.taken_by_others(mgr, lent) == set()
                lease.release()
                assert kit.closed(kv)
            else:
                lease.release()
                assert kit.closed(kv), "the release ends the loan"
                lease.mark_arrived(view.row_masks(True))
                assert kit.closed(kv)
            assert kit.whole_pool_goes_to_others(mgr, lent)


def test_the_pages_stay_until_the_last_of_several_leases_ends(kit, real_manager):
    with real_manager(max_tokens=kit.POOL_TOKENS) as mgr:
        request = kit.published(mgr, SOURCE, PROMPT)
        kv = kit.kv(mgr, request)
        lent = set(kit.pages(kv, 0))
        lender = attach_in_place(mgr)
        # Never polled: a loan opens at the call.
        write = lender.lend_write(request, 0, END)
        read = lender.lend_read(request, 0, 2 * TPB)
        mgr.free_resources(request)
        write.release()
        assert not kit.closed(kv) and kit.taken_by_others(mgr, lent) == set()
        read.release()
        assert kit.closed(kv)


def next_owner_s_row_across_the_kept_close(kit, mgr, early=False):
    """``SOURCE`` is lent, freed, and its index slot taken by another request; then the lease ends
    and the kept cache closes. The new owner's host page table row before and after. ``early``: the
    slot went before the loan, as a context-only worker releases it after the forward."""
    request = kit.published(mgr, SOURCE, PROMPT)
    kv = kit.kv(mgr, request)
    index = mgr.index_mapper.get_index(SOURCE)
    lender = attach_in_place(mgr)
    if early:
        mgr.release_index_slot(SOURCE)
    lease = lender.lend_read(request, 0, END)
    mgr.free_resources(request)
    others = kit.Requests(mgr)
    assert others.allocate(3)
    (other,) = others.held
    assert mgr.index_mapper.get_index(other.py_request_id) == index, "the slot is reused"
    row = mgr.host_kv_cache_block_offsets[0, index * mgr.max_beam_width]
    before = row.clone()
    lease.release()
    assert kit.closed(kv)
    after = row.clone()
    others.free()
    return before, after


@pytest.mark.parametrize("early", [False, True], ids=["at_its_free", "released_early"])
def test_a_lent_request_gives_up_its_index_slot_at_its_free(kit, real_manager, early):
    with real_manager(max_tokens=kit.POOL_TOKENS) as mgr:
        before, after = next_owner_s_row_across_the_kept_close(kit, mgr, early)
    assert torch.equal(after, before), "closing the kept cache wrote into the next owner's row"


def free_lent_attached(self, request_id, kv_cache):
    """``KVCacheManagerV2._free_lent`` without its detach: the lent cache stays attached to the
    page-index buffer row it held."""
    if request_id in self._early_freed_index_requests:
        self._early_freed_index_requests.discard(request_id)
        return
    self.index_mapper.remove_sequence(request_id)


def test_the_row_oracle_sees_a_close_writing_into_the_next_owner(kit, real_manager, monkeypatch):
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2

    monkeypatch.setattr(KVCacheManagerV2, "_free_lent", free_lent_attached)
    with real_manager(max_tokens=kit.POOL_TOKENS) as mgr:
        before, after = next_owner_s_row_across_the_kept_close(kit, mgr)
    assert not torch.equal(after, before), "the oracle cannot see a write into the row"


def test_a_kept_cache_closes_before_its_stats_exclusion_is_cleared(kit, real_manager):
    with real_manager(max_tokens=kit.POOL_TOKENS) as mgr:
        request = kit.published(mgr, SOURCE, PROMPT)
        kv = kit.kv(mgr, request)
        mgr.impl.mark_stats_excluded(SOURCE)
        lender = attach_in_place(mgr)
        lease = lender.lend_read(request, 0, END)
        impl = mgr.impl
        spy = mgr.impl = kit.StatsSpy(impl, kv)
        try:
            mgr.free_resources(request)
            assert spy.cleared == [] and impl.is_stats_excluded(SOURCE)
            lease.release()
            assert spy.cleared == [(SOURCE, True)], "cleared before the close"
            assert not impl.is_stats_excluded(SOURCE)
        finally:
            mgr.impl = impl


# -- the manager's shutdown -------------------------------------------------------------------


@pytest.mark.parametrize("freed", [True, False], ids=["request_freed", "request_live"])
def test_shutdown_keeps_what_a_lease_still_lends_until_exit(kit, real_manager, monkeypatch, freed):
    with real_manager(max_tokens=kit.POOL_TOKENS) as mgr:
        request = kit.published(mgr, SOURCE, PROMPT)
        kv = kit.kv(mgr, request)
        dev = kit.DevicePages(mgr)
        lender = attach_in_place(mgr)
        lease = lender.lend_read(request, 0, END)
        slots = [kit.pages(kv, 0)[o] for o in lease.poll().runs[0].ordinals.tolist()]
        content = [dev.read(0, slot) for slot in slots]
        if freed:
            mgr.free_resources(request)
        spy = mgr.impl = kit.ShutdownSpy(mgr.impl)
        warnings = kit.lender_warnings(monkeypatch)
        mgr.shutdown()
        assert not spy.shut_down, "the pools holding a lent page were destroyed"
        assert warnings, "keeping caches until exit is logged"
        kept = kit.retained()
        assert any(o is spy for o in kept) and any(o is kv for o in kept)
        assert not kit.closed(kv)
        now = kit.digest([dev.read(0, slot) for slot in slots])
        assert now == kit.digest(content), "the lent bytes stay readable"
        lease.release()
        assert not kit.closed(kv), "a release after the shutdown closes nothing"
        mgr.shutdown()
        assert not spy.shut_down, "a later shutdown keeps them too"


@pytest.mark.parametrize("leases", [0, 2], ids=["never_lent", "every_lease_ended"])
def test_without_an_open_loan_free_and_shutdown_are_as_without_a_lender(
    kit, real_manager, monkeypatch, leases
):
    with real_manager(max_tokens=kit.POOL_TOKENS) as mgr:
        request = kit.published(mgr, SOURCE, PROMPT)
        kv = kit.kv(mgr, request)
        lent = set(kit.pages(kv, 0))
        lender = attach_in_place(mgr)
        for lend in (lender.lend_read, lender.lend_write)[:leases]:
            lend(request, 0, END).release()
        mgr.free_resources(request)
        assert kit.closed(kv), "freed at once"
        assert kit.taken_by_others(mgr, lent), "the freed pages go to the next requests"
        kept = len(kit.retained())
        spy = mgr.impl = kit.ShutdownSpy(mgr.impl)
        warnings = kit.lender_warnings(monkeypatch)
        mgr.shutdown()
        assert spy.shut_down and warnings == [] and len(kit.retained()) == kept


# -- references and threads -------------------------------------------------------------------


def test_a_dropped_lease_ends_no_loan_whatever_thread_collects_it(kit, real_manager, monkeypatch):
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    ended = []
    end_loan = _lender.InPlace._end_loan

    def spy(self, kv_cache):
        ended.append(threading.get_ident())
        return end_loan(self, kv_cache)

    monkeypatch.setattr(_lender.InPlace, "_end_loan", spy)
    with real_manager(max_tokens=kit.POOL_TOKENS) as mgr:
        request = kit.published(mgr, SOURCE, PROMPT)
        kv = kit.kv(mgr, request)
        lender = attach_in_place(mgr)
        cycle = [lender.lend_read(request, 0, END)]
        cycle.append(cycle)  # only a collector frees it
        mgr.free_resources(request)
        del cycle
        kit.on_thread(gc.collect, "collector", cuda=True)
        assert ended == [], "collecting a lease ended its loan"
        assert not kit.closed(kv) and kit.kv(mgr, request) is None
        mgr.shutdown()
        assert any(o is kv for o in kit.retained()), "the shutdown keeps what is still lent"


def test_the_check_catches_a_lease_whose_collection_ends_its_loan(kit, real_manager, monkeypatch):
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    init = _lender._InPlaceLease.__init__

    def hold_lender_strongly(self, lender, *args, **kwargs):
        init(self, lender, *args, **kwargs)
        # The collector clears a weak reference inside the garbage before any finalizer runs.
        self._lender = lambda: lender

    def release_when_collected(self):
        self.release()

    monkeypatch.setattr(_lender._InPlaceLease, "__init__", hold_lender_strongly)
    monkeypatch.setattr(_lender._InPlaceLease, "__del__", release_when_collected, raising=False)
    with pytest.raises(AssertionError, match="collecting a lease ended its loan"):
        test_a_dropped_lease_ends_no_loan_whatever_thread_collects_it(
            kit, real_manager, monkeypatch
        )


def test_open_leases_keep_neither_the_lender_nor_the_manager_alive(kit):
    torch.cuda.init()
    gc.collect()
    mgr = kit.make_manager()
    request = kit.published(mgr, SOURCE, PROMPT)
    lender = attach_in_place(mgr)
    read = lender.lend_read(request, 0, END)
    write = lender.lend_write(request, 0, END)
    view = write.poll()
    watched = (weakref.ref(mgr), weakref.ref(lender))
    mgr._stream.synchronize()
    del mgr, lender
    gc.collect()
    assert [ref() for ref in watched] == [None, None], "a lease kept its lender or manager alive"
    write.mark_arrived(view.row_masks(True))
    for lease in (write, write, read):
        lease.release()
    gc.collect()
    torch.cuda.empty_cache()


def test_threads_take_turns_and_a_release_closes_a_freed_cache_on_its_own(kit, real_manager):
    with real_manager(max_tokens=kit.POOL_TOKENS) as mgr:
        request = kit.published(mgr, SOURCE, PROMPT)
        kv = kit.kv(mgr, request)
        before = set(threading.enumerate())
        lender = kit.on_thread(lambda: attach_in_place(mgr), "builder", cuda=True)["value"]

        def executor_loop():
            lease = lender.lend_read(request, 0, END)
            rows = lease.poll().num_rows
            mgr.free_resources(request)
            lease.release()  # closes the freed cache inside this call, on this thread
            return rows, kit.closed(kv)

        looped = kit.on_thread(executor_loop, "executor-loop", cuda=True)
        assert looped.get("value") == (ceil_blocks(END), True)
        assert set(threading.enumerate()) <= before, "the lender started a thread"
        assert "error" not in kit.on_thread(mgr.shutdown, "shutdown", cuda=True)


# -- the manager's page-index buffer under a cache that outlives the manager --------------------
# A cache writes -1 for every block into its manager's host page-index buffer as it closes. A new
# tensor reclaims a freed buffer's address with a canary the cache must neither read nor write.

FREED = "touched the page-index buffer freed with its manager"


def fresh_device():
    torch.cuda.init()
    gc.collect()
    torch.cuda.empty_cache()


def check_a_lease_outliving_its_manager(kit, kind="in_place", free_first=False):
    """A lease outlives its manager and lender, collected without a shutdown: an in-place write or a
    staging read, each holding the request's cache. ``free_first`` frees the request first, which
    detaches its cache from the buffer. A staging lease also keeps its staging memory as it was."""
    fresh_device()
    mgr = kit.make_manager()
    request = kit.published(mgr, SOURCE, PROMPT)
    kv = kit.kv(mgr, request)
    blocks = int(kv.num_blocks)
    if kind == "staging":
        whole = END // TPB * TPB  # a staging lease covers whole blocks
        lender = attach_staging(mgr, scope=b"index-buffer", staging=StagingOptions(whole))
        lease = lender.lend_read(request, 0, whole)
    else:
        lender = attach_in_place(mgr)
        lease = lender.lend_write(request, 0, END)
    mgr._stream.synchronize()  # a staging read is ready once its copy into staging ran
    view = lease.poll()
    assert view is not None
    if kind == "staging":
        (run,) = view.runs
        address, slot_bytes = int(run.addresses[0]), lender.parts[run.part].slot_bytes
        parts, content = lender.parts, kit.host_bytes(address, slot_bytes)
    row = kit.index_row(mgr, request)
    own = list(kv.get_base_page_indices(0)[:blocks])
    assert row.values[:blocks] == own and -1 not in own, "the cache writes elsewhere"
    if free_first:
        mgr.free_resources(request)
    watched = [weakref.ref(o) for o in (mgr.host_kv_cache_block_offsets, mgr, lender)]
    mgr._stream.synchronize()
    del mgr, lender
    gc.collect()
    assert watched[1]() is None and watched[2]() is None, "the manager or lender was not collected"
    canary = None if watched[0]() is not None else kit.reclaim(row)
    if watched[0]() is None:
        assert canary is not None, "inconclusive: no allocation reused the freed buffer's address"
    read = list(kv.get_base_page_indices(0)[:blocks])
    if kind == "staging":
        assert kit.staging_kept(parts), "the staging memory went with its manager"
        now = kit.digest(kit.host_bytes(address, slot_bytes))
        assert now == kit.digest(content), "the kept staging memory changed"
    del kv  # the lease now holds the cache's last reference
    lease.release()
    lease.release()
    del lease  # a staging lease keeps its cache past its release
    gc.collect()
    written = kit.canary_written(canary, row)
    assert not (canary is not None and read == [kit.CANARY] * blocks) and not written, (
        f"the lent cache {FREED}: it read {read} and its close wrote {written} (cell, value)"
    )


def test_a_freed_request_s_cache_is_detached_before_its_manager_goes(kit, monkeypatch):
    """Without the lender's keep of the page-index buffer, which would hide a missing detach, the
    buffer goes with the manager: only the free's detach keeps the lent cache off it."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    monkeypatch.setattr(_lender.InPlace, "_keep_index_buffer", lambda self, manager: None)
    check_a_lease_outliving_its_manager(kit, free_first=True)


def test_the_check_catches_a_manager_freeing_a_lent_cache_attached(kit, monkeypatch):
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2

    monkeypatch.setattr(KVCacheManagerV2, "_free_lent", free_lent_attached)
    with pytest.raises(AssertionError, match=FREED):
        test_a_freed_request_s_cache_is_detached_before_its_manager_goes(kit, monkeypatch)


def check_a_cache_kept_at_shutdown(kit):
    """The manager shuts down with a loan open, which keeps the cache until exit, and is collected.
    At the process exit the keep list goes before a lease that something still holds; the cache
    that lease holds must close without touching the freed page-index buffer."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    fresh_device()
    before = kit.retained()
    mgr = kit.make_manager()
    request = kit.published(mgr, SOURCE, PROMPT)
    kv = kit.kv(mgr, request)
    blocks = int(kv.num_blocks)
    lender = attach_in_place(mgr)
    lease = lender.lend_read(request, 0, END)
    assert lease.poll() is not None
    row = kit.index_row(mgr, request)
    mgr._stream.synchronize()
    mgr.shutdown()
    assert any(o is kv for o in kit.retained()), "the shutdown keeps the lent cache"
    buffer = weakref.ref(mgr.host_kv_cache_block_offsets)
    del mgr, lender, request
    gc.collect()
    added = [o for o in kit.retained() if not any(o is b for b in before)]
    for owner in added:  # what the process exit does to the keep list
        _lender._let_go(owner)
    del added, owner
    gc.collect()
    canary = None if buffer() is not None else kit.reclaim(row)
    if buffer() is None:
        assert canary is not None, "inconclusive: no allocation reused the freed buffer's address"
    read = list(kv.get_base_page_indices(0)[:blocks])
    del kv, lease  # the lease held its cache's last reference
    gc.collect()
    written = kit.canary_written(canary, row)
    assert not (canary is not None and read == [kit.CANARY] * blocks) and not written, (
        f"the kept cache {FREED}: it read {read} and its close wrote {written} (cell, value)"
    )


class _Reclaimer:
    """A manager attribute set after the buffer and before the lender, so the manager's collection
    drops it between the two: it reclaims the freed buffer's address before the lender goes."""

    def __init__(self, kit, row, out):
        self._kit, self._row, self._out = kit, row, out

    def __del__(self):
        self._out.append(self._kit.reclaim(self._row))


def check_a_dropped_lease_on_a_collected_manager(kit):
    """A lease dropped unreleased leaves its loan with the lender; the manager is collected without
    a shutdown, its attributes in order. The cache the lender held closes after the buffer went."""
    fresh_device()
    mgr = kit.make_manager()
    request = kit.published(mgr, SOURCE, PROMPT)
    row = kit.index_row(mgr, request)
    out = []
    mgr._reclaimer = _Reclaimer(kit, row, out)
    lender = attach_in_place(mgr)
    lender.lend_read(request, 0, END)  # dropped unreleased
    del lender
    mgr._stream.synchronize()
    watched, buffer = weakref.ref(mgr), weakref.ref(mgr.host_kv_cache_block_offsets)
    del mgr
    gc.collect()
    assert watched() is None
    canary = out[0] if out else None
    if buffer() is None:
        assert canary is not None, "inconclusive: the freed address was not reclaimed"
    written = kit.canary_written(canary, row)
    assert not written, f"the cache the lender held {FREED}: its close wrote {written}"


def check_a_last_release_after_its_manager(kit, cache_held=False):
    """The manager is collected without a shutdown or a free while a loan is open; the caller still
    holds the lender and releases the last lease. The cache closes as the release drops its last
    reference or, with ``cache_held``, as the test drops its own."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    fresh_device()
    mgr = kit.make_manager()
    request = kit.published(mgr, SOURCE, PROMPT)
    kv = kit.kv(mgr, request)
    blocks = int(kv.num_blocks)
    lender = attach_in_place(mgr)
    lease = lender.lend_write(request, 0, END)
    assert lease.poll() is not None
    row = kit.index_row(mgr, request)
    own = list(kv.get_base_page_indices(0)[:blocks])
    assert row.values[:blocks] == own and -1 not in own, "the cache writes elsewhere"
    buffer, manager = weakref.ref(mgr.host_kv_cache_block_offsets), weakref.ref(mgr)
    mgr._stream.synchronize()
    del mgr
    gc.collect()
    assert manager() is None, "the manager was not collected"
    out = []

    def reclaim_if_freed():
        gc.collect()
        if buffer() is None and not out:
            out.append(kit.reclaim(row))

    reclaim_if_freed()  # a buffer the manager's collection freed
    if cache_held:
        lease.release()
        reclaim_if_freed()
        del kv
    else:
        end_loan = _lender.InPlace._end_loan

        def end_loan_then_reclaim(self, kv_cache):
            # Reclaims a buffer freed here, before the release drops the cache's last reference.
            end_loan(self, kv_cache)
            reclaim_if_freed()

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(_lender.InPlace, "_end_loan", end_loan_then_reclaim)
            del kv  # the lease and the loan now hold the cache's last references
            lease.release()
    gc.collect()
    canary = out[0] if out else None
    if buffer() is None:
        assert canary is not None, "inconclusive: no allocation reused the freed buffer's address"
    written = kit.canary_written(canary, row)
    assert not written, f"the lent cache {FREED}: its close wrote {written} (cell, value)"


@pytest.mark.parametrize("cache_held", [False, True], ids=["release_drops_cache", "cache_held"])
def test_the_check_catches_a_lender_letting_the_buffer_go_after_its_manager(
    kit, monkeypatch, cache_held
):
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    def let_go_regardless(self):
        _lender._let_go(self._index_buffer)
        self._index_buffer = None

    monkeypatch.setattr(_lender.InPlace, "_let_go_index_buffer", let_go_regardless)
    with pytest.raises(AssertionError, match=FREED):
        check_a_last_release_after_its_manager(kit, cache_held)


# Every way a cache can outlive its manager; none may touch the page-index buffer freed with it.
OUTLIVING = {
    "staging_lease_outlives": lambda kit: check_a_lease_outliving_its_manager(kit, "staging"),
    "lease_outlives": check_a_lease_outliving_its_manager,
    "kept_at_shutdown": check_a_cache_kept_at_shutdown,
    "dropped_lease": check_a_dropped_lease_on_a_collected_manager,
    "last_release_drops_cache": check_a_last_release_after_its_manager,
    "last_release_cache_held": lambda kit: check_a_last_release_after_its_manager(kit, True),
}


@pytest.mark.parametrize("check", list(OUTLIVING))
def test_a_cache_outliving_its_manager_touches_no_freed_index_buffer(kit, check):
    OUTLIVING[check](kit)


# A cache kept at the shutdown outlives the keep list, and with it the buffer the lender kept: the
# manager's detach alone guards it.
@pytest.mark.parametrize("check", [check for check in OUTLIVING if check != "kept_at_shutdown"])
def test_the_checks_catch_a_lender_not_keeping_the_index_buffer(kit, monkeypatch, check):
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    for cls in (_lender.Staging, _lender.InPlace):
        monkeypatch.setattr(cls, "_keep_index_buffer", lambda self, manager: None)
    with pytest.raises(AssertionError, match=FREED):
        OUTLIVING[check](kit)


def test_the_check_catches_a_manager_keeping_a_cache_attached_at_shutdown(kit, monkeypatch):
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2

    def attached(self):  # the kept caches leave the map with their page indices still written
        kept = self._sharing._on_shutdown(self.impl)
        for request_id in [rid for rid, kv_cache in self.kv_cache_map.items() if kv_cache in kept]:
            del self.kv_cache_map[request_id]
        return bool(kept)

    monkeypatch.setattr(KVCacheManagerV2, "_keep_lent_until_exit", attached)
    with pytest.raises(AssertionError, match=FREED):
        check_a_cache_kept_at_shutdown(kit)


def test_the_index_buffer_is_kept_only_while_a_loan_is_open(kit, real_manager):
    with real_manager(max_tokens=kit.POOL_TOKENS) as mgr:
        request = kit.published(mgr, SOURCE, PROMPT)
        before = len(kit.retained())
        buffer = mgr.host_kv_cache_block_offsets
        lender = attach_in_place(mgr)
        assert not any(o is buffer for o in kit.retained()), "kept with no loan open"
        leases = [lender.lend_read(request, 0, END), lender.lend_write(request, 0, END)]
        assert any(o is buffer for o in kit.retained()), "not kept while a loan is open"
        leases[0].release()
        assert any(o is buffer for o in kit.retained()), "let go with a loan still open"
        leases[1].release()
        assert len(kit.retained()) == before, "kept after the last loan ended"


def test_the_check_catches_a_lender_keeping_the_index_buffer_for_good(
    kit, real_manager, monkeypatch
):
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    monkeypatch.setattr(_lender.InPlace, "_let_go_index_buffer", lambda self: None)
    with pytest.raises(AssertionError, match="kept after the last loan ended"):
        test_the_index_buffer_is_kept_only_while_a_loan_is_open(kit, real_manager)


def switching_in_let_go(step, call, switch):
    """Run ``call``; before the ``step``-th line it runs inside the keep list's let-go, run
    ``switch``, as another thread may. Whether ``call`` got that far."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    seen, switched = 0, False

    def line(frame, event, arg):
        nonlocal seen, switched
        if event == "line" and not switched:
            seen += 1
            if seen == step:
                switched = True
                switch()
        return line

    def tracer(frame, event, arg):
        if switched or frame.f_code is not _lender._let_go.__code__:
            return None
        return line

    previous = sys.gettrace()
    sys.settrace(tracer)
    try:
        call()
    finally:
        sys.settrace(previous)
    return switched


def test_a_last_release_survives_another_lender_s_at_any_line(kit, real_manager):
    # Lenders of managers on different threads let go of their page-index buffers on their own;
    # the other one doing so before any line of one let-go breaks neither. Each line is tried.
    with real_manager() as first, real_manager() as second:
        managers = (first, second)
        requests = [kit.published(mgr, SOURCE, PROMPT) for mgr in managers]
        lenders = [attach_in_place(mgr) for mgr in managers]
        buffers = [mgr.host_kv_cache_block_offsets for mgr in managers]
        step, switched = 0, True
        while switched:
            step += 1
            leases = [lender.lend_read(r, 0, END) for lender, r in zip(lenders, requests)]
            switched = switching_in_let_go(step, leases[0].release, leases[1].release)
            if not switched:  # past the let-go's last line
                leases[1].release()
            kept = [b for b in buffers if any(o is b for o in kit.retained())]
            assert not kept, f"a buffer stayed kept when the releases interleaved at line {step}"
        assert step > 1, "the let-go ran no line to interleave"

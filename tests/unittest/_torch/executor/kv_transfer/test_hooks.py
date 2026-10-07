# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``KVTransferHooks``: the loop hooks over the real coordinator and effects, with fakes for
the reader and the backends. Covered: which requests are fetch candidates and which finished
prefills are offered for publish; ``has_pending_work`` for idle detection and ``pace_idle``;
``close()`` order, its timeout against a hanging backend, the shutdown drain and the status
dump's JSON schema; capped rejections; the executor's one scheduler call and its protected
set; the host-first path through the scheduler seam. The effects have ``test_effects.py``,
the release gate ``test_release_gate.py``.
"""

import json
import threading
import time
from types import SimpleNamespace

import pytest
from engine_fakes import (
    CONTEXT_INIT,
    TOKEN_END,
    TPB,
    EngineRig,
    FakeKVCache,
    FakeKVCacheManager,
    HangingClose,
    extent_names,
    finish_prefill,
    make_request,
)

from tensorrt_llm._torch.disaggregation.remote_cache import DEFER, FetchPlan
from tensorrt_llm._torch.pyexecutor.kv_transfer.effects import (
    KV_FETCH_IN_PROGRESS,
    EngineRequestView,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState

pytestmark = pytest.mark.cpu_only

RECORD_KEYS = {
    "request_id",
    "direction",
    "state",
    "try_index",
    "attempts",
    "outcomes",
    "deadline",
    "expired",
    "token_end",
    "gave_up_launching",
    "peer_launch_seen_at",
    "has_landing",
    "resource_wait_since",
    "retries_left",
    "consecutive_launch_failures",
    "retry_cap",
    "committed_names",
}
"""The keys of one record entry of the coordinator's status dump."""


# =============================================================================================
# Publish selection
# =============================================================================================


class TestPublishSelection:
    def test_only_finished_prefills_with_pages_are_offered(self, rig):
        done = make_request(1, 100)
        finish_prefill(done)
        rig.kv.kv_cache_map[1] = FakeKVCache(history_length=100)
        mid = make_request(2, 100)  # prefill not over
        rig.kv.kv_cache_map[2] = FakeKVCache(history_length=64)
        failed = make_request(3, 100)  # failed: its pages are gone
        finish_prefill(failed)
        failed.state = LlmRequestState.GENERATION_COMPLETE
        first_token_done = make_request(4, 100, max_new_tokens=1)  # finished with prefill
        finish_prefill(first_token_done)
        rig.kv.kv_cache_map[4] = FakeKVCache(history_length=100)
        first_token_done.state = LlmRequestState.GENERATION_COMPLETE
        dummy = make_request(5, 100)
        finish_prefill(dummy)
        rig.kv.kv_cache_map[5] = FakeKVCache(history_length=100)
        dummy.is_dummy_request = True

        rig.hooks.publish_committed_blocks([done, mid, failed, first_token_done, dummy])

        assert [c for c in rig.reader.calls if c[0] == "publish_extent_and_chunk"] == [
            ("publish_extent_and_chunk", 1),
            ("publish_extent_and_chunk", 4),
        ]
        assert rig.publisher.count("publish") == 2
        assert all(a.payload.is_last for a in rig.publisher.attempts)

    def test_generation_only_request_is_not_offered(self, rig):
        from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestType

        req = make_request(1, 100)
        req.py_llm_request_type = LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY
        assert req.is_generation_only_request
        finish_prefill(req)
        rig.kv.kv_cache_map[1] = FakeKVCache(history_length=100)
        rig.hooks.publish_committed_blocks([req])
        assert rig.publisher.count("publish") == 0

    def test_a_suspended_cache_is_not_offered(self, rig):
        """A request whose cache was suspended (evicted to a lower tier) has no pages on the
        device to read; it is skipped rather than published from memory that is not there."""
        req = make_request(1, 100)
        finish_prefill(req)
        cache = FakeKVCache(history_length=100)
        cache.is_active = False
        rig.kv.kv_cache_map[1] = cache
        rig.hooks.publish_committed_blocks([req])
        assert rig.publisher.count("publish") == 0
        assert rig.reader.calls == []

    def test_no_publisher_configured_offers_nothing(self):
        rig = EngineRig(publish=False)
        req = make_request(1, 100)
        finish_prefill(req)
        rig.kv.kv_cache_map[1] = FakeKVCache(history_length=100)
        rig.hooks.publish_committed_blocks([req])
        assert rig.reader.calls == []
        assert rig.executor._terminate_request(req) is None
        assert rig.terminations() == 1  # nothing to hold it


# =============================================================================================
# Idle detection, pacing, candidates
# =============================================================================================


class TestIdleAndCandidates:
    """``has_pending_work`` and ``pace_idle`` for the loop's idle detection, and which requests
    are fetch candidates at all."""

    def test_has_pending_work_reports_fetches_and_publishes(self, rig):
        """True while this rank has work, and for one more round after it ends: the round
        that ends it gathers the ranks' words while the record still exists."""
        assert rig.hooks.has_pending_work() is False
        req = make_request(1, 100)
        rig.plan(req)  # planned, not launched: waiting for pages on this layer's clock
        assert rig.hooks.has_pending_work() is True
        rig.reserve(req, TOKEN_END)
        attempt = rig.launch(req)
        assert rig.hooks.has_pending_work() is True
        attempt.deliver_all()
        rig.advance(req)
        assert rig.coord.has_pending_work() is False and rig.hooks.has_pending_work() is True
        rig.advance(req)
        assert rig.hooks.has_pending_work() is False
        pub = rig.publish(req)
        assert rig.hooks.has_pending_work() is True
        pub.deliver_all()
        rig.advance()
        assert rig.coord.has_pending_work() is False and rig.hooks.has_pending_work() is True
        rig.advance()
        assert rig.hooks.has_pending_work() is False

    def test_pace_idle_sleeps_only_when_a_backend_can_make_progress(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
        rig = EngineRig(probe_answer=None)  # the store never answers: the request is deferred
        rig.hooks.pace_idle()
        assert sleeps == []
        req = make_request(1, 100)
        rig.advance(req)
        assert rig.hooks.fetch_answer(req) is DEFER
        rig.hooks.pace_idle()
        assert sleeps == [0.001]
        # An in-flight transfer paces too.
        rig2 = EngineRig()
        req2 = make_request(2, 100)
        rig2.plan_reserve_launch(req2)
        rig2.hooks.pace_idle()
        assert sleeps == [0.001, 0.001]
        rig2.store.attempts[-1].deliver_all()
        rig2.advance(req2)
        rig2.hooks.pace_idle()  # the landing round's gathered word: one more yield
        assert sleeps == [0.001, 0.001, 0.001]
        rig2.advance(req2)
        rig2.hooks.pace_idle()
        assert sleeps == [0.001, 0.001, 0.001]

    def test_pace_idle_sleeps_while_a_fetch_waits_for_pages_or_a_request_is_held(self, monkeypatch):
        """A planned fetch the scheduler finds no pages for, and a finished request held until
        the ranks agree, both move on this layer's clocks alone: the loop must not spin through
        them at full speed (the deadlock detector would count a thousand passes in a second)."""
        sleeps = []
        monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
        rig = EngineRig()
        req = make_request(1, 100)
        rig.plan(req)  # planned, waiting for pages the scheduler never finds
        assert rig.hooks.has_pending_work() is True
        rig.hooks.pace_idle()
        assert sleeps == [0.001]
        rig.executor._terminate_request(req)  # held until the agreement, one round away
        assert rig.hooks.has_pending_work() is True
        rig.hooks.pace_idle()
        assert sleeps == [0.001, 0.001]
        rig.advance()
        assert rig.terminations() == 1 and not rig.coord.has_pending_work()
        rig.hooks.pace_idle()  # that round's gathered word: one more yield
        assert sleeps == [0.001, 0.001, 0.001]
        rig.advance()
        assert not rig.hooks.has_pending_work()
        rig.hooks.pace_idle()
        assert sleeps == [0.001, 0.001, 0.001]

    def test_candidates_are_first_chunk_context_init_non_dummy_only(self, rig):
        first = make_request(1, 100)
        dummy = make_request(2, 100)
        dummy.is_dummy_request = True
        continuation = make_request(3, 100)
        continuation.context_chunk_size = 32
        continuation.move_to_next_context_chunk()
        assert not continuation.is_first_context_chunk
        gen_init = make_request(4, 100)
        gen_init.state = LlmRequestState.DISAGG_GENERATION_INIT
        short = make_request(5, 20)  # nothing nameable: planned None, never probed

        rig.advance(first, dummy, continuation, gen_init, short)

        assert isinstance(rig.hooks.fetch_answer(first), FetchPlan)
        assert rig.hooks.fetch_answer(short) is None
        assert rig.hooks.fetch_answer(gen_init) is None  # gen-init receives are the transceiver's
        probed = {name for _, (name, _) in ((m, a) for m, a in rig.store.calls if m == "probe")}
        assert len(probed) == 1  # only ``first`` reached the store
        for other in (dummy, continuation):
            assert rig.coord.fetch_answer(EngineRequestView(other)) is DEFER  # never planned


# =============================================================================================
# close(): order, timeout, status dump
# =============================================================================================


def wait_for_closer_thread_to_exit(timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while any(t.name == "kv-transfer-close" for t in threading.enumerate()):
        assert time.monotonic() < deadline, "the close thread did not exit"
        time.sleep(0.01)


class TestClose:
    def test_close_stops_backends_then_frees_held_and_parked_then_writes_the_dump(
        self, tmp_path, monkeypatch
    ):
        dump_path = tmp_path / "kvt.json"
        rig = EngineRig(status_dump_path=str(dump_path))
        held = make_request(1, 100)
        rig.publish(held)
        rig.executor._terminate_request(held)
        parked = make_request(2, 100)
        rig.plan_reserve_launch(parked)
        assert rig.held_and_parked() == ({1}, {2})
        real_write = rig.hooks._write_status_dump
        monkeypatch.setattr(
            rig.hooks,
            "_write_status_dump",
            lambda: (rig.trace.append("dump"), real_write())[1],
        )

        rig.hooks.close()

        assert rig.trace[0] == "backend.close"
        assert sorted(rig.trace[1:3]) == [("free", 1), ("free", 2)]
        assert rig.trace[3] == "dump"
        assert rig.executor._do_terminate_request.call_count == 0  # not through terminate
        assert dump_path.exists()
        # Idempotent: a second close does nothing more.
        rig.hooks.close()
        assert rig.trace.count("backend.close") == 1
        assert rig.executor._free_request_resources.call_count == 2

    def test_close_with_nothing_held_frees_nothing(self, rig):
        rig.hooks.close()
        assert rig.trace == ["backend.close"]
        rig.executor._free_request_resources.assert_not_called()

    def test_close_gives_up_a_hanging_backend_after_the_timeout(self, tmp_path, hooks_logger):
        """The backends did not stop: a request whose record is still in flight keeps its pages
        (a backend may still be writing them); one whose transfers are all over is freed; the
        dump is still written."""
        hanging = HangingClose()
        dump_path = tmp_path / "kvt.json"
        rig = EngineRig(close_timeout_s=0.2, backend_close=hanging, status_dump_path=str(dump_path))
        held = make_request(1, 100)
        rig.publish(held)
        rig.executor._terminate_request(held)
        parked = make_request(2, 100)
        rig.plan_reserve_launch(parked)
        try:
            started = time.monotonic()
            rig.hooks.close()
            elapsed = time.monotonic() - started
            assert hanging.entered.is_set()
            assert 0.2 <= elapsed < 5.0
            hooks_logger.error.assert_called_once()
            assert "did not close within" in hooks_logger.error.call_args.args[0]
            kept = sorted(c.args[1] for c in hooks_logger.warning.call_args_list)
            assert kept == [1, 2]
            rig.executor._free_request_resources.assert_not_called()
            assert dump_path.exists()
            assert 1 in rig.kv.kv_cache_map and 2 in rig.kv.kv_cache_map
        finally:
            hanging.gate.set()
            wait_for_closer_thread_to_exit()

    def test_close_within_the_timeout_logs_no_error(self, hooks_logger):
        rig = EngineRig(close_timeout_s=5.0)
        rig.hooks.close()
        hooks_logger.error.assert_not_called()
        wait_for_closer_thread_to_exit()

    def test_stop_check_sees_a_publish_submitted_and_held_after_this_rounds_advance(self, rig):
        """Round N: the loop head's ``advance_round`` runs before the forward; the publish is
        submitted after the forward and the gate holds the finished request in the response
        pass. Round N+1 consumes the shutdown item: its stop check, with no advance in between,
        must keep the loop running on this rank's own word, since no gathered word has seen the
        record yet."""
        rig.advance()
        held = make_request(1, 100)
        rig.publish(held)
        rig.executor._terminate_request(held)
        rig.executor.is_shutdown = True
        assert rig.coord.any_rank_pending is False
        assert rig.hooks.any_rank_has_pending_work()

    def test_shutdown_drain_lands_the_held_publish_and_close_finds_nothing_left(self, tmp_path):
        dump_path = tmp_path / "kvt.json"
        rig = EngineRig(status_dump_path=str(dump_path))
        held = make_request(1, 100)
        attempt = rig.publish(held)
        rig.executor._terminate_request(held)
        rig.executor.is_shutdown = True
        assert rig.hooks.any_rank_has_pending_work()

        attempt.deliver_all()
        rig.advance()  # the drain round: the publish lands, the held request is terminated
        assert rig.records() == [] and rig.coord.status_dump()["finished_pending"] == []
        assert rig.terminations() == 1
        assert rig.hooks.any_rank_has_pending_work()  # that round's word still saw the record
        rig.advance()
        assert not rig.hooks.any_rank_has_pending_work()

        rig.hooks.close()
        assert rig.trace == ["backend.close"]  # nothing left to free
        with open(dump_path, encoding="utf-8") as f:
            coordinator = json.load(f)["coordinator"]
        assert coordinator["records"] == [] and coordinator["finished_pending"] == []
        assert coordinator["any_rank_pending"] is False

    def test_shutdown_drain_stops_counting_its_own_work_after_close_timeout(self, clock):
        """The attempt never finishes. ``close_timeout_s`` after the first round at shutdown
        with nothing left but transfers this rank reports itself drained, and its word turns
        False with the record still in flight, so a hung store cannot keep the loop alive;
        idle pacing still sees the record."""
        rig = EngineRig(close_timeout_s=5.0)
        held = make_request(1, 100)
        rig.publish(held)
        rig.executor._terminate_request(held)
        rig.executor.is_shutdown = True
        rig.executor.active_requests = [make_request(2, 100)]  # a request still running
        rig.advance()  # not quiescent: the drain clock does not start
        clock["t"] += 10.0
        rig.executor.active_requests = []
        rig.advance()  # the first quiescent round at shutdown starts the drain clock
        assert rig.hooks.any_rank_has_pending_work()
        clock["t"] += 4.9
        rig.advance()
        assert rig.hooks.any_rank_has_pending_work() and not rig.coord.any_rank_drained
        clock["t"] += 0.1
        rig.advance()  # past the deadline: this rank reports itself drained
        assert rig.coord.any_rank_drained
        assert not rig.hooks.any_rank_has_pending_work()
        assert rig.hooks.has_pending_work()
        assert [r["state"] for r in rig.records()] == ["IN_FLIGHT"]

    def test_status_dump_schema(self, tmp_path):
        dump_path = tmp_path / "kvt.json"
        rig = EngineRig(status_dump_path=str(dump_path))
        held = make_request(1, 100)
        rig.publish(held)
        rig.executor._terminate_request(held)
        parked = make_request(2, 100)
        rig.plan_reserve_launch(parked)

        rig.hooks.close()

        with open(dump_path, encoding="utf-8") as f:
            dump = json.load(f)
        assert set(dump) == {"started_at", "pid", "rank", "coordinator", "backends"}
        assert dump["started_at"] == 123.5
        import os

        assert dump["pid"] == os.getpid()
        assert dump["rank"] is None  # the rig does not name a rank; the assembly does
        coordinator = dump["coordinator"]
        assert set(coordinator) == {
            "plan_authority",
            "any_rank_pending",
            "any_rank_drained",
            "records",
            "decided_plans",
            "finished_pending",
        }
        assert coordinator["plan_authority"] == "ALL_RANKS"
        assert coordinator["any_rank_pending"] is True  # the last advance saw the planned fetch
        assert coordinator["finished_pending"] == [1]
        assert isinstance(coordinator["decided_plans"], int)
        records = {(r["request_id"], r["direction"]): r for r in coordinator["records"]}
        assert set(records) == {(1, "publish"), (2, "fetch")}
        for record in records.values():
            assert set(record) == RECORD_KEYS
        assert records[(1, "publish")]["state"] == "IN_FLIGHT"
        assert records[(2, "fetch")]["state"] == "IN_FLIGHT"
        assert records[(2, "fetch")]["token_end"] == 96
        assert records[(1, "publish")]["token_end"] is None
        assert dump["backends"] == [
            {
                "name": "store",
                "type": "fake",
                "roles": ["fetch", "publish"],
                "landing": "device",  # the handle's default; a host-first backend says "host"
                "counters": {"fetch_hits": 1, "publish_stored": 1},  # read at dump time
            }
        ]

    def test_no_dump_path_writes_nothing(self, tmp_path):
        rig = EngineRig(status_dump_path=None)
        rig.hooks.close()
        assert list(tmp_path.iterdir()) == []

    def test_status_dump_is_readable_before_close(self, rig):
        dump = rig.hooks.status_dump()
        assert dump["coordinator"] == {
            "plan_authority": "ALL_RANKS",
            "any_rank_pending": False,
            "any_rank_drained": False,
            "records": [],
            "decided_plans": 0,
            "finished_pending": [],
        }
        assert dump["backends"][0]["name"] == "store"


# =============================================================================================
# Rejected submissions, the one scheduler call, the recompute-pause guard
# =============================================================================================


class TestRejectedSubmissions:
    def test_consecutive_rejections_are_capped_then_the_request_computes_locally(self, rig):
        """Back-pressure is a reason to try again, not forever: after a bounded run of
        ``SubmissionRejected`` the rank gives the plan up and answers the scheduler DEFER; the
        next round's agreement spends the retry on a fresh plan, a second run gives that up
        too, and the agreement after it settles on local compute."""
        rig.store.reject_next_calls = 100
        req = make_request(1, 100)
        for _ in range(12):
            rig.advance(req)
            plan = rig.hooks.fetch_answer(req)
            if plan is None:
                break
            if plan is DEFER:
                continue  # given up: waiting for the agreement, nothing reserved this round
            assert isinstance(plan, FetchPlan)
            rig.reserve(req, plan.token_end)
            rig.hooks.launch_reserved_fetches([req])
        assert rig.hooks.fetch_answer(req) is None, "rejections were never capped"
        assert rig.store.count("fetch") == 6  # 3 per plan, two plans
        assert rig.store.attempts == []  # nothing escaped
        assert rig.executor._revert_ctx_alloc.call_count == rig.store.count("fetch")
        assert req.state == CONTEXT_INIT and not rig.hooks.owns(req)
        assert rig.records() == []

    def test_publish_expiry_of_a_running_request_only_warns(self, clock):
        rig = EngineRig(publish_timeout_s=10.0)
        rig.wire_engine_error_path()
        req = make_request(1, 100)
        attempt = rig.publish(req)
        clock["t"] += 11.0
        rig.advance()
        rig.executor._handle_errors.assert_not_called()
        assert [r["expired"] for r in rig.records()] == [True]  # the warning is the only word
        attempt.deliver_all()
        rig.advance()
        assert rig.records() == []
        rig.executor._terminate_request(req)  # ends normally later
        assert rig.terminations() == 1


class TestScheduleActiveRequests:
    """The engine's one scheduler call. Without the KV transfer layer it is upstream's
    two-argument call, so a scheduler that knows nothing of the layer still runs; with the layer
    attached the requests with a transfer in flight ride along as the protected set."""

    def test_schedule_without_kv_transfer_makes_upstreams_two_argument_call(self, rig):
        rig.executor.kv_transfer = None
        req = make_request(1, 100)
        rig.executor.active_requests = [req]
        rig.executor.inflight_req_ids = {7}

        class UpstreamScheduler:
            def schedule_request(self, active_requests, inflight_request_ids):
                return ("scheduled", active_requests, inflight_request_ids)

        rig.executor.scheduler = UpstreamScheduler()
        assert rig.executor._schedule_active_requests() == ("scheduled", [req], {7})

    def test_schedule_with_kv_transfer_passes_the_pages_in_flight_as_protected(self, rig):
        publishing = make_request(1, 100)
        rig.publish(publishing)  # still running: not finished, publish IN_FLIGHT
        plain = make_request(2, 100)
        rig.executor.active_requests = [publishing, plain]
        rig.executor.inflight_req_ids = set()
        seen = {}

        class RecordingScheduler:
            def schedule_request(
                self, active_requests, inflight_request_ids, *, protected_from_eviction_request_ids
            ):
                seen.update(
                    active=active_requests,
                    inflight=inflight_request_ids,
                    protected=protected_from_eviction_request_ids,
                )
                return "scheduled"

        rig.executor.scheduler = RecordingScheduler()
        assert rig.executor._schedule_active_requests() == "scheduled"
        assert seen == dict(active=[publishing, plain], inflight=set(), protected={1})
        assert seen["protected"] == rig.coord.inflight_request_ids()


class TestRecomputePauseProtection:
    """A request whose publish is in flight must keep its pages: the executor's recompute-pause
    teardown skips it (the scheduler never picks it as a victim either; see the scheduler seam
    tests)."""

    def test_terminate_recompute_paused_requests_skips_a_request_with_a_publish_in_flight(
        self, rig
    ):
        publishing = make_request(1, 100)
        rig.publish(publishing)  # still running: not finished, publish IN_FLIGHT
        assert 1 in rig.coord.inflight_request_ids()
        plain = make_request(2, 100)
        batch = SimpleNamespace(recompute_paused_requests=[publishing, plain])

        rig.executor._terminate_recompute_paused_requests(batch)

        rig.executor._free_request_resources.assert_called_once_with(plain)
        assert 1 in rig.kv.kv_cache_map

    def test_terminate_recompute_paused_requests_frees_once_the_publish_landed(self, rig):
        req = make_request(1, 100)
        attempt = rig.publish(req)
        attempt.deliver_all()
        rig.advance()
        assert rig.coord.inflight_request_ids() == frozenset()
        rig.executor._terminate_recompute_paused_requests(
            SimpleNamespace(recompute_paused_requests=[req])
        )
        rig.executor._free_request_resources.assert_called_once_with(req)


class TestCandidatesAndPublishers:
    def test_plan_fetch_answers_none_for_every_non_candidate(self, rig):
        """Scenario: a planner is attached, yet dummies, chunk continuations, gen-init and
        DISAGG_CONTEXT_INIT_AND_TRANS requests take the ordinary path: ``fetch_answer`` is None."""
        dummy = make_request(1, 100)
        dummy.is_dummy_request = True
        continuation = make_request(2, 100)
        continuation.context_chunk_size = 32
        continuation.move_to_next_context_chunk()
        gen_init = make_request(3, 100)
        gen_init.state = LlmRequestState.DISAGG_GENERATION_INIT
        ctx_and_trans = make_request(4, 100)
        ctx_and_trans.state = LlmRequestState.DISAGG_CONTEXT_INIT_AND_TRANS
        for request in (dummy, continuation, gen_init, ctx_and_trans):
            assert rig.hooks.fetch_answer(request) is None
            rig.advance(request)
            assert rig.hooks.fetch_answer(request) is None
        assert rig.store.count("probe") == 0
        assert rig.coord.status_dump()["decided_plans"] == 0

    def test_max_tokens_one_request_is_published(self, rig):
        """A request that finishes with its first token still publishes: its prefill ended and
        its pages are there (the publish selection reads the prefill cursor and the pages, not
        the request state)."""
        req = make_request(1, 100, max_new_tokens=1)
        finish_prefill(req)
        req.py_decoding_iter = 1
        req.state = LlmRequestState.GENERATION_COMPLETE
        rig.kv.kv_cache_map[1] = FakeKVCache(history_length=100)
        rig.hooks.publish_committed_blocks([req])
        assert rig.publisher.count("publish") == 1
        extent = rig.publisher.attempts[0].payload
        assert extent.is_last and len(extent.units) == 3  # (100 - 1) // 32 nameable blocks


# =============================================================================================
# The engine fakes mirror the wrapper where the hooks depend on it
# =============================================================================================


class TestEngineFakesMirrorTheWrapper:
    def test_reserving_again_for_a_request_with_a_cache_keeps_the_cache(self):
        """The wrapper's ``reserve_transfer_pages`` reuses a request's cache and never lowers its
        history; the fake does the same, so a second reservation (the scheduler asking again for
        a plan whose launch did not go through) keeps the committed tokens of the first."""
        kv = FakeKVCacheManager(TPB)
        req = make_request(1, 100)
        assert kv.reserve_transfer_pages(req, TOKEN_END)
        first = kv.kv_cache_map[1]
        first.num_committed_tokens = 32
        assert kv.reserve_transfer_pages(req, 64)
        assert kv.kv_cache_map[1] is first
        assert first.history_length == TOKEN_END and first.num_committed_tokens == 32
        assert kv.reserve_transfer_pages(req, 128)
        assert kv.kv_cache_map[1] is first and first.history_length == 128
        assert first.capacity >= 128


# =============================================================================================
# Host-first fetch through the scheduler seam
# =============================================================================================


class TestHostFirstSchedulerSeam:
    """A ``LandsOnHost`` store: the landing starts when the plan is decided, with no pages; the
    scheduler is asked to reserve only once the landing is complete, and keeps being asked
    while it cannot; the placement into the pages then parks the request as a fetch does."""

    def scheduler_round(self, rig: EngineRig, req) -> bool:
        """The scheduler's fetch path for one request in one round: ask for the plan, reserve
        pages for it, queue the request for launch when the reservation went through."""
        plan = rig.hooks.fetch_answer(req)
        if not isinstance(plan, FetchPlan):
            return False
        if not rig.kv.reserve_transfer_pages(req, plan.token_end):
            return False
        req.py_ctx_pre_resize_cap = 0
        rig.slots.add(req)
        rig.hooks.launch_reserved_fetches([req])
        return True

    def test_landing_starts_with_the_plan_and_wants_no_pages_yet(self):
        rig = EngineRig(host_first=True)
        req = make_request(1, 100)
        rig.advance(req)
        assert rig.store.count("fetch_to_host") == 1 and rig.hooks.fetch_answer(req) is DEFER
        assert rig.kv.count("reserve_transfer_pages") == 0
        assert req.state == CONTEXT_INIT and not rig.hooks.owns(req)
        assert rig.hooks.has_pending_work()  # a landing paces the idle loop ...
        assert rig.hooks.inflight_request_ids() == frozenset()  # ... but protects no page
        assert [r["state"] for r in rig.records()] == ["LANDING"]
        assert rig.executor._try_cancel_request(req) is True  # not parked: cancellable

    def test_landed_request_is_asked_for_pages_every_round_and_placed_once_they_come(self):
        rig = EngineRig(host_first=True)
        req = make_request(1, 100)
        rig.advance(req)
        landing = rig.store.landings[0]
        landing.deliver_all()
        rig.advance(req)
        assert [r["state"] for r in rig.records()] == ["LANDED"]

        rig.kv.reserve_answer = False
        for _ in range(3):
            assert self.scheduler_round(rig, req) is False
            rig.advance(req)
        assert rig.kv.count("reserve_transfer_pages") == 3
        assert [r["state"] for r in rig.records()] == ["LANDED"]
        assert rig.store.count("place") == 0 and req.state == CONTEXT_INIT
        assert landing.closes == 0

        rig.kv.reserve_answer = True
        assert self.scheduler_round(rig, req) is True
        assert rig.store.count("place") == 1
        assert req.state == KV_FETCH_IN_PROGRESS and rig.coord.parked_request_ids() == {1}
        assert rig.hooks.owns(req) and rig.hooks.inflight_request_ids() == {1}
        (placed,) = rig.store.attempts
        assert extent_names(placed.payload) == rig.reader.unit_names(req, range(3))

        placed.deliver_all()
        rig.advance(req)
        assert req.state == CONTEXT_INIT and req.context_current_position == TOKEN_END
        assert landing.closes == 1 and not rig.hooks.owns(req)
        assert [r["state"] for r in rig.records()] == ["DELIVERED"]

    def test_request_finished_while_landing_releases_the_landing_at_once_then_terminates(self):
        """The landing names no page and goes now; the record stays one round to vote so every
        rank terminates the request together, and the layer terminates it with the agreement."""
        rig = EngineRig(host_first=True)
        req = make_request(1, 100)
        rig.advance(req)
        landing = rig.store.landings[0]
        rig.executor._terminate_request(req)
        assert rig.terminations() == 0 and rig.coord.held_request_ids() == {1}
        assert landing.closes == 1 and not rig.coord.has_backend_work()
        assert rig.hooks.has_pending_work()  # held: the loop must not sleep on its queue
        rig.advance()
        rig.executor._do_terminate_request.assert_called_once_with(req)
        assert rig.records() == [] and rig.reader.forgotten == [1]
        rig.advance()
        assert rig.terminations() == 1

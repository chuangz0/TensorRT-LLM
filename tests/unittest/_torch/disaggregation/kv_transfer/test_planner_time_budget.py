# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""U4 (integration plan §11, §10 #5): the planner's wall-clock probe budget.

``Planner(probe_timeout_s=, clock=)`` with an injected clock: a request waiting on a store probe is
deferred until ``probe_timeout_s`` has passed since its first deferral, then planned without the
store. ``probe_budget_rounds=None`` leaves the wait to the clock alone; with both set, whichever
budget runs out first ends the wait. Import-light like the ``kv_transfer`` suite.
"""

import pytest

__extra_import_path__ = ["~/tensorrt_llm/_torch"]
from disaggregation.orchestration.kv_transfer_interfaces import DEFER, FetchSource  # noqa: E402
from disaggregation.orchestration.remote_cache import FetchPlan, Planner  # noqa: E402
from fakes import TPB, FakeFetches, FakeReader, FakeRequest, full_attention  # noqa: E402

pytestmark = pytest.mark.cpu_only


class Clock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def make_planner(clock, **kw) -> tuple[Planner, FakeReader]:
    reader = FakeReader(groups=[full_attention(0)])
    store = FakeFetches(name="store", single_destination=True)
    planner = Planner([FetchSource("store", store, None)], reader, TPB, clock=clock, **kw)
    return planner, reader


UNANSWERED = {}  # the store has not answered the probe yet
STILL_UNANSWERED = {"store": None}


def test_defers_until_probe_timeout_then_plans_without_the_store():
    clock = Clock()
    planner, _ = make_planner(clock, probe_budget_rounds=None, probe_timeout_s=0.5)
    req = FakeRequest(1, prompt_len=29)

    assert planner.decide(req, UNANSWERED) is DEFER  # first deferral starts the clock
    clock.now += 0.2
    assert planner.decide(req, STILL_UNANSWERED) is DEFER
    clock.now += 0.2999
    assert planner.decide(req, UNANSWERED) is DEFER  # 0.4999 s: still inside the budget
    clock.now += 0.0001
    assert planner.decide(req, UNANSWERED) is None  # 0.5 s: budget spent, compute locally


def test_rounds_none_does_not_end_the_wait_by_count():
    clock = Clock()
    planner, _ = make_planner(clock, probe_budget_rounds=None, probe_timeout_s=1.0)
    req = FakeRequest(1, prompt_len=29)
    for _ in range(200):
        assert planner.decide(req, UNANSWERED) is DEFER  # clock stands still
    clock.now += 1.0
    assert planner.decide(req, UNANSWERED) is None


def test_an_answer_within_the_budget_is_used():
    clock = Clock()
    planner, reader = make_planner(clock, probe_budget_rounds=None, probe_timeout_s=0.5)
    req = FakeRequest(1, prompt_len=29)
    assert planner.decide(req, UNANSWERED) is DEFER
    clock.now += 0.3
    plan = planner.decide(req, {"store": reader.unit_names(req, range(7))})
    assert isinstance(plan, FetchPlan) and plan.source == "store" and plan.token_end == 28


def test_budget_is_per_request_and_restarts_after_a_decision():
    clock = Clock()
    planner, _ = make_planner(clock, probe_budget_rounds=None, probe_timeout_s=0.5)
    first, second = FakeRequest(1, prompt_len=29), FakeRequest(2, prompt_len=29)
    assert planner.decide(first, UNANSWERED) is DEFER
    clock.now += 0.4
    assert planner.decide(second, UNANSWERED) is DEFER  # its own clock starts now
    clock.now += 0.1
    assert planner.decide(first, UNANSWERED) is None  # 0.5 s for the first
    assert planner.decide(second, UNANSWERED) is DEFER  # 0.1 s for the second
    # The first request's deferral state is dropped with the decision: asked again, it waits anew.
    assert planner.decide(first, UNANSWERED) is DEFER
    clock.now += 0.4
    assert planner.decide(second, UNANSWERED) is None
    assert planner.decide(first, UNANSWERED) is DEFER
    clock.now += 0.1
    assert planner.decide(first, UNANSWERED) is None


def test_forget_restarts_the_clock():
    clock = Clock()
    planner, _ = make_planner(clock, probe_budget_rounds=None, probe_timeout_s=0.5)
    req = FakeRequest(1, prompt_len=29)
    assert planner.decide(req, UNANSWERED) is DEFER
    clock.now += 0.4
    planner.forget(req.py_request_id)
    assert planner.decide(req, UNANSWERED) is DEFER
    clock.now += 0.4  # 0.8 s since the original first deferral, 0.4 s since the restart
    assert planner.decide(req, UNANSWERED) is DEFER
    clock.now += 0.1
    assert planner.decide(req, UNANSWERED) is None


def test_both_budgets_set_rounds_run_out_first():
    clock = Clock()
    planner, _ = make_planner(clock, probe_budget_rounds=2, probe_timeout_s=10.0)
    req = FakeRequest(1, prompt_len=29)
    assert planner.decide(req, UNANSWERED) is DEFER
    clock.now += 0.001
    assert planner.decide(req, UNANSWERED) is DEFER
    clock.now += 0.001
    assert planner.decide(req, UNANSWERED) is None  # two rounds charged, clock far from 10 s


def test_both_budgets_set_clock_runs_out_first():
    clock = Clock()
    planner, _ = make_planner(clock, probe_budget_rounds=100, probe_timeout_s=0.5)
    req = FakeRequest(1, prompt_len=29)
    assert planner.decide(req, UNANSWERED) is DEFER
    clock.now += 0.6
    assert planner.decide(req, UNANSWERED) is None  # one round charged, clock past 0.5 s


def test_zero_timeout_allows_exactly_one_deferral():
    """The engine's default is a short positive budget; zero is the edge the assembly may pass
    when the operator wants no store wait at all -- one round to see the answer, then local."""
    clock = Clock()
    planner, _ = make_planner(clock, probe_budget_rounds=None, probe_timeout_s=0.0)
    req = FakeRequest(1, prompt_len=29)
    assert planner.decide(req, UNANSWERED) is DEFER
    assert planner.decide(req, UNANSWERED) is None


def test_no_budget_at_all_waits_indefinitely():
    clock = Clock()
    planner, _ = make_planner(clock, probe_budget_rounds=None, probe_timeout_s=None)
    req = FakeRequest(1, prompt_len=29)
    for _ in range(50):
        clock.now += 100.0
        assert planner.decide(req, UNANSWERED) is DEFER

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The planner's wall-clock probe budget.

``Planner(probe_timeout_s=)`` measured on the ``now`` each ``decide`` receives, the loop clock the
coordinator passes to ``advance``: a request waiting on a store probe is deferred until
``probe_timeout_s`` has passed since its first deferral, then planned without the store.
Import-light like the ``kv_transfer`` suite.
"""

import pytest

__extra_import_path__ = ["~/tensorrt_llm/_torch"]
from disaggregation.remote_cache import DEFER, FetchPlan, FetchSource, Planner  # noqa: E402
from fakes import TPB, FakeFetches, FakeReader, FakeRequest, full_attention  # noqa: E402

pytestmark = pytest.mark.cpu_only


def make_planner(**kw) -> tuple[Planner, FakeReader]:
    reader = FakeReader(groups=[full_attention(0)])
    store = FakeFetches(name="store", single_destination=True)
    planner = Planner([FetchSource("store", store, None)], reader, TPB, **kw)
    return planner, reader


UNANSWERED = {}  # the store has not answered the probe yet: the coordinator records nothing


def test_defers_until_probe_timeout_then_plans_without_the_store():
    planner, _ = make_planner(probe_timeout_s=0.5)
    req = FakeRequest(1, prompt_len=29)

    assert planner.decide(req, UNANSWERED, now=100.0) is DEFER  # first deferral starts the clock
    assert planner.decide(req, UNANSWERED, now=100.2) is DEFER
    assert planner.decide(req, UNANSWERED, now=100.4999) is DEFER  # still inside the budget
    assert planner.decide(req, UNANSWERED, now=100.5) is None  # budget spent, compute locally


def test_an_answer_within_the_budget_is_used():
    planner, reader = make_planner(probe_timeout_s=0.5)
    req = FakeRequest(1, prompt_len=29)
    assert planner.decide(req, UNANSWERED, now=100.0) is DEFER
    plan = planner.decide(req, {"store": reader.unit_names(req, range(7))}, now=100.3)
    assert isinstance(plan, FetchPlan) and plan.source == "store" and plan.token_end == 28


def test_budget_is_per_request_and_restarts_after_a_decision():
    planner, _ = make_planner(probe_timeout_s=0.5)
    first, second = FakeRequest(1, prompt_len=29), FakeRequest(2, prompt_len=29)
    assert planner.decide(first, UNANSWERED, now=100.0) is DEFER
    assert planner.decide(second, UNANSWERED, now=100.4) is DEFER  # its own clock starts now
    assert planner.decide(first, UNANSWERED, now=100.5) is None  # 0.5 s for the first
    assert planner.decide(second, UNANSWERED, now=100.5) is DEFER  # 0.1 s for the second
    # The first request's deferral state is dropped with the decision: asked again, it waits anew.
    assert planner.decide(first, UNANSWERED, now=100.5) is DEFER
    assert planner.decide(second, UNANSWERED, now=100.9) is None
    assert planner.decide(first, UNANSWERED, now=100.9) is DEFER
    assert planner.decide(first, UNANSWERED, now=101.0) is None


def test_forget_restarts_the_clock():
    planner, _ = make_planner(probe_timeout_s=0.5)
    req = FakeRequest(1, prompt_len=29)
    assert planner.decide(req, UNANSWERED, now=100.0) is DEFER
    planner.forget(req.py_request_id)
    assert planner.decide(req, UNANSWERED, now=100.4) is DEFER
    # 0.8 s since the original first deferral, 0.4 s since the restart.
    assert planner.decide(req, UNANSWERED, now=100.8) is DEFER
    assert planner.decide(req, UNANSWERED, now=100.9) is None


def test_zero_timeout_allows_exactly_one_deferral():
    """The engine's default is a short positive budget; zero is the edge the assembly may pass
    when the operator wants no store wait at all -- one round to see the answer, then local."""
    planner, _ = make_planner(probe_timeout_s=0.0)
    req = FakeRequest(1, prompt_len=29)
    assert planner.decide(req, UNANSWERED, now=100.0) is DEFER
    assert planner.decide(req, UNANSWERED, now=100.0) is None


def test_no_budget_at_all_waits_indefinitely():
    planner, _ = make_planner(probe_timeout_s=None)
    req = FakeRequest(1, prompt_len=29)
    for round_index in range(50):
        assert planner.decide(req, UNANSWERED, now=100.0 + 100.0 * round_index) is DEFER

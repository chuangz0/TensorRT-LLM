# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fakes satisfy the contract protocols structurally, and the optional protocols are
non-empty so that a plain object does not accidentally satisfy them (design §7.5)."""

import sys

import pytest

__extra_import_path__ = ["~/tensorrt_llm/_torch"]
from disaggregation.base.cache_backend import (  # noqa: E402
    Attempt,
    CacheExtent,
    Delivered,
    Fetches,
    Publishes,
    RegistersPools,
    Unit,
)
from disaggregation.orchestration.kv_transfer.interfaces import (  # noqa: E402
    CarriesAux,
    Landing,
    LandsOnHost,
    PlacesPieces,
)
from fakes import (  # noqa: E402
    FakeAttempt,
    FakeAuxAttempt,
    FakeFetches,
    FakeLandsOnHost,
    FakePlacingPublishes,
    FakePublishes,
    FakeRequest,
    FakeRoute,
    KVTransferCoordinator,
)

pytestmark = pytest.mark.cpu_only


def test_modules_under_test_were_not_imported_through_tensorrt_llm():
    # This suite imports the coordination layer straight from ``tensorrt_llm/_torch`` and never
    # needs ``tensorrt_llm`` itself; a parent conftest may import it, this directory does not.
    assert (
        KVTransferCoordinator.__module__ == "disaggregation.orchestration.kv_transfer.coordinator"
    )
    assert Fetches.__module__ == "disaggregation.base.cache_backend"
    through_package = (
        "tensorrt_llm._torch.disaggregation.base.views",
        "tensorrt_llm._torch.disaggregation.orchestration.kv_transfer",
        "tensorrt_llm._torch.disaggregation.remote_cache",
    )
    assert not any(name.startswith(through_package) for name in sys.modules)
    # One copy of the contract: the fakes and the coordinator see the same classes.
    assert isinstance(FakeFetches(), Fetches)


def test_fake_fetches_satisfies_fetches_fully():
    fake = FakeFetches()
    assert isinstance(fake, Fetches)
    for member in ("fetch", "quiesce", "settle", "probe", "open_route"):
        assert callable(getattr(fake, member))
    assert not isinstance(fake, Publishes)  # it has no ``publish``


def test_fake_publishes_satisfies_publishes():
    assert isinstance(FakePublishes(), Publishes)
    assert isinstance(FakePlacingPublishes(), Publishes)
    assert not isinstance(FakePublishes(), Fetches)


def test_places_pieces_is_non_empty():
    assert isinstance(FakePlacingPublishes(), PlacesPieces)
    assert not isinstance(FakePublishes(), PlacesPieces)
    assert not isinstance(object(), PlacesPieces)


def test_carries_aux_is_non_empty():
    extent = CacheExtent(name=b"x", units=(), is_last=True)
    assert isinstance(FakeAuxAttempt(extent, {"a": 1}), CarriesAux)
    assert isinstance(FakeAuxAttempt(extent, {}), Attempt)
    assert not isinstance(FakeAttempt(extent), CarriesAux)
    assert not isinstance(object(), CarriesAux)


def test_attempt_protocol():
    extent = CacheExtent(name=b"x", units=(), is_last=True)
    assert isinstance(FakeAttempt(extent), Attempt)
    assert not isinstance(object(), Attempt)


def test_fakes_do_not_claim_registers_pools():
    assert not isinstance(FakeFetches(), RegistersPools)
    assert not isinstance(FakePublishes(), RegistersPools)


def test_single_destination_refuses_routes():
    store = FakeFetches(single_destination=True)
    with pytest.raises(NotImplementedError):
        store.open_route({"peer": "x"})
    worker = FakeFetches()
    route = worker.open_route({"peer": "x"})
    assert isinstance(route, FakeRoute)
    route.close()
    route.close()
    assert route.closed == 2  # the fake counts; the coordinator is what must close once


def test_fake_request_has_the_request_view_surface():
    req = FakeRequest(1, prompt_len=10)
    for attr in (
        "py_request_id",
        "prompt_len",
        "is_gen_init",
        "is_gen_first_context",
        "route_hints",
    ):
        assert hasattr(req, attr)
    assert req.route_hints == {}


def test_lands_on_host_and_landing_are_non_empty_and_disjoint_from_fetches():
    host = FakeLandsOnHost()
    assert isinstance(host, LandsOnHost)
    assert not isinstance(host, Fetches)  # no ``fetch``, no ``open_route``
    assert not isinstance(FakeFetches(), LandsOnHost)  # no ``fetch_to_host``
    assert not isinstance(object(), LandsOnHost)
    landing = host.fetch_to_host(b"fetch:1", [b"u"])
    assert isinstance(landing, Landing)
    assert not isinstance(object(), Landing)
    assert isinstance(landing.place(CacheExtent(name=b"x", units=(), is_last=True)), Attempt)


# ---- the harness invariants the fakes enforce ----


def one_unit_extent() -> CacheExtent:
    return CacheExtent(name=b"x", units=(Unit(name=b"u", local_group=0, local=0),), is_last=True)


def test_fakes_refuse_quiesce_before_outcome():
    """A real backend's ``quiesce`` waits for a live attempt on the engine thread; the strict
    fakes fail the test instead, so a coordinator that asks too early is caught at once. Once
    the attempt has its outcome the same call is answered."""
    store = FakeFetches(strict_quiesce=True)
    live = store.fetch(one_unit_extent())
    with pytest.raises(AssertionError, match="quiesce asked under 1 live attempt"):
        store.quiesce([live])
    live.deliver_all()
    assert store.quiesce([live]) is True

    publisher = FakePublishes(strict_quiesce=True)
    pending = publisher.publish(one_unit_extent())
    with pytest.raises(AssertionError, match="live attempt"):
        publisher.quiesce([pending])

    host = FakeLandsOnHost(strict_quiesce=True)
    placement = host.fetch_to_host(b"n", [b"u"]).place(one_unit_extent())
    with pytest.raises(AssertionError, match="live attempt"):
        host.quiesce([placement])


def test_fake_landing_refuses_a_placement_after_release():
    """The coordinator forgets a landing when it releases it; a placement on a released landing
    is a harness invariant broken, and the fake fails the test rather than answering."""
    host = FakeLandsOnHost()
    landing = host.fetch_to_host(b"n", [b"u"])
    landing.release()
    with pytest.raises(AssertionError, match="after release"):
        landing.place(one_unit_extent())


def test_fakes_deliver_an_empty_extent_at_once_without_consuming_a_script():
    """An empty extent completes at once: ``Landing.place`` promises it, and the fetch fake
    mirrors the blob backend's ``admit_delivery`` (the ``Fetches`` contract leaves it open).
    The scripted outcome stays queued for the next extent that names something."""
    empty = CacheExtent(name=b"x", units=(), is_last=True)
    store = FakeFetches()
    store.script(Delivered(frozenset({b"u"})))
    assert store.fetch(empty).poll() == Delivered(frozenset())
    assert store.fetch(one_unit_extent()).poll() == Delivered(frozenset({b"u"}))

    host = FakeLandsOnHost()
    host.script_place(Delivered(frozenset({b"u"})))
    landing = host.fetch_to_host(b"n", [b"u"])
    assert landing.place(empty).poll() == Delivered(frozenset())
    assert landing.place(one_unit_extent()).poll() == Delivered(frozenset({b"u"}))

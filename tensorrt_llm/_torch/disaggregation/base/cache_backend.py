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
"""What a cache backend is asked to do, and when the caller may reuse the memory it named.

This is the surface a backend implements. It names content rather than requests: the same bytes
serve different requests, which is what reuse means, so a backend keyed on a request id could not
be a store.

How the bytes are laid out locally -- pages, layer groups, token ranges -- is not here. That
belongs to whatever builds a ``CacheExtent``, and a backend never sees it.

TODO:
1. Something outside the engine cannot implement this until the file leaves this package.
2. A unit's name and an extent's name are both opaque, so nothing here can check that the two
   levels were derived consistently.
3. Reuse across a different TP/PP split needs names derived below one rank's shard. Until then two
   deployments that shard differently compute different names and miss each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Protocol, Sequence, Union, runtime_checkable

__all__ = [
    "Attempt",
    "CacheExtent",
    "Cancelled",
    "Delivered",
    "Failed",
    "Fetches",
    "Outcome",
    "Publishes",
    "RegistersPools",
    "Registration",
    "Route",
    "SubmissionRejected",
    "Unit",
]


@dataclass(frozen=True, kw_only=True)
class Unit:
    """One piece of content, named once and delivered whole or not at all.

    Immutable, because handing one to ``fetch`` or ``publish`` lends it out and a backend may read
    it on another thread for as long as the delivery lasts. Keyword-only, because three of its
    fields would otherwise be positional and two of those are integers.
    """

    name: bytes
    """Opaque to the backend, which may only compare it for equality.

    Re-encoding is allowed while it stays reversible and equality-preserving. ``bytes`` carries its
    own length; spliced into a larger key it does not, so a re-encoding must stay separable.
    """

    local_group: int
    """Which of this process's layer groups the unit belongs to.

    Purely local. It must not be sent to a peer or compared against a peer's: the two sides order
    their layer groups independently, so the same ordinal need not mean the same layers. What the
    two sides share -- the group's identity -- is folded into ``name``.
    """

    local: int
    """The region id within that layer group. Region ids are not unique across groups."""

    def __post_init__(self):
        if self.local_group < 0 or self.local < 0:
            raise ValueError(f"negative local coordinate ({self.local_group}, {self.local})")


@dataclass(frozen=True, kw_only=True)
class CacheExtent:
    """One ask: which content, and which of this rank's units carry it.

    ``name`` is shared by both sides; ``units`` are not -- each side's units sit in its own memory.

    Immutable for the same reason a ``Unit`` is, and the reason is the whole of it: a backend that
    may read this on another thread needs it to still say what it said. A sequence passed in is
    copied to a tuple, so a caller keeping its own list and changing it afterwards changes nothing
    here.
    """

    name: bytes
    units: tuple[Unit, ...]
    is_last: bool
    """Whether this ends a pipelined series.

    Only a backend that works in series may read it. A store addresses by name and has no series,
    so it must not read this or infer anything from it.

    No default: leaving it out would end a series early, and in silence.
    """

    def __post_init__(self):
        object.__setattr__(self, "units", tuple(self.units))
        names = [unit.name for unit in self.units]
        if len(set(names)) != len(names):
            raise ValueError("two units in one extent share a name")


@dataclass(frozen=True)
class Delivered:
    """The backend answered. Not that the content arrived -- ``served`` says what did."""

    served: frozenset[bytes]
    """The names actually delivered, a subset of what was asked.

    No default: an empty set is a real answer about what happened, not the absence of one.

    Empty is a legal answer, and means different things by direction: fetching, the backend does not
    hold the content and the caller should compute it; publishing, nothing was taken.
    """


@dataclass(frozen=True)
class Failed:
    """Something went wrong. Not holding the content is not this -- that is an empty ``served``."""

    reason: str


@dataclass(frozen=True)
class Cancelled:
    """The delivery ended without delivering, because someone stopped it.

    This interface offers no way to start a cancel; a delivery may still be cancelled for reasons
    outside it, and the backend needs to be able to say so.
    """

    by_peer: bool
    """A local cancel is an ordinary end, a peer's is a transfer error, and only the backend knows
    which happened -- so there is no default."""


Outcome = Union[Delivered, Failed, Cancelled]
"""How one delivery ended. ``None`` from ``poll`` means it has not ended yet.

Reaching an outcome is the *logical* end. It does not mean the backend has stopped touching the
memory; only ``quiesce`` answering true guarantees that. Nor does going quiet mean an outcome has
been reached; only ``settle`` returning guarantees that.
"""


class SubmissionRejected(Exception):
    """A submission was refused outright, and nothing came of it.

    Raising this is a promise, not just a report: no asynchronous work was started, no peer was
    told, and none of the caller's memory was touched. The caller therefore has nothing to
    quiesce, which is the whole reason a submission may raise at all.

    Once any of those has escaped, the call must answer with an ``Attempt`` and report the failure
    through it. An exception at that point would leave the caller holding memory it cannot learn
    when to reuse.
    """


@runtime_checkable
class Attempt(Protocol):
    """One delivery of one extent."""

    def poll(self) -> Optional[Outcome]:
        """Non-blocking. ``None`` while no outcome has been reached."""
        ...


class Route(Protocol):
    """Where to read from for one request, in the terms of the backend that made it.

    Opaque: a caller may hold one and hand it back, and nothing else. Only the backend that made it
    can read it, which is what lets one carry a connection, an epoch or a source generation without
    any of those appearing here.

    It says where to read, not that anything is being held there. Nothing on this interface
    reserves content, so holding one is not a claim on what it points at.

    **One route belongs to one request.** A backend that plans a request's sources before its
    fetches needs somewhere to keep that plan, and it needs to be told when the request is over --
    which is what ``close`` is. Keying by source instead leaves nowhere to put a plan that is true
    for one request only, and nowhere to end it.
    """

    def close(self) -> None:
        """The request this route was opened for is over. Idempotent once it has succeeded.

        Not a cancel: deliveries already started keep running, and the memory they name is still
        released only by ``quiesce``. This says the plan will not be used again, so whatever the
        backend keyed on the request may go.

        **The caller stops fetching along a route before closing it.** A fetch submitted afterwards
        is refused rather than served, which is a caller's ordering mistake and not something the
        backend recovers from.

        **A close that raises has not closed.** The handle stays open and the caller may try again;
        counting a failed release as done would strand whatever it stood for with nothing left to
        reach it.
        """
        ...


@runtime_checkable
class Fetches(Protocol):
    """Somewhere cache can be pulled from.

    One instance serves one backend and lives for the process. It is bound to no name; the name
    arrives with each call.

    **Nothing here serialises.** Two calls may be in flight at once against one instance, for
    different requests and in either direction, and one ``Attempt`` may be handed to ``quiesce`` or
    ``settle`` more than once. A backend built on a client that cannot take that guards itself;
    there is no member for declining, because the useful statement turned out to be unsayable --
    a bound on overlapping calls cancels the interleaving that a blocking ``settle`` exists to
    allow, and a bound on how long each touch of the client lasts is not something a caller can
    honour from out here.

    **Every member here is required; what varies is the answer.** A backend that cannot probe more
    cheaply than fetching says so by answering ``None``, and one with a single source says so by
    refusing a route -- neither says it by leaving the method out. Declaring a member and calling
    it optional does not work: a protocol member is required by structural typing whatever its
    body says, so the two readings would disagree and only one of them is checkable.
    """

    def fetch(self, extent: CacheExtent, *, route: Optional[Route] = None) -> Attempt:
        """Start pulling ``extent``, optionally along a route this backend made earlier.

        A backend must refuse a route it did not make, and one that cannot route at all must
        refuse any route. Reading from somewhere else instead is a disagreement nothing reports.

        Raising ``SubmissionRejected`` is allowed only while nothing has escaped; past that point
        the failure belongs in an ``Attempt``.
        """
        ...

    def quiesce(self, attempts: Iterable[Attempt]) -> bool:
        """Block until those deliveries touch none of the caller's memory, and say whether they do.

        Mandatory. A backend that copies synchronously satisfies it by returning ``True``; an
        asynchronous one joins the work it started.

        **False is an answer, not an error.** It means the backend cannot establish that the
        memory is untouched, and no amount of further waiting is promised to change that. The
        caller must then keep every span named by those deliveries out of use -- not freed, not
        rebuilt, not handed to another request -- until a later call on the same attempts answers
        ``True``. That may never come. A backend whose transport gives no proof of release, or
        whose peer went away mid-write, has no honest answer other than ``False``.

        It answers where the memory stands and nothing else. A delivery that failed goes quiet
        exactly like one that succeeded, so ``True`` says nothing about what arrived.
        """
        ...

    def settle(self, attempts: Iterable[Attempt]) -> None:
        """Block until each of those deliveries has reached an outcome, which ``poll`` then gives.

        Mandatory, and a separate question from ``quiesce``: reaching an outcome is the logical
        end, going quiet is the physical one, and neither implies the other. A peer can stop
        touching the memory while word of what happened is still in flight, and can know what
        happened while its writers run on.

        It is the blocking form of ``poll`` and nothing more: it does not bound how long the
        backend takes to get there. A caller that needs a bound polls and times itself.

        Neither wait may make its return depend on a delivery it was not given. Carrying an
        unrelated one along on the way is harmless -- a shared completion queue reaps what it
        reaps -- but waiting for one would put a caller freeing a single page behind every
        unrelated transfer in flight.
        """
        ...

    def probe(self, name: bytes, units: Sequence[bytes]) -> Optional[frozenset[bytes]]:
        """Which of these units this backend holds, without moving anything.

        ``None`` means this backend cannot answer more cheaply than fetching -- the honest answer
        for a peer, where the same question costs a round trip either way. That answer is how a
        backend declines, and writing it costs one line; leaving the method out is not an option,
        because a caller cannot tell an absent member from a broken one.

        It exists because a caller cannot ask by fetching: ``fetch`` needs the destination named,
        and the caller has to decide whether the content is worth a destination before it commits
        one. Only this direction has that problem -- a publication's source is memory the caller
        already holds.

        **A failure raises.** An empty set means "I hold none of these" and must not be how an
        unreachable backend answers; folding the two makes an outage read as an empty cache and
        turns a cluster-wide fault into a hit rate of zero with no other symptom.

        **The answer is advisory.** It may be stale before the caller acts on it, so a later fetch
        may serve fewer units than this promised. That is already what ``Delivered.served`` being a
        subset means; nothing here reserves anything.

        What a hit is worth -- a contiguous prefix, every layer group present, how many tokens --
        is the caller's to work out. This answers only which names are held, and only names it was
        asked about.
        """
        ...

    def open_route(self, hint: Mapping[str, object]) -> Route:
        """Turn a routing hint this deployment attached to a request into something ``fetch`` reads.

        It exists because one instance serves the whole process: a peer's route used to ride on a
        per-request object, and there is no longer one for it to ride on.

        **It may start control-plane preparation for this request and must not wait for it.**
        Handing a source's name to a connector, or asking for a peer's metadata, is allowed and is
        why a deployment bothers to route eagerly at all. Blocking until any of it finishes is not:
        the caller is on the engine's thread. Nor may it move any payload, take a hold on anything
        at the source, or touch the caller's cache memory -- nothing here reserves content, so a
        route is a plan and never a claim. Whatever it started, ``Route.close`` ends.

        ``hint`` is opaque to this interface and its shape belongs to the deployment. Three
        failures, three answers, because a caller does something different with each:

        * a backend with one source has nothing to route, and raises ``NotImplementedError``;
        * a hint missing a field, or naming a source this backend does not know, is bad input, and
          raises ``ValueError``;
        * a hint that was fine but whose preparation failed raises whatever the backend's own
          transport raised. That one may be worth retrying; the other two never are.

        ``SubmissionRejected`` is not among them. It carries a promise about memory the caller may
        have to quiesce, and this is not a delivery.
        """
        ...


@runtime_checkable
class Publishes(Protocol):
    """Somewhere cache can be offered from: the side that generated it."""

    def publish(self, extent: CacheExtent) -> Attempt:
        """Offer ``extent``.

        Each unit becomes readable whole or not at all; that is the implementation's to guarantee
        and this interface has no field for it. The boundary is the unit, not the extent -- a unit
        is what is named, what is served, and what a store keys on, and different units of one
        extent may become readable at different times.

        Raising ``SubmissionRejected`` is allowed only while nothing has escaped, as in the other
        direction.
        """
        ...

    def quiesce(self, attempts: Iterable[Attempt]) -> bool:
        """As ``Fetches.quiesce``."""
        ...

    def settle(self, attempts: Iterable[Attempt]) -> None:
        """As ``Fetches.settle``."""
        ...


class Registration(Protocol):
    """One registered span, and the way to take it back.

    Unlike a route, this really is a resource: the backend's transport holds something for as long
    as the span is registered. Giving it a handle is what lets a pool be torn down and rebuilt,
    and what says who ends it.
    """

    def close(self) -> None:
        """Release this registration. Idempotent once it has succeeded.

        **It does not wait, and it is not a second memory-safety guarantee.** Every delivery naming
        memory in this span must already have been through ``quiesce``; closing does not end one in
        flight, and a backend part way through a transfer keeps writing. ``quiesce`` remains the
        only thing that brings the memory to rest. So the caller's order is fixed: stop submitting
        deliveries that name this span, ``quiesce`` the ones already submitted, then close.

        **A close that raises has not closed.** The span is still registered and the caller may try
        again. A backend must therefore not count a failed release as done, or the transport keeps
        memory the caller believes it owns again, with nothing left to hand it over.

        After it returns the span is no longer registered, so a delivery naming it fails rather
        than reading memory nobody handed over. The caller may then free or rebuild the span.

        Closing is by handle, not by span: a handle closed twice must not reach a registration made
        after it. Closing, registering the same span again, then closing the stale handle is a pool
        rebuild followed by a legal redundant close, and matching by address would take the live
        one down.
        """
        ...


@runtime_checkable
class RegistersPools(Protocol):
    """Optional. A backend that needs the caller's memory handed to its transport implements it.

    The caller registers every relevant pool before its first ``fetch`` or ``publish``.
    """

    def register_pool(self, address: int, size: int) -> Registration:
        """Register one span of memory this backend may access.

        The only place an address appears. Everywhere else the backend sees local coordinates and
        resolves them through a mapping of its own, which registration does not supply.

        One span is registered once. A span overlapping one already registered is refused rather
        than merged: two handles over one range leaves neither able to say when the transport may
        let go. Registering a span that was registered and then closed is ordinary.

        Failure raises and registers nothing, so a caller registering several pools closes the
        handles it already holds and is back where it started. There is no half-registered state
        for it to reason about.

        The span must be mapped when it is registered and must stay mapped until the handle is
        closed. A caller whose pool is backed by virtual memory that is mapped and unmapped
        underneath it registers what is mapped now and re-registers when that changes; this
        interface has no way to learn of the change by itself.
        """
        ...

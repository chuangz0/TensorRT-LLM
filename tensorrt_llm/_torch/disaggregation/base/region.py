from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntFlag, auto
from typing import List, NamedTuple, Optional, Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class IndexRange:
    """
    Represents a closed interval [start, end], with both bounds >= 0.
    Commonly used for indexing layers, heads, or tokens.
    """

    start: int
    end: int

    def __post_init__(self):
        if not (isinstance(self.start, int) and isinstance(self.end, int)):
            raise TypeError("start and end must be integers")
        if self.start < 0 or self.end < 0:
            raise ValueError("start and end must be >= 0")
        if self.end < self.start:
            raise ValueError("end must be >= start")


class MemRegion(NamedTuple):
    """Describes a block of memory by starting pointer and size in bytes."""

    ptr: int
    bytes: int


class MemRegionGroup(NamedTuple):
    """Describes a block of memory by starting pointer and size in bytes."""

    ptrs: np.ndarray  # dtype=np.int64
    bytes_per_region: int


class DataLayout(IntFlag):
    """Possible orders for storing data in memory."""

    HND = auto()  # (head, seq_len, dim)
    NHD = auto()  # (seq_len, head, dim)


@dataclass(frozen=True)
class RegionSpec:
    """
    Specifies a (potentially partial) region of the cache.
    Extend this base class for additional axis or specialization.
    """

    layers: Optional[IndexRange] = None


@dataclass(frozen=True)
class KVRegionSpec(RegionSpec):
    """
    Specifies a region within the Key/Value cache, with optional axes.
    """

    heads: Optional[IndexRange] = None
    tokens: Optional[IndexRange] = None


class SpecRegion(NamedTuple):
    """
    Associates a memory region with its semantic specifier.
    """

    memory: MemRegion | MemRegionGroup
    spec: RegionSpec = None


class RegionExtractorBase(ABC):
    """
    Interface for extracting region descriptors from some backing store.
    """

    @abstractmethod
    def extract(self, region_ids: Optional[np.ndarray] = None) -> List[SpecRegion]:
        """
        Args:
            region_ids: (Optional) np.ndarray of integer region identifiers to extract.
        Returns:
            List of Regions for corresponding regions.
        """
        ...


class SpecRegionPair(NamedTuple):
    """
    Maps a source descriptor to a destination descriptor
    (e.g., when copying or reindexing regions).
    """

    src: SpecRegion
    dst: SpecRegion


class RegionMapperBase(ABC):
    """
    Maps a batch of region descriptors to corresponding destination(s).
    """

    @abstractmethod
    def map(self, src_regions: SpecRegion, dst_regions: SpecRegion) -> SpecRegionPair:
        """
        Args:
            src_regions: List of source Regions.
            dst_regions: List of destination Regions.
        Returns:
            List of RegionPairs mapping source to destination.
        """
        ...


Segment = tuple[int, int]
"""``(address, size)`` of one contiguous memory range: a part of a unit, or of a registered span."""


class RegionResolver(Protocol):
    """Resolve ``(local_group, local)`` to the segments that hold the unit, in a fixed order.

    The mapping from a unit's local coordinates to memory, which the cache backend contract leaves
    to the backend. The order is part of the stored byte layout: segments are concatenated into one
    object, so a resolver that reorders them between two processes makes their bytes disagree.
    Whatever fixes the order must therefore be folded into the layout fingerprint the key scheme
    carries.

    Raises ``KeyError`` or ``ValueError`` for coordinates it does not know.
    """

    def __call__(self, local_group: int, local: int) -> Sequence[Segment]: ...

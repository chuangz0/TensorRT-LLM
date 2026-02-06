from collections import defaultdict
from dataclasses import dataclass
from typing import List

import numpy as np

from tensorrt_llm._torch.disaggregation.base.region import (
    DataLayout,
    DataRole,
    MemRegionGroup,
    RegionExtractorBase,
    SpecRegion,
)
from tensorrt_llm._torch.disaggregation.native.region.page import (
    BUFFER_ENTRY_DTYPE,
    KVCachePageTable,
    PoolDescriptor,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import get_size_in_bytes
from tensorrt_llm.bindings import DataType


@dataclass
class KVPoolAttrs:
    """Attributes for a single (primary) KV memory pool."""

    pool_ptrs: List[int]
    block_bytes: List[int]


class KVRegionExtractorV1(RegionExtractorBase):
    """
    Descriptor and region extractor for KV cache pool managed by KVCacheManager.
    Provides region descriptors for adapting block-wise view.
    """

    def __init__(self, kv_arg: KVCacheManager | KVPoolAttrs | KVCachePageTable):
        if isinstance(kv_arg, KVPoolAttrs):
            self._kv_pool_attrs = kv_arg
        elif isinstance(kv_arg, KVCacheManager):
            self._kv_pool_attrs = self._attrs_from_manager(kv_arg)
        elif isinstance(kv_arg, KVCachePageTable):
            self._kv_pool_attrs = self._attrs_from_page_table(kv_arg)
        else:
            raise TypeError(
                f"kv_arg must be KVCacheManager, KVPoolAttrs, or KVCachePageTable, "
                f"got {type(kv_arg)}"
            )
        self._data_layout = DataLayout.HND

    @staticmethod
    def _attrs_from_page_table(page_table: KVCachePageTable) -> KVPoolAttrs:
        ptrs = []
        block_sizes = []

        assert len(page_table.pools) == 1, "Multiple pool groups not supported in this extractor"
        pool_group = page_table.pools[0]
        pool_desc = pool_group[0]
        ptrs.append(pool_desc.base_address)
        block_sizes.append(pool_desc.slot_bytes)

        return KVPoolAttrs(pool_ptrs=ptrs, block_bytes=block_sizes)

    @staticmethod
    def _attrs_from_manager(manager: KVCacheManager) -> KVPoolAttrs:
        try:
            pools = manager.get_unique_primary_pool()
        except Exception as ex:
            raise ValueError(f"Failed to get pool(s): {ex}")

        pool_list = list(pools) if isinstance(pools, (list, tuple)) else [pools]
        elem_bytes = get_size_in_bytes(1, manager.dtype)
        ptrs, block_sizes = [], []

        for p in pool_list:
            if hasattr(p, "data_ptr") and callable(p.data_ptr):
                try:
                    ptr = int(p.data_ptr())
                except Exception as ex:
                    raise ValueError(f"Fail to call data_ptr(): {ex}")
            elif isinstance(p, int):
                ptr = int(p)
            else:
                raise ValueError(f"Pool object lacks 'data_ptr' and is not int: {p!r}")
            ptrs.append(ptr)

            try:
                if hasattr(p, "__getitem__") and hasattr(p[0], "numel"):
                    n = int(p[0].numel())
                elif hasattr(p, "numel") and callable(p.numel):
                    n = int(p.numel())
                else:
                    raise RuntimeError("Cannot determine element count")
            except Exception as ex:
                raise ValueError(f"Failed to get block size from {p!r}: {ex}")

            block_sizes.append(n * elem_bytes)

        return KVPoolAttrs(pool_ptrs=ptrs, block_bytes=block_sizes)

    def extract(self, region_ids: List[int]) -> SpecRegion:
        """
        Given a list of region_ids, returns a single SpecRegion,
        whose memory is a MemRegionGroup containing all blocks described
        by region_ids.
        """
        assert len(self._kv_pool_attrs.pool_ptrs) == 1
        pool_idx = 0
        attrs = self._kv_pool_attrs
        ptrs = [
            attrs.pool_ptrs[pool_idx] + block_id * attrs.block_bytes[0] for block_id in region_ids
        ]
        memory = MemRegionGroup(ptrs=ptrs, bytes_per_region=attrs.block_bytes[0])
        return SpecRegion(memory=memory)


def build_page_table(kv_cache_manager) -> KVCachePageTable:
    if kv_cache_manager.dtype == DataType.NVFP4:
        raise NotImplementedError("NVFP4 quantization not supported")

    tokens_per_block = kv_cache_manager.tokens_per_block
    num_layers = kv_cache_manager.num_local_layers

    pool_id_to_layers = defaultdict(list)
    for layer_idx in range(num_layers):
        pool_id = int(kv_cache_manager.kv_cache_pool_mapping[layer_idx][0].item())
        pool_id_to_layers[pool_id].append(layer_idx)

    pool_groups = []
    for pool_id in sorted(pool_id_to_layers.keys()):
        layers = pool_id_to_layers[pool_id]
        base_addr = int(kv_cache_manager.kv_cache_pool_pointers[pool_id][0].item())

        layer_idx = layers[0]
        elements = (
            kv_cache_manager.tokens_per_block
            * kv_cache_manager.num_kv_heads_per_layer[layer_idx]
            * kv_cache_manager.head_dim
        )
        buffer_size = get_size_in_bytes(elements, kv_cache_manager.dtype)
        is_key_only = kv_cache_manager.kv_factor == 1

        stride = buffer_size * kv_cache_manager.kv_factor
        slot_bytes = stride * len(layers)

        entries = []
        for i, layer_idx in enumerate(layers):
            base_offset = i * stride
            entries.append((layer_idx, int(DataRole.KEY), base_offset, buffer_size))
            if not is_key_only:
                entries.append(
                    (layer_idx, int(DataRole.VALUE), base_offset + buffer_size, buffer_size)
                )

        pool_descriptor = PoolDescriptor(
            base_address=base_addr,
            slot_bytes=slot_bytes,
            num_slots=kv_cache_manager.blocks_in_primary_pool,
            buffer_entries=np.array(entries, dtype=BUFFER_ENTRY_DTYPE),
        )
        pool_groups.append([pool_descriptor])

    return KVCachePageTable(
        tokens_per_block=tokens_per_block, num_layers=num_layers, pools=pool_groups
    )

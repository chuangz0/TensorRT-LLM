# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A padding dummy never reuses committed blocks.

The attention-DP empty-batch pad (``_pad_empty_attention_dp_batch``) adds a generation dummy whose
prompt is ``[1] * token_num``. Token 1 is the BOS of Llama-family tokenizers, so the radix tree
of any rank that served a real request holds a block starting with it; with block reuse on, the
dummy's cache could match into that block and ``add_dummy_requests`` would trip on its
``num_committed_tokens == 0`` invariant. The pad runs whenever a rank has active requests but
schedules none -- rare without a KV transfer layer, routine with one (a request waiting on a
store lookup or parked for a fetch is active and unschedulable).
"""

import gc

import pytest
import torch

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

DataType = tensorrt_llm.bindings.DataType
CacheType = tensorrt_llm.bindings.internal.batch_manager.CacheType

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="allocates real KV cache pools"
)

TPB = 32
DUMMY_ID = 10_000_000


@pytest.fixture
def manager():
    torch.cuda.init()
    gc.collect()
    mgr = KVCacheManagerV2(
        kv_cache_config=KvCacheConfig(max_tokens=2048, enable_block_reuse=True),
        kv_cache_type=CacheType.SELF,
        num_layers=2,
        num_kv_heads=4,
        head_dim=64,
        tokens_per_block=TPB,
        max_seq_len=512,
        max_batch_size=4,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=DataType.HALF,
        vocab_size=32000,
    )
    yield mgr
    mgr.shutdown()
    del mgr
    gc.collect()
    torch.cuda.empty_cache()


def _commit_prompt(manager, request_id, tokens):
    request = LlmRequest(
        request_id=request_id,
        max_new_tokens=4,
        input_tokens=list(tokens),
        sampling_config=SamplingConfig(1),
        is_streaming=False,
    )
    assert manager.prepare_context(request)
    assert manager.resize_context(request, request.context_remaining_length)
    batch = ScheduledRequests()
    batch.append_context_request(request)
    manager.prepare_resources(batch)
    request.move_to_next_context_chunk()
    manager.update_context_resources(batch)
    return request


BOS_THEN_TEXT = [1] + list(range(100, 100 + 4 * TPB))
"""A real prompt: BOS, then text."""
ONES = [1] * (2 * TPB) + list(range(100, 100 + 3 * TPB))
"""A prompt whose first blocks are all ones, so even a dummy longer than a block matches."""


@pytest.mark.parametrize(
    "prompt, token_num",
    [(BOS_THEN_TEXT, None), (ONES, None), (ONES, 2), (ONES, TPB + 1)],
    ids=["bos_gen_pad", "ones_gen_pad", "ones_two", "ones_block_plus_one"],
)
def test_dummy_does_not_reuse_committed_blocks(manager, prompt, token_num):
    request = _commit_prompt(manager, 1, prompt)
    manager.free_resources(request)

    dummies = manager.add_dummy_requests(
        request_ids=[DUMMY_ID],
        token_nums=None if token_num is None else [token_num],
        is_gen=True,
        prepare_resource=True,
    )
    assert dummies is not None and len(dummies) == 1
    manager.free_resources(dummies[0])

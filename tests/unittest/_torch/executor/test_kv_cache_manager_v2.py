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

from types import SimpleNamespace

from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2


class _FakeKVCache:
    def __init__(self, num_committed_tokens: int):
        self.num_committed_tokens = num_committed_tokens
        self.committed_tokens = None
        self.stopped_committing = False

    def commit(self, tokens):
        self.committed_tokens = tokens
        self.num_committed_tokens += len(tokens)

    def stop_committing(self):
        self.stopped_committing = True


def test_try_commit_blocks_commits_uncommitted_tokens_and_stops_at_context_end():
    request = SimpleNamespace(
        py_request_id=1,
        is_dummy_request=False,
        context_current_position=8,
        context_remaining_length=0,
        get_tokens=lambda beam_id: list(range(10)),
    )
    kv_cache = _FakeKVCache(num_committed_tokens=4)
    manager = object.__new__(KVCacheManagerV2)
    manager.enable_block_reuse = True
    manager.is_draft = False
    manager.kv_cache_map = {request.py_request_id: kv_cache}
    manager._augment_tokens_for_block_reuse = lambda tokens, request, start, end: tokens[start:end]

    manager.try_commit_blocks(request)

    assert kv_cache.committed_tokens == [4, 5, 6, 7]
    assert kv_cache.num_committed_tokens == 8
    assert kv_cache.stopped_committing


def test_disagg_role_mapper_kinds_default_to_indexed():
    from tensorrt_llm._torch.disaggregation.resource.page import MapperKind
    from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import Role

    manager = object.__new__(KVCacheManagerV2)

    # K/V default to the TRTLLM head-major layout; the index-key side cache
    # defaults to REPLICATED (every shipped index-K — DSA V1, MiniMax M3 —
    # is TP-replicated). The INDEX_KEY entry is inert unless a subclass
    # registers such buffers.
    assert manager.get_disagg_role_mapper_kinds() == {
        Role.ALL: MapperKind.INDEXED,
        Role.INDEX_KEY: MapperKind.REPLICATED,
    }

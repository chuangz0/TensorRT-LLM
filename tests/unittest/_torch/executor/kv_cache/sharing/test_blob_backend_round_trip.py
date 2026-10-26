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
"""The blob backend between two real staging lenders.

One manager publishes a request's KV blocks through its lender's staging slots; another fetches
them into a request of its own. A test lease holder drives the lender as the runtime does: lend,
wait for the view, submit it to the backend as it stands, wait until the attempt quiesces, then
release a read lease or return a write lease with what arrived. The store is the in-memory one, or
with ``TRTLLM_MOONCAKE_REAL=1`` a real Mooncake pool, one for the module
(``TRTLLM_MOONCAKE_PROTOCOL``, default rdma). The caches: FP16 and NVFP4 multi-head attention, and
DeepSeek sparse attention's MLA latent with its indexer keys. Allocates device pools.
"""

from __future__ import annotations

import os
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.disaggregation.blob.backend import BlobBackend, BlobConfig, RegionBinding
from tensorrt_llm._torch.disaggregation.blob.keys import encode_key
from tensorrt_llm._torch.disaggregation.blob.mooncake import MooncakeDriver
from tensorrt_llm._torch.disaggregation.blob.mooncake_setup import MooncakeConfig, open_sdk
from tensorrt_llm._torch.disaggregation.testing.memory_store import (
    Commit,
    MemoryDriver,
    MemoryStore,
)
from tensorrt_llm._torch.disaggregation.testing.object_trailer import Checksum, encode_trailer
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import (
    LeaseView,
    MemoryKind,
    ResumeBounds,
    RowSelection,
    StagingOptions,
    attach_lender,
)
from tensorrt_llm.bindings import DataType

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="allocates KV cache pools")

TPB = 32
PROMPT = list(range(1000, 1097))  # three whole blocks and one token
END = 96
BLOCKS = END // TPB
SOURCE, TARGET = 1, 2
SCOPE = b"blob-backend-round-trip"
KEY_PREFIX = "round-trip"
WAIT_S = 60.0
PROTOCOL = os.environ.get("TRTLLM_MOONCAKE_PROTOCOL", "rdma")
DEVICES = os.environ.get("TRTLLM_MOONCAKE_DEVICES", "")


class Wake:
    """The ``WakeFn`` a test hands the backend, with bounded waits on what it signals."""

    def __init__(self) -> None:
        self._woken = threading.Condition()

    def __call__(self) -> None:
        with self._woken:
            self._woken.notify_all()

    def settled(self, attempt):
        """The attempt's outcome, once it also quiesced."""
        with self._woken:
            done = self._woken.wait_for(
                lambda: attempt.poll() is not None and attempt.quiesced(), WAIT_S
            )
        assert done, "the attempt did not settle"
        return attempt.poll()


class Stores:
    """Drivers over one store: the in-memory one, or with ``master`` a real Mooncake pool.
    ``close`` closes the Mooncake clients only once every backend built over them shut down
    cleanly."""

    def __init__(self, master: str | None = None) -> None:
        self._store = MemoryStore() if master is None else None
        self._master = master
        self._clients = []
        self.backends = []

    def open_driver(self):
        if self._store is not None:
            return MemoryDriver(self._store)
        sdk = open_sdk(MooncakeConfig(self._master, protocol=PROTOCOL, device_name=DEVICES))
        self._clients.append(sdk)
        return MooncakeDriver(sdk)

    def commit(self, object_key: np.ndarray, data: bytes) -> None:
        """Commits ``data`` straight into the in-memory store, with the driver's trailer."""
        key = encode_key(KEY_PREFIX, object_key.tobytes())
        trailer = encode_trailer(key.encode(), len(data), checksum=Checksum.CRC32, payload=data)
        assert self._store.commit(key, data, trailer) is Commit.STORED

    def close(self) -> None:
        if all(backend.shutdown(WAIT_S) == frozenset() for backend in self.backends):
            for client in self._clients:
                client.close()


@pytest.fixture(scope="module")
def mooncake_master(tmp_path_factory):
    """A real Mooncake pool for the module's tests; its master's address. The caches' keys never
    collide: each cache's page layout names a namespace of its own."""
    if os.environ.get("TRTLLM_MOONCAKE_REAL") != "1":
        pytest.skip("needs a real Mooncake pool; set TRTLLM_MOONCAKE_REAL=1")
    from test_common.mooncake_utils import running_pool

    run_dir = str(tmp_path_factory.mktemp("pool"))
    with running_pool(run_dir, 256 << 20, protocol=PROTOCOL, device_name=DEVICES) as master:
        yield master.address


@pytest.fixture(params=["memory", "mooncake"])
def stores(request):
    if request.param == "memory":
        yield Stores()
        return
    opened = Stores(master=request.getfixturevalue("mooncake_master"))
    yield opened
    opened.close()


def dsa_manager():
    """DeepSeek-V3.2's cache, small: each layer's MLA latent and indexer keys, two pools of one pool
    group."""
    from tensorrt_llm._torch.attention.backends.sparse.dsa import DSACacheManagerV2
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig, KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    return DSACacheManagerV2(
        kv_cache_config=KvCacheConfig(enable_block_reuse=True, max_tokens=2048, host_cache_size=0),
        kv_cache_type=CacheType.SELFKONLY,
        num_layers=2,
        num_kv_heads=1,
        head_dim=576,
        tokens_per_block=TPB,
        max_seq_len=256,
        max_batch_size=4,
        mapping=Mapping(world_size=1, rank=0, tp_size=1),
        dtype=DataType.BF16,
        vocab_size=32000,
        execution_stream=torch.cuda.Stream(),
        sparse_attention_config=DeepSeekSparseAttentionConfig(
            index_n_heads=4, index_head_dim=128, index_topk=64
        ),
        pretrained_config=SimpleNamespace(num_hidden_layers=2, qk_rope_head_dim=64),
    )


def cache_manager(kit, cache: str):
    """A small manager of ``cache`` ("fp16", "nvfp4" or "dsa"), shut down when the block exits."""
    if cache == "dsa":
        return kit.managed(dsa_manager)
    dtype = DataType.NVFP4 if cache == "nvfp4" else DataType.HALF
    return kit.managed(lambda: kit.make_manager(dtype=dtype))


def attach(mgr):
    return attach_lender(mgr, scope=SCOPE, staging=StagingOptions(END))


def build(stores: Stores, lender, wake: Wake):
    """A backend over ``lender``'s regions, built on this thread: a driver, the lender's hold, the
    regions registered under it, and the backend. Returns the binding, whose ``release_hold``
    gives the hold back once the backend shut down, and the backend."""
    binding = RegionBinding(lender.regions, stores.open_driver(), lender.hold_regions())
    binding.register()
    config = BlobConfig(key_prefix=KEY_PREFIX, workers=2, rows_per_batch=1)
    backend = BlobBackend(binding, wake, config)
    stores.backends.append(backend)
    return binding, backend


def publish(stores, kit, mgr, select=None):
    """Publishes the source request's blocks (those ``select`` keeps); returns the view lent."""
    source = kit.published(mgr, SOURCE, PROMPT)
    lender, wake = attach(mgr), Wake()
    binding, backend = build(stores, lender, wake)
    lease = lender.lend_read(
        source, 0, END, memory_kind=MemoryKind.HOST, rows=RowSelection.NAMED, ready_event=None
    )
    view = kit.ready_view(lease, mgr)
    submitted = (
        view
        if select is None
        else LeaseView(tuple(g.select(select) for g in view.groups), view.start, view.end)
    )
    outcome = wake.settled(backend.publish(submitted, None))
    assert outcome.fault is None
    assert all(mask.all() for mask in outcome.delivered_masks)
    lease.release()
    assert backend.shutdown(WAIT_S) == frozenset()
    binding.release_hold()
    return source, view


def fetch(stores, kit, mgr):
    """Fetches every block of the target request; returns the target, lender, lease, outcome,
    backend and binding, the lease not yet returned."""
    target = kit.admitted(mgr, TARGET, PROMPT)
    lender, wake = attach(mgr), Wake()
    binding, backend = build(stores, lender, wake)
    lease = lender.lend_write(
        target, 0, END, memory_kind=MemoryKind.HOST, rows=RowSelection.NAMED, ready_event=None
    )
    view = kit.ready_view(lease, mgr)
    kit.fill_sentinel(mgr, target)
    outcome = wake.settled(backend.fetch(view, None))
    return target, lender, lease, outcome, backend, binding


@pytest.mark.parametrize("cache", ["fp16", "nvfp4", "dsa"])
def test_publish_then_fetch_restores_pages(kit, stores, cache: str) -> None:
    """Every byte of each page comes back, from every pool of its pool group: NVFP4's block scales
    and DSA's indexer keys are pools of their own."""
    with cache_manager(kit, cache) as mgr_a, cache_manager(kit, cache) as mgr_b:
        (pool_group,) = mgr_a.impl.pool_group_descs
        assert len(pool_group.pools) == (1 if cache == "fp16" else 2)
        source, _ = publish(stores, kit, mgr_a)
        target, lender, lease, outcome, backend, binding = fetch(stores, kit, mgr_b)
        assert outcome.fault is None
        kit.return_write(lender, lease, outcome.delivered_masks)
        assert kit.settled_bounds(lender, mgr_b, target) == ResumeBounds(END, 0)
        for ordinal in range(BLOCKS):
            got = kit.digest(kit.page(mgr_b, target, 0, ordinal))
            assert got == kit.digest(kit.page(mgr_a, source, 0, ordinal)), "a fetched page differs"
        assert backend.shutdown(WAIT_S) == frozenset()
        binding.release_hold()


def test_partial_fetch_marks_only_delivered_rows(kit, real_manager) -> None:
    stores = Stores()
    with real_manager() as mgr_a, real_manager() as mgr_b:
        first_block_only = np.arange(BLOCKS) == 0
        source, view = publish(stores, kit, mgr_a, select=first_block_only)
        for object_key in view.groups[0].object_keys[1:]:  # the later blocks are the wrong size
            stores.commit(object_key, b"short")
        target, lender, lease, outcome, backend, binding = fetch(stores, kit, mgr_b)
        assert outcome.delivered_masks[0].tolist() == [True, False, False]
        assert "object is 5 bytes" in outcome.fault
        kit.return_write(lender, lease, outcome.delivered_masks)
        # Usable up to the first gap.
        assert kit.settled_bounds(lender, mgr_b, target) == ResumeBounds(TPB, 0)
        sentinel = kit.digest(bytes([kit.SENTINEL]) * kit.DevicePages(mgr_b).page_bytes(0))
        pages = [kit.digest(kit.page(mgr_b, target, 0, ordinal)) for ordinal in range(BLOCKS)]
        assert pages == [kit.digest(kit.page(mgr_a, source, 0, 0)), sentinel, sentinel]
        assert backend.shutdown(WAIT_S) == frozenset()
        binding.release_hold()

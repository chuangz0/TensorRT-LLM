<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 对 `docs/shared/README.md`(kv-shared-draft)的修订提案

> 给 PR 描述与 `origin/feat/kv-shared-draft` 作者的留言用。README 不在本分支上(`KV_TRANSFER_ALIGNMENT_PLAN.zh.md` §1 选 B),
> 修订不在这里落地,只在此成文。README 自己规定"当前与它有哪些出入,另行成文",故分两类。编号沿用 ALIGNMENT_PLAN §2.2。

## A 类:改终态设计,需要作者批准

**A1. `orchestration/remote_cache.py` 承载 `Planner`。**
README §5 给 `remote_cache.py` 的职责是"远端缓存的取用策略"。本分支落地的 `Planner` 正是它:决定取不取、问谁、取到哪并归并(`merge`、`retry_hint_from`),
也决定 gen-init 的短路与 gen-first 的 `DEFER`。它**不**持有目的区域——调度器经 `reserve_transfer_pages(req, token_end)` 预留页;**不**构造 extent——
`KVv2ResourceReader.fetch_extent` 做。README 中与路由提示相关的 `hint.py` 尚不存在,随 worker 后端一并拆出(见 B2)。
提议:README §5 的 `remote_cache.py` 一行加注"实现为 `Planner`;决策输入只用所有 rank 相同的量"。

**A5. `blob/fetch.py` + `blob/publish.py` → `blob/backend.py`。**
README §5 把 blob 后端拆成 `fetch.py` 与 `publish.py`。本分支的 `BlobStoreBackend` 是一个类同时实现 `Fetches` / `Publishes` / `RegistersPools`:
三者共用同一个 `StoreClient`、同一个 staging pool、同一张 registration 表(登记的 pool 跨度既被 fetch 的目的地检查用,也被 publish 的来源检查用),
拆成两个文件只会让共享状态变成第三个模块。提议:README §5 的 `blob/` 条目改为 `backend.py`(三个契约面)+ `client.py`(`StoreClient` Protocol)+
`keys.py` / `staging.py` / `worker_pool.py`(实现细节)+ `mooncake.py`(驱动:配置、开客户端、注册表工厂;后续驱动与其并列)。
若作者坚持拆分,`fetch.py` / `publish.py` 可以是对 `backend.py` 的纯搬家,不改行为。

## B 类:现状注记,进 README 文末"当前落点与上图的出入"段

**B2. `hint.py` 尚不存在。** 路由今天只有 `FetchSource.hint_key`(装配表里每个后端认哪个提示键)+ `KVTransferCoordinator.launch_fetches` 里的一次
`open_route(hint)`。blob 后端 `hint_key = None`。有 worker 后端接入协调层时再拆。

**B3. `resource/kv_v2_reader.py`。** `ResourceReader` 的 KV v2 实现:请求 → 块键(`context_block_keys`)、层组(`group_specs`)、fetch extent、publish extent。
README 无对应条目;它是 `resource/` 层"请求 → 可命名内容"的入口。

**B4. 装配层。** `backends/config.py`(装配表 YAML → `KVTransferConfig` / `BackendEntry`;`TRTLLM_KV_TRANSFER_CONFIG` 指向它)与
`backends/registry.py`(`type` → 工厂 → `BackendHandle`;内置表按名懒加载)。README 只写"装配表一行加一个后端",没写装配表长什么样。

**B6. "在飞传输的登记"有两份。** 旧路 `orchestration/transfer_manager.py`(`AsyncTransferManager`,服务 transceiver),新路
`orchestration/records.py`(`TransferRecord` 表,服务 `KVTransferCoordinator`)。统一后后者取代前者(设计 §12.1)。

**B7. 引擎侧三文件。** `pyexecutor/kv_transfer_effects.py`(唯一写请求状态处)、`kv_transfer_binding.py`(循环每轮调用的对象:
`advance_round` / `launch_reserved_fetches` / `publish_committed_blocks` / `on_request_finished` / `is_tracking` / `pace_idle` / `close`)、
`kv_transfer_assembly.py`(装配与范围守卫)。README 无引擎层;旧路对应 `pyexecutor/disagg_adapter.py`。

**B9. 新契约的落点。** 在 `feat/mooncake-store-backend` 上,SPEC §7 的十三个名字从 `base/cache_backend.py` 导出,与 kv-shared-draft 的
`base/backend.py` **逐字节同步**(模块 docstring 末尾有核对命令);`base/backend.py` 仍是配对路径(`transceiver.py`、`native/`)的旧契约,
直到配对路径迁到新类型后换名(ALIGNMENT_PLAN §6 第 4 步)。`resource/naming.py` 与 kv-shared-draft 只差一行 import(`..base.cache_backend` vs `..base`)。

## 本轮已按 README 落地的名字(供对照)

| README §5 | 本分支 |
|---|---|
| `backends/blob/` | `backends/blob/{backend,client,keys,staging,worker_pool,mooncake}.py` |
| `orchestration/remote_cache.py` | 同名,类 `Planner` |
| `resource/region.py`(页表 → 指针) | 同名,`KVv2RegionResolver` + `layout_fingerprint` |
| `base/region.py` | 追加 `RegionResolver` Protocol 与 `Segment` |
| `backends/config.py` | 同名,`KVTransferConfig` / `BackendEntry` / `load_kv_transfer_config` |

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 对 `docs/shared/README.md`(kv-shared-draft)的修订提案

> 给 PR 描述与 `origin/feat/kv-shared-draft` 作者的留言用。README 不在本分支上(`KV_TRANSFER_ALIGNMENT_PLAN.zh.md` §1 选 B),修订不在这里落地,只在此成文。
> 对照版本:README `147ed68276e`,代码组织在 **§4**(此前 `2ea958f8817` 的 §5;`KV_TRANSFER_COORDINATOR_DESIGN.zh.md` 里的 "README §5" 均指旧版)。README §4 写明"加粗的是目标形态中尚待新增的文件",本文按它的四条组织规则与目录清单逐条对照本分支(`feat/mooncake-store-backend`)的落点,分两类:**A 类**改终态设计,需要作者批准;**B 类**是现状注记,只说明本分支今天落在哪。

## README §4 的四条规则与本分支

| 规则 | 本分支 |
|---|---|
| 同级后端之间不得直接依赖;共用的东西提到公共契约一层 | 遵守。`backends/blob/` 不 import 任何其他后端;两类 host-landing 后端会共用的实现件(`Copier` / `CudaCopier`)放在 `backends/host_copy.py`,不在 `base/`——它是实现不是契约(见 A2) |
| 目录按协议复用关系组织:复用某后端适配层的产品,作为其驱动置于其下 | 遵守。Mooncake 是 `backends/blob/drivers/mooncake.py`,与 `drivers/memory.py` 并列;两者共用 `blob/` 的适配层 |
| 同名模块分别表示抽象与实现 | 遵守一处:`base/region.py`(`RegionResolver` 协议)与 `resource/region.py`(`KVv2RegionResolver`)。契约本身落在 `base/cache_backend.py` 而非 README 的 `base/backend.py`(B5) |
| 后端只经契约面取得内容标识、区域与保护,不读 KVManager 内部 | 遵守。后端只见 `RegionResolver` 给的 `(address, size)` 与 unit 名;KV v2 只被 `resource/kv_v2_reader.py`、`resource/region.py` 触达 |

## A 类:改终态设计,需要作者批准

**A1. 远端缓存策略的位置:`disaggregation/remote_cache.py::Planner` vs README §4 的 `pyexecutor/kv_cache/remote_policy.py`。**
README `147ed68276e` 把"远端缓存策略:给出取得／发布意图、内容范围与资源约束"放到 KVManager 侧(`pyexecutor/kv_cache/remote_policy.py`,待新增),与其 §3 "远端缓存策略是 KVManager 侧的一项能力"一致。本分支的 `Planner` 落在 `disaggregation/remote_cache.py`(`2ea958f8817` README §5 的名字):它决定取不取、问谁、按所有分页层组取到哪(`servable_blocks`)并归并(`merge`,其 B 也是重试提示),也决定 gen-init 的短路与 gen-first 的 `DEFER`;它只经 `ResourceReader` 读 KV v2,**不**持有目的区域(调度器经 `reserve_transfer_pages(req, token_end)` 预留页),**不**构造 extent(`KVv2ResourceReader.fetch_extent` 做),决策输入只用所有 rank 相同的量。
提议二选一:(a) README 承认策略可以在传输侧(`disaggregation/`)落地,只要它只经导出面读 KVManager——这是本分支的现状;或 (b) 本分支把 `remote_cache.py` 搬为 `pyexecutor/kv_cache/remote_policy.py`,搬家时保住 `Planner` 不 import `tensorrt_llm` 的性质。未定前本分支不搬。

**A2. `backends/` 这一层。**
README §4 把 `native/`、`blob/`、`kvcr/`、`nixl/` 直接放在 `disaggregation/` 下,没有 `backends/`。本分支有 `backends/{config,registry,host_copy}.py` + `backends/blob/`:`config.py`(装配表 YAML → `KVTransferConfig` / `BackendEntry`,超时缺省只在此一处)与 `registry.py`(`type` → 工厂 → `BackendHandle`,内置表按名懒加载)是 README §6 所说"创建 KVManager、TransferManager 和各 backend 的装配层",`host_copy.py` 是 host-landing 后端共用的拷贝件。
提议:README §4 增加 `backends/` 一级,把装配层的两个模块与各后端目录收在一起;否则本分支需把 `blob/` 上提一层,`config.py` / `registry.py` / `host_copy.py` 另找落点。

**A3. `blob/fetch.py` + `blob/publish.py` → `blob/{store,backend,host_landing,factory,keys,staging,worker_pool}.py` + `drivers/`。**
README §4 把 blob 后端拆成 `fetch.py` 与 `publish.py`。本分支的 `BlobStoreBackend` 是一个类同时实现 `Fetches` / `Publishes` / `RegistersPools`:三者共用同一个 `BlobStore`、同一张 registration 表(登记的 pool 跨度既被 fetch 的目的地检查用,也被 publish 的来源检查用),拆成两个文件只会让共享状态变成第三个模块。`landing: host` 形态 `HostLandingBlobBackend`(`LandsOnHost`)组合一个 `BlobStoreBackend`,自成一个模块 `host_landing.py`。
提议:README §4 的 `blob/` 条目改为 `store.py`(`BlobStore` 协议 + `PutStatus` / `GetStatus` / `BlobStoreError`:驱动实现的面)+ `backend.py`(三个契约面 + `BlobStoreConfig`)+ `host_landing.py`(host-first 形态)+ `factory.py`(所有驱动共用的工厂骨架)+ `keys.py` / `staging.py` / `worker_pool.py`(实现细节)+ `drivers/`(每个存储一个模块:`mooncake.py`、`memory.py`;新存储只加一个模块与注册表一行)。README "一个存储驱动;其他字节存储产品与之并列"仍成立,只多 `drivers/` 一层。若作者坚持拆分,`fetch.py` / `publish.py` 可以是对 `backend.py` 的纯搬家,不改行为。

## B 类:现状注记

**B1. TransferManager 的落点。** README §4 的目标文件 `disaggregation/transfer_manager.py`(候选后端选择与次序、在途提交、句柄与保护的关联、回退与收尾)在本分支由 `orchestration/kv_transfer/`(`coordinator.py::KVTransferCoordinator` + `records.py::TransferRecord` 表 + `interfaces.py` + `build.py`)承担,"候选来源与次序"在 `remote_cache.py::Planner`;"请求退出后继续收尾"即 `hold_for_transfer` 与记录出表后的 `terminate_request`。统一后(设计 §12)旧路 `orchestration/transfer_manager.py`(`AsyncTransferManager`)由它取代,与 README "并入 TransferManager 后移除"一致。

**B2. 路由提示。** README §7 的"由路由提示确定对端"在本分支只有 `FetchSource.hint_key`(装配表里每个后端认哪个提示键)+ `KVTransferCoordinator.launch_reserved_fetches` 里的一次 `open_route(hint)`;blob 后端 `hint_key = None`。有 worker 后端接入协调层时再具体化。

**B3. `resource/kv_v2_reader.py` 与 `resource/region.py`。** README §4 说 `resource/` 是过渡期 helper,"职责迁入 KVManager 侧导出模块(`export.py`)后退役"。本分支的 `KVv2ResourceReader`(请求 → 块键、层组、fetch extent、publish extent)与 `KVv2RegionResolver` + `layout_fingerprint`(页 → 内存段、pool 跨度、布局指纹)就是那个导出面今天的落点,放在 `resource/`,只经 KV v2 wrapper 的公开方法触达。

**B4. 装配层。** `backends/config.py`(`TRTLLM_KV_TRANSFER_CONFIG` 指向的 YAML → `KVTransferConfig` / `BackendEntry`)与 `backends/registry.py`(`type` → 工厂 → `BackendHandle`)。README §6 只写"装配层负责安排这些步骤",没写装配表长什么样;目录归属见 A2。

**B5. 契约的落点。** README §4 的契约文件是 `base/backend.py`。本分支上 SPEC §7 的十三个名字从 `base/cache_backend.py` 导出,与 kv-shared-draft 的 `base/backend.py` **逐字节同步**(模块 docstring 末尾有核对命令);本分支的 `base/backend.py` 仍是配对路径(`transceiver.py`、`native/`)的旧契约,直到配对路径迁到新类型后换名(ALIGNMENT_PLAN §6 第 4 步)。`resource/naming.py` 与 kv-shared-draft 只差一行 import。协调层消费的只读视图 `RequestView` / `GroupSpec` / `ResourceReader` 在 `base/views.py`,README 无对应条目。

**B6. 引擎侧三文件。** `pyexecutor/kv_transfer/effects.py`(唯一写请求状态处)、`kv_transfer/hooks.py`(循环每轮调用的对象 `KVTransferHooks`,引擎循环每个钩子点一个方法:`advance_round` / `export_plan_answers` / `adopt_plan_answers` / `plan_fetch` / `launch_reserved_fetches` / `publish_committed_blocks` / `on_request_finished` / `owns` / `has_pending_work` / `inflight_request_ids` / `pace_idle` / `close`)、`kv_transfer/assembly.py`(装配与范围守卫)。README §7 的 "Executor / Coordinator 即架构中的请求编排"对应这一层;旧路对应 `pyexecutor/disagg_adapter.py`。

**B7. "在飞传输的登记"有两份。** 旧路 `orchestration/transfer_manager.py`(`AsyncTransferManager`,服务 transceiver),新路 `orchestration/kv_transfer/records.py`(`TransferRecord` 表,服务 `KVTransferCoordinator`)。统一后后者取代前者(设计 §12.1)。

## 本分支的落点(对照 README §4 `147ed68276e`)

| README §4 | 本分支 |
|---|---|
| `pyexecutor/kv_cache/remote_policy.py`(待新增) | `disaggregation/remote_cache.py`,类 `Planner`(A1) |
| `pyexecutor/kv_cache/export.py`(待新增) | `disaggregation/resource/kv_v2_reader.py` + `resource/region.py`(B3) |
| `disaggregation/transfer_manager.py`(待新增) | `disaggregation/orchestration/kv_transfer/{coordinator,records,interfaces,build}.py` + `remote_cache.py`(B1) |
| `disaggregation/blob/`(待新增) | `disaggregation/backends/blob/{store,backend,host_landing,factory,keys,staging,worker_pool}.py` + `drivers/{mooncake,memory}.py`(A2、A3) |
| —(装配层无条目) | `disaggregation/backends/{config,registry,host_copy}.py`(A2、B4) |
| `base/backend.py`(契约) | `base/cache_backend.py`(新契约,逐字节同步)+ `base/views.py` + `base/region.py` 追加 `RegionResolver` / `Segment`(B5) |
| —(引擎层无条目) | `pyexecutor/kv_transfer/{assembly,hooks,effects}.py`(B6) |

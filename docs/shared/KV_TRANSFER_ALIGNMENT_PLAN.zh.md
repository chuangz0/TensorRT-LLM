<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 与 `origin/feat/kv-shared-draft` 对齐(v4:只做 store 侧)

> 范围:`feat/mooncake-store-backend`(HEAD c0fd92e8d13 之后)向 `origin/feat/kv-shared-draft`(2ea958f8817,merge-base dcc95a8bf52)
> 对齐**目录、命名与文档**。用户决定:**本分支不碰配对路径**——`transceiver.py`、`native/*`、`base/transfer.py`、`base/backend.py`
> 一行不改。术语以其 `docs/shared/README.md` §5–§7 与 `CACHE_BACKEND_SPEC.md` §7 为准;下文"README"均指它。

## 0. 一分钟版

- **契约选 B:不合并他们的 5 个提交。** 旧 `base/backend.py` 继续服务 transceiver;新契约留在 `base/cache_backend.py`
  (与他们的 `backend.py` 逐字节相同,已核实),加一段头注释说明它是 SPEC 契约、等配对路径迁移后换名。`resource/naming.py`
  与他们只差一行 import,保持不动、写进文档。
- **目录按 README §5 走,五处改名:** `backends/store/` → `backends/blob/`(`BlobStoreBackend` + `mooncake.py` 驱动);
  `orchestration/planner.py` → `remote_cache.py`(类名 `Planner` 不变);`resource/kv_v2_layout.py` → `resource/region.py`
  (含 `layout_fingerprint`);`backends/store/regions.py::RegionResolver` → `base/region.py`;`backends/kv_transfer_config.py` → `backends/config.py`。
- **可读性改名五条**(§4);`hold_for_transfer` 不改(它对在飞 fetch 也 hold)。
- **README 修订分两类**(§2.2);**文档刷新 + 删两份未跟踪草稿**(§5)。
- **S0 已完成**(正确性修补 = c0fd92e8d13)。剩余 S1–S5 估 **1 个工作日**(改名 0.5 + 文档 0.5)。
- 配对路径迁移、`backends/worker/`、契约换名全部移到 §6 后续。

## 1. 契约:B,以及它留下的事实

| 事实 | 依据 |
|---|---|
| `base/cache_backend.py` == 他们的 `base/backend.py`,零差异 | `git diff --no-index <git show origin/feat/kv-shared-draft:…/base/backend.py> base/cache_backend.py` 为空 |
| `resource/naming.py` 与他们只差 :53 一行:`from ..base.cache_backend import Unit` vs `from ..base import Unit` | `diff` |
| 他们的 `base/__init__` 只导出 SPEC §7 十三个名字;合并会让 `transceiver.py:28`、`native/transfer.py:40` 与 8 个测试文件 `ImportError`,并让 `native/handle.py::TaskHandle.poll` 构造的 `Failed(..., reports_pending=)` / `Cancelled(..., reports_pending=)` 和 `Delivered(token_end=)` 在新类型上 `TypeError` | grep;这是选 B 的直接原因 |
| 他们的 `base/region.py` 与 merge-base **相同**(`IndexRange`/`MemRegion`/`RegionSpec`/`RegionExtractorBase`/`RegionMapperBase`),没有 `RegionResolver`;把我们的 `RegionResolver`/`Segment` 加进去是纯新增,将来同步无冲突 | `git diff --stat dcc95a8bf52 origin/feat/kv-shared-draft -- base/region.py` 为空 |
| 我们的 `resource/cache_reuse.py` 加了 `get_block_ordinals`,他们加了 `get_block_ids_with_ordinals`/`block_keys`——**本轮不碰**,留给 §6 的合并 | 两侧 diff |

**B 的代价(写明,不隐藏):** 两份契约模块并存;他们每改契约要手工同步 `cache_backend.py`(S1 加一条可重复执行的核对命令);
`kv_transfer_interfaces.py:35` 的 `TYPE_CHECKING` 从 `..base.backend` 取 `Chunk` 是有意的(`PlacesPieces.place(chunk)` 用的就是配对路径今天的 `Chunk`)。

## 2. 目录与命名对齐 README §5

### 2.1 改名表

| 现在 | 目标 | 为什么 |
|---|---|---|
| `base/cache_backend.py` | 不改名;头注释加:"SPEC `CACHE_BACKEND_SPEC.md` §7 的十三个名字;与 kv-shared-draft `base/backend.py` 逐字节同步;配对路径迁到新类型后此文件改名 `backend.py`" | §1 |
| `backends/store/` | `backends/blob/` | README:"Mooncake 复用 `blob/` 的适配层,故位于 `blob/` 之下";`store/backend.py` 只依赖 `StoreClient` Protocol,正是那层适配层 |
| `backends/store/backend.py::MooncakeStoreBackend` | `backends/blob/backend.py::BlobStoreBackend` | 类不含 Mooncake 特有逻辑(接入计划 §6:156 当时写"本轮不改") |
| `backends/store/mooncake.py`(工厂)+ `client.py::open_mooncake_client` + `config.py::MooncakeStoreConfig` | `backends/blob/mooncake.py`(工厂 + 打开客户端 + 配置) | README:"`mooncake.py` 一个驱动实现;后续驱动与其并列"。`StoreClient` Protocol 留 `blob/client.py` |
| `backends/store/regions.py::RegionResolver`、`Segment` | `base/region.py`(追加) | README 组织规则一:多后端共用的抽象放 `base/`;`registry.py:33` 已跨目录 import 它 |
| `backends/store/{keys,staging,worker_pool}.py` | `backends/blob/…` | 随目录走 |
| `backends/kv_transfer_config.py` | `backends/config.py` | 目录已限定范围;`KVTransferConfig`/`BackendEntry`/`load_kv_transfer_config` 名不变 |
| `orchestration/planner.py` | `orchestration/remote_cache.py`,`Planner`/`FetchPlan`/`merge`/`retry_hint_from` 名不变 | 用户已定:文件名对齐 README,类名说它做什么。导入点 14 处见 §2.3 |
| `resource/kv_v2_layout.py`(`KVv2RegionResolver` + `layout_fingerprint`) | `resource/region.py`,两者都去 | README:`resource/region.py` "页表 → 指针";`layout_fingerprint` 依赖 `kv_extractor.build_page_table_from_manager`,不能进他们那份纯 numpy 的 `naming.py` |
| `resource/kv_v2_reader.py`、`resource/naming.py`、`backends/registry.py` | 不变 | README 无对应条目(§2.2 B 类注记);`naming.py` 是他们的名字 |
| `orchestration/{kv_transfer_coordinator,records,kv_transfer_interfaces}.py` | 不变 | 目标名 `coordinator.py`/`interfaces.py` 被旧实现占用(§3、§6) |
| 测试目录 `tests/unittest/_torch/disaggregation/store_backend/` | `blob_backend/` | 随实现走 |

### 2.2 README 修订:提出来,不默默偏离

README 自己规定"当前与它有哪些出入,另行成文",所以分两类(编号沿用 v1,故不连续)。README 不在我们分支里(选 B),修订以
**PR 描述 + 给 kv-shared-draft 作者的 issue/评论**提出。

**A 类:改终态设计,要作者批准**
1. `remote_cache.py` 承载 `Planner`:决定取不取、问谁、取到哪并归并,也决定 gen-init 短路与 gen-first `DEFER`;不持有目的区域
   (调度器经 `prepare_disagg_gen_init(req, token_end)`)、不构造 extent(`KVv2ResourceReader.fetch_extent`)。`hint.py` 随 worker 后端一并拆出。
5. `blob/fetch.py` + `publish.py` → `blob/backend.py`:一个类同时实现 `Fetches`/`Publishes`/`RegistersPools`,三者共用客户端、staging pool、registration 表。

**B 类:现状注记,进 README 文末"当前落点与上图的出入"段**
2. `hint.py` 尚不存在:路由只有 `FetchSource.hint_key` + `KVTransferCoordinator.launch_fetches` 里一次 `open_route(hint)`。
3. `resource/kv_v2_reader.py`:请求 → 块键、层组、fetch/publish extent(`ResourceReader` 的 KV v2 实现)。
4. 装配层:`backends/config.py`(装配表 YAML)、`backends/registry.py`(type → 工厂 → `BackendHandle`)。
6. "在飞传输的登记"有两份:旧路 `transfer_manager.py`、新路 `records.py`;统一后后者取代前者(设计 §12.1)。
7. 引擎侧:`pyexecutor/kv_transfer_effects.py`(唯一写请求状态处)、`kv_transfer_binding.py`(循环每轮调用的对象)、`kv_transfer_assembly.py`(装配);旧路对应 `pyexecutor/disagg_adapter.py`。
9. **新契约的落点:** 在 `feat/mooncake-store-backend` 上,SPEC §7 的十三个名字从 `base/cache_backend.py` 导出,与 kv-shared-draft 的 `base/backend.py` 逐字节同步;`base/backend.py` 仍是配对路径的旧契约,直到 `transceiver.py`/`native/` 迁移。

### 2.3 `planner` → `remote_cache` 的 14 个导入点(已核实)

| 类别 | 文件 |
|---|---|
| 生产(4) | `orchestration/kv_transfer_coordinator.py:46`;`orchestration/records.py:31`(`TYPE_CHECKING`);`pyexecutor/kv_transfer_assembly.py:40`;`pyexecutor/scheduler/scheduler_v2.py:40`(`TYPE_CHECKING`) |
| 测试 `kv_transfer/`(6) | `fakes.py:35`、`test_planner.py:14`、`test_coordinator.py:15`、`test_consensus.py:16`、`test_merge_rule.py:26`、`test_planner_time_budget.py:15` |
| 测试其他(4) | `store_backend/test_with_coordinator.py:18`;`executor/kv_transfer/test_kv_v2_reader_layout.py:25`、`test_kv_transfer_effects_binding.py:44`;`test_scheduler_kv_fetch_seam.py:587` 是 **monkeypatch 字串** `"tensorrt_llm._torch.disaggregation.orchestration.planner"`,grep import 语句找不到 |

`test_planner.py`/`test_planner_time_budget.py` 文件名保留:测的是 `Planner` 类。

## 3. 本轮不动的部分

| 什么 | 为什么 |
|---|---|
| `base/backend.py`、`base/transfer.py`、`transceiver.py`、`native/`、`nixl/`、`resource/cache_reuse.py` 的合并 | 用户决定;T1/T2 的用例盯着它们 |
| `orchestration/coordinator.py`、`transfer_manager.py`、`admission.py`、`pp_termination.py`、`interfaces.py` | 配对路径今天的引擎接入;处置依赖 `backends/worker/`(§6) |
| `pyexecutor/kv_transfer_{effects,binding,assembly}.py` 的文件名 | README 无引擎层;已用 README 词汇(§2.2 #7) |
| `kv_cache_transceiver.py` | 与本线无关 |

## 4. 可读性改名

| 现在 | 目标 | 在哪 | 理由 |
|---|---|---|---|
| `prepare_disagg_gen_init(req, token_end=None)` | 新增 `reserve_transfer_pages(req, token_end)` 承载实现;`prepare_disagg_gen_init(req)` 保留为一行别名给 gen-init 调用点(`KVCacheV2Scheduler` gen-init 分支 :802、`transceiver.py` 报错文字 :1291/1316) | `kv_cache_manager_v2.py::prepare_disagg_gen_init` | fetch 接缝 `KVCacheV2Scheduler._try_take_fetch_path`(:634)调的是"为一次传输预留页",与 disagg 无关;别名让 T4 与 transceiver 零改动 |
| `KVTransferEngineBinding.advance_transfers` | `advance_round` | `kv_transfer_binding.py` | 与 `KVTransferCoordinator.advance(candidates, now)` 签名不同不能同名;`advance_round` 说明"每轮一次、先挑候选再转调" |
| `KVTransferCoordinator.publish_context_progress` / `Binding.publish_completed_contexts` | 两层同名 `publish_committed_blocks` | coordinator、binding | 同一件事(binding 只多一层过滤);"committed blocks"是发布对象(设计 §3.2 ④) |
| `BackendHandle.fetches`/`.publishes`(对象) vs `BackendEntry.fetches`/`.publishes`(布尔) | `BackendHandle.fetcher`/`.publisher`;`BackendEntry.serves_fetch`/`.serves_publish` | `registry.py::BackendHandle`、`kv_transfer_config.py::BackendEntry`、`mooncake.py::build_mooncake_backend`、`kv_transfer_assembly.py::attach_kv_transfer` | 同名不同类型 |
| `KVTransferEffects.hold_for_transfer` | **不改** | — | 对在飞 **fetch** 也 hold:`notify_request_finished` 里 `(rid, "fetch") in self._records or (rid, "publish") in self._records` 都触发;`test_coordinator.py:150–165` 断言之。设计 §7.3 的 `hold_for_publish` 是文档错(§5.1)。**既有褶皱:** `PyExecutorKVTransferEffects.hold_for_transfer` 一律置 `KV_PUBLISH_IN_PROGRESS`(`kv_transfer_effects.py:241`),fetch hold 也如此——记进设计 §4.2 待决,本轮不改 |
| `RecordState.PLANNED` 用于 publish 记录 | **不改,不加 `PENDING`** | `records.py::RecordState` | 第六个状态给同一条边两个名字是复杂度不是可读性 |
| `Binding.launch_reserved_fetches`/`Coordinator.launch_fetches`;`notify_request_finished`/`on_request_finished` | 不变 | — | 已区分 |

## 5. 文档刷新清单

### 5.1 `KV_TRANSFER_COORDINATOR_DESIGN.zh.md`

| 位置 | 现在 | 改成 |
|---|---|---|
| 页首、§3.1:133 | "公共契约 `base/backend.py`(未冻结)" | "公共契约 `base/cache_backend.py`(与 kv-shared-draft `base/backend.py` 逐字节同步;配对路径迁移后换名)" |
| §3.2、§7.1、§7.3、§9.2、附录 C | `hold_for_publish` | `hold_for_transfer`,写明"fetch 在飞的已结束请求同样 hold";§4.2 待决加"fetch hold 也置 `KV_PUBLISH_IN_PROGRESS`" |
| §3.2、§7.1 | `publish_context_progress(reqs)`;`launch_fetches(queue)` | `publish_committed_blocks(reqs, finished, now)`;`launch_fetches(queue, now)` |
| §7.1 构造签名 | `(sources, publishers, planner, effects, queue, registry, dist)` | `(sources, publishers, planner, reader, effects, queue, dist, *, fetch_timeout_s, publish_timeout_s, attention_dp, queue_budget, gather)`;补 `notify_request_finished`、`inflight_request_ids` |
| §7.2 | 6 字段 `FetchPlan`;`Planner(sources, reader, tokens_per_block)`;标题下注 `planner.py` | 11 字段(补 `unit_names`、`group_plans`、`block_keys`、`reuse_end`、`tokens_per_block`);`Planner(..., probe_budget_rounds, probe_timeout_s, clock)`;补 `decide(req, probe_answers, *, retry_hint)`、`probe_query`、`forget`;文件 `orchestration/remote_cache.py` |
| §7.3 | `orchestration/interfaces.py`;`pyexecutor/disagg_adapter.py` | `orchestration/kv_transfer_interfaces.py`;`pyexecutor/kv_transfer_effects.py`;注明旧路两文件待统一 |
| §7.5 | "`Chunk` 是 `resource/page.py` 已有类型" | 今天 `Chunk` 在 `base/backend.py`(旧契约),kv-shared-draft 已搬到 `resource/page.py`;我们随配对路径迁移一并跟进 |
| §7.4、§9.4、§9.5 | "store 后端" | "blob 后端(Mooncake 驱动)" |
| §12.2 第 0 步 | "把 kv-shared-draft 的 … 合进目标分支" | 改为"契约以逐字节副本落在 `cache_backend.py`;合并与配对路径迁移见 ALIGNMENT_PLAN §6" |
| 附录 A | 无 blob | 补一条 |

### 5.2 `KV_TRANSFER_ENGINE_INTEGRATION_PLAN.zh.md`

| 位置 | 改成 |
|---|---|
| §3 图 | `advance_round · launch_reserved_fetches · publish_committed_blocks`;`backends/blob/{backend,mooncake}.py`;`backends/config.py`;`orchestration/remote_cache.py`;`resource/region.py`(含 `layout_fingerprint`) |
| §4 命名表 | 上述改名;`BackendHandle.fetcher/publisher`;`BackendEntry.serves_fetch/serves_publish`;新增 `reserve_transfer_pages`;`Planner(probe_budget_rounds=, probe_timeout_s=, clock=)` |
| §6:156 | "改名决定 … 本轮不改" → 已改为 `BlobStoreBackend` |
| §11、§12 | `test_planner_time_budget.py` 现在在 `kv_transfer/`;`store_backend/test_store_backend_contract.py` → 随目录 `blob_backend/`;新增 `test_worker_pool.py`、`executor/kv_transfer/test_kv_transfer_hook_points.py`;测试数 345 → 561(含 test_kv_transfer_hook_points.py、test_worker_pool.py) |
| §14 | 行号基于 f974a61764a,过期。改"函数名 + 相邻语句"锚点;HEAD 更新 |

### 5.3 其他

- `KV_TRANSFER_COORDINATOR_DESIGN.notes.zh.md` §2 的源码位置全是行号,同样改锚点。
- **删除**(未跟踪,`rm`,无 commit):`docs/shared/DESIGN.zh.rewrite.md`(9/2 草稿,被 README/SPEC 取代)、
  `docs/shared/KV_TRANSFER_ENGINE_DESIGN.zh.md`(自称"待讨论的终态方案",已被两份现行文档覆盖)。已 grep:无引用。
- 工作区另有无关的未跟踪文件 `tensorrt_llm/_torch/pyexecutor/py_kv_cache_transceiver copy.py`——不提交,提醒用户处置。

## 6. 后续(不在本计划):配对路径迁移 → `backends/worker/` → 契约换名

本计划留下的前置与顺序:
1. **合并 kv-shared-draft**:冲突只在 `base/backend.py`、`resource/cache_reuse.py`(两侧各加了一个位置表 API:`get_block_ordinals` vs
   `get_block_ids_with_ordinals`/`block_keys`,建议统一到前者并向作者提议)、`resource/naming.py`(取他们的 import)。
2. **配对路径上新类型**:`transceiver.py`、`native/transfer.py`、`test_backend_contract.py`、`tests/unittest/disaggregated/` 7 文件改从
   `resource.page` 取 `Chunk/TokenRange/CacheKind`;`native/handle.py::TaskHandle`、`NothingPublished`、`native/fetch.py::CancelledBeforePublication`
   需要自己的结局族(`Placed/PlaceFailed/PlaceCancelled`,携带 `reports_pending`),`PeerFetch.fetch(extent, *, src)`/`PeerPublish.publish(extent)`
   改为按 `Chunk` 放置。`PlacesPieces` 是 `@runtime_checkable`,只看有没有 `place`;配对放置入口若同名会结构匹配——用不同方法名或
   `records.py::TransferRecord.merged_served` 的守卫防混接。
3. **`backends/worker/`**:搬 `PeerFetch`/`PeerPublish`,补 `Fetches` 五个必需成员,`PlaceOutcome → Outcome` 映射后再实现 `PlacesPieces`。
4. **契约换名**:删 `base/cache_backend.py`,`base/backend.py` 取新契约,8 处 import 改 `from ..base import`(`registry.py`、`blob/backend.py`、
   `kv_transfer_coordinator.py`、`records.py`、`kv_v2_reader.py`、`naming.py`、`kv_transfer_interfaces.py` + T3 测试)。
5. 之后:`Planner` 短路规则接上调度器路由(设计 §12.2 第 1–2 步),T1 的 339 个用例对照附录 C 迁到 `KVTransferCoordinator`,删旧协调层三文件。

## 7. 步骤、检查点、风险、回滚

### 7.1 检查点用的测试

| 代号 | 命令 | 基线 |
|---|---|---|
| T1 旧协调层 | `pytest tests/unittest/_torch/disaggregation --ignore=…/{kv_transfer,store_backend,engine_integration,e2e}` | **339**(collect 核实) |
| T3 新协调层 | `pytest tests/unittest/_torch/disaggregation/{kv_transfer,store_backend,engine_integration} tests/unittest/_torch/executor/kv_transfer` | **561**(核实;S2 后目录名变 `blob_backend`) |
| T4 KV v2 wrapper | `pytest tests/unittest/_torch/executor/kv_cache` | 任务描述说 513;本机整目录 925 —— S1 pin 出集合写进本节 |
| E1/E2 | `pytest tests/unittest/_torch/disaggregation/e2e`(GPU + `mooncake_master` + TinyLlama) | 2 用例 |

T2(`tests/unittest/disaggregated`)本轮不受影响:没有一步碰它 import 的模块。

### 7.2 步骤(S0 已完成 = c0fd92e8d13;一步一个 commit)

| 步 | 做什么 | 验证 |
|---|---|---|
| S1 契约副本核对 | `cache_backend.py` 头注释(§2.1 首行);把核对命令写进模块 docstring 末尾:`git show origin/feat/kv-shared-draft:tensorrt_llm/_torch/disaggregation/base/backend.py \| git diff --no-index - tensorrt_llm/_torch/disaggregation/base/cache_backend.py`(应仅剩头注释那几行);pin T4 | 命令跑通且差异只在注释;T3 绿 |
| S2 目录与文件改名 | §2.1 表全部 `git mv` + 类名 `BlobStoreBackend`;`mooncake.py` 吸收 `open_mooncake_client` 与 `MooncakeStoreConfig`;`RegionResolver`/`Segment` 追加进 `base/region.py`;`kv_v2_layout.py` → `resource/region.py`。**连带清单:** (1) `registry.py::_BUILTIN_FACTORIES` 字串 `".store.mooncake:build_mooncake_backend"` → `".blob.mooncake:…"`;(2) `test_kv_transfer_config_registry.py` 子进程断言的模块名 `disaggregation.backends.store.mooncake`/`.store.backend`(:391–397)与 `mooncake_module` fixture 的 `import_module("disaggregation.backends.store.mooncake")`(:409);(3) `engine_integration/conftest.py:12` 的 `__extra_import_path__` 中 `"../store_backend"`(`test_with_coordinator.py:15` 指向 `../kv_transfer`,不受影响);(4) §2.3 的 14 个 `planner` 导入点(含 monkeypatch 字串);(5) e2e YAML 的 `type: mooncake` 不变 | T3 561、T4、E1/E2 绿;`grep -rn "backends.store\|backends/store\|kv_v2_layout\|kv_transfer_config\.py\|backends\.kv_transfer_config\|store_backend\|orchestration.planner\|from .planner" tensorrt_llm tests/unittest/_torch` 为空 |
| S3 可读性改名 | §4 表前四条;`reserve_transfer_pages` 落地 + 别名 | T1、T3、T4 绿;`grep -rn "advance_transfers\|publish_completed_contexts\|\.fetches\b\|\.publishes\b" tensorrt_llm` 为空;调用点按符号核对:`PyExecutor._prepare_and_schedule_batch`(`advance_round`、`launch_reserved_fetches`)、`PyExecutor._executor_loop`/`_executor_loop_overlap`(`publish_committed_blocks`)、`KVCacheV2Scheduler._try_take_fetch_path`(`reserve_transfer_pages`) |
| S4 文档 | §5.1–5.3;README 修订按 §2.2 两类写进 PR 描述并给作者留言;`rm` 两份草稿 | 文档里每个文件名 `ls` 得到、每个方法名 `grep` 得到 |
| S5 收尾 | PR 描述:改名理由、README A/B 两类修订、grep 结论、测试数、§6 后续;`pre-commit` | 四组全绿 |

### 7.3 风险与回滚

| 风险 | 对策 / 回滚 |
|---|---|
| 他们改契约后 `cache_backend.py` 落后 | S1 的核对命令写在模块 docstring,每次同步先跑;差异只允许在头注释 |
| `store→blob` 改名打断用户 YAML | 不会:YAML 键是 `type: mooncake`,不含目录名;`TRTLLM_KV_TRANSFER_CONFIG` 不变 |
| `planner`→`remote_cache` 漏改 monkeypatch 字串 | S2 连带清单 (4) 单列;S2 grep 门含 `orchestration.planner` |
| README A 类修订被作者否决 | #1 只影响一行注记文字,文件名已按 README;#5 拆 `fetch.py`/`publish.py` 是纯搬家 |
| T4 的"513"无法复现 | S1 pin;pin 不出则以 `executor/kv_cache` 整目录 925 为基线 |
| 任一步失败 | 每步一个 commit,`git reset --hard <上一步 sha>`;S1–S3 无行为改动 |

## 8. 核实情况

**已核实:** `cache_backend.py` 与他们 `backend.py` 零差异(`git diff --no-index`);`naming.py` 仅差 :53 import;他们 `base/region.py` 与 merge-base
相同、无 `RegionResolver`;他们 5 个提交不碰 `transceiver.py`/`native/`;合并会导致的 ImportError/TypeError 位置(§1,作为选 B 的依据);
`registry.py:33` 跨目录 import `store.regions`;`kv_v2_layout.py` 导入 `kv_extractor`,`naming.py` 只导入 numpy/hashlib/`Unit`;
`orchestration.planner` 的 14 个导入点(含 `test_scheduler_kv_fetch_seam.py:587` 字串);`_BUILTIN_FACTORIES` 字串、registry 测试的模块名断言、
`engine_integration/conftest.py:12`;`notify_request_finished` 对 fetch 记录也调 `hold_for_transfer`,`kv_transfer_effects.py:241` 置
`KV_PUBLISH_IN_PROGRESS`;`prepare_disagg_gen_init` 两个调用点;`py_executor.py` 三个调用点所在函数;两份草稿无引用;S0 已提交为 c0fd92e8d13;
T1 = 339、T3 = 561。

**未核实:** T4 的"513"对应哪个集合;E1/E2 在本机是否仍可跑;kv-shared-draft 作者对 A 类两条修订的态度。

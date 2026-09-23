<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 与 `origin/feat/kv-shared-draft` 对齐:契约、目录、命名、文档

> 范围:`feat/mooncake-store-backend`(HEAD d4e3fc4761b + 工作区里未提交的正确性修补)向
> `origin/feat/kv-shared-draft`(2ea958f8817,merge-base dcc95a8bf52)对齐。本文只做计划,不改代码。术语以其
> `docs/shared/README.md` §5–§7 与 `CACHE_BACKEND_SPEC.md` §7 为准;下文"README"均指它。

## 0. 一分钟版

- **契约选 A:现在合并,不再养两份契约。** 我们的 `base/cache_backend.py` 与他们的 `base/backend.py` 逐字节相同,
  留两份只有维护成本。合并冲突 3 个文件、都小。代价是我们替他们把配对路径(`transceiver.py` / `native/`)搬到新类型上:
  这不只是改 import——新契约的 `Failed(reason)` / `Cancelled(by_peer)` 没有 `reports_pending`,而 `native/handle.py` 的
  `TaskHandle.poll` / `_rebuild`、`NothingPublished`、`native/fetch.py::CancelledBeforePublication` 都带着它构造,合并后是
  运行时 `TypeError`。所以配对路径要有**自己的一族放置结局**(`Placed / PlaceFailed / PlaceCancelled`)和**自己的协议名**
  (`PairedPlacer.place(chunk) -> PlaceAttempt`,都在 `base/transfer.py`),并明说 `PeerFetch` / `PeerPublish` 在 worker 后端统一之前
  **不是**契约的 `Fetches` / `Publishes`,也不是协调层的 `PlacesPieces`——后者继续只产出契约 `Outcome`,orchestration 只认一种结局。
  估 2.5–3 个工作日。
- **目录按 README §5 走:** `backends/store/` → `backends/blob/`(Mooncake 是它下面的驱动);`resource/kv_v2_layout.py` →
  `resource/region.py`(`layout_fingerprint` 随之,不进 `naming.py`);`store/regions.py::RegionResolver` 上提到 `base/region.py`;
  `PlacesPieces` 留在 orchestration(见上);**`orchestration/planner.py` → `remote_cache.py`,类名 `Planner` 不变**(用户已定,§2.3)。
- **旧协调层与引擎侧三文件本轮不动**(§3)。
- **可读性改名六条**(§4);`hold_for_transfer` **不改**——它对在飞 fetch 也 hold,设计文档要改过来。
- **文档:** 两份设计文档 + 接入计划逐条更新(§5),删两份未跟踪的过时草稿;README 修订分"要作者批的终态改动"与"现状注记"两类(§2.2)。
- **S0 = 先把正确性修补那一轮(25 个已改文件 + 6 个未跟踪文件)提交干净**,再合并;检查点五组测试(§7.1);回滚 = `reset` 回 S0 的 sha。

## 1. 契约:A 还是 B

### 1.1 现状(已核实)

| 事实 | 依据(符号) |
|---|---|
| `cache_backend.py` == 他们的 `backend.py`,零差异;我们的 `resource/naming.py` 与他们只差 `Unit` 的 import 来源 | `diff` |
| `merge-tree` 冲突:`base/backend.py`、`resource/cache_reuse.py`、`resource/naming.py`(add/add) | `git merge-tree --write-tree HEAD origin/feat/kv-shared-draft` |
| 冲突 1:我们在旧 `backend.py` 的 `Chunk.block_ids_per_layer_groups` / `CacheExtent.local` docstring 写了"位置表 + `-1` 洞"(14 行);他们把 `Chunk/TokenRange/CacheKind` 搬去 `resource/page.py` | 两侧 diff |
| 冲突 2:我们给 `CacheReuseAdapter` 加抽象 `get_block_ordinals`(位置表);他们加 `get_block_ids_with_ordinals`(过滤后 ids + 序号)与 `block_keys` | 两侧 diff |
| 他们的 `base/__init__` 只导出 SPEC §7 的十三个名字;`CacheKind/Chunk/TokenRange` 不再从 `base` 导出;他们的 5 个提交只碰 `tensorrt_llm/` 6 个文件 + `docs/shared/` 4 个文件,**不碰** `transceiver.py` / `native/` | 他们的 diff --stat |
| 合并后 `ImportError`:`transceiver.py:28`(`CacheExtent, CacheKind, Chunk, TokenRange`)、`native/transfer.py:40`(`Chunk`);测试:`tests/unittest/disaggregated/` 7 个文件(`test_chunked_transfer`、`test_kv_transfer`、`test_kv_transfer_mp`、`test_peer_fetch`、`test_peer_publish`、`test_task_handle`、`test_transfer_ownership_regressions`)+ `tests/unittest/_torch/disaggregation/test_backend_contract.py`(**共 8 个**直接从 `base` 取旧名) | grep `disaggregation.base import` |
| 合并后语义不符——**构造侧**:`transceiver.py::_create_cache_extent` / `_build_prefill_extent`(:390、:884)造旧 `CacheExtent(name=rid, local=Chunk)`;`_describe_local` 造 `Chunk`;调用点 `PeerPublish(session, req).publish(extent)`(:931)与 `fetches.fetch(extent)`(:963、:1046) | grep |
| **后端侧**:`native/fetch.py::PeerFetch.fetch(extent, *, src=None)` 读 `extent.local`、`extent.local.token_range.end`(新 `CacheExtent` 无 `local`;契约参数是 `route=`);`native/publish.py::PeerPublish.publish` 同样读 `extent.local`(:50–51);`refuse_extent_for_another_request` 读 `extent.name` 比 rid | 代码 |
| **结局侧**:`native/handle.py::TaskHandle.__init__(session, task, token_end)`;`poll` 造 `Delivered(token_end=)`、`Cancelled(by_peer=, reports_pending=True)`、`Failed(reason=, reports_pending=True)`;`_rebuild` 每次重造 `reports_pending=owed`;`NothingPublished` 造 `Failed(..., reports_pending=False)`;`native/fetch.py::CancelledBeforePublication` 造 `Cancelled(by_peer, reports_pending=False)`。新契约 `Failed` / `Cancelled` 无 `reports_pending`,`Delivered` 只有 `served` | `handle.py:45–65,75–84,118–123`;`fetch.py:53` |
| `Delivered.token_end` 与 `reports_pending` 在**生产代码无读者**;读者全在测试:`test_task_handle.py`(`_owes_a_report` 助手 :57,断言 :130/142/156/214/268/271/295/339/390/492)、`test_peer_fetch.py`(`_owes_a_report` :39,:194/206/225)、`test_backend_contract.py`(`_owes_a_report` :31,:109–157) | grep `.token_end`、`reports_pending` |
| 旧 `backend.py` 的 `Outcome` docstring 写着"两条轴"(`reports_pending` 只答"还欠不欠报告",**不是释放条件**);新契约把同一思想改写成 `poll`(逻辑结束)/ `quiesce`(物理静默)两轴 | `backend.py:197–215` vs 新 `Outcome` docstring |
| 他们的 `base/transfer.py`(配对会话:`TxSession.send(chunk)` / `RxSession.receive(chunk)`)只把 `Chunk` 改为 `TYPE_CHECKING` 自 `resource.page` 导入——配对路径的放置单位仍是 `Chunk`,留在会话层 | 他们的 `transfer.py` diff |

### 1.2 两个选项

**A. 合并他们的 5 个提交,删 `cache_backend.py`,我们补齐配对路径。**
- 好:一份契约,`base/__init__` 导出面就是 SPEC §7 的"可执行事实";`orchestration/`、`backends/`、`resource/` 8 处 import 改
  `from ..base import`;`backends/worker/` 统一时配对路径已在正确的类型上。
- 坏:要动 `transceiver.py`、`native/{fetch,publish,handle,transfer}.py`、`base/transfer.py`,以及 8 个测试文件;不是本分支主题。
- 量:import 13 处;放置结局一族(3 个 dataclass + 1 个 Union,搬 docstring)+ 两个协议(`PairedPlacer`、`PlaceAttempt`);
  `TaskHandle` / `NothingPublished` / `CancelledBeforePublication` 改造结局类型;`PeerFetch` / `PeerPublish` 入口改 `place(chunk)`;
  `transceiver.py` 5 个调用点;测试改 import 8 文件、改断言约 20 处。

**B. 两份契约共存到 worker 后端统一时再合。**
- 好:本轮零风险。
- 坏:他们每改契约我们手工同步一份逐字节副本;`kv_transfer_interfaces.py:35` 已出现"运行时用 `cache_backend`、类型检查用
  `backend`"的混用;README"十三个名字从同一模块导出"对我们不成立;设计 §3.1 的 `base/backend.py` 与代码对不上。统一时这些活
  一样要做,只是更晚、diff 更大。

**推荐 A。** 配对路径的改动是"类型搬家 + 入口换名 + 结局换族",不改会话状态机与传输逻辑;而且把 README §2 的"放置入口"
落成代码(`PlacesPieces.place(chunk)`),是 §6 统一工作的前置。

### 1.3 A 的冲突解法(逐文件)

| 文件 | 取法 |
|---|---|
| `base/backend.py` | 全取他们的。我们那 14 行"位置表 + `-1` 洞"docstring **搬到** `resource/page.py::Chunk`;旧 `Outcome` docstring 里"不是释放条件"那段搬到 `base/transfer.py` 的 `PlaceOutcome`(§7 S2) |
| `base/__init__.py`、`base/transfer.py`、`resource/page.py` | 自动合并,取他们的 |
| `resource/naming.py` | 取他们的(`from ..base import Unit`) |
| `resource/cache_reuse.py` | **一个位置表 API,两个名字都留,一份实现。** 抽象方法只有我们的 `get_block_ordinals`(位置表,`-1` 洞);他们的 `get_block_ids_with_ordinals` 改为基类**非抽象**默认实现:`ord = self.get_block_ordinals(...)`;`kept = np.nonzero(ord >= 0)[0]`;返回 `(ord[kept], kept)`;**删掉**他们 `_CacheReuseAdapterV2` 里那份 override(与我们 V2 的 `get_block_ordinals` 是同一个 `get_aggregated_page_indices(group_idx, valid_only=False)` 调用)。**调用方用哪个:** 要按 token 范围切片的(transceiver、reader)用 `get_block_ordinals`;要"哪些序号有页"的(命名)用 `get_block_ids_with_ordinals`。他们的 `block_keys` 保留(V2 → `context_block_keys`,V1 → `[]`);`KVv2ResourceReader.block_keys` 直接调 wrapper,不受影响 |
| `docs/shared/README*.md`、`CACHE_BACKEND_SPEC*.md` | 新增,直接落地;我们对 README 的修订见 §2.2 |

## 2. 目录与命名对齐 README §5

### 2.1 改名表

| 现在 | 目标 | 为什么 |
|---|---|---|
| `base/cache_backend.py` | 删除;`from ..base import ...` | §1 |
| `backends/store/` | `backends/blob/` | README:"Mooncake 复用 `blob/` 的适配层,故位于 `blob/` 之下"。`store/backend.py` 只依赖 `StoreClient` Protocol,正是那层适配层 |
| `backends/store/backend.py::MooncakeStoreBackend` | `backends/blob/backend.py::BlobStoreBackend` | 类不含 Mooncake 特有逻辑(接入计划 §6 当时写"本轮不改") |
| `backends/store/mooncake.py`(工厂)+ `client.py::open_mooncake_client` + `config.py::MooncakeStoreConfig` | `backends/blob/mooncake.py`(工厂 + 打开客户端 + 配置) | README:"`mooncake.py` 一个驱动实现;后续驱动与其并列"。`StoreClient` Protocol 留 `blob/client.py` |
| `backends/store/regions.py::RegionResolver`、`Segment` | `base/region.py` | README 组织规则一:多后端共用的抽象放 `base/`;`registry.py:33` 已跨目录 import 它 |
| `backends/store/{keys,staging,worker_pool}.py` | `backends/blob/…` | 随目录走 |
| `backends/kv_transfer_config.py` | `backends/config.py` | 目录已限定范围;类名/函数名不变 |
| `backends/registry.py` | 不变 | README 没有装配层文件,见 §2.2 注记 |
| `resource/kv_v2_layout.py`(`KVv2RegionResolver` + `layout_fingerprint`) | `resource/region.py`,**两者都去** | README:`resource/region.py` "页表 → 指针";`layout_fingerprint` 依赖 `kv_extractor.build_page_table_from_manager`,进 `naming.py` 会把 `kv_extractor` 拖进他们那份纯 numpy 的命名模块 |
| `resource/kv_v2_reader.py`、`resource/naming.py` | 不变 | 前者 README 无条目(§2.2 注记);后者是他们的名字 |
| `orchestration/kv_transfer_interfaces.py::PlacesPieces`(`place(chunk) -> Attempt`,产出契约 `Outcome`) | **不动** | 它是协调层的可选协议,`records.py::_is_failure` / `TransferRecord.merged_served` 按 `isinstance(outcome, Failed/Delivered)` 读结局;若让它产出 `PlaceOutcome`,一个接上的 `PeerPublish` 会被 `merged_served` 当成空 `served`——静默未命中 |
| (新增)`base/transfer.py::PairedPlacer.place(chunk) -> PlaceAttempt`、`PlaceAttempt.poll() -> Optional[PlaceOutcome]` | 新增 | 配对路径自己的放置入口与句柄类型;与 `PlacesPieces` 同形不同结局族,名字分开以免混接。统一时(§6)由 `backends/worker/` 做 `PlaceOutcome → Outcome` 的映射后再实现 `PlacesPieces` |
| `orchestration/{kv_transfer_coordinator,records,kv_transfer_interfaces}.py` | 本轮不变 | 目标名 `coordinator.py` / `interfaces.py` 被旧实现占用(§3、§6) |
| `orchestration/planner.py` | `orchestration/remote_cache.py`,类名 `Planner`、`FetchPlan`、`merge`、`retry_hint_from` 不变 | 用户已定:文件名对齐 README §5,类名说它做什么;导入点清单见 §2.3 与 S3 |
| 测试目录 `tests/unittest/_torch/disaggregation/store_backend/` | `blob_backend/` | 随实现走 |

### 2.2 README 与我们代码不合之处:提修订,不默默偏离

README 自己规定"分阶段走到哪一步、当前与它有哪些出入,另行成文",所以分两类(编号沿用 v1,故不连续)。合并后 README 在我们分支里。

**A 类:改终态设计,要 kv-shared-draft 作者批准**(改 §5 目录图正文;PR 描述单列,未批准前不动对应代码目录):

1. **`remote_cache.py` 的职责描述与我们的 `Planner` 不符,加一行。** README 说它"规划 extent、持有目的区域、选择后端";我们的
   `Planner.decide` 做来源决策、gen-init 短路、gen-first `DEFER`、`merge` 归并;**不**构造 extent(`KVv2ResourceReader.fetch_extent`),
   **不**持有目的区域(调度器经 `prepare_disagg_gen_init(req, token_end)`)。修订文字:"`remote_cache.py` 承载 `Planner`:决定取不取、
   问谁、取到哪并归并,也决定 gen-init 短路与 gen-first 的 DEFER;不持有目的区域、不构造 extent。`hint.py` 随 worker 后端一并拆出"。
5. **`blob/fetch.py` + `publish.py` → `blob/backend.py`。** 一个类同时实现 `Fetches` / `Publishes` / `RegistersPools`,因为三者共用同一个
   客户端、staging pool、registration 表;拆两个文件只会共享一堆私有状态。
8. **配对路径的结局族与协议名。** README §5 说配对路径的 placement/session/outcome 未迁入契约。我们把它落为 `base/transfer.py` 的
   `PairedPlacer.place(chunk) -> PlaceAttempt` 与 `Placed / PlaceFailed / PlaceCancelled`(§7 S2),并声明 `PeerFetch` / `PeerPublish`
   只实现 `PairedPlacer`,不是契约的 `Fetches` / `Publishes`。请作者确认这是他们想要的方向,还是希望结局最终进契约。

**B 类:现状注记,写进 README 文末"当前落点与上图的出入"段**(不改终态,不需批准):

2. `hint.py` 尚不存在:路由只有 `FetchSource.hint_key` + `KVTransferCoordinator.launch_fetches` 里一次 `open_route(hint)`;多来源(设计 §10.4)时再抽。
3. `resource/kv_v2_reader.py`:请求 → 块键、层组、fetch/publish extent(`ResourceReader` 的 KV v2 实现);README 的 `resource/` 没有这一项。
4. 装配层文件:`backends/config.py`(装配表 YAML)、`backends/registry.py`(type → 工厂 → `BackendHandle`)。
6. "在飞传输的登记"今天有两份:旧路 `transfer_manager.py`、新路 `records.py`(`TransferRecord` 表);统一后后者取代前者(设计 §12.1)。
7. 引擎侧文件:`pyexecutor/kv_transfer_effects.py`(effects 实现,唯一写请求状态处)、`kv_transfer_binding.py`(循环每轮调用的对象)、
   `kv_transfer_assembly.py`(装配);旧路对应 `pyexecutor/disagg_adapter.py`。

### 2.3 已定:`orchestration/planner.py` → `orchestration/remote_cache.py`,类名 `Planner` 不变

用户决定:文件名对齐 README §5,类名说它做什么;README 加一行(§2.2 #1);`hint.py` 随 worker 后端一并拆出。
`git mv` 之后要改的导入点(grep `orchestration.planner` / `from .planner`,共 14 处,已核实):

| 类别 | 文件 |
|---|---|
| 生产(4) | `orchestration/kv_transfer_coordinator.py:46`(`from .planner import FetchPlan, Planner, merge, retry_hint_from`);`orchestration/records.py:31`(`TYPE_CHECKING`);`pyexecutor/kv_transfer_assembly.py:40`;`pyexecutor/scheduler/scheduler_v2.py:40`(`TYPE_CHECKING`,`FetchPlan`) |
| 测试 `kv_transfer/`(6) | `fakes.py:35`、`test_planner.py:14`、`test_coordinator.py:15`、`test_consensus.py:16`、`test_merge_rule.py:26`、`test_planner_time_budget.py:15`(均 `from disaggregation.orchestration.planner import …`) |
| 测试其他(4) | `store_backend/test_with_coordinator.py:18`;`executor/kv_transfer/test_kv_v2_reader_layout.py:25`、`test_kv_transfer_effects_binding.py:44`;`test_scheduler_kv_fetch_seam.py:587` 是**字串**(`"tensorrt_llm._torch.disaggregation.orchestration.planner"`,给 monkeypatch 用),grep import 语句找不到,要单独改 |

测试文件名 `test_planner.py` / `test_planner_time_budget.py` 保留:它们测的是 `Planner` 类。

## 3. 本轮留在 README §5 之外的部分

| 什么 | 为什么留 | 何时收 |
|---|---|---|
| `orchestration/coordinator.py`(`DisaggTransferCoordinator`)、`transfer_manager.py`、`admission.py`、`pp_termination.py`、`interfaces.py` | 配对路径今天的引擎接入,T1 的 339 个用例盯着它;设计 §12.1 的处置依赖 `backends/worker/` 存在 | §6 统一 |
| `transceiver.py`、`native/`、`nixl/` | README 明说"后端平铺在 `disaggregation/` 之下,尚未收进 `backends/`"。本轮只做类型与结局搬家(S2),不搬目录 | §6 统一为 `backends/worker/` |
| `pyexecutor/kv_transfer_{effects,binding,assembly}.py` | README 没有引擎层;已用 README 词汇(effects、装配);见 §2.2 #7 | 统一时与 `disagg_adapter.py` 合并 |
| `kv_cache_transceiver.py`(C++ transceiver 包装) | 与本线无关 | — |

## 4. 可读性改名(不与 README 冲突)

| 现在 | 目标 | 在哪 | 理由 |
|---|---|---|---|
| `prepare_disagg_gen_init(req, token_end=None)` | 新增 `reserve_transfer_pages(req, token_end)` 承载实现;`prepare_disagg_gen_init(req)` 保留为一行别名给 gen-init 调用点(`scheduler_v2.py` 的 gen-init 分支 :802、`transceiver.py` 报错文字 :1291/1316) | `kv_cache_manager_v2.py::prepare_disagg_gen_init` | fetch 接缝(`scheduler_v2.py:634`)调的是"为一次传输预留页",与 disagg 无关;别名让 T4 与 C++ 侧文字零改动 |
| `KVTransferEngineBinding.advance_transfers` | `advance_round` | `kv_transfer_binding.py` | 与 `KVTransferCoordinator.advance(candidates, now)` 签名不同不能同名;`advance_round` 说明"每轮一次、先挑候选再转调" |
| `KVTransferCoordinator.publish_context_progress` / `Binding.publish_completed_contexts` | 两层同名 `publish_committed_blocks` | coordinator、binding | 同一件事(binding 只多一层过滤)用同一个名字;"committed blocks"是发布对象(设计 §3.2 ④"先 commit 再发布") |
| `Binding.launch_reserved_fetches` / `Coordinator.launch_fetches` | 不变 | — | 已符合"同动词 + binding 说明多做了什么" |
| `KVTransferEffects.hold_for_transfer` | **不变** | `kv_transfer_interfaces.py`、`kv_transfer_effects.py`、`KVTransferCoordinator.notify_request_finished` | 它对在飞 **fetch** 也 hold:`notify_request_finished` 里 `(rid, "fetch") in self._records or (rid, "publish") in self._records` 都触发;`test_coordinator.py:150–165` 断言 fetch 在飞时 `effects.names()[-1:] == ["hold_for_transfer"]`。设计 §7.3 的 `hold_for_publish` 是文档错,改文档(§5.1)。**既有褶皱:** `PyExecutorKVTransferEffects.hold_for_transfer` 一律置 `KV_PUBLISH_IN_PROGRESS`(`kv_transfer_effects.py:241`),对 fetch hold 也如此——本轮不改,记进设计 §4.2 待决 |
| `BackendHandle.fetches` / `.publishes`(对象) vs `BackendEntry.fetches` / `.publishes`(布尔) | `BackendHandle.fetcher` / `.publisher`;`BackendEntry.serves_fetch` / `.serves_publish` | `registry.py::BackendHandle`、`kv_transfer_config.py::BackendEntry`、`mooncake.py::build_mooncake_backend`、`kv_transfer_assembly.py::attach_kv_transfer` | 同名不同类型 |
| `Coordinator.notify_request_finished` / `Binding.on_request_finished -> bool` | 不变 | — | 一层通知、一层回答"能否现在终止" |
| `RecordState.PLANNED` 用于 publish 记录 | **不改,不加 `PENDING`** | `records.py::RecordState` | 第六个状态给同一条边两个名字是复杂度不是可读性;整体改名要连带设计 §4.1 状态图和 T3 里的字串断言,收益不抵 |

## 5. 文档刷新清单

### 5.1 `KV_TRANSFER_COORDINATOR_DESIGN.zh.md`

| 位置 | 现在 | 改成 |
|---|---|---|
| 页首、§3.1:133 | "公共契约 `base/backend.py`(未冻结)" | 合并后为真;补"已与 kv-shared-draft 2ea958f8817 合并,十三个名字从 `base/__init__` 导出" |
| §3.2 图与表、§7.1、§7.3、§9.2、附录 C | `hold_for_publish` | `hold_for_transfer`,并写明"fetch 在飞的已结束请求同样 hold";§4.2 待决加"fetch hold 也置 `KV_PUBLISH_IN_PROGRESS`" |
| §3.2、§7.1 | `publish_context_progress(reqs)`;`launch_fetches(queue)` | `publish_committed_blocks(reqs, finished, now)`;`launch_fetches(queue, now)` |
| §7.1 构造签名 | `(sources, publishers, planner, effects, queue, registry, dist)` | `(sources, publishers, planner, reader, effects, queue, dist, *, fetch_timeout_s, publish_timeout_s, attention_dp, queue_budget, gather)`;补 `notify_request_finished`、`inflight_request_ids` |
| §7.2 `FetchPlan` / `Planner` | 6 字段;`Planner(sources, reader, tokens_per_block)` | 11 字段(补 `unit_names`、`group_plans`、`block_keys`、`reuse_end`、`tokens_per_block`);`Planner(..., probe_budget_rounds, probe_timeout_s, clock)`;补 `decide(req, probe_answers, *, retry_hint)`、`probe_query`、`forget` |
| §7.3 标题与正文 | `orchestration/interfaces.py`;`pyexecutor/disagg_adapter.py` | `orchestration/kv_transfer_interfaces.py`;`pyexecutor/kv_transfer_effects.py`;注明旧路两文件待统一 |
| §7.5 | `PlacesPieces` 在 `orchestration/interfaces.py`;"`Chunk` 是 `resource/page.py` 已有类型" | `orchestration/kv_transfer_interfaces.py`,并写明它产出契约 `Outcome`,与 `base/transfer.py::PairedPlacer`(产出 `PlaceOutcome`)的区别;后者合并后为真 |
| §7.4、§9.4、§9.5 | "store 后端" | "blob 后端(Mooncake 驱动)" |
| §12.2 第 0 步 | "把 kv-shared-draft 的 … 合进目标分支" | 标已完成,是 merge 不是复制,附带 S2 的配对路径适配 |
| 附录 A | 无 blob / Placed | 补两条 |

### 5.2 `KV_TRANSFER_ENGINE_INTEGRATION_PLAN.zh.md`

| 位置 | 改成 |
|---|---|
| §3 图 | `advance_round · launch_reserved_fetches · publish_committed_blocks`;`backends/blob/{backend,mooncake}.py`;`backends/config.py`;`resource/region.py`(含 `layout_fingerprint`) |
| §4 命名表 | 上述改名;`BackendHandle.fetcher/publisher`;`BackendEntry.serves_fetch/serves_publish`;新增 `reserve_transfer_pages`;`Planner(probe_budget_rounds=, probe_timeout_s=, clock=)` |
| §6:156 | "改名决定 … 本轮不改" → 已改为 `BlobStoreBackend` |
| §11、§12 | `test_planner_time_budget.py` 已从 `engine_integration/` 移到 `kv_transfer/`;`store_backend/test_backend_contract.py` → `test_store_backend_contract.py` → 随目录 `blob_backend/`;新增 `test_worker_pool.py`、`executor/kv_transfer/test_kv_transfer_hook_points.py`;测试数 345 → 534 |
| §14 | 行号全部基于 f974a61764a,过期。改"函数名 + 相邻语句"锚点;HEAD 更新 |

### 5.3 其他

- `KV_TRANSFER_COORDINATOR_DESIGN.notes.zh.md` §2 的源码位置全是行号,同样改锚点。
- **删除**(未跟踪,`rm`,无 commit):`docs/shared/DESIGN.zh.rewrite.md`(9/2 草稿,被 README/SPEC 取代)、
  `docs/shared/KV_TRANSFER_ENGINE_DESIGN.zh.md`(自称"待讨论的终态方案",已被 COORDINATOR_DESIGN + INTEGRATION_PLAN 覆盖)。已 grep:无引用。
- 工作区另有一个与本线无关的未跟踪文件 `tensorrt_llm/_torch/pyexecutor/py_kv_cache_transceiver copy.py`——**不要**随 S0 提交,提醒用户处置。

## 6. 后续:统一为一个协调层 + `backends/worker/`(不在本计划)

本计划留下的前置:
- 配对路径已在 `resource.page.Chunk` + `PairedPlacer.place(chunk)` 上;`backends/worker/fetch.py` 只需搬 `PeerFetch` 并补 `Fetches` 五个必需成员
  (可命名整块走 `fetch(extent, route)`)。
- **接任何放置型后端到协调层之前,先做 `PlaceOutcome → Outcome` 的映射**(`Placed` → `Delivered(served=…)`、`PlaceFailed` → `Failed`、
  `PlaceCancelled` → `Cancelled`),放在 worker 后端的 `Attempt` 包装里;`records.py::TransferRecord.merged_served` 与 `_is_failure` 只认契约
  `Outcome`,S2 在 `merged_served` 加一条 `assert isinstance(a.outcome, (Delivered, Failed, Cancelled))` 式的守卫 + TODO,把混接从静默未命中变成断言。
- `records.py` 已是"在飞传输的登记";`transfer_manager.py` 的释放职责已由 `hold_for_transfer` 承接。
- `PlaceOutcome` 与契约 `Outcome` 并列;统一时决定 `reports_pending` 这条轴是并入 `quiesce` 还是留在会话层(README §5 待办、§2.2 #8)。
- 剩下:`Planner` 短路规则接上调度器路由(设计 §12.2 第 1–2 步),T1 的 339 个用例对照附录 C 迁到 `KVTransferCoordinator`,删旧协调层三文件。

## 7. 步骤、检查点、风险、回滚

### 7.1 检查点用的五组测试

| 代号 | 命令 | 基线 |
|---|---|---|
| T1 旧协调层 | `pytest tests/unittest/_torch/disaggregation --ignore=…/{kv_transfer,store_backend,engine_integration,e2e}` | **339**(`--collect-only` 核实;含 `test_backend_contract.py`) |
| T2 配对路径 | `pytest tests/unittest/disaggregated`(45 文件;7 个直接从 `base` 取旧名) | S0 时 pin;部分需 GPU |
| T3 新协调层 | `pytest tests/unittest/_torch/disaggregation/{kv_transfer,store_backend,engine_integration} tests/unittest/_torch/executor/kv_transfer` | **534**(核实;接入计划写 345 已过期) |
| T4 KV v2 wrapper | `pytest tests/unittest/_torch/executor/kv_cache` | 任务描述说 513;本机 collect 整目录 925、`test_kv_cache_manager_v2.py` 133;**S0 pin 出 513 对应集合,写进本节** |
| E1/E2 | `pytest tests/unittest/_torch/disaggregation/e2e`(GPU + `mooncake_master` + TinyLlama) | 2 用例 |

### 7.2 步骤(一步一个 commit;每步末跑本步触及的组,S2/S3 末跑全部)

| 步 | 做什么 | 在哪 | 为什么 | 验证 |
|---|---|---|---|---|
| **S0 先提交正确性修补那一轮** | (1) `git add` 工作区 **25** 个已修改/改名/删除的跟踪文件(`kv_transfer_coordinator.py`、`records.py`、`kv_transfer_interfaces.py`、`kv_transfer_config.py`、`store/backend.py`、`kv_cache_manager_v2.py`、`kv_transfer_binding.py`、`kv_transfer_effects.py`、`py_executor.py`、`scheduler_v2.py`、**`tensorrt_llm/runtime/kv_cache_manager_v2/_block_radix_tree.py`、`…/_core/_kv_cache_manager.py`** + 13 个测试文件);(2) **加入 6 个未跟踪但被依赖的文件**:`backends/store/worker_pool.py`(`store/backend.py:54` `from .worker_pool import DaemonWorkerPool`——不提交则 T3 整组 ImportError)、`store_backend/test_worker_pool.py`、`kv_transfer/test_planner_time_budget.py`(从 `engine_integration/` 移来)、`engine_integration/conftest.py`(定义 `__extra_import_path__ = ["~/tensorrt_llm/_torch", "../store_backend"]`)、`executor/kv_transfer/conftest.py`、`executor/kv_transfer/test_kv_transfer_hook_points.py`;(3) 不提交 `py_kv_cache_transceiver copy.py`;(4) pin T2/T4 基线数;记下 S0 的 sha 作回滚点 | 工作区 | 正确性修补还在并行落地,合并必须从干净树开始,否则冲突与修补混在一个 diff | `git status --short --untracked-files=all tensorrt_llm/_torch/disaggregation tensorrt_llm/_torch/pyexecutor tensorrt_llm/runtime/kv_cache_manager_v2 tests/unittest/_torch/disaggregation tests/unittest/_torch/executor tests/unittest/disaggregated` 只剩那个 `copy.py`;T1/T3/T4 绿 |
| S1 合并契约 | `git merge origin/feat/kv-shared-draft`,按 §1.3 解 3 个冲突;删 `base/cache_backend.py`;import 改 `from ..base import`:`registry.py`、`store/backend.py`、`kv_transfer_coordinator.py`、`records.py`、`kv_v2_reader.py`、`naming.py`、`kv_transfer_interfaces.py`(其 :35 `TYPE_CHECKING` 改自 `..resource.page`);测试侧 `kv_transfer/{fakes,test_consensus,test_contract_fakes,test_coordinator}.py`、`store_backend/{store_fakes,test_backend_semantics,test_backend_staging,test_real_master,test_store_backend_contract}.py`、`executor/kv_transfer/{engine_fakes,test_kv_transfer_effects_binding}.py` | `base/`、`resource/`、`orchestration/`、`backends/`、T3 测试 | 契约是其余改名的地基 | T3 534 绿;`grep -rn cache_backend tensorrt_llm/_torch/disaggregation tensorrt_llm/_torch/pyexecutor tests/unittest/_torch/{disaggregation,executor}` 为空。**T1/T2 此时预期红**(`transceiver.py:28` ImportError),S2 修;S1、S2 连续提交,PR 内不单独合 S1 |
| S2 配对路径上新类型与新结局族 | **(a) 类型搬家:** `transceiver.py:28`、`native/transfer.py:40`、`test_backend_contract.py:16`、`tests/unittest/disaggregated/` 7 文件改从 `resource.page` 导入 `Chunk/TokenRange/CacheKind`。**(b) 放置协议(`base/transfer.py`,新增):** `PairedPlacer(Protocol).place(chunk: Chunk) -> PlaceAttempt`;`PlaceAttempt(Protocol).poll() -> Optional[PlaceOutcome]`。`PlacesPieces` **不动**,留在 `kv_transfer_interfaces.py`(§2.1 理由);`records.py::TransferRecord.merged_served` 加守卫 + TODO(§6)。**(c) 放置结局族(`base/transfer.py`):** `Placed`(无字段——`token_end` 无生产读者,删)、`PlaceFailed(reason: str, reports_pending: bool)`、`PlaceCancelled(by_peer: bool, reports_pending: bool)`、`PlaceOutcome = Union[...]`;旧 `Outcome` docstring 里"`reports_pending` 只答欠不欠报告,**不是释放条件**"那段搬到 `PlaceOutcome` 上;`TaskHandle.__init__(session, task)` 去掉 `token_end`,`poll` / `_rebuild` 造 `Placed` / `PlaceFailed` / `PlaceCancelled`;`NothingPublished`、`CancelledBeforePublication` 同改;`native/` 内不再 import 契约的 `Delivered/Failed/Cancelled/Outcome/Attempt`。**(d) 入口换名:** `PeerFetch.fetch(extent, *, src)` → `place(chunk: Chunk) -> PlaceAttempt`(去掉 `src` 分支);`PeerPublish.publish(extent)` → `place(chunk)`;**删除** `refuse_extent_for_another_request`——`place(chunk)` 没有 name 可比,而 `transceiver.py` 构造 chunk 用的 `_describe_local(req)` 与 `PeerFetch(worker, req)` / `PeerPublish(session, req)` 绑定的是同一个 `req`,校验对象不存在了。模块 docstring 写明:**`PeerFetch` / `PeerPublish` 只实现 `PairedPlacer`,不是契约的 `Fetches` / `Publishes`,也不得接进协调层(`PlacesPieces` 是 `runtime_checkable`,同名 `place` 会结构匹配,名字不同不是防线——见 §7.3 风险表),到 `backends/worker/` 统一为止。** **(e) 调用点:** `transceiver.py` 删 `_create_cache_extent`,`_build_prefill_extent(req) -> Optional[CacheExtent]` 改为 `-> Optional[Chunk]`(去掉 `CacheExtent(name=rid, local=…)` 外壳,`None` 分支不变),`:931 PeerPublish(...).publish(extent)`、`:963/:1046 fetches.fetch(extent)` 改 `place(chunk)`。**(f) 测试:** `test_task_handle.py:57` / `test_peer_fetch.py:39` / `test_backend_contract.py:31` 的 `_owes_a_report` 助手改 `outcome is None or getattr(outcome, "reports_pending", False)`;`isinstance(..., Delivered)` → `Placed`;删 `token_end` 断言 3 处;`test_backend_contract.py` 里新契约不再有的形状测试(`Delivered(reports_pending=True)` 抛错等 :109–126)改为测 `PlaceOutcome` 族 | `transceiver.py`、`native/{fetch,publish,handle,transfer}.py`、`base/transfer.py`、`records.py`(守卫)、8 个测试文件 | 让 README"配对路径的 placement 走独立放置入口"成为代码;不改会话状态机与传输逻辑;orchestration 只认一种结局类型 | T1 339、T2、T3 534、T4 绿;`grep -rn "CacheExtent(\|\.fetch(\|\.publish(" tensorrt_llm/_torch/disaggregation/transceiver.py tensorrt_llm/_torch/disaggregation/native/` 为空(只剩 `place(`);`grep -rn "Delivered(token_end\|reports_pending" tensorrt_llm/_torch/disaggregation/native tests/unittest/disaggregated` 只命中 `Place*` 的构造与断言;`grep -rn "from tensorrt_llm._torch.disaggregation.base import" tensorrt_llm/_torch/disaggregation/native` 为空 |
| S3 目录与文件改名 | 按 §2.1:`git mv backends/store backends/blob`;`BlobStoreBackend`;`mooncake.py` 吸收 `open_mooncake_client` 与 `MooncakeStoreConfig`;`regions.py` 内容并入 `base/region.py`;`kv_v2_layout.py` → `resource/region.py`;`kv_transfer_config.py` → `config.py`;`git mv orchestration/planner.py orchestration/remote_cache.py`;测试目录 `store_backend/` → `blob_backend/`。**改名连带清单:** (1) `registry.py::_BUILTIN_FACTORIES` 的字串 `".store.mooncake:build_mooncake_backend"` → `".blob.mooncake:…"`;(2) `test_kv_transfer_config_registry.py` 子进程断言的模块名 `disaggregation.backends.store.mooncake` / `.store.backend`(:391–397)与 `mooncake_module` fixture 的 `importlib.import_module("disaggregation.backends.store.mooncake")`(:409);(3) `engine_integration/conftest.py:12` 的 `__extra_import_path__` 中 `"../store_backend"`(`store_backend/test_with_coordinator.py:15` 指向的是 `"../kv_transfer"`,不受影响);(4) `planner` → `remote_cache` 的 14 个导入点(§2.3 表),其中 `test_scheduler_kv_fetch_seam.py:587` 是 monkeypatch 字串;(5) e2e YAML 的 `type: mooncake` 不变 | `backends/`、`resource/`、`base/region.py`、`orchestration/`、测试 | 纯搬家,与 S4 分开便于 review 与回滚 | T3 534、T4、E1/E2 绿;`grep -rn "backends.store\|backends/store\|kv_v2_layout\|kv_transfer_config\|store_backend\|orchestration.planner\|from .planner" tensorrt_llm tests/unittest/_torch docs/shared` 为空(`docs/shared` 中允许本文件与设计文档里"改名自 …"的历史提及) |
| S4 可读性改名 | §4 表六条;`reserve_transfer_pages` 落地 + 别名 | wrapper、binding、coordinator、registry、config | 行为不变的 rename,单独一 commit | T1、T3、T4 绿;`grep -rn "advance_transfers\|publish_completed_contexts\|\.fetches\b\|\.publishes\b" tensorrt_llm` 为空;调用点按符号核对:`PyExecutor._prepare_and_schedule_batch`(`advance_transfers`、`launch_reserved_fetches`)、`PyExecutor._executor_loop` 与 `_executor_loop_overlap`(`publish_completed_contexts`)、`KVCacheV2Scheduler._try_take_fetch_path`(`prepare_disagg_gen_init(req, plan.token_end)` → `reserve_transfer_pages`) |
| S5 文档 | §5.1–5.3;README 按 §2.2 分类修订(A 类进 PR 描述待批,B 类进"当前落点"段);`rm` 两份草稿 | `docs/shared/` | 代码稳定后一次写完 | 文档里每个文件名 `ls` 得到、每个方法名 `grep` 得到 |
| S6 收尾 | PR 描述:改名理由、README A/B 两类修订、grep 结论、测试数;`pre-commit` | — | — | 五组全绿 |

### 7.3 风险与回滚

| 风险 | 对策 / 回滚 |
|---|---|
| S1 之后到 S2 之前 T1/T2 红 | 预期内,两步连续提交。**回滚 = `git reset --hard <S0 sha>`**(merge commit 不用 `revert`,否则以后再合 kv-shared-draft 会被 revert 记录挡住) |
| S2(c) 结局族与契约 `Failed` / `Cancelled` 同名易混;S2(b) 有人把 `PeerPublish` 直接接进协调层的 publisher 表 | 前缀 `Place*` 强制区分;`native/` 内不再 import 契约类型(S2 验证 grep);**注意 `PlacesPieces` 是 `@runtime_checkable`,`isinstance` 只看有没有 `place` 属性,名字不同不阻止匹配**——带 `place()` 的 `PeerPublish` 在结构上会被判真,随后协调层调 `.publish()` 抛 `AttributeError`。防线是:注册表(`registry.py::build_backends`)从不产出 `PeerPublish` 进 `KVTransferCoordinator._publishers` + `merged_served` 守卫把混接变成断言而非静默未命中。若要在结构上也区分,可把配对方法命名为 `place_piece`(可选设计项,S2 时决定) |
| `resource/cache_reuse.py` 下次同步 kv-shared-draft 会**再次冲突**(我们改写了他们刚加的 `get_block_ids_with_ordinals` 默认实现并删了 V2 override) | 合并 PR 描述里单列这一处,并向 kv-shared-draft 作者提"位置表 API 统一到 `get_block_ordinals`"的建议;若他们接受,下次同步无冲突 |
| S2(d) 删 `refuse_extent_for_another_request` 后丢一层防护 | 它防的是"传错 extent";`place(chunk)` 的 chunk 由 transceiver 用同一 `req` 现场构造,没有别的来源;`test_peer_fetch.py` 若有对应用例随之删 |
| `test_backend_contract.py`(T1 内)大半测旧契约形状 | 保留 `Chunk`/`TokenRange` 校验(import 自 `resource.page`),结局部分改测 `PlaceOutcome`;新契约形状已由 T3 `test_contract_fakes.py` 覆盖,不重复 |
| 正确性修补一轮还在并行落地,S0 之后又出现新的未跟踪文件 | S0 的验证命令(`git status --untracked-files=all` 五个目录)在 S1 之前**再跑一次**;有新文件先并进 S0 |
| `cache_reuse.py` 合并后 V1 的 `get_block_ids_with_ordinals` 默认实现经 `get_block_ordinals` 走 `get_batch_cache_block_ids`(GPU) | 既有路径,T2 覆盖;只改组合不改逻辑 |
| `store→blob` 改名打断用户 YAML | 不会:YAML 键是 `type: mooncake`,不含目录名;`TRTLLM_KV_TRANSFER_CONFIG` 不变 |
| README A 类修订被作者否决 | #1:文件名已按 README 取 `remote_cache.py`,否决只影响那一行注记文字,代码不动;#5:拆 `fetch.py`/`publish.py` 是纯搬家;#8:结局进契约是统一时的事,不影响本轮 |
| T4 的"513"集合无法复现 | S0 pin;pin 不出则以 `executor/kv_cache` 整目录 925 为基线并写进 §7.1 |

## 8. 核实情况

**已对照代码核实:** `cache_backend.py` 与他们 `backend.py` 零差异;`naming.py` 仅差 import;`merge-tree` 三处冲突及内容;他们
`base/__init__` 十三个导出名与 SPEC §7 一致;他们 `base/transfer.py` 只改 `Chunk` 为 `TYPE_CHECKING` 导入;他们 5 个提交不碰
`transceiver.py`/`native/`;合并后 ImportError 的文件(2 生产 + 8 测试);`TaskHandle.poll`/`_rebuild`、`NothingPublished`、
`CancelledBeforePublication` 都以 `reports_pending=` 构造结局,新契约无此字段;`records.py::_is_failure` 与
`TransferRecord.merged_served` 用 `isinstance(outcome, Failed/Delivered)` 读契约结局(`PlaceOutcome` 会被当成空 `served`);三个测试文件
的助手名是 `_owes_a_report`;`_build_prefill_extent(req) -> Optional[CacheExtent]`;工作区已改跟踪文件 25 个(含
`runtime/kv_cache_manager_v2/_block_radix_tree.py`、`_core/_kv_cache_manager.py`);`test_with_coordinator.py:15` 的 extra path 指向
`../kv_transfer` 而非 `../store_backend`;`PeerFetch.fetch`(`src=`,读 `extent.local`)与
`PeerPublish.publish`(读 `extent.local`)签名;`refuse_extent_for_another_request` 比 `extent.name` 与 `get_unique_rid(request)`;
`Delivered.token_end` 与 `reports_pending` 生产代码无读者、测试读者清单;`notify_request_finished` 对 fetch 记录也调
`hold_for_transfer`,`test_coordinator.py:150–165` 断言之;`hold_for_transfer` 置 `KV_PUBLISH_IN_PROGRESS`(`kv_transfer_effects.py:241`);
`store/backend.py:54` 导入未跟踪的 `worker_pool.py`;工作区 25 个已改跟踪文件与 7 个未跟踪文件(含无关的 `copy.py`);
`_BUILTIN_FACTORIES` 字串、`test_kv_transfer_config_registry.py` 的模块名断言、两处 `"../store_backend"`;`kv_v2_layout.py` 导入
`kv_extractor`,`naming.py` 只导入 numpy/hashlib/`Unit`;`prepare_disagg_gen_init` 两个调用点;`orchestration.planner` 的 14 个导入点
(含 `test_scheduler_kv_fetch_seam.py:587` 的字串);两份草稿无引用;T1 = 339、T3 = 534。

**未核实:** T4 的"513"对应哪个集合;T2 用例数与 GPU 需求;`test_peer_fetch.py` 是否有专测 `refuse_extent_for_another_request`
的用例(S2(d) 若有随删);E1/E2 在本机是否仍可跑;kv-shared-draft 作者对 A 类三条修订的态度。

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KV 传输协调层接入引擎:实施计划

> 状态:v4(已实施;目录与命名按 `KV_TRANSFER_ALIGNMENT_PLAN.zh.md` 对齐 kv-shared-draft README §5)。设计依据 `KV_TRANSFER_COORDINATOR_DESIGN.zh.md`(下称「设计」,引用写作 设计§n)。
> 代码基线曾为 `f974a61764a`;文中残留的行号以该 commit 为准,**已过期**——定位以函数名 + 相邻语句为准(§14)。探索分支 `wip/engine-integration-exploration`(`2c2c4fdc1a0`)证明了接线点可行,本计划沿用其接线点、重排其结构与命名。
> 术语沿用设计附录 A;本文不重述设计,只说明**接到哪、加什么、为什么、怎么验**。

## 1. 一分钟版

1. `KVTransferCoordinator`(已实现)与 Mooncake store 后端(已实现)通过**三个新文件**接进 `PyExecutor`:`kv_transfer_effects.py`(引擎侧协议实现)、`kv_transfer_binding.py`(循环调用的对象 `KVTransferEngineBinding`)、`kv_transfer_assembly.py`(装配与范围守卫)。
2. 共享引擎文件只加**一行调用 + 一行注释**,共 13 处(§5:py_executor 9、scheduler_v2 2、scheduler 1、creator 1);全部受 `self.kv_transfer is not None` 保护,环境变量不设即零行为差异。
3. 后端由 `TRTLLM_KV_TRANSFER_CONFIG=<yaml>` 描述,`backends:` 列表经注册表按 `type` 构造;本特性范围内 "mooncake" 只出现在 `backends/blob/mooncake.py`(驱动)与注册表的一行内置项里。
4. 与现有 disagg 协调器共存:gen-init 请求仍全归 `DisaggTransferCoordinator`;disagg ctx-only 请求对 fetch 路径是普通 context 请求,context 结束后**既发给 gen worker 又发布到 store**。终止由**释放门**把关(§9):门只问本层的记录表,**不问 disagg 发送**——disagg 已有自己的释放点,且在 partial-reuse 早终止模式下发送方根本不会来终止请求。
5. 请求状态用设计§12.2 的别名表:`KV_FETCH_IN_PROGRESS ≡ DISAGG_GENERATION_TRANS_IN_PROGRESS(9)`、`KV_PUBLISH_IN_PROGRESS ≡ DISAGG_CONTEXT_TRANS_IN_PROGRESS(21)`,两者都在调度器可调度区间之外,调度器排除逻辑**零改动**。
6. KV v2 wrapper 只改三处(§7);调度器只多问一个只读钩子 `plan_fetch`,多回一个 `fetch_launch_queue`。
7. 范围守卫:KV v2 + `enable_block_reuse` + full attention + TP=PP=CP=1 + 无 ADP + 无 spec-decode + 无 KV connector + 无 draft 管理器 + beam 1;其余装配期拒绝并说明原因。
8. 测试三层:GPU 上对真 `KVCacheManagerV2` 的 reader/resolver/fingerprint 单测;假执行器的 effects 单测;e2e 两个(双实例同 prompt、disagg 共存),按 `test_llm_pytorch.py::test_llm_disagg_gen_cancelled` 的单进程双 `LLM()` 模式写,自起 `mooncake_master` 与长寿命 segment provider,计数经 `TRTLLM_KV_TRANSFER_STATUS_DUMP` 写 JSON 读取。
9. 实施分 6 步,每步独立可测;既有 339 + 345 + 513 个测试全程保持绿色。

## 2. 范围与不做什么

| 做 | 不做(本轮) |
|---|---|
| store 后端的 fetch(含 probe → DEFER)与 publish,单 rank | worker 后端接入协调层(设计§12.2 阶段 1 的 gen-init 迁移);gen-init 仍走 `DisaggTransferCoordinator` |
| 与现有 disagg 共存,ctx-only 请求双发 + 只问本层的释放门 | 删除 KV connector / admission / transfer_manager(设计§12.1) |
| 别名状态常量 | C++ 枚举收敛(设计§11 #1) |
| KV v2 wrapper 三个小改动 | 设计§8.1 #3、#4(`match_keys`、`create_kv_cache(reuse_keys)`)与 `pin_by_keys`:只有 worker 后端应答 demand 才需要 |
| full attention(TinyLlama) | VSWA、SSM、spec-decode、TP/PP/CP>1、ADP、beam>1、KV v1、C++ transceiver |
| 已改名 `BlobStoreBackend`(`backends/blob/backend.py`),Mooncake 特有部分收进 `blob/mooncake.py`;见§6 末「改名决定」 | 第二个 blob 驱动 |
| 配置走环境变量 + YAML | `LlmArgs` schema 改动(不改 → 不触发 golden manifest) |

## 3. 模块与边界

```text
 PyExecutor 循环 / KVCacheV2Scheduler            (共享文件:每处一行调用)
      │ advance_round · launch_reserved_fetches · publish_committed_blocks
      │ on_request_finished · is_tracking · plan_fetch · pace_idle · close
      ▼
 pyexecutor/kv_transfer_binding.py   KVTransferEngineBinding  ← 循环看到的唯一对象
 pyexecutor/kv_transfer_effects.py   PyExecutorKVTransferEffects · EngineRequestView
 pyexecutor/kv_transfer_assembly.py  attach_kv_transfer · check_engine_supports_kv_transfer
      │ Protocol(RequestView / KVTransferEffects / EngineQueue / DistLike)
      ▼
 disaggregation/orchestration/       kv_transfer_coordinator.py (KVTransferCoordinator) · remote_cache.py (Planner,加时间预算)
      │ Fetches / Publishes 契约(base/cache_backend.py)   │ ResourceReader
      ▼                                                  ▼
 disaggregation/backends/            disaggregation/resource/
   config.py              (YAML → KVTransferConfig)        kv_v2_reader.py  (KVv2ResourceReader)
   registry.py            (type → factory → BackendHandle) region.py        (KVv2RegionResolver · layout_fingerprint)
   blob/mooncake.py       (type: mooncake 的驱动:配置、开客户端、工厂)   │ 只经 wrapper 公开方法
   blob/backend.py …      (BlobStoreBackend;RegionResolver 在 base/region.py)  ▼
                                                    pyexecutor/kv_cache/kv_cache_manager_v2.py (3 处小改)
```

边界规则(设计§3.1 四条边界的落地):

- 循环与调度器只 import `kv_transfer_binding` 的**类型**(`TYPE_CHECKING`)或完全不 import(scheduler_v2 用鸭子类型 `self.kv_transfer_planner`),协调层不经包顶层被引入 —— `kv_transfer/test_contract_fakes.py` 的导入卫生断言保持成立。
- 只有 `kv_transfer_effects.py` 写请求状态;只有 `resource/kv_v2_reader.py` 与 `resource/region.py` 调 KV v2 wrapper;后端只见 `RegionResolver` 给的 `(address, size)`。
- `EngineRequestView` 包住 `LlmRequest`,给协调层三个常量属性(`is_gen_init=False`、`is_gen_first_context=False`、`route_hints={}`),其余属性透传;effects 用 `.request` 解包后写。

## 4. 命名表

| 名字 | 类型 | 在哪 | 一句话含义 |
|---|---|---|---|
| `TRTLLM_KV_TRANSFER_CONFIG` | 环境变量 | 读于 `py_executor_creator` | 指向 YAML;未设 = 不装配,零行为差异 |
| `TRTLLM_KV_TRANSFER_STATUS_DUMP` | 环境变量(测试用) | 读于 `kv_transfer_assembly` | 设了则 `close()` 时把协调层 `status_dump()` + 各后端 counters 写成 JSON;路径中的 `{pid}` 替换为 worker 进程号 |
| `KVTransferConfig` | dataclass | `backends/config.py` | 整份 YAML:`backends` 装配表 + 协调层限值 |
| `BackendEntry` | dataclass | 同上 | 装配表一行:`name, type, hint_key, roles, options`;`serves_fetch` / `serves_publish` 是 roles 的布尔视图 |
| `load_kv_transfer_config(path)` | 函数 | 同上 | 读 YAML,校验,返回 `KVTransferConfig` |
| `fetch_timeout_s` / `publish_timeout_s` | 配置键 | YAML 顶层 | 记录 deadline,秒;缺省 None = 不超时 |
| `probe_timeout_s` | 配置键 | YAML 顶层 | 请求等 store 查找的最长时间(秒),过后按本地计算 |
| `close_timeout_s` | 配置键 | YAML 顶层 | `close()` 等后端关闭的上限(秒,缺省 30);超时放弃后端、继续释放请求资源 |
| `BackendRegistry` | 类 | `backends/registry.py` | `type` 名 → `BackendFactory`;内置类型按名懒加载 |
| `BackendFactory` | 类型别名 | 同上 | `(BackendEntry, BackendBuildContext) -> BackendHandle` |
| `BackendBuildContext` | dataclass | 同上 | 工厂可能需要的引擎事实:resolver、fingerprint、`max_unit_bytes`、`device_index` |
| `BackendHandle` | dataclass | 同上 | 装配看到的后端:`fetcher / publisher / pool_registrar / close / counters`(对象,与 `BackendEntry` 的布尔 `serves_*` 区分) |
| `build_backends(config, ctx)` / `close_backends(handles)` | 函数 | 同上 | 按装配表逐行构造;失败关掉已建的 |
| `register_backend_type(name, factory)` | 函数 | 同上 | 外部后端登记入口(设计§10.3) |
| `build_mooncake_backend` | 函数 | `backends/blob/mooncake.py` | `type: mooncake` 的工厂;拒绝 `hint_key`。同文件:`MooncakeStoreConfig`、`open_mooncake_client` |
| `BlobStoreBackend` | 类 | `backends/blob/backend.py` | `Fetches` / `Publishes` / `RegistersPools` 三面合一,只依赖 `StoreClient` Protocol |
| `RegionResolver` / `Segment` | Protocol / 类型别名 | `base/region.py` | `(local_group, local) → Sequence[(address, size)]`;多后端共用 |
| `KVv2ResourceReader` | 类 | `resource/kv_v2_reader.py` | `ResourceReader` 实现:块 key、层组、fetch extent、publish extent |
| `KVv2RegionResolver` | 类 | `resource/region.py` | `(local_group, page) → [(address, size)]`;也给出 pool 跨度与最大 unit 字节数 |
| `layout_fingerprint(manager, page_table)` | 函数 | 同上 | 布局摘要,进 store key;两侧字节含义不同则必不同 |
| `KV_FETCH_IN_PROGRESS` | 常量 | `pyexecutor/kv_transfer_effects.py` | 别名 `DISAGG_GENERATION_TRANS_IN_PROGRESS`:前缀在取回中,调度器不碰 |
| `KV_PUBLISH_IN_PROGRESS` | 常量 | 同上 | 别名 `DISAGG_CONTEXT_TRANS_IN_PROGRESS`:请求已结束,页被传输占着 |
| `EngineRequestView` | 类 | 同上 | `RequestView` 实现,包一个 `LlmRequest` |
| `PyExecutorKVTransferEffects` | 类 | 同上 | `KVTransferEffects` 实现;`held_request_ids` 记录被扣住的请求 |
| `EngineWorkQueue` | 类 | 同上 | `EngineQueue` 实现(后端线程 post,引擎线程 drain);本轮无 poster,为契约完整而存在 |
| `SingleRankDist` | 类 | 同上 | `DistLike` 实现,单 rank:gather 结果就是自己 |
| `KVTransferEngineBinding` | 类 | `pyexecutor/kv_transfer_binding.py` | 循环调用的对象:选请求、包 view、转调协调层、管后端关闭("binding" 而非 "hooks",避免与 connector 的 layer hook 混淆) |
| `.advance_round(active_requests)` | 方法 | 同上 | 循环头(设计§3.2 ①),每轮一次:先挑候选再转调 `advance(candidates, now)`;取一次 `time.monotonic()` |
| `.plan_fetch(req)` / `.DEFER` | 方法/属性 | 同上 | 调度器钩子(设计§5);gen-init 答 None |
| `.launch_reserved_fetches(queue)` | 方法 | 同上 | 调度后(设计§3.2 ③) |
| `.publish_committed_blocks(ctx_requests)` | 方法 | 同上 | context 提交且 forward 完成后(设计§3.2 ④);筛掉 gen-only / dummy / 已失败,转调协调层同名方法 |
| `.on_request_finished(req) -> bool` | 方法 | 同上 | 释放门:告知请求结束;True = 引擎现在可以终止,False = 本层扣住,稍后由本层终止 |
| `.is_tracking(req) -> bool` | 方法 | 同上 | 请求有本层记录(fetch 在飞或被扣住);取消路径用,不看状态;dummy 从不进记录表,故调用方无需先过滤 dummy |
| `.pace_idle()` | 方法 | 同上 | 空转且只有后端能推进时睡 1 ms |
| `.has_transfer_in_flight()` | 方法 | 同上 | 空闲判定用 |
| `.close()` | 方法 | 同上 | 在 `close_timeout_s` 内关后端(等在飞交付结束 = 静默)→ 释放仍被扣住/parked 请求的资源 → 写 status dump |

绑定对象每次调用都新建 `EngineRequestView(request)`;协调层以 `py_request_id` 为键,不对 view 或 `LlmRequest` 的对象身份做任何假设。
| `attach_kv_transfer(executor, config_path, …)` | 函数 | `pyexecutor/kv_transfer_assembly.py` | 守卫 → 建 reader/resolver → 建后端 → 登记 pool → 建协调层 → 挂到 executor 与 scheduler |
| `check_engine_supports_kv_transfer(…)` | 函数 | 同上 | 范围守卫;不满足抛 `ValueError`,消息含原因 |
| `PyExecutor.kv_transfer` | 属性 | `py_executor.py` | 类级缺省 `None`,类型 `Optional[KVTransferEngineBinding]`;`attach_kv_transfer` 赋值 |
| `PyExecutor._kv_fetch_launch_queue` | 属性 | 同上 | 本轮调度器为 fetch 预留了页的请求 |
| `KVCacheV2Scheduler.kv_transfer_planner` | 属性 | `scheduler_v2.py` | 鸭子类型,缺省 `None` |
| `SchedulerOutput.fetch_launch_queue` | 字段 | `scheduler.py` | 缺省空列表;V1 调度器不感知 |
| `reserve_transfer_pages(req, token_end)` | 方法 | `kv_cache_manager_v2.py` | 设计§8.1 #1:为一次传输预留页;`token_end=None` 即 gen-init 的整 prompt |
| `prepare_disagg_gen_init(req)` | 方法(别名) | 同上 | `reserve_transfer_pages(req, None)`;gen-init 调用点与 transceiver 的报错文字不改 |
| `context_block_keys(req)` | 方法 | 同上 | 设计§8.1 #2 |
| `Planner(probe_budget_rounds=, probe_timeout_s=, clock=)` | 参数 | `orchestration/remote_cache.py` | 轮预算与时间预算先到者为准;`clock` 缺省 `time.monotonic`,与 `advance(now)` 同源 |

## 5. 接线点逐条

每处:**位置**(file:function + 相邻语句;行号是 `f974a61764a` 时的,已过期,仅供检索)→ **加什么**(一行调用)→ **为什么** → **怎么验**。所有调用都在 `if self.kv_transfer is not None:` 下,注释一行引设计§。

| # | 位置 | 加什么 | 为什么 | 怎么验 |
|---|---|---|---|---|
| 1 | `py_executor.py:_prepare_and_schedule_batch` 3717,`self.disagg.poll_gen_transfers()` 之后 | `self.kv_transfer.advance_round(self.active_requests)` | 设计§3.2 ①:本轮调度看到上轮落地。两条循环(4173 / 5010)都经此函数;overlap 循环在 5039 `_wait_for_model_engine_input_copy` 之后才调它,页表写入安全 | 假执行器单测:调用一次 `advance`;e2e 中 `fetch_hits > 0` |
| 2 | 同函数 3799 `_schedule()` 返回之后 | `self.kv_transfer.launch_reserved_fetches(self._kv_fetch_launch_queue)` | 设计§3.2 ③;`_schedule` 6322 把 `scheduler_output.fetch_launch_queue` 存到 `self._kv_fetch_launch_queue` | 单测:队列非空 → 请求进 `KV_FETCH_IN_PROGRESS` |
| 3 | `py_executor.py:_executor_loop` 4382,`_update_v2_context_resources` 之后、`_send_kv_async` 4383 之前 | `self.kv_transfer.publish_committed_blocks(scheduled_batch.context_requests)` | **发布不变量:只在产生这些页的 forward 已在 host 侧完成、且 commit 之后发布**(设计§8.1)。非 overlap 循环里 4370 `_update_requests(sample_state)` 已同步过采样结果,forward 必已完成;4382 提交本 batch。放在 `_send_kv_async` 前,ctx-only 请求两条发送同轮启动 | e2e E1:`publish_stored == 已提交整块数` |
| 4 | `py_executor.py:_executor_loop_overlap` 5213,`_update_requests(self.previous_batch.sample_state)` 5204 之后、紧邻 `_send_kv_async(self.previous_batch...)` | `self.kv_transfer.publish_committed_blocks(self.previous_batch.scheduled_requests.context_requests)` | 同一不变量。overlap 下 5276 的 `_update_v2_context_resources(scheduled_batch)` 提交的是**当前** batch,其 forward 仍在执行流上;store 后端在自己的线程/流上拷页,无事件依赖 → 此处发布会读到未写完的页。`previous_batch` 的 commit 发生在上一轮 5276,其 forward 在 5204 `_update_requests` 同步采样结果时已完成,因此发布 `previous_batch` 是唯一安全点(与探索分支一致) | e2e E1 以 `disable_overlap_scheduler=False` 跑一遍(输出相同 + `fetch_hits` 满即证明发布的页是完整的) |
| 5 | `py_executor.py:_terminate_request` 7860 函数首行 | `if not request.is_dummy_request and not self.kv_transfer.on_request_finished(request): return` | 设计§4.3 释放点、§4.2 规则 1:门只问本层记录表(§9);dummy 请求不参与传输,与 7865 对 PP 终止处理器的豁免同理 | 释放门单测(§11 U2 五种情形);e2e E2 |
| 6 | `py_executor.py:_fetch_and_enqueue_requests` 5646 `idle = (...)` | 追加 `and not self.kv_transfer.has_transfer_in_flight()` | 被扣住的请求不算 live;不加则空闲时阻塞在请求队列上,publish 永不被 reap | 单测:一个 held 请求 + 空队列 → timeout 为 0 |
| 7 | `py_executor.py:_executor_loop` 4441 与 `_executor_loop_overlap` 5332,`self.disagg.pace_idle()` 之后 | `self.kv_transfer.pace_idle()` | 空转轮约 1 ms;不睡则 probe 等待期间 CPU 空转、日志刷屏 | 单测:in-flight 或 deferred 时 sleep 被调 |
| 8 | `py_executor.py:_try_cancel_request` 7924,`kv_cache_transceiver is None` 判断之前 | `if self.kv_transfer.is_tracking(request): return False`(无需 dummy 过滤:dummy 不会进记录表,答 False 只针对被跟踪的请求) | `_is_request_in_transmission` 7911 按状态判断,会把我们 parked 的请求交给 transceiver 取消。归属看记录不看状态;取回落地/失败后下一轮取消照常进行 | 单测:parked 请求取消 → 保留在 `canceled_req_ids`;落地后被取消 |
| 9 | `py_executor.py:shutdown` 1690,`torch.cuda.synchronize()` 之后、managers `shutdown()` 之前 | `self.kv_transfer.close()` | 后端可能登记了 KV pool,且可能仍有 parked / 扣住的请求占着页:`close()` 先在 `close_timeout_s` 内等后端关闭(`BlobStoreBackend.close` 的 `pool.shutdown(wait=True)` 即静默;Mooncake 客户端调用与 `client.py`/`backend.py` 均**无**自带超时,所以这个上限是唯一的兜底,超时则记 ERROR、放弃后端、继续),再对每个仍被跟踪的请求调 `_free_request_resources`,最后写 status dump;全部在 `manager.shutdown()` 之前 | e2e 干净退出;U2:关闭时一个 held 请求 → `free_resources` 被调一次;后端 `close` 挂住 → 超时后仍释放 |
| 10 | `scheduler_v2.py:_schedule_loop` 493,`peft_pages = budget.peft_pages_needed(req)` 之前(pending_ctx 循环内) | `answer = self.kv_transfer_planner.plan_fetch(req)`;`DEFER → continue`;有计划 → `_try_reserve_fetch_pages(req, plan)` 成功则 `fetch_launch_queue.append(req); continue` | 设计§5:在 `prepare_context_cache` 之前问,DEFER 不付建/删 cache;有计划的请求像 gen-init(371–396)一样不计 `num_requests/num_tokens`。注意该循环里两处**既有** `continue` 先于本钩子:465–470(chunking 开启且 chunk token 预算耗尽)与 483–491(首个新块已被本轮某请求贡献)—— 此时请求本轮不被规划,下一轮仍是候选,不影响正确性,只是取回晚一轮 | `test_kv_cache_v2_scheduler.py` 现有用例不变;新增:planner 答 plan → 请求进 `fetch_launch_queue`、不进 `scheduled_ctx`;chunk 预算耗尽时 planner 不被调用 |
| 11 | `scheduler_v2.py` 新方法 `_try_reserve_fetch_pages`(放在 `_try_schedule_disagg_gen_init` 713 旁) | 调 `reserve_transfer_pages(req, plan.token_end)`;失败则 `free_resources` + `rewind_context_after_cache_drop`,返回 `SKIP`;调用方对 `SKIP` 执行 `continue`(本轮不调度该请求,下一轮计划仍在,再试) | 与 gen-init 同一条分配路径(730),多传 `token_end`;SKIP 语义与 731–733 一致 | 同上 |
| 12 | `scheduler.py:SchedulerOutput` 79 | 字段 `fetch_launch_queue`,缺省 `[]` | V1 调度器继续构造八字段输出 | 现有 scheduler 单测 |
| 13 | `py_executor_creator.py:_create_py_executor_impl` 1065 `start_worker()` 之前 | `if os.environ.get("TRTLLM_KV_TRANSFER_CONFIG"): from .kv_transfer_assembly import attach_kv_transfer; attach_kv_transfer(...)`(字面字符串,不在模块顶层 import 常量,保持懒加载) | 设计§7.4:装配期一次性构造;懒 import 保持导入卫生 | 未设环境变量时 `grep` 确认无协调层模块被加载 |

`py_executor.py` 另需两行状态:类级 `kv_transfer: Optional["KVTransferEngineBinding"] = None`(`object.__new__(PyExecutor)` 的测试读得到)与 `__init__` 里 `self._kv_fetch_launch_queue: List[LlmRequest] = []`。

## 6. 新文件

| 文件 | 职责 | 公开接口 | 预算(行) |
|---|---|---|---|
| `disaggregation/backends/config.py` | YAML → 数据类;校验 roles、name 唯一、未知键 | `KVTransferConfig`、`BackendEntry`、`load_kv_transfer_config`、`KV_TRANSFER_CONFIG_ENV`、`BACKEND_ROLES` | ≤150 |
| `disaggregation/backends/registry.py` | type → factory;内置类型按名懒加载 | `BackendRegistry`、`BackendFactory`、`BackendBuildContext`、`BackendHandle`、`build_backends`、`close_backends`、`register_backend_type` | ≤150 |
| `disaggregation/backends/blob/mooncake.py` | `type: mooncake` 驱动:配置、开 client、按需开 staging、返回 handle;唯一 import Mooncake 绑定的地方 | `MooncakeStoreConfig`、`open_mooncake_client`、`build_mooncake_backend` | ≤200 |
| `disaggregation/base/region.py`(追加) | 多后端共用的 `(local_group, local) → 段` 抽象 | `RegionResolver`、`Segment` | +20 |
| `disaggregation/resource/kv_v2_reader.py` | `ResourceReader` 实现;块 key 缓存(按 request_id,prompt_len 变则失效) | `KVv2ResourceReader`(含 `forget_request`) | ≤200 |
| `disaggregation/resource/region.py` | 页 → 内存段;pool 跨度;布局指纹 | `KVv2RegionResolver`(`__call__`、`pool_memory_spans`、`max_unit_bytes`)、`layout_fingerprint` | ≤100 |
| `pyexecutor/kv_transfer_effects.py` | 引擎侧四个 Protocol 的实现 + 两个别名常量 | 见命名表 | ≤260 |
| `pyexecutor/kv_transfer_binding.py` | 循环视角的 API;候选/发布请求筛选;关闭顺序 | `KVTransferEngineBinding` | ≤200 |
| `pyexecutor/kv_transfer_assembly.py` | 范围守卫、装配、pool 登记、status dump、日志 | `attach_kv_transfer`、`check_engine_supports_kv_transfer` | ≤180 |

三点实现约定:

- `kv_v2_reader.publish_description` 读**提交后**的页(`kv_cache.num_committed_tokens`、`get_aggregated_page_indices(group, valid_only=False)`),按 `_stale_block_range(group, history_length)` 剔除 stale 块(full attention 下为空范围);chunk 恒为 `None`(store 不按位置搬)。
- `fetch_extent` 只为 `plan.group_plans` 里、且调度器已分到页的 ordinal 造 unit;`CacheExtent.name = b"fetch:<rid>"`,`is_last=True`。
- `EngineRequestView.__getattr__` 透传,使 `resource/` 能把 view 直接交给 wrapper 方法。

**改名决定(`MooncakeStoreBackend` → `BlobStoreBackend`):已改。** 按 kv-shared-draft README §5,"Mooncake 复用 `blob/` 的适配层,故位于 `blob/` 之下":`backends/blob/backend.py::BlobStoreBackend` 只依赖 `StoreClient` Protocol,不含 Mooncake 特有逻辑;Mooncake 特有的三件事——`MooncakeStoreConfig`、`open_mooncake_client`(唯一 import 绑定处)、注册表工厂 `build_mooncake_backend`——收在 `blob/mooncake.py` 一个驱动文件里,后续驱动与其并列。`StoreClient` 的返回码约定(`OBJECT_NOT_FOUND = -704`、`batch_is_exist` 的 1/0/负数)仍照抄 Mooncake,这是第二个驱动出现时要泛化的点。**"mooncake" 隔离规则只约束本特性新增与触及的代码**(`backends/`、`resource/`、`pyexecutor/kv_transfer_*`、调度器与循环的接线点):其中该词只出现在 `blob/mooncake.py`、`blob/backend.py` 的线程名前缀与注册表内置表的一行。用户 YAML 不受影响:键是 `type: mooncake`,不含目录名。

## 7. KV v2 wrapper 改动(`pyexecutor/kv_cache/kv_cache_manager_v2.py`)

| # | 方法 | 改动 | 设计依据 | 验证 |
|---|---|---|---|---|
| 1 | 新增 `reserve_transfer_pages(req, token_end)`;`prepare_disagg_gen_init(req)` 保留为一行别名(`token_end=None`) | `None` 走原路(history = prompt_len,含 draft 与 extra);给了则 `history = target = max(history, token_end)`,并置 `kv_cache.enable_swa_scratch_reuse = False`。`resize(capacity, history)`。fetch 接缝 `_try_reserve_fetch_pages` 调新名;gen-init 分支与 transceiver 报错文字继续用别名 | §8.1 #1;§6.3 规则 1(history 一次声明到 token_end) | GPU 单测:`get_history_length == token_end`;`revert_allocate_context` 后 `kv_cache_map` 无该请求 |
| 2 | 新增 `context_block_keys(req) -> list[bytes]` | `_context_reuse_tokens(req)` + `ReuseScope(lora_task_id, _derive_reuse_salt(cache_salt))` → `sequence_to_blockchain_keys`,跳过 root,取 `len(tokens) // tpb` 个 | §8.1 #2 | GPU 单测:两个同 prompt 请求 key 相同;prompt 改一个 token 后从该块起不同;数量 = `(prompt_len - 1) // tpb` |
| 3 | `release_index_slot` 5039 | 已在 `_early_freed_index_requests` 的 request_id 直接返回(主管理器也如此,不只 draft) | 两个传输方(disagg 发送 409–430 已调一次;`hold_for_transfer` 可能再调)共享一个请求 | 单测:连续调两次不抛;`index_mapper.remove_sequence` 只被调一次 |

不改:`_settle_context_cursor` 1086(直接复用)、`try_commit_blocks` 5012、`revert_allocate_context` 3379、`probe_context_reuse` 3463、`_stale_block_range` 3753、`get_history_length` 3672。

## 8. 配置与注册表格式

```yaml
# TRTLLM_KV_TRANSFER_CONFIG=/path/to/kv_transfer.yaml
fetch_timeout_s: 30          # 可选;None = 不超时
publish_timeout_s: 60        # 可选
probe_timeout_s: 0.05        # 可选;等 store 查找的时间上限,过后按本地计算

backends:                    # 顺序即 fetch 优先级(设计§7.4)
  - name: shared-store       # FetchSource.name;记录在计划与 attempt 上
    type: mooncake           # 注册表键;本特性新增代码里只有 backends/blob/mooncake.py 与注册表内置表认识它
    roles: [fetch, publish]  # 子集;缺省两者都有
    # hint_key 缺省 None:store 目的地唯一;mooncake 工厂拒绝非 None
    # ---- 以下为 type 特有字段,原样交给工厂(MooncakeStoreConfig.from_dict)----
    master_server_address: 127.0.0.1:50051
    protocol: tcp
    local_hostname: 127.0.0.1
    global_segment_size: 0   # 引擎不贡献段;对象活在 segment provider 里(§10 #8)
    stage_through_host: true # TCP 传输走 pinned host 中转;KV pool 不登记
    namespace: tinyllama-e2e
```

注册表:`BackendRegistry.factory_for(type)` 先查显式登记,再查内置表 `{"mooncake": ".blob.mooncake:build_mooncake_backend"}`(相对本包的字符串,import 延后到首次使用);未知类型报错并列出已知类型。加后端 = 新目录 + 内置表一行(或运行期 `register_backend_type`),协调层、hooks、调度器不改(设计§10.3)。

`BackendHandle.pool_registrar`:`stage_through_host=False` 时为后端自身(`RegistersPools`),装配把 `KVv2RegionResolver.pool_memory_spans()` 逐个 `register_pool`;为 True 时 `None`。

**测试用状态导出** `TRTLLM_KV_TRANSFER_STATUS_DUMP=/tmp/kvt-{pid}.json`:设了则 `KVTransferEngineBinding.close()` 末尾写一份 JSON:`{"started_at": <attach 时的 time.time()>, "pid": …, "coordinator": status_dump(), "backends": [{"name", "type", "roles", "counters": {...}}]}`。`{pid}` 由 worker 进程号替换,同一测试里的多个 `LLM()` 各写一份;测试按 `started_at` 排序即得创建顺序。不设则不写;非测试代码不读它。

## 9. 共存规则与释放门

**归属规则**(设计§3.1、§4.2):

1. gen-init 请求(状态 8)从不在 `CONTEXT_INIT`,天然不是候选;`KVTransferEngineBinding.plan_fetch` 再对 `is_disagg_generation_init_state` 答 `None` 兜底。`DisaggTransferCoordinator` 对它们的处理一字不改。
2. gen-first 的 ctx 请求在 `DISAGG_CONTEXT_WAIT_SCHEDULER`(7)等待,同样不是候选;`prepare_context_schedulable` 放行后才进入 fetch 路径。
3. 候选 = `state == CONTEXT_INIT and is_first_context_chunk and not is_dummy_request and not is_disagg_generation_init_state` 且计划未定(这四个条件写在 `advance_round` 的筛选函数 `_is_fetch_candidate` 里,最后一条是对规则 1 的显式兜底);disagg ctx-only 与普通请求一视同仁。
4. 发布筛选:`context_remaining_length == 0`、有 `_KVCache`、非 `GENERATION_COMPLETE`(失败路径)、非 `is_generation_only_request`、非 dummy。gen worker 因而不为 disagg 请求发布。
5. 两个协调器都会写别名状态;**谁拥有请求由各自的记录表决定(本层:`TransferRecord` 表;disagg:`AsyncTransferManager.requests_in_transfer()`),任何代码不得靠读状态判断归属。**

**释放门的原则:门只问本层。** `_terminate_request` 是引擎对一个请求的释放点;谁调它,就是在说"引擎这边不再需要它"。disagg 发送有自己的释放点(`release_transfer` 520–557 → `effects.terminate_request`),并且 disagg 已经处理了"页在发送中而请求被终止"的情形:`start_transfer`(`transfer_manager.py` 58–91)把块 `store_blocks_for_reuse` 钉进 reuse tree,`free_resources` 不会碰它们。因此本层**不**查 `requests_in_transfer()`:查了反而在下面情形 E 中把请求扣死(评审第 1 轮的发现)。

释放门 `on_request_finished(R)` 的行为:`coordinator.notify_request_finished(view)`;若 R 随后在 `held_request_ids`(publish 仍在飞 → `hold_for_transfer` 被调)则返回 False,否则 True。`effects.terminate_request(R)`(协调层在记录全部 RELEASED 后调)直接调 `executor._do_terminate_request(R)`(7877)—— 此时 R 必已离开 `active_requests`(`_handle_responses` 8238 只把未完成的请求放回,被扣住的请求都是已完成的),所以不做任何 `active_requests` 操作;**不**经 `_terminate_request`(否则重入门时 `held_request_ids` 尚未清除,误判为仍扣住),**不**查 disagg。

五种情形(`R` 为请求;"publish"指本层记录;"send"指 disagg 发送):

| 情形 | 谁先到 `_terminate_request` | 门的答复 | 谁最终 `_do_terminate_request` |
|---|---|---|---|
| A 普通请求,publish 已落地 | `_handle_responses` 8235 | True | 引擎,立即 |
| B 普通请求,publish 在飞 | `_handle_responses` 8235 | False(扣住,状态 21) | 协调层,publish 落地后 |
| C ctx-only,send 先完成,publish 在飞 | disagg `release_transfer` 557 | False(扣住;seq slot / index slot 已被 disagg 释放,effect 跳过) | 协调层,publish 落地后 |
| D ctx-only,publish 先落地,send 在飞 | disagg `release_transfer`(send 完成时):R 仍活跃则经 `stage_transfer_response` → `_pending_response_terminations` → `_flush_pending_transfer_responses` 1268 `_terminate_request`;否则 557 直接 `effects.terminate_request` → `_terminate_request` | True(publish 记录早已 RELEASED,`_finish_publish` 对未结束的请求不做事) | 引擎,立即 |
| E ctx-only,`force_terminate_ctx_for_partial_reuse`(668:`enable_partial_reuse_for_disagg` 且 PP=1) | `_handle_responses` 8233 在 prefill 完成当轮,send **仍在飞** | 视 publish:落地 → True,引擎立即终止(页由 reuse tree 的 pin 保活,与今天相同);在飞 → False,协调层落地后终止。`release_transfer` 556 在此模式下**不会**再调 `terminate_request`,所以若门在此查 `requests_in_transfer()`,R 将永远无人终止。**注**:650–653 把该开关限定为 `not _is_kv_manager_v2`,本轮范围(KV v2)内 E 实际不可达;门仍按"不查 disagg"设计,以免将来 V2 打开该开关时门要重写;U2 用假执行器合成 E | 引擎或协调层,各恰一次 |

**ctx-only 请求全程**(disagg ctx worker + store,`R` 为请求,情形 C/D):

```text
① 到达 CONTEXT_INIT → advance: probe → DEFER 一两轮 → FetchPlan(token_end = 连续命中前缀末端)
② 调度器: prepare_context_cache(本地 reuse) → reserve_transfer_pages(R, token_end) → fetch_launch_queue
③ launch_reserved_fetches: fetch(extent) → park_for_fetch: state = KV_FETCH_IN_PROGRESS(9)
④ advance: Delivered → unpark: state = CONTEXT_INIT; _settle_context_cursor(R, token_end); try_commit_blocks
   (失败: quiesce → give_back_fetch_pages → 重试一次或按本地计算)
⑤ 调度器按 num_committed_tokens 落游标,算剩余 [token_end, prompt_len)
⑥ context 结束同一轮:
   _update_v2_context_resources(commit) → publish_committed_blocks(R) → publish 记录 IN_FLIGHT
   _send_kv_async → send_completed_context: release_index_slot, start_transfer(state=21), respond_and_send_async
   _handle_responses: is_disagg_context_transmission_state → 不终止,移出 active_requests(既有行为 8226)
⑦ 情形 C:disagg 发送完 → reap_context_sends → release_transfer → end_transfer=True → effects.terminate_request → _terminate_request(R)
        → on_request_finished(R): notify_request_finished → publish 仍在飞 → hold_for_transfer(R) → False → return
   之后 store 发布完 → advance → _finish_publish → _finish_if_released → effects.terminate_request(R) → _do_terminate_request(R)
   情形 D:store 先落地 → 记录 RELEASED,R 未结束,无动作;disagg 发送完 → _terminate_request(R) → 门答 True → _do_terminate_request(R)
⑧ free_resources 恰好一次
```

`hold_for_transfer` 效果**不读** `requests_in_transfer()`,无条件释放 seq slot 并调 `release_index_slot`:`SlotManager.remove_slot`(`resource_manager.py` 2694–2697)对未知 id 是 no-op,所以 disagg `start_transfer` 已释放过的 seq slot 再释放一次无害;spec 资源管理器不在范围内(守卫拒绝);`release_index_slot` 因 §7 #3 幂等。本层因此对 disagg 记录表零读取。状态写 `KV_PUBLISH_IN_PROGRESS` 与 disagg 已写的值相同,无副作用。

**关闭时的释放**(§5 #9):`close()` 顺序 = 在 `close_timeout_s` 内关后端(在飞交付结束即静默;超时则记 ERROR 放弃)→ 对 `held_request_ids` 与所有 parked(有 fetch 记录)的请求各调一次 `executor._free_request_resources` → 写 status dump。这些请求此时已不在 `active_requests`(held)或不会再被调度(parked),不走 `_do_terminate_request` 以免动 `result_wait_queues`。

## 10. 已知约束与对策

| # | 约束 | 对策 |
|---|---|---|
| 1 | C++ `setPrepopulatedPromptLen` 断言 `prepopulated < promptLen`;`getContextChunkSize` 断言请求处于 context 状态 | `unpark` 顺序固定:先 `state = CONTEXT_INIT`,再 `_settle_context_cursor`;Planner 已保证 `token_end ≤ ⌊(prompt_len-1)/tpb⌋·tpb < prompt_len`;`unpark` 另断言 `get_history_length(R) >= token_end`(接线错误早暴露) |
| 2 | `_KVCache` 非线程安全 | 协调层、reader、effects 全在引擎线程;后端线程只碰 `StoreClient` 与 staging;`EngineWorkQueue` 本轮无 poster |
| 3 | `kv_transfer/test_contract_fakes.py` 断言协调层不经包顶层加载 | scheduler_v2 鸭子类型;creator 懒 import;`py_executor.py` 只 `TYPE_CHECKING` import |
| 4 | `object.__new__(PyExecutor)` 的测试 | `kv_transfer` 类级缺省 `None`;`_kv_fetch_launch_queue` 只在 `_schedule` 赋值后被读 |
| 5 | Planner probe 预算按轮计,空闲轮 ~1 ms,本地 Mooncake 查找即耗尽 | 双保险:Planner 加 `probe_timeout_s`(缺省 0.05)与可注入 `clock`,首次 DEFER 记时间,超时视为未命中;保留 `probe_budget_rounds` 参数以兼容 345 个既有测试(两者取"先到者") + hooks `pace_idle` 在 deferred/in-flight 时睡 1 ms |
| 6 | `release_index_slot` 对主管理器不幂等 | §7 #3 |
| 7 | 状态 9 的 parked 请求被 `_is_request_in_transmission` 当成 disagg 传输 | §5 #8:`is_tracking()` 先判 |
| 8 | Mooncake 对象活在客户端段里,publisher 退出即带走 key | 引擎 `global_segment_size: 0`;e2e 起 `mooncake_master` + **segment provider**(一个 `MooncakeDistributedStore.setup(global_segment_size=N)` 后空转的子进程),两者活过所有引擎实例 |
| 9 | 空闲循环阻塞在请求队列 | §5 #6 |
| 10 | 有 `FetchPlan` 的请求受 `if budget.requests_full: break`(460)影响,gen-init 因在更早阶段(371)不受 | 本轮接受(取回请求少);记录为后续项 |
| 11 | fetch 落地的请求 `context_current_position` 直接来自游标,`unpark` 后 `py_ctx_pre_resize_cap` 必须清空 | `unpark` 置 `R.py_ctx_pre_resize_cap = None`,否则后续 `_revert_ctx_alloc` 会把已有内容的页缩掉 |
| 12 | TCP 传输能否直接读 GPU 登记内存未验证 | e2e 用 `stage_through_host: true`;RDMA 路径留待有网卡的环境 |
| 13 | `revert_allocate_context` 3379 在 `py_ctx_pre_resize_cap is None` 时直接返回 True(`reserve_transfer_pages` 只在容量真的增长时才记 pre_cap;cache 被 resume 且容量够时不记)→ `give_back_fetch_pages` 后 cache 仍活着、history 仍声明到 `token_end`,而页里没有数据 | `give_back_fetch_pages` 在 `_revert_ctx_alloc` 之后检查:请求仍在 `kv_cache_map` 则 `kv_cache_manager.free_resources(R)` + `rewind_context_after_cache_drop(R, tpb)`(`llm_request.py` 1666;与 `_try_reserve_fetch_pages` 失败路径同一套),请求作为全新首 chunk 重入。full attention 下 history 偏高本身不致错(无 stale 范围、默认 `all_reusable` 不要求 commit 终点等于 history),但重新走 reuse match 更简单也更省页,统一丢弃 |
| 14 | `unpark` 之后请求以 `is_first_context_chunk` 重入调度,`reserve_transfer_pages`/`prepare_context_cache` 会按 `num_committed_tokens` 重新落游标;若 `try_commit_blocks` 没提交到 `token_end`,游标会回退到本地 reuse 深度、已取回的页被当作未算 | 范围守卫加 `enable_block_reuse=True`(`try_commit_blocks` 5013 在关闭 reuse 时直接返回);`unpark` 在 commit 后检查 `kv_cache.num_committed_tokens >= token_end`,不满足记 WARNING 并继续(请求退回按本地 reuse 深度重算,慢但正确;不用硬断言,避免一个请求拖垮引擎) |
| 15 | 时间来源不一致会让 deadline 与 probe 预算各说各话 | binding 每轮取一次 `now = time.monotonic()` 传 `advance(candidates, now)`;Planner 的 `clock` 缺省即 `time.monotonic`,装配不另注入;测试用 `kv_transfer/fakes.py` 的假时钟同时替换两处 |

## 11. 测试计划

单元(CPU,无 GPU 者标 `cpu_only`):

| ID | 文件 | 覆盖 |
|---|---|---|
| U0 | `tests/unittest/_torch/disaggregation/engine_integration/test_kv_transfer_config_registry.py` | YAML 解析/校验;注册表:未知 type、重复登记、假工厂、失败回滚 `close`;`mooncake` 内置项按名懒加载(用 monkeypatched `open_mooncake_client`,断言模块名 `disaggregation.backends.blob.mooncake`) |
| U1 | `tests/unittest/_torch/executor/kv_transfer/test_kv_v2_reader_layout.py`(GPU) | 对真 `KVCacheManagerV2`(小配置,如 4 层 / tpb 32 / 256 页):`context_block_keys` 数量与一致性;`layout_fingerprint` 同配置稳定、改 tpb/dtype/头数即变;`KVv2RegionResolver` 段数 = pool 数、跨度不重叠;`reserve_transfer_pages(token_end)` 后 `fetch_extent` 的 unit 数 = `(token_end/tpb - reuse_end)`;两请求经 `try_commit_blocks` 后 `publish_description` 命名相同 |
| U2 | `tests/unittest/_torch/executor/kv_transfer/test_kv_transfer_effects_binding.py`(另有 `test_kv_transfer_assembly_guard.py` 守卫、`test_kv_transfer_hook_points.py` 按源码文本核对 13 个接线点) | `object.__new__(PyExecutor)` + 假 `resource_manager`/`kv_cache_manager`(不需要假 `async_transfer_manager`:本层不读它),真 `KVTransferCoordinator` + `kv_transfer/fakes.py` 的 `FakeFetches/FakePublishes`:park/unpark 状态与游标(含 §10 #14 的 WARNING 路径);`give_back_fetch_pages` 调 `_revert_ctx_alloc`,cache 残留时 `free_resources` + rewind(§10 #13);释放门五种情形 A–E(§9)各 `_do_terminate_request` 恰好一次,E 下 `release_transfer` 不再来也不泄漏;`is_tracking` 对 parked 为 True;dummy 请求绕过门;`has_transfer_in_flight` 影响 idle;`close()` 释放 held 请求并写 dump,后端 `close` 挂住时超时后仍释放。overlap 发布只取 `previous_batch` 这一点不在此单测(需要整条循环),由 E1 的 `disable_overlap_scheduler=False` 参数化覆盖 |
| U3 | `tests/unittest/_torch/executor/kv_transfer/test_scheduler_kv_fetch_seam.py`(`test_kv_cache_v2_scheduler.py` 不改) | planner 答 DEFER → 不建 cache;答 plan → 进 `fetch_launch_queue`、不计 `num_requests`;`reserve_transfer_pages` 失败 → 请求留在 CONTEXT_INIT;gen-init 仍走 `prepare_disagg_gen_init(req)` 原调用形状;导入卫生(monkeypatch 字串含 `orchestration.remote_cache`) |
| U4 | `kv_transfer/test_planner_time_budget.py` | `probe_timeout_s` 与注入 clock:超时后无 store 答案 → None |
| U5 | `blob_backend/test_worker_pool.py` | `DaemonWorkerPool`:daemon 线程、`shutdown(wait=True)` 等在飞任务、异常不吞 |

e2e(`tests/unittest/_torch/disaggregation/e2e/`,GPU,`pytest.importorskip("mooncake.store")`,无 `mooncake_master` 则 skip;模型 `/workspaces/tekit/.models/TinyLlama-1.1B-Chat-v1.0`,可由 `TINYLLAMA_MODEL_PATH` 覆盖):

- 进程模型:照 `tests/unittest/llmapi/test_llm_pytorch.py:1296 test_llm_disagg_gen_cancelled`——**一个测试进程里直接创建两个 `LLM()`**(ctx 与 gen,各自 `CacheTransceiverConfig(backend="NIXL", transceiver_runtime="PYTHON")`),标 `@pytest.mark.private_mpi_session`、`@pytest.mark.timeout`。不再自造子进程 + Pipe 的驱动。两个环境变量(`TRTLLM_KV_TRANSFER_CONFIG`、`TRTLLM_KV_TRANSFER_STATUS_DUMP`)在创建 `LLM()` 前用 `monkeypatch.setenv` 设定,由 MPI spawn 的 worker 继承(待 S4 验证,见§14)。
- 公共夹具 `mooncake_cluster.py`:`start_master()`(复用 `blob_backend/test_real_master.py` 的 free-port + TCP 就绪等待,20 s)、`start_segment_provider(master, bytes=1 GiB)`(`multiprocessing` 子进程,`MooncakeDistributedStore().setup("127.0.0.1", "P2PHANDSHAKE", N, 16<<20, "tcp", "", master)` 后 `Event.wait()`,teardown `terminate`;它是唯一贡献段的客户端,活过所有 `LLM()`)、`write_kv_transfer_yaml(tmp_path, master, namespace)`(§8 的样例,`probe_timeout_s: 1.0`)、`read_status_dumps(glob) -> list` 按 `started_at` 排序。每实例 `KvCacheConfig(use_kv_cache_manager_v2=True, enable_block_reuse=True, free_gpu_memory_fraction=0.2, tokens_per_block=32)`,`SamplingParams(max_tokens=16, temperature=0)`。
- **E1 双实例同 prompt**(`test_store_fetch_two_instances`):实例 A 生成(prompt ≥ 6 块)→ `A.shutdown()` → 实例 B 同 prompt 生成 → `B.shutdown()`。断言:两次输出 token 相同;dump[0](A)`publish_stored == (prompt_len-1)//tpb`;dump[1](B)`fetch_hits == (prompt_len-1)//tpb`、`fetch_misses == 0`、`decided_plans`/记录表为空(无泄漏)。参数化 `disable_overlap_scheduler ∈ {True, False}`。超时 300 s。
- **E2 disagg 共存**(`test_store_with_disagg_ctx_gen`):ctx 与 gen 同时存活,同一份 YAML。① ctx 处理 `context_only` 请求 P → gen 用返回的 `disaggregated_params` 处理 `generation_only` → 输出 O1;② ctx 再处理同 prompt P 的 `context_only` → gen 生成 O2;③ 对照 O0:同进程内先用**无 store、无 transceiver** 的单实例算一次(创建在两者之前,`monkeypatch.delenv` 保证它不装配)。断言:O1 == O2 == O0;ctx dump `fetch_hits > 0`(第二次请求命中)、`publish_stored > 0`、记录表为空;gen dump `publish_stored == 0`、`fetch_hits == 0`;ctx 端 `get_stats` 的 `kvCacheStats.usedNumBlocks` 在两次请求后回到基线(与 1367–1380 的等待循环同法)—— 这是"每个请求 `free_resources` 恰一次、无泄漏"的外部可见判据。情形 E 在 KV v2 下不可达(§9),只由 U2 覆盖。超时 600 s。

保持绿色的既有套件(以 `--collect-only` 计,HEAD f86fd5c4edf 之后的工作树):T1 `tests/unittest/_torch/disaggregation/`(不含 `kv_transfer/`、`blob_backend/`、`engine_integration/`、`e2e/`)339;T3 `kv_transfer/` + `blob_backend/` + `engine_integration/` + `tests/unittest/_torch/executor/kv_transfer/` 561(计划 v3 时为 345,新增 `test_worker_pool.py`、`engine_integration/`、`executor/kv_transfer/` 五文件后为此数);T4 `tests/unittest/_torch/executor/kv_cache/` + `executor/test_py_executor.py`(v3 写的"513"集合无法复现,以此整目录为基线;其中 53 个失败与本线无关、改动前后一致);E1/E2 `disaggregation/e2e/` 3 用例(E1 参数化 overlap/no_overlap)。

## 12. 实施顺序与检查点

每步一个 commit,独立可测;每步末跑本步新增 + 上述三组套件。

| 步 | 内容 | 检查点 |
|---|---|---|
| S0 | `backends/config.py`、`registry.py`、`blob/mooncake.py`;U0 | 纯 Python,无引擎依赖;T3 绿 |
| S1 | wrapper 三改(§7)+ `kv_v2_reader.py` + `resource/region.py`;U1 | GPU 单测绿;T4 绿(wrapper 改动向后兼容) |
| S2 | Planner `probe_timeout_s` / `clock`;U4 | 345 绿(默认值不改变既有轮计数行为) |
| S3 | `kv_transfer_effects.py`、`kv_transfer_binding.py`、`kv_transfer_assembly.py`(含 status dump、`close()` 释放顺序);§5 全部接线点;U2、U3 | 未设环境变量:T1 + T3 + T4 绿,`test_contract_fakes` 导入卫生绿;设环境变量指向非法 YAML:装配期 `ValueError` 含原因;全仓 `grep DISAGG_GENERATION_TRANS_IN_PROGRESS\|DISAGG_CONTEXT_TRANS_IN_PROGRESS` 逐处核对读状态的代码,结论写进 PR 描述 |
| S4 | e2e 夹具 + E1;**先验证**环境变量能否到达 MPI spawn 的 worker(不能则改为在 YAML 路径上用固定约定,如 `~/.trtllm/kv_transfer.yaml`,并记录) | E1 两个参数化用例绿;探索分支的 20 块结果复现 |
| S5 | E2 + 释放门情形 C/D/E 补强 | E2 绿;`usedNumBlocks` 回到基线 |
| S6 | 收尾:三个新模块的模块 docstring 即目录说明(`tensorrt_llm/_torch/disaggregation/` 下无 README,不新建);PR 描述写 rationale 与 grep 结论 | pre-commit 通过 |

## 13. 风险与回滚

| 风险 | 概率/影响 | 对策 / 回滚 |
|---|---|---|
| 环境变量到不了 MPI spawn 的 worker | 低(探索 smoke 已在 shell 设变量跑通;测试内 `monkeypatch.setenv` 早于 spawn) / E1 不装配 | S4 首项验证;备选见 §12 S4 |
| 同进程两个 `LLM()` 的 worker 各写 dump 互相覆盖 | 低 | 路径含 `{pid}`,按 `started_at` 排序 |
| TCP 传输 + GPU 直读不可用 | 中 / E1 fetch 失败 | `stage_through_host: true`(缺省即如此写在 e2e YAML) |
| overlap 下发布读到 forward 未写完的页 | 已消除 | §5 #4 只发布 `previous_batch`;E1 overlap 参数化 |
| probe 时间预算太短,本地 Mooncake 也答不完 → 全部本地计算,`fetch_hits == 0` | 低 / E1 断言失败 | e2e 用 `probe_timeout_s: 1.0`;生产缺省 0.05 |
| `hold_for_transfer` 与 disagg `start_transfer` 重复释放 seq slot | 无 | `SlotManager.remove_slot` 对未知 id 是 no-op(2694–2697);U2 五种情形 |
| 后端 `close()` 挂住导致 shutdown 不返回 | 低 | `close_timeout_s`(§5 #9) |
| partial-reuse 早终止(情形 E)下请求被本层扣死 | 已消除 | 门不查 `requests_in_transfer()`;U2 情形 E |
| 别名状态 9 被其他读状态的代码误判 | 低 | 全仓 `grep DISAGG_GENERATION_TRANS_IN_PROGRESS` 逐处核对(S3 内完成,结果写进 PR 描述) |
| 行号漂移 | 必然 | 接线以函数名 + 相邻语句定位,行号仅供检索 |

**回滚**:不设 `TRTLLM_KV_TRANSFER_CONFIG` 即回到基线行为——所有接线点受 `kv_transfer is not None` 保护,wrapper 三改向后兼容(缺省参数、幂等化)。代码回滚 = revert S3 之后的 commit;S0–S2 无引擎副作用。

---

## 14. 对照代码核实情况

行号已随实施漂移,不再维护;下列锚点按**函数名 + 相邻语句**给出,`grep -n "def <名字>"` 即得(HEAD f86fd5c4edf 之后的工作树核实;`tests/unittest/_torch/executor/kv_transfer/test_kv_transfer_hook_points.py` 按源码文本逐条守着这些接线点)。

**已核实**:
- `py_executor.py`:`_prepare_and_schedule_batch`(`poll_gen_transfers()` → `advance_round` → `check_transfer_timeouts()`;`_schedule()` 之后 `launch_reserved_fetches(self._kv_fetch_launch_queue)`);非 overlap `_executor_loop`(`_update_requests(sample_state)` → `_update_v2_context_resources(scheduled_batch)` → `publish_committed_blocks(scheduled_batch.context_requests)` → `_send_kv_async`);overlap `_executor_loop_overlap`(`_update_requests(self.previous_batch.sample_state)` → `publish_committed_blocks(self.previous_batch.scheduled_requests.context_requests)` → `_send_kv_async(previous_batch)`;当前 batch 的 `_update_v2_context_resources(scheduled_batch)` 在其后);两循环末尾 `self.disagg.pace_idle()` → `self.kv_transfer.pace_idle()`;`_fetch_and_enqueue_requests` 的 idle 判定追加 `has_transfer_in_flight()`;`_terminate_request` 首行的释放门(dummy 豁免同函数);`_do_terminate_request`;`_try_cancel_request` 在 `kv_cache_transceiver is None` 判断前的 `is_tracking`;`_is_request_in_transmission`;`_handle_responses` 的 `force_terminate_for_partial_reuse` 分支与 ctx-only 不终止分支;`force_terminate_ctx_for_partial_reuse` 及其前提 `enable_partial_reuse_for_disagg`(含 `not _is_kv_manager_v2`);`_schedule` 存 `fetch_launch_queue` 并传 `protected_from_eviction_request_ids=self.kv_transfer.inflight_request_ids()`;`_terminate_recompute_paused_requests` 跳过在飞请求;`shutdown` 里 `torch.cuda.synchronize()` 之后、managers `shutdown()` 之前的 `kv_transfer.close()`。
- `scheduler_v2.py`:`_schedule_loop` pending_ctx 循环里 `_try_take_fetch_path` 先于 `peft_pages_needed`,两处既有 `continue`(chunk 预算耗尽、首个新块已被贡献)在其前;gen-init 分支调 `prepare_disagg_gen_init(req)`;`_try_reserve_fetch_pages` 调 `reserve_transfer_pages(req, plan.token_end)`,失败 `free_resources` + `rewind_context_after_cache_drop`。
- `kv_cache_manager_v2.py`:`_settle_context_cursor`;`revert_allocate_context`(`py_ctx_pre_resize_cap is None → return True`);`probe_context_reuse`;`reserve_transfer_pages`(只在容量增长时记 pre_cap;`prepare_disagg_gen_init` 为其别名);`get_history_length`;`_stale_block_range`;`try_commit_blocks`(关 reuse 即返回);`release_index_slot` 对 `_early_freed_index_requests` 幂等;`context_block_keys`。`rewind_context_after_cache_drop` 在 `llm_request.py`。`py_executor_creator._create_py_executor_impl` 在 `start_worker()` 之前懒 import `attach_kv_transfer`。
- 状态枚举值 7 / 8 / 9 / 10 / 21。`AsyncTransferManager.start_transfer` 钉块 + 状态写法;`release_transfer` 在 `force_terminate` 下不调 `terminate_request`(情形 E 的依据);`SchedulerOutput` 八字段 + `fetch_launch_queue`。协调层 API 与 `_finish_if_released` 的终止调用顺序;`Planner.decide` 双预算(`probe_budget_rounds` / `probe_timeout_s`)。`test_llm_pytorch.py::test_llm_disagg_gen_cancelled` 在一个进程内创建 ctx 与 gen 两个 `LLM()`(`private_mpi_session`),并用 `get_stats` 的 `usedNumBlocks` 等待释放。`BlobStoreBackend.close` 以 `pool.shutdown(wait=True)` 等在飞交付,`blob/client.py`/`blob/backend.py` 中无任何超时参数。`SlotManager.remove_slot` 对未知 id 无操作。`_flush_pending_transfer_responses` 经 `_terminate_request` 终止 staged 请求。`tensorrt_llm/_torch/disaggregation/` 下无 README。本机:A6000 一块、`mooncake.store` 可导入、`~/.local/bin/mooncake_master` 存在、TinyLlama 权重在位。既有测试数见 §11。

**未核实**:`monkeypatch.setenv` 设的环境变量是否到达 MPI spawn 的 worker(S4 首项;探索 smoke 是在 shell 层设的);TCP 传输是否接受 GPU 登记缓冲(一律走 staging);探索分支 smoke 所用 YAML 未随 commit 提交;`sequence_to_blockchain_keys` 对多模态 digest token 的行为未在 GPU 上验证(U1 覆盖纯文本)。

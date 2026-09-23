<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KV 传输协调层设计:配套笔记

> 配套 `KV_TRANSFER_COORDINATOR_DESIGN.zh.md`(下称主文)。这里放主文读者不必第一遍就读的东西:与 vLLM / SGLang 的逐条对照、正文主张的源码位置、评审中否决的方案。

## 1. 与 vLLM / SGLang 的对照

调研了 vLLM V1 KV connector(含 Mooncake Store、LMCache、NIXL 后端)与 SGLang HiCache(host 池 + Mooncake L3)的源码和 issue(本地 checkout:`~/Documents/projects/{vllm,sglang}`)。它们踩过的坑归纳成 12 条约束,逐条对照主文:

| # | 教训(来源) | 主文 |
|---|---|---|
| 1 | lookup 命中但 get 失败反复发生(vLLM #55297、LMCache #2204、SGLang #24018):exists 与 get 不原子,store 在两者之间驱逐 | **部分避开**。`probe` 只是建议,`served` 是子集,少到 = 失败重试**一次**再退回本地计算,不会无限重算。真正的原子性要 store 给读者租约,是契约变化(§11 #9) |
| 2 | 数据到达前就把 `num_computed_tokens` 推进,导致整套 `invalid_block_ids` / `failed_recving` / 回退机制(vLLM #19329、#53298、#54870;`vllm/v1/core/sched/scheduler.py` 在调度时设 `num_computed_tokens = local + external`,请求在 `WAITING_FOR_REMOTE_KVS` 等) | **避开**。请求在 `KV_FETCH_IN_PROGRESS` 不可调度;游标只在 `unpark` 时推进,且只推进到共识后的 B |
| 3 | 失败不是一等结局、没有看门狗、请求卡在等待态(vLLM #57530、#50984、#46283) | **避开**。fetch 与 publish 记录都有 `deadline`;`Failed` 是契约的一等结局;一个请求只做一次重试决定 |
| 4 | 块的存活靠调度器可见的延迟释放集合,而不是 KV manager 里的引用计数;vLLM Mooncake connector 最终改成 `BlockPool.touch()` + `_pinned_saves`,SGLang 用 `protect_host` / `lock_ref` | **避开**。存活由活请求的 `_KVCache` 锁和应答侧 `pin_by_keys` 的 `_KVCache` 提供,都是 KV v2 自己的 hold/lock;`Page.hold()` 返回共享的 holder,两个重叠的应答共享存活 |
| 5 | LMCache 在 `wait_for_save` 里同步 D2H,占 forward 关键路径;Mooncake connector 用 CUDA event 让写线程等 | **后端内部事**。主文 §10.1 的中转在后端自己的 stream 上做;契约需要非阻塞静默查询才能把早放页兑现(§11 #4) |
| 6 | 写线程计数泄漏、异常后 ZMQ REP 卡死导致调度器 hang(vLLM #40900 评审、#57530) | 后端实现指南:一个线程池,`finally` 必释放;`pin_by_keys` 的任何异常路径 `finally: close()` |
| 7 | store 页大小与引擎块大小假设相等(SGLang `hicache_storage.py` TODO);LMCache 丢弃不满 chunk 的部分 | **避开**。unit 就是一个块,名字含 `tokens_per_block`;store 页大于块时由 store 后端聚合,契约不知道 |
| 8 | key 里缺并行布局导致 TP 错切(vLLM #40900 的 TP-rank bug);SGLang MLA 用 TP 无关的 key | **避开**。SPEC §4.2 不变式 3:凡两侧不一致即读错字节的量进名字;TP 切分不同表现为未命中而不是错数据。跨 TP 重切分只在 worker 后端的 mapper 里做 |
| 9 | 各 rank 独立的后台线程各自 all-reduce,导致死锁和树分叉(SGLang #22607,代码注释"This is so tricky") | **避开**。主文 §3.1 线程规则:后端线程不做集合通信、不碰 KV v2;决定都在引擎线程的一次集合通信里 |
| 10 | 部分成功无法表达(SGLang `batch_set` 返回 bool);准入前缀要取跨 rank 的 MIN | **避开**。`served` 按 unit 报告;B 取跨 rank 的 MIN |
| 11 | SGLang `write_through_selective` 按命中次数(≥2)门控 host 备份,L3 写跟在 host ack 之后;首次出现的前缀被驱逐前来不及进 L3(#39444) | **避开**。每个已提交块都发布;要不要按策略少发,是 `publish_committed_blocks` 的一个开关,不影响正确性 |
| 12 | 投机解码的 draft 状态不在缓存单元里,命中反而比冷 prefill 慢(SGLang #31600) | **未覆盖**。gen-init 经 aux 带首 token 与 draft token;跨请求取回时 draft KV 是否成为 unit,待定(§11 #10) |

另外两条 vLLM 的接口级教训直接体现在契约里:`(0, True)` 这类"命中为零但异步"的非法组合,契约用"units 可为空且不是错误"消掉;整段命中必须重算最后一个 token,主文用 `token_end ≤ ⌊(prompt_len − 1)/tpb⌋·tpb` 与 gen-init 的 aux 首 token 分别处理。

## 2. 主文主张的源码位置

路径相对 `tensorrt_llm/`;runtime 指 `runtime/kv_cache_manager_v2/`。定位用**函数名 + 相邻语句**,不用行号(行号随每次改动漂移);`grep -n "def <名字>"` 即可找到。

| 主张 | 位置 |
|---|---|
| history 不能回退;新块跳过 stale 范围 | runtime `_core/_kv_cache.py` `_KVCache.resize`:history 单调检查,以及为新块跳过 stale 范围的分支 |
| `commit()` 终点须等于 history(`commit_min_snapshot`) | `_KVCache.commit` 里的 `commit_min_snapshot` 检查;开关在 `_torch/pyexecutor/kv_cache/kv_cache_manager_v2.py` `KVCacheManagerV2._build_base_config`,Mamba 强制开于 `mamba_cache_manager.py` `MambaHybridCacheManagerV2._build_cache_config`;默认策略 `all_reusable` 见 `llmapi/llm_args.py` `BlockReuseConfig` |
| `commit()` 推进 history | `_KVCache.commit` 末尾 |
| 提交时放开 stale 页、rebasing | `_KVCache._commit_block` |
| SSM 快照挂到 tree block | `_KVCache._snapshot_ssm_to_tree_block`,由 `_commit_block` 调用 |
| `resume()` 的门与延迟拷贝,返回 False 不抛 | `_KVCache.resume`;`_KVCache.close` 释放 scratch 与 holder |
| serve 用的 `_KVCache` 会进统计 | `_KVCache.close`(更新 tuner 统计) |
| radix tree children 按 key 索引 | runtime `_block_radix_tree.py` `BlockRadixTree._match_token_path`;`Block.tokens`;`BlockRadixTree._prune_match` 的 SSM 截断;`_get_matched_tokens` 于 `_core/_kv_cache.py` |
| `hold()` 只防 drop;HELD 页在非最后一级可迁移 | runtime `_page.py` `Page.hold`;runtime `_storage_manager.py` `StorageManager.is_evictable` |
| `_settle_context_cursor` 与 C++ 断言 | `kv_cache_manager_v2.py` `_settle_context_cursor`;`cpp/include/tensorrt_llm/batch_manager/llmRequest.h` `setPrepopulatedPromptLen` 的断言 |
| `reserve_transfer_pages`(`prepare_disagg_gen_init` 是其 gen-init 别名)与 scratch 开关 | `kv_cache_manager_v2.py` `KVCacheManagerV2.prepare_context_cache`、`KVCacheManagerV2.reserve_transfer_pages`(`token_end` 给定时置 `enable_swa_scratch_reuse = False`);`_torch/disaggregation/transceiver.py` `KvCacheTransceiverV2._assert_disagg_history_declared` |
| `expect_snapshot_points` 每轮重建;`prompt_len` 缺省与 `point > position` 过滤 | `_torch/pyexecutor/py_executor.py` 两处 `kv_cache_manager.prepare_expect_snapshot_points(...)` 调用(`_executor_loop_pp` 与调度前的批准备);`mamba_cache_manager.py` `MambaHybridCacheManager.prepare_expect_snapshot_points`、`MambaHybridCacheManagerV2.prepare_expect_snapshot_points`、`MambaHybridCacheManagerV2.try_commit_blocks` |
| admission 被 bypass | `_torch/disaggregation/orchestration/coordinator.py` `transfer_window_bypass_eligible` 及其在 `py_executor.py` 的调用点 |
| 共识范围与 allgather | `transceiver.py` `KvCacheTransceiverV2._init_sync_policy`、`_ctx_consensus`、`_gen_consensus`、`check_gen_transfer_status` |
| 今天没有页在 NIXL 在飞时被释放 | `native/transfer.py` `TxSession.is_completed`、`RxSession.is_completed`;`transceiver.py` `KvCacheTransceiverV2._collect_done`、`_ownership_blocks_retirement`、`check_context_transfer_status`、`cancel_request` |
| demand 在监听线程处理、只碰预建的 session | `native/transfer.py` `Sender._start_listener`、`Sender._respond_with_kv`、`Sender._save_peer_req_info`;`native/messenger.py` `ZMQMessenger.listener` |
| gen 侧 session 完成即 close 释放 aux slot | `transceiver.py` `KvCacheTransceiverV2.check_gen_transfer_status` 里完成分支的 `close()`;`native/transfer.py` `RxSession.close` |
| PP 跟随者的调和;`_pp_retry_until_can_schedule` 只查 `scheduled_batch` | `py_executor.py` `PyExecutor._pp_schedule_and_propagate`、`_executor_loop_pp`、`_pp_retry_until_can_schedule`;`_torch/pyexecutor/scheduler/scheduler.py` `SerializableSchedulerOutput` |
| gen-init 落地后的批级准备;gen-init 不计入预算 | `py_executor.py` 处理 `DISAGG_GENERATION_TRANS_COMPLETE` 请求的批级准备;`_torch/pyexecutor/scheduler/scheduler_v2.py` `KVCacheV2Scheduler._schedule_loop` 的 gen-init 分支与 `_try_schedule_disagg_gen_init` |
| `AsyncTransferManager.start_transfer` 释放 seq slot / spec 资源 | `orchestration/transfer_manager.py` `start_transfer` |
| `_send_kv_async` 每轮调用;pipelined 的 `_build_prefill_extent` 输入与策略 | `py_executor.py` `_executor_loop` 与 `_executor_loop_overlap` 里的 `_send_kv_async` 调用;`orchestration/coordinator.py` `DisaggTransferCoordinator.send_completed_context`;`transceiver.py` `KvCacheTransceiverV2._build_prefill_extent` |
| 调度器排除的状态 | `_torch/pyexecutor/scheduler/scheduler.py` `RequestScheduler` 的可调度状态判定 |
| connector 的 per-layer hook | `py_executor.py` `PyExecutor._maybe_init_kv_connector_manager` |
| 空 `@runtime_checkable` Protocol 对任何对象判真 | Python 3.12 行为,评审中在本仓库环境验证 |

## 3. 评审中否决或修正的方案

| 方案 | 为什么不要 |
|---|---|
| 协调层在调度器之前做独立准入(块预算 + FCFS) | Python 路径今天已 bypass admission,证明不需要第二本账;而且会让取回优先于本地 prefill 抢空闲块 |
| 协调层持有 pin,`quiesce` 后释放 | v1 所有端点都由活请求的 `_KVCache` 持有,pin 是冗余;真正需要按名钉住的只有应答 demand 一侧 |
| 部分到达就地提交到 B | history 不能回退、窗口组在 `stale(token_end)` 没有页,`commit()` 断言只是次要理由 |
| 辅助线程跑 `quiesce` / `probe` | 记录表被两个线程写;纯 Python KV v2 非线程安全。改为释放点同步 `quiesce`(结局已知时立即返回)、后端自己异步 `probe` |
| `RunsOnEngineThread.step()` 让后端在引擎线程调 `_KVCache` | 后端越过 `resource/` 直接碰 KV v2,违反边界。改为 `EngineQueue.post(fn)` + `resource/pin_by_keys` |
| `PublishesIncrementally` 空标记协议 | 空的 `@runtime_checkable` Protocol 对任何对象 `isinstance` 都是 True。并入 `PlacesPieces` |
| `StreamsLayers` 列在 v1 可选协议里 | 第一版不实现,列在那里让 v1 表面失实;挪到扩展点,身份靠 `attempts` 传 |
| 共识"答案不一致取 None" | store probe 异步,rank 先后不一,会把请求永久降成本地计算。改为任一 DEFER 则 DEFER |
| 计划答案随每 rank 本地命中变化 | 各 rank radix tree 不同,会导致调度分叉。本地命中只裁 unit |
| 只有 fetch 记录有 deadline | 今天 `check_transfer_timeouts` 给 ctx 发送超时,去掉是倒退 |
| `attempts` 列表配一个 `outcome` | 分块发送与重试下没定义。改为 `AttemptRecord` 各带 outcome |
| 拆成主文 / 笔记两份时把 §11 待决问题搬走 | 正文 12 处「待定」引用它;搬走只会变成跨文档链接 |

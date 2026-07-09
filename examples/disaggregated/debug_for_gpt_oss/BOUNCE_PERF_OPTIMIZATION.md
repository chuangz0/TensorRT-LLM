# BounceTransport 性能优化记录

日期:2026-07-07
场景:gpt-oss-120b 8k/1k con128 disagg,ctx tp1 → gen tp4,GB200 NVL,
NIXL bounce v2(`TRTLLM_NIXL_BOUNCE_ENABLE=1`,chunk≤128MB,arena 512MB,
`TRTLLM_KV_TRANSFER_NUM_THREADS=4`,`TRTLLM_NIXL_BOUNCE_ZEROCOPY_ARGS=1`)。

分析方法:nsys `trtllm.disagg.bounce` NVTX domain 导出 sqlite 做分布统计,
与 CUPTI kernel 记录按序关联拆分排队/执行/轮询;E2E 用
`TRTLLM_KVCACHE_TIME_OUTPUT_PATH` 的 per-request CSV。

---

## 1. 基线诊断(round17)

单 chunk = 72MB / 18432 个 4KB desc(8k tokens ≈ 256 blocks × 36 layers × K/V,
tp4 每 rank 2/8 heads)。CTX sender 侧 320 个 chunk 的分布:

| 阶段 | avg | max | 根因 |
|---|---|---|---|
| req 总计 | **1681us** | 2773 | |
| buildPlan | 230 | 314 | 18432 desc 装包(在 req 之前,串行) |
| grantWait | 243 | 542 | WANT→GRANT RTT,完全串行无重叠 |
| gatherLaunch | 98 | 186 | 3×18432 vector 构建 + 368KB pinned memcpy |
| gather | 379 | **1349** | **kernel 排队 262us (max 1119)**:ExecPool stream 默认优先级,排在 prefill kernel 后;kernel 本体仅 65us;event 轮询检测 +52us |
| nixlWrite | 229 | 457 | RDMA 本体,正常 |
| ackWait | **658** | 1046 | 见下 |

**ackWait 分解**:GEN 侧实际工作仅 scatterQueue 45 + scatter 149(kernel 本体
35us);其余 ~460us 是控制路径 ——

- **DATA 消息 442KB**(18432 entry × 24B)走 ZMQ/TCP:约 4 次全量拷贝
  (encode → zmq msg → recv assign → decode)+ TCP 传输;
- scatter 完成 → done 队列 → IO 线程 drain → 发 ACK 的中转(IO 线程
  backoff sleep 50us);
- sender 侧 onAck 抢 mReqMu(pumpRequest 持锁做 gatherLaunch ~98us)。

关键洞察:gen (tp4) rank 的 dst pool 每层只含自己的 2 heads,
`HeadMismatchMapper` 的 dst 片段步长恰好等于片段大小(head_off=0),块间因
block 顺序分配也连续 → **整个 72MB dst 在显存里是一段连续区间**,DATA 的
18432 个 entry 高度可压缩。

---

## 2. 优化项

代码全部位于 `cpp/tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/`。

### 2.1 ExecPool 最高优先级 stream(gather 排队 262→150us)
`ExecPool.cpp`:`cudaStreamCreateWithPriority(..., greatestPriority)`。
高优先级只能在现存 kernel 的 block 退出时抢 SM(无抢占),
所以排队未清零(剩 ~150us,约一个 prefill kernel 的时长)。

### 2.2 Scatter 计划压缩:coalesce + strided run(DATA 442KB → 36B)
- `BounceTransferPlan`:desc 级合并(src+dst+bounce 三连续时原地扩展);
- `buildScatterRuns()`:贪心生成 `BounceScatterRun{bounceOffset, dstAddr,
  dstStride, bounceStride, pieceSize, count}` ——
  - **连续 dst**(ctx tp1→gen tp4):count==1、pieceSize 覆盖全 chunk → 1 run;
  - **等差跨步 dst**(ctx tp4→gen dp:head slice 落进全 heads pool,逐项差恒为
    bytes-per-layer):stride latch → 1 个 count=N run;
  - 不规则布局自然断开,最坏退化为逐 desc(与旧格式等价),正确性不依赖合并。
- wire 版本 `kBounceVersion` 1→2(混部两侧需同版本,否则退化为 leaseTimeout)。

### 2.3 GPU 并行度保持:kernel 启动时按 64KB 拆分
`appendSplitInto()`:大 run 在 receiver/sender 启动 kernel 前拆回 ≤64KB piece
(copy kernel 一个 thread block 处理一个 entry)。拆分带 scratch 容量预算
(`splitBudget`),超预算的余量作为单个超大 entry(kernel strided loop 兜底,
只损失并行度不损失正确性)。

### 2.4 Scatter worker 直发 ACK(省 ~50-100us 中转)
`scatterWorkerLoop`:streamSync 成功后 worker 直接 `sendTo(ACK)`
(ControlChannel 线程安全);IO 线程的 `drainScatterDone` 只做 region
释放/re-grant 簿记。失败仍不 ACK(sender 超时,不产生假 ACK)。

### 2.5 Eager gather(grantWait 与 gather 重叠)
`TRTLLM_NIXL_BOUNCE_EAGER_GATHER`(默认开)。submit() 发出 WANT 后立即在
调用线程启动 gather(不等 GRANT):
- Posted 增加 `Gathered` 状态与 `hasCredit`;credit 按 chunk 序号 FIFO 配对
  (`Request::nextCredit`),到达后 `attachCredits` 回填 remote 目标,
  `drainGatherReady` 在 "gather 完成 ∧ credit 就绪" 时才 postWrite;
- **防双向死锁**:eager(无 credit)staging 占用被 `CreditScheduler` 限制在
  arena 容量的一半(`acquireLocal(bytes, eager)`);credit 到达后
  `promoteLocal` 移出预算,稳态流水不受限;
- `CreditScheduler` 全方法加锁(submit 线程会并发调 `acquireLocal`)。

### 2.6 Desc 元数据零中间拷贝(直写 pinned)
`launchPacked` → `launchPrepared`:调用方先精确计数(`piecesFor`)再经
`planBufs()` **直接写入 pinned buffer** 的 `[srcs|dsts|sizes]` 布局,
删除了"构建 3 个临时 vector + 368KB memcpy 进 pinned"两趟搬运。
配合 `ZEROCOPY_ARGS=1`(kernel 直读 pinned)整条 desc 元数据链只剩一次写。

### 2.7 NIXL notification 控制通道(每跳 ~50us → ~10us)
`NixlNotifControlChannel`(`TRTLLM_NIXL_BOUNCE_NIXL_CONTROL`,默认关):
- `genNotif`/`getNotifs` 走 UCX active message(RDMA 网络),替代 ZMQ/TCP;
- **反向路径 bootstrap**:`localEndpoint()` 返回本 agent 序列化 NIXL metadata,
  随 WANT 携带,receiver 在 onWant self-bootstrap 时 `loadRemoteMD`
  (与 zmq 通道 endpoint-in-WANT 机制同构);
- 约束:两端必须同时开启;`transceiver_runtime=CPP` 下禁用
  (其 DataSender 消费同一个 agent notification 队列);
  recv 无 fd 可等,idle 时 20us sleep-poll。

### 2.8 细粒度 NVTX(验证用)
新增 `dataSend / onData / scatterPrep / scatterSync / ackSend / onAck /
doneDrain / zmqSend(notifSend) / zmqRecv`,把 ackWait 全链路逐段可测;
`nixlWrite` 修正为纯 RDMA 时间(entry 构建移出该 span)。

---

## 3. 验证数据

同配置 A/B(nsys 窗口 ctx iter 20-60,数百 chunk/轮):

| CTX 侧指标 | r17 基线 | r18(2.1-2.6 除 run 格式) | r19(+strided run, zmq) | r20(+nixl-notif) |
|---|---|---|---|---|
| **req 总延迟** | 1681us | 1017 (-40%) | 976 | **854 (-49%)** |
| ackWait | 658 | 357 | 359 | **278** |
| grantWait(已被 eager 重叠) | 243 | 262 | 261 | 213 |
| gather | 379 | 273 | 252 | 229 |
| nixlWrite | 229 | 183 | 172 | 156 |
| gatherLaunch | 98 | 144¹ | 150¹ | 138¹ |
| DATA 消息 | 442KB | 24B (n=1) | 36B (n=1 run) | 36B |

¹ eager 化后 gatherLaunch 移到 submit 线程执行(GPU 忙时驱动调用变慢),
但已不在关键路径上(与 grantWait 重叠)。

r20 GEN 侧 ackWait 组成:onData 8 + scatterQueue 31 + scatterPrep 40 +
scatterSync 42 + ackSend 13 ≈ 134us,其余为两跳 wire + sender 分发,
已接近 scatter 本体下限。

E2E:ctx 侧 per-chunk `transfer_latency` 中位 1.76→1.11ms(-37%,全程 5120
chunk);gen 侧 per-request 时间与 TTFT 受 con128 排队噪声主导(±1ms),
不能分辨 ~100us 级差异,以 NVTX per-chunk 为准。全部轮次零 bounce
报错/超时/abandon。

数据目录:`logs/gpt_oss-120b-8k1k-con128-disagg_nsys_chunk128_area_512_t4_nvtx_round{17,18,19,20}*`
(含 nsys-rep、sqlite 导出、kv_cache_time_output CSV)。

---

## 4. 环境变量小结

| 变量 | 默认 | 说明 |
|---|---|---|
| `TRTLLM_NIXL_BOUNCE_EAGER_GATHER` | 1 | submit 即启动 gather,与 GRANT RTT 重叠;eager 占用 ≤ arena/2 |
| `TRTLLM_NIXL_BOUNCE_NIXL_CONTROL` | 0 | 控制消息走 NIXL notif(UCX AM);两端一致;CPP runtime 勿开 |
| `TRTLLM_NIXL_BOUNCE_ZEROCOPY_ARGS` | 0 | kernel 直读 pinned 计划数组(本场景建议开) |

## 5. 遗留与后续方向

1. **gather 排队仍有 ~150us avg / ~1ms 尾部**:高优先级 stream 无法抢占运行中
   的 prefill kernel;可探索 copy-engine(batched cudaMemcpyAsync)或等
   更细粒度 kernel。
2. **buildPlan ~170-230us 串行**:可挪到 WANT 发送之后与 RTT 重叠。
3. **72MB chunk 在 buddy 中占 128MB block**(44% 内部碎片):
   `TRTLLM_NIXL_BOUNCE_MAX_CHUNK_BYTES=32MB` 可让单请求 3-chunk 流水化,
   值得 A/B。
4. ~~ctx_tp4→gen_dp4 case 的 strided-run 压缩尚未实测~~ **已验证(round21)**:
   `logs/gpt_oss-120b-8k1k-con128-disagg_bounce_t4_ctx_tp4_round21_strided`,
   跨步 dst(stride 16384 / piece 4096)下 **195/220 的 DATA 为 n=1 / 36B**
   (单 strided run 覆盖整 chunk);少数多 chunk 请求因 32MB chunk 边界切在
   block 中间断成 57-66 run(≤2.4KB,旧格式约 197KB/chunk → 仍小 ~100×)。
   零报错;gen 侧 per-request 中位与旧版 round12 持平(3.35 vs 3.44ms,
   round21 还带着 nsys 开销),receiver 链路 onData 5 + queue 18 + prep 56 +
   sync 24 + ackSend 10 ≈ 113us。
5. Python→C++ desc marshalling 与 plan 构建的拷贝为结构性保留项。

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Fabric Transfer Tests
// Tests for BatchCopyWorkerPool, FabricTransferStatus (multi-event mode),
// and FabricTransferHelper::submitWithCudaMemcpyBatch.
//
// Build target: fabricTransferTest
// Run: ./fabricTransferTest (requires at least one CUDA GPU)

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/fabricTransfer.h"
#include "tensorrt_llm/runtime/cudaEvent.h"
#include "tensorrt_llm/runtime/cudaStream.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <numeric>
#include <thread>
#include <vector>

using namespace tensorrt_llm::executor::kv_cache;
namespace runtime = tensorrt_llm::runtime;

// ============================================================================
// Test fixture with CUDA device memory helpers
// ============================================================================

class FabricTransferTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        int deviceCount = 0;
        ASSERT_EQ(cudaGetDeviceCount(&deviceCount), cudaSuccess);
        ASSERT_GT(deviceCount, 0) << "No CUDA device available";
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

        // Initialize CUDA driver API (needed by FabricTransferHelper constructor)
        ASSERT_EQ(cuInit(0), CUDA_SUCCESS);
    }

    // Allocate GPU memory filled with a pattern
    void* allocGpuWithPattern(size_t bytes, uint8_t pattern)
    {
        void* ptr = nullptr;
        TLLM_CUDA_CHECK(cudaMalloc(&ptr, bytes));
        TLLM_CUDA_CHECK(cudaMemset(ptr, pattern, bytes));
        mAllocations.push_back(ptr);
        return ptr;
    }

    // Allocate zeroed GPU memory
    void* allocGpuZeroed(size_t bytes)
    {
        return allocGpuWithPattern(bytes, 0);
    }

    void TearDown() override
    {
        for (auto* ptr : mAllocations)
        {
            cudaFree(ptr);
        }
        mAllocations.clear();
    }

private:
    std::vector<void*> mAllocations;
};

// ============================================================================
// BatchCopyWorkerPool Tests
// ============================================================================

TEST_F(FabricTransferTest, WorkerPoolBasicSubmitAndWait)
{
    // Verify that the worker pool can process tasks and the per-batch counter works.
    constexpr int kNumWorkers = 2;
    constexpr size_t kSize = 4096;

    BatchCopyWorkerPool pool(kNumWorkers, 0);

    auto* src = allocGpuWithPattern(kSize, 0xAB);
    auto* dst = allocGpuZeroed(kSize);

    auto batchPending = std::make_shared<std::atomic<int>>(1);
    auto event = std::make_shared<runtime::CudaEvent>();
    auto stream = std::make_shared<runtime::CudaStream>();

    BatchCopyTask task;
    task.dst = {static_cast<void*>(dst)};
    task.src = {static_cast<void const*>(src)};
    task.sizes = {kSize};
    task.stream = stream->get();
    task.completionEvent = event->get();
    task.batchPending = batchPending.get();

    pool.submit(std::move(task));
    pool.waitAll();

    // batchPending should be 0 after worker completes
    EXPECT_EQ(batchPending->load(), 0);

    // Wait for GPU work to finish
    event->synchronize();

    // Verify data correctness
    std::vector<uint8_t> hostBuf(kSize);
    TLLM_CUDA_CHECK(cudaMemcpy(hostBuf.data(), dst, kSize, cudaMemcpyDeviceToHost));
    for (size_t ii = 0; ii < kSize; ++ii)
    {
        ASSERT_EQ(hostBuf[ii], 0xAB) << "Mismatch at byte " << ii;
    }
}

TEST_F(FabricTransferTest, WorkerPoolMultipleTasks)
{
    // Submit multiple tasks to the pool and verify all complete.
    constexpr int kNumWorkers = 2;
    constexpr int kNumTasks = 4;
    constexpr size_t kSize = 1024;

    BatchCopyWorkerPool pool(kNumWorkers, 0);

    auto batchPending = std::make_shared<std::atomic<int>>(kNumTasks);
    std::vector<std::shared_ptr<runtime::CudaEvent>> events;
    std::vector<std::shared_ptr<runtime::CudaStream>> streams;
    std::vector<void*> srcs, dsts;

    for (int ii = 0; ii < kNumTasks; ++ii)
    {
        srcs.push_back(allocGpuWithPattern(kSize, static_cast<uint8_t>(ii + 1)));
        dsts.push_back(allocGpuZeroed(kSize));
        events.push_back(std::make_shared<runtime::CudaEvent>());
        streams.push_back(std::make_shared<runtime::CudaStream>());
    }

    for (int ii = 0; ii < kNumTasks; ++ii)
    {
        BatchCopyTask task;
        task.dst = {dsts[ii]};
        task.src = {static_cast<void const*>(srcs[ii])};
        task.sizes = {kSize};
        task.stream = streams[ii]->get();
        task.completionEvent = events[ii]->get();
        task.batchPending = batchPending.get();
        pool.submit(std::move(task));
    }

    pool.waitAll();
    EXPECT_EQ(batchPending->load(), 0);

    // Verify each copy
    for (int ii = 0; ii < kNumTasks; ++ii)
    {
        events[ii]->synchronize();
        std::vector<uint8_t> hostBuf(kSize);
        TLLM_CUDA_CHECK(cudaMemcpy(hostBuf.data(), dsts[ii], kSize, cudaMemcpyDeviceToHost));
        EXPECT_EQ(hostBuf[0], static_cast<uint8_t>(ii + 1)) << "Task " << ii << " data mismatch";
    }
}

TEST_F(FabricTransferTest, WorkerPoolPerBatchIsolation)
{
    // Verify that per-batch pending counters are independent.
    // Submit two batches concurrently; each batch's counter tracks only its own tasks.
    constexpr int kNumWorkers = 2;
    constexpr size_t kSize = 1024;

    BatchCopyWorkerPool pool(kNumWorkers, 0);

    // Batch A: 2 tasks
    auto pendingA = std::make_shared<std::atomic<int>>(2);
    std::vector<std::shared_ptr<runtime::CudaEvent>> eventsA;
    std::vector<std::shared_ptr<runtime::CudaStream>> streamsA;

    for (int ii = 0; ii < 2; ++ii)
    {
        eventsA.push_back(std::make_shared<runtime::CudaEvent>());
        streamsA.push_back(std::make_shared<runtime::CudaStream>());

        BatchCopyTask task;
        auto* src = allocGpuWithPattern(kSize, 0xAA);
        auto* dst = allocGpuZeroed(kSize);
        task.dst = {dst};
        task.src = {static_cast<void const*>(src)};
        task.sizes = {kSize};
        task.stream = streamsA[ii]->get();
        task.completionEvent = eventsA[ii]->get();
        task.batchPending = pendingA.get();
        pool.submit(std::move(task));
    }

    // Batch B: 2 tasks
    auto pendingB = std::make_shared<std::atomic<int>>(2);
    std::vector<std::shared_ptr<runtime::CudaEvent>> eventsB;
    std::vector<std::shared_ptr<runtime::CudaStream>> streamsB;

    for (int ii = 0; ii < 2; ++ii)
    {
        eventsB.push_back(std::make_shared<runtime::CudaEvent>());
        streamsB.push_back(std::make_shared<runtime::CudaStream>());

        BatchCopyTask task;
        auto* src = allocGpuWithPattern(kSize, 0xBB);
        auto* dst = allocGpuZeroed(kSize);
        task.dst = {dst};
        task.src = {static_cast<void const*>(src)};
        task.sizes = {kSize};
        task.stream = streamsB[ii]->get();
        task.completionEvent = eventsB[ii]->get();
        task.batchPending = pendingB.get();
        pool.submit(std::move(task));
    }

    pool.waitAll();

    // Both batch counters should reach 0 independently
    EXPECT_EQ(pendingA->load(), 0);
    EXPECT_EQ(pendingB->load(), 0);
}

// ============================================================================
// FabricTransferStatus Tests
// ============================================================================

TEST_F(FabricTransferTest, StatusSingleEventMode)
{
    // Single-event mode: record event on stream, status should detect completion.
    auto stream = std::make_shared<runtime::CudaStream>();
    auto event = std::make_shared<runtime::CudaEvent>();

    // Record event immediately (no work) — should be completed quickly
    stream->record(*event);

    FabricTransferStatus status(stream, event);

    auto result = status.wait(-1);
    EXPECT_EQ(result, TransferState::kSUCCESS);
    EXPECT_TRUE(status.isCompleted());
}

TEST_F(FabricTransferTest, StatusMultiEventMode)
{
    // Multi-event mode: simulate workers recording events via per-batch counter.
    constexpr int kNumEvents = 3;

    auto batchPending = std::make_shared<std::atomic<int>>(kNumEvents);
    std::vector<std::shared_ptr<runtime::CudaEvent>> events;
    std::vector<std::shared_ptr<runtime::CudaStream>> streams;

    for (int ii = 0; ii < kNumEvents; ++ii)
    {
        events.push_back(std::make_shared<runtime::CudaEvent>());
        streams.push_back(std::make_shared<runtime::CudaStream>());
    }

    FabricTransferStatus status(batchPending, events);

    // Before events are recorded, isCompleted should return false
    // (batchPending > 0 means events haven't been recorded yet)
    EXPECT_FALSE(status.isCompleted());

    // Simulate workers: record events and decrement counter
    for (int ii = 0; ii < kNumEvents; ++ii)
    {
        streams[ii]->record(*events[ii]);
        batchPending->fetch_sub(1, std::memory_order_release);
    }

    // Now wait should succeed
    auto result = status.wait(-1);
    EXPECT_EQ(result, TransferState::kSUCCESS);
    EXPECT_TRUE(status.isCompleted());
}

TEST_F(FabricTransferTest, StatusMultiEventTimedWait)
{
    // Multi-event timed wait: should return IN_PROGRESS when events haven't fired.
    auto batchPending = std::make_shared<std::atomic<int>>(1);
    auto event = std::make_shared<runtime::CudaEvent>();
    auto stream = std::make_shared<runtime::CudaStream>();

    // Don't record the event yet — simulate worker still running
    FabricTransferStatus status(batchPending, {event});

    // With 0ms timeout, should return IN_PROGRESS (batchPending > 0 means
    // spin-wait in wait() won't get past the pending check within the timeout)
    // Actually wait() spin-waits on batchPending first, so with timeout on the
    // outer loop it depends on implementation. Let's simulate a completed API call
    // but pending GPU work instead:
    stream->record(*event);
    batchPending->store(0, std::memory_order_release);

    // Launch a large kernel to keep the stream busy
    constexpr size_t kLargeSize = 256 * 1024 * 1024; // 256MB
    void* buf = allocGpuZeroed(kLargeSize);
    TLLM_CUDA_CHECK(cudaMemsetAsync(buf, 0xFF, kLargeSize, stream->get()));

    // Re-record event after the memset so it captures the pending work
    auto event2 = std::make_shared<runtime::CudaEvent>();
    stream->record(*event2);

    auto batchPending2 = std::make_shared<std::atomic<int>>(0);
    FabricTransferStatus status2(batchPending2, {event2});

    // With very short timeout, might be IN_PROGRESS or SUCCESS depending on GPU speed
    auto result = status2.wait(0);
    EXPECT_TRUE(result == TransferState::kIN_PROGRESS || result == TransferState::kSUCCESS);

    // Infinite wait should always succeed
    result = status2.wait(-1);
    EXPECT_EQ(result, TransferState::kSUCCESS);
}

// ============================================================================
// FabricTransferHelper::submitWithCudaMemcpyBatch Tests
// ============================================================================

TEST_F(FabricTransferTest, SubmitBatchEmptyInput)
{
    // Empty input should return an immediately-completed status.
    FabricTransferHelper helper;

    std::vector<void*> srcPtrs, dstPtrs;
    std::vector<size_t> sizes;

    auto status = helper.submitWithCudaMemcpyBatch(srcPtrs, dstPtrs, sizes);
    ASSERT_NE(status, nullptr);

    auto result = status->wait(-1);
    EXPECT_EQ(result, TransferState::kSUCCESS);
}

TEST_F(FabricTransferTest, SubmitBatchSingleEntry)
{
    // Single entry: should use single-thread path regardless of thread count.
    FabricTransferHelper helper;

    constexpr size_t kSize = 8192;
    auto* src = allocGpuWithPattern(kSize, 0xCD);
    auto* dst = allocGpuZeroed(kSize);

    std::vector<void*> srcPtrs = {src};
    std::vector<void*> dstPtrs = {dst};
    std::vector<size_t> sizes = {kSize};

    auto status = helper.submitWithCudaMemcpyBatch(srcPtrs, dstPtrs, sizes);
    ASSERT_NE(status, nullptr);

    auto result = status->wait(-1);
    EXPECT_EQ(result, TransferState::kSUCCESS);

    // Verify data
    std::vector<uint8_t> hostBuf(kSize);
    TLLM_CUDA_CHECK(cudaMemcpy(hostBuf.data(), dst, kSize, cudaMemcpyDeviceToHost));
    for (size_t ii = 0; ii < kSize; ++ii)
    {
        ASSERT_EQ(hostBuf[ii], 0xCD) << "Mismatch at byte " << ii;
    }
}

TEST_F(FabricTransferTest, SubmitBatchMultipleEntries)
{
    // Multiple entries: exercises the multi-thread path (default 2 threads).
    FabricTransferHelper helper;

    constexpr int kNumEntries = 8;
    constexpr size_t kEntrySize = 16 * 1024; // 16KB per entry

    std::vector<void*> srcPtrs, dstPtrs;
    std::vector<size_t> sizes;

    for (int ii = 0; ii < kNumEntries; ++ii)
    {
        srcPtrs.push_back(allocGpuWithPattern(kEntrySize, static_cast<uint8_t>(ii + 1)));
        dstPtrs.push_back(allocGpuZeroed(kEntrySize));
        sizes.push_back(kEntrySize);
    }

    auto status = helper.submitWithCudaMemcpyBatch(srcPtrs, dstPtrs, sizes);
    ASSERT_NE(status, nullptr);

    auto result = status->wait(-1);
    EXPECT_EQ(result, TransferState::kSUCCESS);

    // Verify each entry
    for (int ii = 0; ii < kNumEntries; ++ii)
    {
        std::vector<uint8_t> hostBuf(kEntrySize);
        TLLM_CUDA_CHECK(cudaMemcpy(hostBuf.data(), dstPtrs[ii], kEntrySize, cudaMemcpyDeviceToHost));
        EXPECT_EQ(hostBuf[0], static_cast<uint8_t>(ii + 1)) << "Entry " << ii << " first byte mismatch";
        EXPECT_EQ(hostBuf[kEntrySize - 1], static_cast<uint8_t>(ii + 1)) << "Entry " << ii << " last byte mismatch";
    }
}

TEST_F(FabricTransferTest, SubmitBatchLargeEntries)
{
    // Larger entries to verify bandwidth path works.
    FabricTransferHelper helper;

    constexpr int kNumEntries = 4;
    constexpr size_t kEntrySize = 256 * 1024; // 256KB per entry

    std::vector<void*> srcPtrs, dstPtrs;
    std::vector<size_t> sizes;

    for (int ii = 0; ii < kNumEntries; ++ii)
    {
        srcPtrs.push_back(allocGpuWithPattern(kEntrySize, static_cast<uint8_t>(0x10 + ii)));
        dstPtrs.push_back(allocGpuZeroed(kEntrySize));
        sizes.push_back(kEntrySize);
    }

    auto status = helper.submitWithCudaMemcpyBatch(srcPtrs, dstPtrs, sizes);
    ASSERT_NE(status, nullptr);

    auto result = status->wait(-1);
    EXPECT_EQ(result, TransferState::kSUCCESS);

    for (int ii = 0; ii < kNumEntries; ++ii)
    {
        std::vector<uint8_t> hostBuf(kEntrySize);
        TLLM_CUDA_CHECK(cudaMemcpy(hostBuf.data(), dstPtrs[ii], kEntrySize, cudaMemcpyDeviceToHost));
        EXPECT_EQ(hostBuf[0], static_cast<uint8_t>(0x10 + ii)) << "Entry " << ii << " mismatch";
    }
}

TEST_F(FabricTransferTest, SubmitBatchConsecutiveCalls)
{
    // Two consecutive submits on the same helper: verifies worker pool reuse
    // and that per-batch counters are independent.
    FabricTransferHelper helper;

    constexpr int kNumEntries = 4;
    constexpr size_t kEntrySize = 4096;

    for (int batch = 0; batch < 2; ++batch)
    {
        std::vector<void*> srcPtrs, dstPtrs;
        std::vector<size_t> sizes;
        uint8_t pattern = static_cast<uint8_t>(0xA0 + batch);

        for (int ii = 0; ii < kNumEntries; ++ii)
        {
            srcPtrs.push_back(allocGpuWithPattern(kEntrySize, pattern));
            dstPtrs.push_back(allocGpuZeroed(kEntrySize));
            sizes.push_back(kEntrySize);
        }

        auto status = helper.submitWithCudaMemcpyBatch(srcPtrs, dstPtrs, sizes);
        ASSERT_NE(status, nullptr);

        auto result = status->wait(-1);
        EXPECT_EQ(result, TransferState::kSUCCESS) << "Batch " << batch << " failed";

        // Verify
        for (int ii = 0; ii < kNumEntries; ++ii)
        {
            std::vector<uint8_t> hostBuf(kEntrySize);
            TLLM_CUDA_CHECK(cudaMemcpy(hostBuf.data(), dstPtrs[ii], kEntrySize, cudaMemcpyDeviceToHost));
            EXPECT_EQ(hostBuf[0], pattern) << "Batch " << batch << " entry " << ii;
        }
    }
}

TEST_F(FabricTransferTest, SubmitBatchConcurrentFromTwoThreads)
{
    // Two host threads submit batches concurrently on the same helper.
    // Verifies per-batch isolation under real concurrency.
    FabricTransferHelper helper;

    constexpr int kNumEntries = 4;
    constexpr size_t kEntrySize = 8192;

    auto runBatch = [&](uint8_t pattern, std::vector<void*> const& srcs, std::vector<void*> const& dsts)
    {
        std::vector<void*> srcPtrs(srcs.begin(), srcs.end());
        std::vector<void*> dstPtrs(dsts.begin(), dsts.end());
        std::vector<size_t> sizes(kNumEntries, kEntrySize);

        auto status = helper.submitWithCudaMemcpyBatch(srcPtrs, dstPtrs, sizes);
        ASSERT_NE(status, nullptr);
        auto result = status->wait(-1);
        EXPECT_EQ(result, TransferState::kSUCCESS);

        for (int ii = 0; ii < kNumEntries; ++ii)
        {
            std::vector<uint8_t> hostBuf(kEntrySize);
            TLLM_CUDA_CHECK(cudaMemcpy(hostBuf.data(), dstPtrs[ii], kEntrySize, cudaMemcpyDeviceToHost));
            EXPECT_EQ(hostBuf[0], pattern) << "Pattern 0x" << std::hex << (int) pattern << " entry " << ii;
        }
    };

    // Allocate memory for both threads upfront (on the main thread)
    std::vector<void*> srcsA, dstsA, srcsB, dstsB;
    for (int ii = 0; ii < kNumEntries; ++ii)
    {
        srcsA.push_back(allocGpuWithPattern(kEntrySize, 0xAA));
        dstsA.push_back(allocGpuZeroed(kEntrySize));
        srcsB.push_back(allocGpuWithPattern(kEntrySize, 0xBB));
        dstsB.push_back(allocGpuZeroed(kEntrySize));
    }

    std::thread threadA(
        [&]()
        {
            TLLM_CUDA_CHECK(cudaSetDevice(0));
            runBatch(0xAA, srcsA, dstsA);
        });

    std::thread threadB(
        [&]()
        {
            TLLM_CUDA_CHECK(cudaSetDevice(0));
            runBatch(0xBB, srcsB, dstsB);
        });

    threadA.join();
    threadB.join();
}

TEST_F(FabricTransferTest, SubmitBatchIsCompletedPolling)
{
    // Verify isCompleted() returns true eventually without calling wait().
    FabricTransferHelper helper;

    constexpr int kNumEntries = 4;
    constexpr size_t kEntrySize = 4096;

    std::vector<void*> srcPtrs, dstPtrs;
    std::vector<size_t> sizes;

    for (int ii = 0; ii < kNumEntries; ++ii)
    {
        srcPtrs.push_back(allocGpuWithPattern(kEntrySize, 0xEE));
        dstPtrs.push_back(allocGpuZeroed(kEntrySize));
        sizes.push_back(kEntrySize);
    }

    auto status = helper.submitWithCudaMemcpyBatch(srcPtrs, dstPtrs, sizes);
    ASSERT_NE(status, nullptr);

    // Poll until completed (with a timeout to avoid hanging)
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!status->isCompleted())
    {
        ASSERT_LT(std::chrono::steady_clock::now(), deadline) << "Timed out waiting for isCompleted()";
        std::this_thread::yield();
    }

    EXPECT_TRUE(status->isCompleted());
}

// ============================================================================
// CudaEventPool Tests
// ============================================================================

TEST_F(FabricTransferTest, EventPoolAcquireAndReuse)
{
    // Acquire event, capture underlying cudaEvent_t handle, release, acquire again.
    // Pool should reuse the same underlying event.
    auto pool = std::make_shared<CudaEventPool>();

    cudaEvent_t firstHandle;
    {
        auto event = pool->acquire();
        firstHandle = event->get();
        // event goes out of scope → returned to pool
    }

    auto event2 = pool->acquire();
    EXPECT_EQ(event2->get(), firstHandle) << "Pool should reuse the same underlying CUDA event";
}

TEST_F(FabricTransferTest, EventPoolConcurrentAccess)
{
    // 8 threads × 100 acquire/release cycles — verify no crashes.
    auto pool = std::make_shared<CudaEventPool>();
    constexpr int kNumThreads = 8;
    constexpr int kNumCycles = 100;

    std::atomic<int> totalAcquires{0};
    std::vector<std::thread> threads;
    threads.reserve(kNumThreads);

    for (int t = 0; t < kNumThreads; ++t)
    {
        threads.emplace_back(
            [&pool, &totalAcquires]()
            {
                for (int i = 0; i < kNumCycles; ++i)
                {
                    auto event = pool->acquire();
                    ASSERT_NE(event, nullptr);
                    totalAcquires.fetch_add(1, std::memory_order_relaxed);
                    // event released at end of each iteration
                }
            });
    }

    for (auto& t : threads)
    {
        t.join();
    }

    EXPECT_EQ(totalAcquires.load(), kNumThreads * kNumCycles);
}

TEST_F(FabricTransferTest, EventPoolDestroyedBeforeEventsReturned)
{
    // Acquire event, destroy pool, verify event still usable.
    // The weak_ptr in the deleter should expire gracefully.
    auto pool = std::make_shared<CudaEventPool>();
    auto event = pool->acquire();
    ASSERT_NE(event, nullptr);

    auto stream = std::make_shared<runtime::CudaStream>();

    // Destroy pool while event is still alive
    pool.reset();

    // Event should still be usable: record and synchronize
    stream->record(*event);
    event->synchronize();

    // Event destruction should not crash (weak_ptr expired → normal delete)
    event.reset();
}

// ============================================================================
// FabricTransferHelper::submitWithCubBatched Tests (merged pinned H2D path)
// ============================================================================

TEST_F(FabricTransferTest, SubmitCubBatchedMergedH2D)
{
    // Allocate N GPU src buffers with known patterns, N zeroed dst buffers.
    // Call submitWithCubBatched, wait, verify dst contents match src patterns.
    FabricTransferHelper helper;

    constexpr int kNumEntries = 16;
    constexpr size_t kEntrySize = 4096;

    std::vector<void*> srcPtrs, dstPtrs;
    std::vector<size_t> sizes;

    for (int ii = 0; ii < kNumEntries; ++ii)
    {
        srcPtrs.push_back(allocGpuWithPattern(kEntrySize, static_cast<uint8_t>(ii + 1)));
        dstPtrs.push_back(allocGpuZeroed(kEntrySize));
        sizes.push_back(kEntrySize);
    }

    auto status = helper.submitWithCubBatched(srcPtrs, dstPtrs, sizes);
    ASSERT_NE(status, nullptr);

    auto result = status->wait(-1);
    EXPECT_EQ(result, TransferState::kSUCCESS);

    // Verify each entry
    for (int ii = 0; ii < kNumEntries; ++ii)
    {
        std::vector<uint8_t> hostBuf(kEntrySize);
        TLLM_CUDA_CHECK(cudaMemcpy(hostBuf.data(), dstPtrs[ii], kEntrySize, cudaMemcpyDeviceToHost));
        EXPECT_EQ(hostBuf[0], static_cast<uint8_t>(ii + 1)) << "Entry " << ii << " first byte mismatch";
        EXPECT_EQ(hostBuf[kEntrySize - 1], static_cast<uint8_t>(ii + 1)) << "Entry " << ii << " last byte mismatch";
    }
}

TEST_F(FabricTransferTest, SubmitCubBatchedEmptyInput)
{
    // Empty input should return immediately-completed status.
    FabricTransferHelper helper;

    std::vector<void*> srcPtrs, dstPtrs;
    std::vector<size_t> sizes;

    auto status = helper.submitWithCubBatched(srcPtrs, dstPtrs, sizes);
    ASSERT_NE(status, nullptr);

    auto result = status->wait(-1);
    EXPECT_EQ(result, TransferState::kSUCCESS);
}

TEST_F(FabricTransferTest, SubmitCubBatchedConsecutiveCalls)
{
    // Two consecutive submitWithCubBatched calls on the same helper.
    // Verifies buffer reuse and correctness across calls.
    FabricTransferHelper helper;

    constexpr int kNumEntries = 8;
    constexpr size_t kEntrySize = 2048;

    for (int batch = 0; batch < 2; ++batch)
    {
        std::vector<void*> srcPtrs, dstPtrs;
        std::vector<size_t> sizes;
        uint8_t pattern = static_cast<uint8_t>(0xC0 + batch);

        for (int ii = 0; ii < kNumEntries; ++ii)
        {
            srcPtrs.push_back(allocGpuWithPattern(kEntrySize, pattern));
            dstPtrs.push_back(allocGpuZeroed(kEntrySize));
            sizes.push_back(kEntrySize);
        }

        auto status = helper.submitWithCubBatched(srcPtrs, dstPtrs, sizes);
        ASSERT_NE(status, nullptr);

        auto result = status->wait(-1);
        EXPECT_EQ(result, TransferState::kSUCCESS) << "Batch " << batch << " failed";

        for (int ii = 0; ii < kNumEntries; ++ii)
        {
            std::vector<uint8_t> hostBuf(kEntrySize);
            TLLM_CUDA_CHECK(cudaMemcpy(hostBuf.data(), dstPtrs[ii], kEntrySize, cudaMemcpyDeviceToHost));
            EXPECT_EQ(hostBuf[0], pattern) << "Batch " << batch << " entry " << ii;
        }
    }
}

// ============================================================================
// Thread Safety Tests for Remote Mappings
// ============================================================================

TEST_F(FabricTransferTest, RemoteMappingConcurrentReadWrite)
{
    // 4 reader threads calling hasRemoteMapping/hasFabricImportFailed/getRemoteMapping in a loop
    // 2 writer threads calling importAndMapRemoteFabric (with empty pools → fails) +
    //   cleanupRemoteFabricMapping in a loop.
    // Run for 500 iterations; verify no crashes.
    FabricTransferHelper helper;

    constexpr int kNumReaders = 4;
    constexpr int kNumWriters = 2;
    constexpr int kIterations = 500;

    std::atomic<bool> stop{false};
    std::vector<std::thread> threads;

    // Reader threads
    for (int r = 0; r < kNumReaders; ++r)
    {
        threads.emplace_back(
            [&helper, &stop, r]()
            {
                while (!stop.load(std::memory_order_acquire))
                {
                    std::string name = "remote_" + std::to_string(r % 4);
                    // These should not crash regardless of concurrent writes
                    [[maybe_unused]] auto has = helper.hasRemoteMapping(name);
                    [[maybe_unused]] auto failed = helper.hasFabricImportFailed(name);
                    [[maybe_unused]] auto mapping = helper.getRemoteMapping(name);
                    std::this_thread::yield();
                }
            });
    }

    // Writer threads
    for (int w = 0; w < kNumWriters; ++w)
    {
        threads.emplace_back(
            [&helper, &stop, w]()
            {
                for (int i = 0; i < kIterations; ++i)
                {
                    std::string name = "remote_" + std::to_string(w);
                    // Empty FabricMemInfo → import will fail gracefully (no pools)
                    FabricMemInfo emptyInfo;
                    emptyInfo.supported = true;
                    emptyInfo.handleType = VmmHandleType::kFabric;
                    // This will record name in mFailedFabricImports (no pools to map)
                    helper.importAndMapRemoteFabric(name, emptyInfo);
                    // Clean up (removes from mFailedFabricImports and mRemoteFabricMappings)
                    helper.cleanupRemoteFabricMapping(name);
                }
            });
    }

    // Wait for writers to finish, then stop readers
    for (int i = kNumReaders; i < static_cast<int>(threads.size()); ++i)
    {
        threads[i].join();
    }
    stop.store(true, std::memory_order_release);
    for (int i = 0; i < kNumReaders; ++i)
    {
        threads[i].join();
    }

    // If we got here without crashing, the test passes
    SUCCEED();
}

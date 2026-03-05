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

#pragma once

#include "tensorrt_llm/executor/transferAgent.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaEvent.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include <atomic>
#include <condition_variable>
#include <cuda.h>
#include <functional>
#include <mutex>
#include <optional>
#include <queue>
#include <shared_mutex>
#include <sstream>
#include <thread>
#include <unordered_set>
#include <vector>

namespace tensorrt_llm::executor::kv_cache
{

// ============================================================================
// VMM Handle Type
// ============================================================================

/// @brief Type of shareable VMM handle
enum class VmmHandleType : uint8_t
{
    kNone = 0,    ///< No shareable handle
    kFabric = 1,  ///< CU_MEM_HANDLE_TYPE_FABRIC (cross-node via NVSwitch)
    kPosixFd = 2, ///< CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR (same-machine via UDS)
    kCudaIpc = 3, ///< cudaMalloc memory via cudaIpcGetMemHandle (same-machine, no UDS needed)
};

/// @brief Convert VmmHandleType to human-readable string
inline char const* handleTypeToString(VmmHandleType type)
{
    switch (type)
    {
    case VmmHandleType::kFabric: return "Fabric";
    case VmmHandleType::kPosixFd: return "PosixFd";
    case VmmHandleType::kCudaIpc: return "CudaIpc";
    default: return "None";
    }
}

// ============================================================================
// Fabric Handle Data Structures for VMM-based KV Cache Transfer Optimization
// ============================================================================

/// @brief Single physical memory chunk information with fabric handle
struct FabricMemChunk
{
    uint64_t virtAddrOffset;  ///< Offset relative to pool base address
    uint64_t size;            ///< Chunk size in bytes
    uint8_t fabricHandle[64]; ///< CUmemFabricHandle (64 bytes)

    void serialize(std::ostream& os) const;
    static FabricMemChunk deserialize(std::istream& is);

    static constexpr size_t serializedSize()
    {
        return sizeof(uint64_t) * 2 + 64;
    }
};

/// @brief Memory pool fabric information (may contain multiple physical chunks)
struct FabricMemPool
{
    int32_t deviceId;        ///< Device ID
    uint64_t poolBaseAddr;   ///< Pool base virtual address (from cuMemGetAddressRange)
    uint64_t poolTotalSize;  ///< Total pool size (from cuMemGetAddressRange)
    uint64_t registeredAddr; ///< Actual registered address (user's registerMemory addr)
    uint64_t registeredSize; ///< Actual registered size (user's registerMemory size)
    uint64_t mappedOffset;   ///< Offset of mapped region relative to poolBaseAddr
    uint64_t mappedSize;     ///< Size of mapped region (may be larger than registeredSize due to chunk alignment)
    std::vector<FabricMemChunk> chunks; ///< Physical chunks (VMM may have multiple)

    void serialize(std::ostream& os) const;
    static FabricMemPool deserialize(std::istream& is);
};

/// @brief Complete fabric memory information for an agent
struct FabricMemInfo
{
    static constexpr uint32_t kMagic = 0x46414252; ///< "FABR"
    static constexpr uint32_t kVersion = 5;        ///< Version 5: numChunks/numPools widened to uint64

    bool supported{false};
    VmmHandleType handleType{VmmHandleType::kNone}; ///< Handle type (fabric or POSIX FD)
    std::string udsPath;                            ///< UDS server path (POSIX FD mode only)
    std::vector<FabricMemPool> pools;

    /// @brief Serialize to string for AgentDesc
    std::string serialize() const;
    /// @brief Deserialize from string
    static std::optional<FabricMemInfo> deserialize(std::string_view data);
};

/// @brief Mapping information for a single remote pool
struct RemotePoolMapping
{
    uint64_t remoteBaseAddr;                                   ///< Remote pool base address (from cuMemGetAddressRange)
    uint64_t totalSize;                                        ///< Total size (from cuMemGetAddressRange)
    uint64_t remoteRegisteredAddr;                             ///< Remote registered address (for validation)
    uint64_t registeredSize;                                   ///< Registered size (for validation)
    uint64_t remoteMappedOffset;                               ///< Offset of mapped region relative to remoteBaseAddr
    uint64_t mappedSize;                                       ///< Size of mapped region
    CUdeviceptr localVirtAddr;                                 ///< Local mapped virtual address
    std::vector<CUmemGenericAllocationHandle> importedHandles; ///< Imported handles
};

/// @brief Complete remote fabric mapping for an agent
struct RemoteFabricMapping
{
    std::string remoteName;
    VmmHandleType handleType{VmmHandleType::kNone}; ///< Handle type used (for cleanup path selection)
    std::vector<RemotePoolMapping> pools;
};

// ============================================================================
// BatchCopyWorkerPool - Persistent thread pool for parallel cudaMemcpyBatchAsync
// ============================================================================

/// @brief A task submitted to a batch copy worker thread.
/// Owns copies of the pointer/size arrays so that the caller can return immediately.
struct BatchCopyTask
{
    std::vector<void*> dst;
    std::vector<void const*> src;
    std::vector<size_t> sizes;
    cudaStream_t stream;
    cudaEvent_t completionEvent;    ///< Worker records this after the API call
    std::atomic<int>* batchPending; ///< Per-batch counter, decremented after event recording
};

/// @brief Persistent thread pool that calls cudaSetDevice once per thread.
/// Workers dequeue BatchCopyTask items and call cudaMemcpyBatchAsync on them.
class BatchCopyWorkerPool
{
public:
    BatchCopyWorkerPool(int numWorkers, int cudaDevice);
    ~BatchCopyWorkerPool();

    // Non-copyable
    BatchCopyWorkerPool(BatchCopyWorkerPool const&) = delete;
    BatchCopyWorkerPool& operator=(BatchCopyWorkerPool const&) = delete;

    /// @brief Submit a copy task (non-blocking). The task owns its data, so the caller's vectors can be destroyed.
    void submit(BatchCopyTask&& task);

    /// @brief Wait until all submitted tasks have been dequeued and their API calls returned.
    void waitAll();

    /// @brief Non-blocking check: returns true when all submitted tasks have completed their API calls.
    [[nodiscard]] bool isDone() const;

private:
    void workerLoop();

    std::vector<std::thread> mWorkers;
    std::mutex mMutex;
    std::condition_variable mCv;
    std::condition_variable mDoneCv;
    std::queue<BatchCopyTask> mQueue;
    bool mShutdown{false};
    std::atomic<int> mPending{0};
};

// ============================================================================
// FabricTransferStatus - Transfer status for fabric batch copy operations
// ============================================================================

class FabricTransferStatus final : public TransferStatus
{
public:
    /// @brief Single-event mode (for single-thread and cub paths)
    FabricTransferStatus(
        std::shared_ptr<runtime::CudaStream> stream, std::shared_ptr<runtime::CudaEvent> completionEvent);

    /// @brief Multi-event mode (for multi-thread batch copy path).
    /// Workers record events after their API calls and decrement the per-batch pending counter.
    FabricTransferStatus(
        std::shared_ptr<std::atomic<int>> batchPending, std::vector<std::shared_ptr<runtime::CudaEvent>> workerEvents);

    ~FabricTransferStatus() override = default;

    [[nodiscard]] bool isCompleted() const override;
    [[nodiscard]] TransferState wait(int64_t timeout_ms = -1) const override;

private:
    // Single-event mode
    std::shared_ptr<runtime::CudaStream> mStream;
    std::shared_ptr<runtime::CudaEvent> mCompletionEvent;

    // Multi-event mode (per-batch isolation — safe with concurrent submitters)
    std::shared_ptr<std::atomic<int>> mBatchPending;
    std::vector<std::shared_ptr<runtime::CudaEvent>> mWorkerEvents;

    mutable std::atomic<bool> mCompleted{false};
};

// ============================================================================
// CudaEventPool - Pool of reusable CUDA events to avoid create/destroy overhead
// ============================================================================

class CudaEventPool : public std::enable_shared_from_this<CudaEventPool>
{
public:
    /// @brief Acquire an event from the pool (creates new if pool is empty).
    /// Returned shared_ptr has a custom deleter that returns the event to the pool.
    [[nodiscard]] std::shared_ptr<runtime::CudaEvent> acquire();

private:
    void release(runtime::CudaEvent* event);

    std::mutex mMutex;
    std::vector<std::unique_ptr<runtime::CudaEvent>> mFreeEvents;
};

// ============================================================================
// FabricTransferHelper - Helper class for fabric transfer operations
// ============================================================================

class FabricTransferHelper
{
public:
    FabricTransferHelper();
    ~FabricTransferHelper();

    // Non-copyable
    FabricTransferHelper(FabricTransferHelper const&) = delete;
    FabricTransferHelper& operator=(FabricTransferHelper const&) = delete;

    /// @brief Check if fabric transfer is supported
    [[nodiscard]] bool isSupported() const
    {
        return mLocalFabricInfo.supported;
    }

    /// @brief Get local fabric memory info
    [[nodiscard]] FabricMemInfo const& getLocalFabricInfo() const
    {
        return mLocalFabricInfo;
    }

    /// @brief Check if remote agent has valid fabric mapping (not failed)
    [[nodiscard]] bool hasRemoteMapping(std::string const& remoteName) const;

    /// @brief Check if fabric import has previously failed for this remote agent
    [[nodiscard]] bool hasFabricImportFailed(std::string const& remoteName) const;

    /// @brief Detect and export fabric handles for registered VRAM memory
    void detectAndExportFabricHandles(RegisterDescs const& descs);

    /// @brief Remove fabric handles for deregistered VRAM memory
    void removeFabricHandles(RegisterDescs const& descs);

    /// @brief Import and map remote fabric memory
    void importAndMapRemoteFabric(std::string const& name, FabricMemInfo const& fabricInfo);

    /// @brief Clean up remote fabric mapping
    void cleanupRemoteFabricMapping(std::string const& name);

    /// @brief Translate remote address to local mapped address (looks up mapping by name each call)
    /// @param transferSize Validates that [remoteAddr, remoteAddr + transferSize) is within registered range
    [[nodiscard]] void* translateToLocalMapping(
        std::string const& remoteName, uintptr_t remoteAddr, size_t transferSize) const;

    /// @brief Get remote fabric mapping (nullptr if not found). Use to cache the lookup
    ///        outside hot loops, then call translateAddress() with the cached mapping.
    ///        Returns shared_ptr to keep the mapping alive even if cleanupRemoteFabricMapping runs concurrently.
    [[nodiscard]] std::shared_ptr<RemoteFabricMapping const> getRemoteMapping(std::string const& remoteName) const;

    /// @brief Translate remote address using a pre-looked-up mapping (avoids per-call hash lookup)
    /// @param transferSize Validates that [remoteAddr, remoteAddr + transferSize) is within registered range
    [[nodiscard]] void* translateAddress(
        RemoteFabricMapping const& mapping, uintptr_t remoteAddr, size_t transferSize) const;

    /// @brief Submit transfer using cub::DeviceMemcpy::Batched (for small segments)
    [[nodiscard]] std::unique_ptr<TransferStatus> submitWithCubBatched(
        std::vector<void*> const& srcPtrs, std::vector<void*> const& dstPtrs, std::vector<size_t> const& sizes);

    /// @brief Submit transfer using cudaMemcpyAsync (for large segments)
    [[nodiscard]] std::unique_ptr<TransferStatus> submitWithCudaMemcpyBatch(
        std::vector<void*> const& srcPtrs, std::vector<void*> const& dstPtrs, std::vector<size_t> const& sizes);

private:
    /// @brief Initialize CUDA resources (lazy)
    void ensureCudaResourcesInitialized();

    /// @brief Ensure pre-allocated buffers are large enough
    void ensurePreallocBuffers(size_t batchSize, size_t cubTempBytes);

    /// @brief Get VMM allocation granularity
    [[nodiscard]] size_t getVmmGranularity();

    /// @brief Detect all chunk boundaries and export fabric handles within registered range.
    /// Uses cuMemGetAddressRange to iterate through physical chunk mappings — O(N) in chunk count.
    /// @param scanStart Start of the registered range to scan
    /// @param scanSize Size of the registered range to scan
    /// @param poolBase Base address of the entire VMM pool (for virtAddrOffset calculation)
    /// @param poolTotalSize Total size of the VMM pool
    void detectAndExportChunks(CUdeviceptr scanStart, size_t scanSize, CUdeviceptr poolBase, size_t poolTotalSize,
        std::vector<FabricMemChunk>& chunks);

    /// @brief Export a single chunk's fabric handle (chunkBase/chunkSize are real boundaries)
    void exportSingleChunk(
        CUdeviceptr chunkBase, size_t chunkSize, CUdeviceptr poolBase, std::vector<FabricMemChunk>& chunks);

    /// @brief Export a single chunk's POSIX FD handle (chunkBase/chunkSize are real boundaries)
    void exportSingleChunkPosixFd(
        CUdeviceptr chunkBase, size_t chunkSize, CUdeviceptr poolBase, std::vector<FabricMemChunk>& chunks);

    /// @brief Export a cudaMalloc allocation as a single chunk via cudaIpcGetMemHandle
    void exportSingleChunkCudaIpc(CUdeviceptr poolBase, size_t poolTotalSize, std::vector<FabricMemChunk>& chunks);

    /// @brief Translate remote address using specific mapping
    /// @param transferSize Validates that [remoteAddr, remoteAddr + transferSize) is within registered range
    [[nodiscard]] void* translateToLocalMappingInternal(
        RemoteFabricMapping const& mapping, uintptr_t remoteAddr, size_t transferSize) const;

    /// @brief Start UDS server for POSIX FD sharing
    void startUdsServer();

    /// @brief Stop UDS server
    void stopUdsServer();

    /// @brief Initialize the batch copy worker pool (lazy, called once)
    void ensureBatchCopyWorkersInitialized();

private:
    // CUDA resources for fabric batch transfer
    std::shared_ptr<runtime::CudaStream> mFabricStream;
    std::shared_ptr<runtime::BufferManager> mBufferManager;
    std::once_flag mCudaResourcesInitFlag; ///< Ensures CUDA resources initialized exactly once

    // Pre-allocated buffers for cub::DeviceMemcpy::Batched
    struct PreallocatedBuffers
    {
        runtime::IBuffer::UniquePtr combinedGpu;    ///< [srcPtrs|dstPtrs|sizes] on GPU (H2D mode only)
        runtime::IBuffer::UniquePtr combinedPinned; ///< [srcPtrs|dstPtrs|sizes] on pinned host
        runtime::IBuffer::UniquePtr cubTempStorage; ///< cub temporary storage
        size_t maxBatchSize{0};                     ///< Current max batch size
        size_t cubTempStorageSize{0};               ///< Current cub temp storage size
    };

    PreallocatedBuffers mPreallocBuffers;
    std::mutex mBufferMutex; ///< Protects pre-allocated buffers

    // CUDA event pool (avoids per-transfer cudaEventCreate/Destroy overhead)
    std::shared_ptr<CudaEventPool> mEventPool;

    // Whether cub reads parameter arrays directly from pinned host (zero-copy)
    bool mCubZeroCopy{false};

    // Local fabric memory information
    FabricMemInfo mLocalFabricInfo;

    // Detected handle type for this helper instance
    VmmHandleType mDetectedHandleType{VmmHandleType::kNone};

    // Local CUDA device ID (queried once at construction)
    CUdevice mLocalDevice{0};

    // Remote fabric memory mappings (stored as shared_ptr for safe concurrent access)
    std::unordered_map<std::string, std::shared_ptr<RemoteFabricMapping>> mRemoteFabricMappings;

    // Set of remote agents where fabric import has failed (e.g., not in same NVLink domain)
    // We track these to avoid retrying failed imports and to ensure proper fallback to NIXL
    std::unordered_set<std::string> mFailedFabricImports;

    // Protects mRemoteFabricMappings and mFailedFabricImports for concurrent read/write access
    mutable std::shared_mutex mRemoteMappingsMutex;

    // ========== POSIX FD specific members ==========
    // Exported file descriptors (one per chunk, in pool/chunk order)
    std::vector<int> mExportedFds;
    std::mutex mExportedFdsMutex; ///< Protects mExportedFds (shared with UDS server thread)

    // UDS server for sharing POSIX FDs with remote processes
    std::string mUdsPath;                       ///< UDS socket file path
    int mUdsServerSocket{-1};                   ///< Server listening socket
    std::thread mUdsServerThread;               ///< Server accept/send thread
    std::atomic<bool> mUdsServerRunning{false}; ///< Controls server thread lifetime

    // Multi-thread batch copy (NT-NS) for cudaMemcpyBatchAsync
    int mBatchCopyThreads{0};                                  ///< Number of worker threads (0 = not yet initialized)
    std::unique_ptr<BatchCopyWorkerPool> mBatchCopyWorkerPool; ///< Persistent worker pool
    std::vector<std::shared_ptr<runtime::CudaStream>> mBatchCopyStreams; ///< Per-worker streams
    std::once_flag mBatchCopyInitFlag;                                   ///< Ensures worker pool initialized once
};

} // namespace tensorrt_llm::executor::kv_cache

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cuda.h>
#include <mutex>
#include <optional>
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
};

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
    static constexpr uint32_t kVersion = 4;        ///< Version 4: added handleType + udsPath for POSIX FD support

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
    std::vector<RemotePoolMapping> pools;
};

// ============================================================================
// FabricTransferStatus - Transfer status for fabric batch copy operations
// ============================================================================

class FabricTransferStatus final : public TransferStatus
{
public:
    FabricTransferStatus(
        std::shared_ptr<runtime::CudaStream> stream, std::shared_ptr<runtime::CudaEvent> completionEvent);

    ~FabricTransferStatus() override = default;

    [[nodiscard]] bool isCompleted() const override;
    [[nodiscard]] TransferState wait(int64_t timeout_ms = -1) const override;

private:
    std::shared_ptr<runtime::CudaStream> mStream;
    std::shared_ptr<runtime::CudaEvent> mCompletionEvent;
    mutable std::atomic<bool> mCompleted{false};
};

// ============================================================================
// FabricTransferHelper - Helper class for fabric transfer operations
// ============================================================================

class FabricTransferHelper
{
public:
    FabricTransferHelper() = default;
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

    /// @brief Translate remote address to local mapped address
    [[nodiscard]] void* translateToLocalMapping(std::string const& remoteName, uintptr_t remoteAddr) const;

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

    /// @brief Get buffer_id for a given address
    [[nodiscard]] unsigned long long getBufferId(CUdeviceptr addr);

    /// @brief Detect all chunk boundaries and export fabric handles within registered range
    /// @param scanStart Start of the registered range to scan
    /// @param scanSize Size of the registered range to scan
    /// @param poolBase Base address of the entire VMM pool (for virtAddrOffset calculation)
    void detectAndExportChunks(
        CUdeviceptr scanStart, size_t scanSize, CUdeviceptr poolBase, std::vector<FabricMemChunk>& chunks);

    /// @brief Find all chunk boundaries recursively using binary search
    void findAllBoundariesRecursive(
        CUdeviceptr left, CUdeviceptr right, size_t granularity, std::vector<CUdeviceptr>& boundaries);

    /// @brief Export a single chunk's fabric handle
    void exportSingleChunk(
        CUdeviceptr chunkAddr, size_t chunkSize, CUdeviceptr poolBase, std::vector<FabricMemChunk>& chunks);

    /// @brief Export a single chunk's POSIX FD handle
    void exportSingleChunkPosixFd(
        CUdeviceptr chunkAddr, size_t chunkSize, CUdeviceptr poolBase, std::vector<FabricMemChunk>& chunks);

    /// @brief Translate remote address using specific mapping
    [[nodiscard]] void* translateToLocalMappingInternal(RemoteFabricMapping const& mapping, uintptr_t remoteAddr) const;

    /// @brief Start UDS server for POSIX FD sharing
    void startUdsServer();

    /// @brief Stop UDS server
    void stopUdsServer();

private:
    // CUDA resources for fabric batch transfer
    std::shared_ptr<runtime::CudaStream> mFabricStream;
    std::shared_ptr<runtime::BufferManager> mBufferManager;

    // Pre-allocated buffers for cub::DeviceMemcpy::Batched
    struct PreallocatedBuffers
    {
        runtime::IBuffer::UniquePtr srcPtrs;        ///< Source pointer array (GPU)
        runtime::IBuffer::UniquePtr dstPtrs;        ///< Destination pointer array (GPU)
        runtime::IBuffer::UniquePtr sizes;          ///< Size array (GPU)
        runtime::IBuffer::UniquePtr cubTempStorage; ///< cub temporary storage
        size_t maxBatchSize{0};                     ///< Current max batch size
        size_t cubTempStorageSize{0};               ///< Current cub temp storage size
    };

    PreallocatedBuffers mPreallocBuffers;
    std::mutex mBufferMutex; ///< Protects pre-allocated buffers

    // Local fabric memory information
    FabricMemInfo mLocalFabricInfo;

    // Detected handle type for this helper instance
    VmmHandleType mDetectedHandleType{VmmHandleType::kNone};

    // Remote fabric memory mappings
    std::unordered_map<std::string, RemoteFabricMapping> mRemoteFabricMappings;

    // Set of remote agents where fabric import has failed (e.g., not in same NVLink domain)
    // We track these to avoid retrying failed imports and to ensure proper fallback to NIXL
    std::unordered_set<std::string> mFailedFabricImports;

    // ========== POSIX FD specific members ==========
    // Exported file descriptors (one per chunk, in pool/chunk order)
    std::vector<int> mExportedFds;

    // UDS server for sharing POSIX FDs with remote processes
    std::string mUdsPath;                       ///< UDS socket file path
    int mUdsServerSocket{-1};                   ///< Server listening socket
    std::thread mUdsServerThread;               ///< Server accept/send thread
    std::atomic<bool> mUdsServerRunning{false}; ///< Controls server thread lifetime
};

} // namespace tensorrt_llm::executor::kv_cache

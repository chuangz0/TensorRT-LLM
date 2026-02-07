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

// POSIX FD Transfer Test
// Run with: mpirun -n 2 ./posixFdTransferTest
//
// Tests include:
// 1. FabricTransferHelper level: detection, serialization, cross-process FD transfer, multi-chunk
// 2. NixlTransferAgent level: full end-to-end register → getLocalAgentDesc → loadRemoteAgent → submitTransferRequests

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/fabricTransfer.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/transferAgent.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace tensorrt_llm::executor::kv_cache;

#define CU_CHECK(call)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        CUresult err = call;                                                                                           \
        if (err != CUDA_SUCCESS)                                                                                       \
        {                                                                                                              \
            char const* errStr;                                                                                        \
            cuGetErrorString(err, &errStr);                                                                            \
            fprintf(stderr, "CUDA driver error at %s:%d: %s (code %d)\n", __FILE__, __LINE__, errStr, err);            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                                              \
        }                                                                                                              \
    } while (0)

// ============================================================================
// VMM Allocation Helper
// ============================================================================

struct VmmAllocation
{
    CUmemGenericAllocationHandle handle;
    CUdeviceptr ptr;
    size_t allocSize;
    size_t granularity;

    void cleanup()
    {
        if (ptr != 0)
        {
            cuMemUnmap(ptr, allocSize);
            cuMemAddressFree(ptr, allocSize);
            ptr = 0;
        }
        if (handle != 0)
        {
            cuMemRelease(handle);
            handle = 0;
        }
    }
};

/// @brief Allocate VMM memory with CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
VmmAllocation allocateVmmPosixFd(size_t requestedSize, int deviceId)
{
    VmmAllocation alloc = {};

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = deviceId;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

    CU_CHECK(cuMemGetAllocationGranularity(&alloc.granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    alloc.allocSize = ((requestedSize + alloc.granularity - 1) / alloc.granularity) * alloc.granularity;

    CU_CHECK(cuMemCreate(&alloc.handle, alloc.allocSize, &prop, 0));
    CU_CHECK(cuMemAddressReserve(&alloc.ptr, alloc.allocSize, alloc.granularity, 0, 0));
    CU_CHECK(cuMemMap(alloc.ptr, alloc.allocSize, 0, alloc.handle, 0));

    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = deviceId;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CU_CHECK(cuMemSetAccess(alloc.ptr, alloc.allocSize, &access, 1));

    return alloc;
}

/// @brief Allocate VMM memory with POSIX FD, using multiple chunks to test multi-chunk handling
struct MultiChunkVmmAllocation
{
    std::vector<CUmemGenericAllocationHandle> handles;
    CUdeviceptr ptr; // Single reserved VA range
    size_t totalSize;
    size_t granularity;
    size_t numChunks;

    void cleanup()
    {
        if (ptr != 0)
        {
            cuMemUnmap(ptr, totalSize);
            cuMemAddressFree(ptr, totalSize);
            ptr = 0;
        }
        for (auto h : handles)
        {
            if (h != 0)
            {
                cuMemRelease(h);
            }
        }
        handles.clear();
    }
};

MultiChunkVmmAllocation allocateMultiChunkVmmPosixFd(size_t chunkSize, size_t numChunks, int deviceId)
{
    MultiChunkVmmAllocation alloc = {};
    alloc.numChunks = numChunks;

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = deviceId;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

    CU_CHECK(cuMemGetAllocationGranularity(&alloc.granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    size_t alignedChunkSize = ((chunkSize + alloc.granularity - 1) / alloc.granularity) * alloc.granularity;
    alloc.totalSize = alignedChunkSize * numChunks;

    // Reserve a single VA range for all chunks
    CU_CHECK(cuMemAddressReserve(&alloc.ptr, alloc.totalSize, alloc.granularity, 0, 0));

    // Create and map each chunk separately
    for (size_t i = 0; i < numChunks; ++i)
    {
        CUmemGenericAllocationHandle handle;
        CU_CHECK(cuMemCreate(&handle, alignedChunkSize, &prop, 0));
        CU_CHECK(cuMemMap(alloc.ptr + i * alignedChunkSize, alignedChunkSize, 0, handle, 0));
        alloc.handles.push_back(handle);
    }

    // Set access for the entire range
    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = deviceId;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CU_CHECK(cuMemSetAccess(alloc.ptr, alloc.totalSize, &access, 1));

    return alloc;
}

// ============================================================================
// Test: POSIX FD Detection
// ============================================================================

bool testPosixFdDetection(int rank)
{
    if (rank != 1)
    {
        return true;
    }

    printf("[Rank %d] === Test: PosixFdDetection ===\n", rank);

    int deviceId = 0;
    TLLM_CUDA_CHECK(cudaSetDevice(deviceId));

    // Check if POSIX FD is supported
    CUdevice device;
    CU_CHECK(cuDeviceGet(&device, deviceId));
    int supported = 0;
    CU_CHECK(cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, device));
    if (!supported)
    {
        printf("[Rank %d] POSIX FD not supported on this device, skipping test\n", rank);
        return true; // Not a failure, just unsupported
    }

    // Allocate VMM memory with POSIX FD
    size_t allocSize = 4 * 1024 * 1024; // 4MB
    auto alloc = allocateVmmPosixFd(allocSize, deviceId);
    printf("[Rank %d] VMM allocated: ptr=0x%lx, size=%zu\n", rank, alloc.ptr, alloc.allocSize);

    // Create helper and detect
    FabricTransferHelper helper;

    // Create a MemoryDesc for the allocated memory
    MemoryDescs descs{MemoryType::kVRAM,
        {MemoryDesc{reinterpret_cast<char*>(alloc.ptr), alloc.allocSize, static_cast<uint32_t>(deviceId)}}};
    helper.detectAndExportFabricHandles(descs);

    // Verify detection
    auto const& info = helper.getLocalFabricInfo();
    bool ok = true;

    if (!info.supported)
    {
        printf("[Rank %d] FAIL: info.supported is false\n", rank);
        ok = false;
    }
    else
    {
        printf("[Rank %d] info.supported = true\n", rank);
    }

    if (info.handleType != VmmHandleType::kPosixFd)
    {
        // If fabric is also supported, it would pick fabric. That's OK for this test.
        if (info.handleType == VmmHandleType::kFabric)
        {
            printf("[Rank %d] handleType = Fabric (fabric takes priority, this is expected)\n", rank);
        }
        else
        {
            printf("[Rank %d] FAIL: unexpected handleType=%d\n", rank, static_cast<int>(info.handleType));
            ok = false;
        }
    }
    else
    {
        printf("[Rank %d] handleType = PosixFd (correct!)\n", rank);
    }

    if (!info.pools.empty())
    {
        printf("[Rank %d] pools=%zu, chunks=%zu\n", rank, info.pools.size(), info.pools[0].chunks.size());
    }

    alloc.cleanup();
    printf("[Rank %d] === PosixFdDetection: %s ===\n", rank, ok ? "PASSED" : "FAILED");
    return ok;
}

// ============================================================================
// Test: POSIX FD Serialization
// ============================================================================

bool testPosixFdSerialization(int rank)
{
    if (rank != 0)
    {
        return true;
    }

    printf("[Rank %d] === Test: PosixFdSerialization ===\n", rank);

    FabricMemInfo info;
    info.supported = true;
    info.handleType = VmmHandleType::kPosixFd;
    info.udsPath = "/tmp/test_uds_path_12345.sock";

    FabricMemPool pool;
    pool.deviceId = 0;
    pool.poolBaseAddr = 0x7f0000000000ULL;
    pool.poolTotalSize = 64 * 1024 * 1024;
    pool.registeredAddr = 0x7f0000000000ULL;
    pool.registeredSize = 4 * 1024 * 1024;
    pool.mappedOffset = 0;
    pool.mappedSize = 4 * 1024 * 1024;

    FabricMemChunk chunk;
    chunk.virtAddrOffset = 0;
    chunk.size = 4 * 1024 * 1024;
    std::memset(chunk.fabricHandle, 0, sizeof(chunk.fabricHandle)); // POSIX FD mode: zeroed
    pool.chunks.push_back(chunk);
    info.pools.push_back(pool);

    // Serialize
    std::string serialized = info.serialize();
    printf("[Rank %d] Serialized size: %zu bytes\n", rank, serialized.size());

    // Deserialize
    auto deserialized = FabricMemInfo::deserialize(serialized);
    bool ok = true;

    if (!deserialized.has_value())
    {
        printf("[Rank %d] FAIL: deserialization returned nullopt\n", rank);
        ok = false;
    }
    else
    {
        if (!deserialized->supported)
        {
            printf("[Rank %d] FAIL: deserialized.supported is false\n", rank);
            ok = false;
        }
        if (deserialized->handleType != VmmHandleType::kPosixFd)
        {
            printf("[Rank %d] FAIL: deserialized.handleType=%d, expected PosixFd\n", rank,
                static_cast<int>(deserialized->handleType));
            ok = false;
        }
        if (deserialized->udsPath != info.udsPath)
        {
            printf("[Rank %d] FAIL: deserialized.udsPath='%s', expected '%s'\n", rank, deserialized->udsPath.c_str(),
                info.udsPath.c_str());
            ok = false;
        }
        if (deserialized->pools.size() != 1)
        {
            printf("[Rank %d] FAIL: deserialized.pools.size()=%zu, expected 1\n", rank, deserialized->pools.size());
            ok = false;
        }
        else
        {
            auto const& p = deserialized->pools[0];
            if (p.deviceId != 0 || p.poolBaseAddr != pool.poolBaseAddr || p.chunks.size() != 1
                || p.chunks[0].size != chunk.size)
            {
                printf("[Rank %d] FAIL: pool data mismatch\n", rank);
                ok = false;
            }
        }
        if (ok)
        {
            printf("[Rank %d] Serialization/deserialization OK: handleType=PosixFd, udsPath='%s', pools=%zu\n", rank,
                deserialized->udsPath.c_str(), deserialized->pools.size());
        }
    }

    printf("[Rank %d] === PosixFdSerialization: %s ===\n", rank, ok ? "PASSED" : "FAILED");
    return ok;
}

// ============================================================================
// Test: Cross-process POSIX FD Transfer (main test)
// ============================================================================

bool testPosixFdTransfer(int rank, int worldSize)
{
    printf("[Rank %d] === Test: PosixFdTransfer ===\n", rank);

    if (worldSize < 2)
    {
        printf("[Rank %d] Need at least 2 ranks, skipping\n", rank);
        return true;
    }

    int deviceId = 0;
    TLLM_CUDA_CHECK(cudaSetDevice(deviceId));

    // Check POSIX FD support
    CUdevice device;
    CU_CHECK(cuDeviceGet(&device, deviceId));
    int posixFdSupported = 0;
    CU_CHECK(cuDeviceGetAttribute(
        &posixFdSupported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, device));

    // Broadcast support check
    MPI_Bcast(&posixFdSupported, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!posixFdSupported)
    {
        printf("[Rank %d] POSIX FD not supported, skipping transfer test\n", rank);
        return true;
    }

    bool ok = true;
    size_t const DATA_SIZE = 4 * 1024 * 1024; // 4MB
    uint8_t const FILL_PATTERN = 0xAB;

    if (rank == 1)
    {
        // ===== EXPORTER =====
        // 1. Allocate VMM memory with POSIX FD
        auto alloc = allocateVmmPosixFd(DATA_SIZE, deviceId);
        printf("[Rank %d] VMM allocated: ptr=0x%lx, size=%zu\n", rank, alloc.ptr, alloc.allocSize);

        // 2. Fill with known pattern
        TLLM_CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(alloc.ptr), FILL_PATTERN, DATA_SIZE));
        TLLM_CUDA_CHECK(cudaDeviceSynchronize());
        printf("[Rank %d] Memory filled with 0x%02X\n", rank, FILL_PATTERN);

        // 3. Create helper and detect/export
        FabricTransferHelper helper;
        MemoryDescs descs{MemoryType::kVRAM,
            {MemoryDesc{reinterpret_cast<char*>(alloc.ptr), alloc.allocSize, static_cast<uint32_t>(deviceId)}}};
        helper.detectAndExportFabricHandles(descs);

        auto const& info = helper.getLocalFabricInfo();
        if (!info.supported)
        {
            printf("[Rank %d] FAIL: fabric info not supported after detection\n", rank);
            alloc.cleanup();
            // Need to signal rank 0
            int metadataSize = 0;
            MPI_Send(&metadataSize, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            return false;
        }

        printf("[Rank %d] Export done: handleType=%s, udsPath=%s, pools=%zu\n", rank,
            info.handleType == VmmHandleType::kPosixFd ? "PosixFd" : "Fabric", info.udsPath.c_str(), info.pools.size());

        // 4. Serialize and send metadata via MPI
        std::string metadata = info.serialize();
        int metadataSize = static_cast<int>(metadata.size());
        MPI_Send(&metadataSize, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(metadata.data(), metadataSize, MPI_BYTE, 0, 1, MPI_COMM_WORLD);
        printf("[Rank %d] Sent metadata (%d bytes) to rank 0\n", rank, metadataSize);

        // 5. Wait for rank 0 to finish importing and verifying
        MPI_Barrier(MPI_COMM_WORLD);

        // 6. Cleanup
        // helper destructor stops UDS server and closes FDs
        alloc.cleanup();
        printf("[Rank %d] Cleanup done\n", rank);
    }
    else if (rank == 0)
    {
        // ===== IMPORTER =====
        // 1. Receive metadata from rank 1
        int metadataSize = 0;
        MPI_Recv(&metadataSize, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (metadataSize == 0)
        {
            printf("[Rank %d] Exporter reported failure, skipping\n", rank);
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }

        std::string metadata(metadataSize, '\0');
        MPI_Recv(metadata.data(), metadataSize, MPI_BYTE, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("[Rank %d] Received metadata (%d bytes) from rank 1\n", rank, metadataSize);

        // 2. Deserialize
        auto fabricInfo = FabricMemInfo::deserialize(metadata);
        if (!fabricInfo.has_value())
        {
            printf("[Rank %d] FAIL: failed to deserialize FabricMemInfo\n", rank);
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }

        printf("[Rank %d] Deserialized: handleType=%s, udsPath=%s, pools=%zu\n", rank,
            fabricInfo->handleType == VmmHandleType::kPosixFd ? "PosixFd" : "Fabric", fabricInfo->udsPath.c_str(),
            fabricInfo->pools.size());

        // 3. Import and map remote memory
        FabricTransferHelper helper;
        helper.importAndMapRemoteFabric("rank1", *fabricInfo);

        if (!helper.hasRemoteMapping("rank1"))
        {
            printf("[Rank %d] FAIL: no remote mapping established\n", rank);
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }
        printf("[Rank %d] Remote mapping established\n", rank);

        // 4. Verify mapped memory is readable
        // Get the mapped local address for the remote memory
        auto const& pool = fabricInfo->pools[0];
        void* localMappedPtr = helper.translateToLocalMapping("rank1", pool.registeredAddr);
        if (localMappedPtr == nullptr)
        {
            printf("[Rank %d] FAIL: translateToLocalMapping returned nullptr\n", rank);
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }
        printf("[Rank %d] Mapped local ptr = %p\n", rank, localMappedPtr);

        // Read back and verify
        std::vector<uint8_t> hostBuf(DATA_SIZE);
        TLLM_CUDA_CHECK(cudaMemcpy(hostBuf.data(), localMappedPtr, DATA_SIZE, cudaMemcpyDeviceToHost));

        bool dataOk = true;
        for (size_t i = 0; i < DATA_SIZE; ++i)
        {
            if (hostBuf[i] != FILL_PATTERN)
            {
                printf("[Rank %d] FAIL: data mismatch at offset %zu: expected 0x%02X, got 0x%02X\n", rank, i,
                    FILL_PATTERN, hostBuf[i]);
                dataOk = false;
                break;
            }
        }
        if (dataOk)
        {
            printf(
                "[Rank %d] Direct read verification PASSED (all %zu bytes == 0x%02X)\n", rank, DATA_SIZE, FILL_PATTERN);
        }
        ok = ok && dataOk;

        // 5. Test batch transfer: copy from mapped remote memory to a local buffer
        char* localDst = nullptr;
        TLLM_CUDA_CHECK(cudaMalloc(&localDst, DATA_SIZE));
        TLLM_CUDA_CHECK(cudaMemset(localDst, 0, DATA_SIZE));

        std::vector<void*> srcPtrs = {localMappedPtr};
        std::vector<void*> dstPtrs = {localDst};
        std::vector<size_t> sizes = {DATA_SIZE};

        auto status = helper.submitWithCudaMemcpyBatch(srcPtrs, dstPtrs, sizes);
        auto result = status->wait();

        if (result != TransferState::kSUCCESS)
        {
            printf("[Rank %d] FAIL: batch transfer did not succeed\n", rank);
            ok = false;
        }
        else
        {
            // Verify transferred data
            std::vector<uint8_t> transferBuf(DATA_SIZE);
            TLLM_CUDA_CHECK(cudaMemcpy(transferBuf.data(), localDst, DATA_SIZE, cudaMemcpyDeviceToHost));

            bool transferOk = true;
            for (size_t i = 0; i < DATA_SIZE; ++i)
            {
                if (transferBuf[i] != FILL_PATTERN)
                {
                    printf("[Rank %d] FAIL: transfer data mismatch at offset %zu: expected 0x%02X, got 0x%02X\n", rank,
                        i, FILL_PATTERN, transferBuf[i]);
                    transferOk = false;
                    break;
                }
            }
            if (transferOk)
            {
                printf("[Rank %d] Batch transfer verification PASSED\n", rank);
            }
            ok = ok && transferOk;
        }

        TLLM_CUDA_CHECK(cudaFree(localDst));

        // 6. Cleanup
        helper.cleanupRemoteFabricMapping("rank1");

        MPI_Barrier(MPI_COMM_WORLD);
        printf("[Rank %d] Cleanup done\n", rank);
    }

    printf("[Rank %d] === PosixFdTransfer: %s ===\n", rank, ok ? "PASSED" : "FAILED");
    return ok;
}

// ============================================================================
// Test: Multi-Chunk POSIX FD Transfer
// ============================================================================

bool testPosixFdMultiChunk(int rank, int worldSize)
{
    printf("[Rank %d] === Test: PosixFdMultiChunk ===\n", rank);

    if (worldSize < 2)
    {
        printf("[Rank %d] Need at least 2 ranks, skipping\n", rank);
        return true;
    }

    int deviceId = 0;
    TLLM_CUDA_CHECK(cudaSetDevice(deviceId));

    // Check POSIX FD support
    CUdevice device;
    CU_CHECK(cuDeviceGet(&device, deviceId));
    int posixFdSupported = 0;
    CU_CHECK(cuDeviceGetAttribute(
        &posixFdSupported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, device));
    MPI_Bcast(&posixFdSupported, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (!posixFdSupported)
    {
        printf("[Rank %d] POSIX FD not supported, skipping multi-chunk test\n", rank);
        return true;
    }

    bool ok = true;
    size_t const CHUNK_SIZE = 2 * 1024 * 1024; // 2MB per chunk
    size_t const NUM_CHUNKS = 3;
    size_t const TOTAL_SIZE = CHUNK_SIZE * NUM_CHUNKS;
    uint8_t const FILL_PATTERN = 0xCD;

    if (rank == 1)
    {
        // ===== EXPORTER =====
        auto alloc = allocateMultiChunkVmmPosixFd(CHUNK_SIZE, NUM_CHUNKS, deviceId);
        printf("[Rank %d] Multi-chunk VMM allocated: ptr=0x%lx, totalSize=%zu, chunks=%zu\n", rank, alloc.ptr,
            alloc.totalSize, alloc.numChunks);

        // Fill with pattern
        TLLM_CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(alloc.ptr), FILL_PATTERN, TOTAL_SIZE));
        TLLM_CUDA_CHECK(cudaDeviceSynchronize());

        // Detect and export
        FabricTransferHelper helper;
        MemoryDescs descs{MemoryType::kVRAM,
            {MemoryDesc{reinterpret_cast<char*>(alloc.ptr), alloc.totalSize, static_cast<uint32_t>(deviceId)}}};
        helper.detectAndExportFabricHandles(descs);

        auto const& info = helper.getLocalFabricInfo();
        if (!info.supported)
        {
            printf("[Rank %d] FAIL: not supported after detection\n", rank);
            alloc.cleanup();
            int metadataSize = 0;
            MPI_Send(&metadataSize, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            return false;
        }

        size_t totalChunks = 0;
        for (auto const& p : info.pools)
        {
            totalChunks += p.chunks.size();
        }
        printf("[Rank %d] Exported: pools=%zu, totalChunks=%zu\n", rank, info.pools.size(), totalChunks);

        // Send metadata
        std::string metadata = info.serialize();
        int metadataSize = static_cast<int>(metadata.size());
        MPI_Send(&metadataSize, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(metadata.data(), metadataSize, MPI_BYTE, 0, 1, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        alloc.cleanup();
    }
    else if (rank == 0)
    {
        // ===== IMPORTER =====
        int metadataSize = 0;
        MPI_Recv(&metadataSize, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (metadataSize == 0)
        {
            printf("[Rank %d] Exporter reported failure, skipping\n", rank);
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }

        std::string metadata(metadataSize, '\0');
        MPI_Recv(metadata.data(), metadataSize, MPI_BYTE, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        auto fabricInfo = FabricMemInfo::deserialize(metadata);
        if (!fabricInfo.has_value())
        {
            printf("[Rank %d] FAIL: deserialization failed\n", rank);
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }

        // Import
        FabricTransferHelper helper;
        helper.importAndMapRemoteFabric("rank1_mc", *fabricInfo);

        if (!helper.hasRemoteMapping("rank1_mc"))
        {
            printf("[Rank %d] FAIL: no mapping for multi-chunk\n", rank);
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }

        // Verify data for each chunk segment
        auto const& pool = fabricInfo->pools[0];
        void* localMappedPtr = helper.translateToLocalMapping("rank1_mc", pool.registeredAddr);
        if (localMappedPtr == nullptr)
        {
            printf("[Rank %d] FAIL: translateToLocalMapping returned nullptr\n", rank);
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }

        std::vector<uint8_t> hostBuf(TOTAL_SIZE);
        TLLM_CUDA_CHECK(cudaMemcpy(hostBuf.data(), localMappedPtr, TOTAL_SIZE, cudaMemcpyDeviceToHost));

        bool dataOk = true;
        for (size_t i = 0; i < TOTAL_SIZE; ++i)
        {
            if (hostBuf[i] != FILL_PATTERN)
            {
                printf("[Rank %d] FAIL: multi-chunk data mismatch at offset %zu: expected 0x%02X, got 0x%02X\n", rank,
                    i, FILL_PATTERN, hostBuf[i]);
                dataOk = false;
                break;
            }
        }
        if (dataOk)
        {
            printf("[Rank %d] Multi-chunk verification PASSED (%zu bytes across %zu chunks)\n", rank, TOTAL_SIZE,
                NUM_CHUNKS);
        }
        ok = ok && dataOk;

        helper.cleanupRemoteFabricMapping("rank1_mc");
        MPI_Barrier(MPI_COMM_WORLD);
    }

    printf("[Rank %d] === PosixFdMultiChunk: %s ===\n", rank, ok ? "PASSED" : "FAILED");
    return ok;
}

// ============================================================================
// Test: NixlTransferAgent end-to-end with POSIX FD VMM memory
// Full flow: registerMemory → getLocalAgentDesc → loadRemoteAgent → submitTransferRequests
// ============================================================================

bool testNixlAgentPosixFdTransfer(int rank, int worldSize)
{
    printf("[Rank %d] === Test: NixlAgentPosixFdTransfer ===\n", rank);

    if (worldSize < 2)
    {
        printf("[Rank %d] Need at least 2 ranks, skipping\n", rank);
        return true;
    }

    int deviceId = 0;
    TLLM_CUDA_CHECK(cudaSetDevice(deviceId));

    // Check POSIX FD support
    CUdevice device;
    CU_CHECK(cuDeviceGet(&device, deviceId));
    int posixFdSupported = 0;
    CU_CHECK(cuDeviceGetAttribute(
        &posixFdSupported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, device));
    MPI_Bcast(&posixFdSupported, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (!posixFdSupported)
    {
        printf("[Rank %d] POSIX FD not supported, skipping NixlAgent test\n", rank);
        return true;
    }

    bool ok = true;
    size_t const DATA_SIZE = 4 * 1024 * 1024; // 4MB
    uint8_t const FILL_PATTERN = 0xEF;
    uint32_t const DEVICE_ID = 0;

    // Create NixlTransferAgent for this rank
    // useListenThread=false since we exchange AgentDesc directly via MPI
    std::string agentName = "nixl_fd_agent_" + std::to_string(rank);
    BaseAgentConfig config{agentName, true, false, false, false};

    std::unique_ptr<BaseTransferAgent> agent;
    try
    {
        agent = std::unique_ptr<BaseTransferAgent>(createNixlTransferAgent(&config));
    }
    catch (std::exception const& e)
    {
        printf("[Rank %d] SKIP: could not create NixlTransferAgent: %s\n", rank, e.what());
        MPI_Barrier(MPI_COMM_WORLD);
        return true; // Not a failure, env may not support NIXL
    }

    if (!agent)
    {
        printf("[Rank %d] SKIP: createNixlTransferAgent returned nullptr\n", rank);
        MPI_Barrier(MPI_COMM_WORLD);
        return true;
    }

    printf("[Rank %d] NixlTransferAgent created: '%s'\n", rank, agentName.c_str());

    if (rank == 1)
    {
        // ===== EXPORTER (Rank 1) =====
        // 1. Allocate VMM POSIX FD memory and fill with pattern
        auto alloc = allocateVmmPosixFd(DATA_SIZE, deviceId);
        TLLM_CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(alloc.ptr), FILL_PATTERN, DATA_SIZE));
        TLLM_CUDA_CHECK(cudaDeviceSynchronize());
        printf("[Rank %d] VMM allocated and filled: ptr=0x%lx, size=%zu, pattern=0x%02X\n", rank, alloc.ptr,
            alloc.allocSize, FILL_PATTERN);

        // 2. Register VMM memory with NixlTransferAgent
        //    This triggers detectAndExportFabricHandles internally
        MemoryDescs vmmDescs{
            MemoryType::kVRAM, {MemoryDesc{reinterpret_cast<void*>(alloc.ptr), alloc.allocSize, DEVICE_ID}}};
        agent->registerMemory(vmmDescs);
        printf("[Rank %d] registerMemory done\n", rank);

        // 3. Get AgentDesc (contains NIXL blob + serialized FabricMemInfo with POSIX FD / UDS path)
        auto agentDesc = agent->getLocalAgentDesc();
        auto const& descStr = agentDesc.getBackendAgentDesc();
        printf("[Rank %d] getLocalAgentDesc: %zu bytes\n", rank, descStr.size());

        // 4. Send AgentDesc string to rank 0 via MPI
        int descSize = static_cast<int>(descStr.size());
        MPI_Send(&descSize, 1, MPI_INT, 0, 10, MPI_COMM_WORLD);
        MPI_Send(descStr.data(), descSize, MPI_BYTE, 0, 11, MPI_COMM_WORLD);

        // Also send remote src address and size so rank 0 can build the TransferRequest
        uintptr_t remoteSrcAddr = static_cast<uintptr_t>(alloc.ptr);
        size_t remoteSrcSize = alloc.allocSize;
        MPI_Send(&remoteSrcAddr, sizeof(remoteSrcAddr), MPI_BYTE, 0, 12, MPI_COMM_WORLD);
        MPI_Send(&remoteSrcSize, sizeof(remoteSrcSize), MPI_BYTE, 0, 13, MPI_COMM_WORLD);
        printf("[Rank %d] Sent AgentDesc and address info to rank 0\n", rank);

        // 5. Wait for rank 0 to complete transfer and verification
        MPI_Barrier(MPI_COMM_WORLD);

        // 6. Cleanup
        agent->deregisterMemory(vmmDescs);
        alloc.cleanup();
        printf("[Rank %d] Cleanup done\n", rank);
    }
    else if (rank == 0)
    {
        // ===== IMPORTER / READER (Rank 0) =====
        // 1. Allocate local GPU destination buffer (regular cudaMalloc)
        char* localDst = nullptr;
        TLLM_CUDA_CHECK(cudaMalloc(&localDst, DATA_SIZE));
        TLLM_CUDA_CHECK(cudaMemset(localDst, 0, DATA_SIZE));

        // 2. Register local destination memory with NixlTransferAgent
        MemoryDescs localDescs{MemoryType::kVRAM, {MemoryDesc{localDst, DATA_SIZE, DEVICE_ID}}};
        agent->registerMemory(localDescs);
        printf("[Rank %d] registerMemory done for local dst buffer\n", rank);

        // 3. Receive AgentDesc from rank 1
        int descSize = 0;
        MPI_Recv(&descSize, 1, MPI_INT, 1, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::string descStr(descSize, '\0');
        MPI_Recv(descStr.data(), descSize, MPI_BYTE, 1, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        uintptr_t remoteSrcAddr = 0;
        size_t remoteSrcSize = 0;
        MPI_Recv(&remoteSrcAddr, sizeof(remoteSrcAddr), MPI_BYTE, 1, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&remoteSrcSize, sizeof(remoteSrcSize), MPI_BYTE, 1, 13, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("[Rank %d] Received AgentDesc (%d bytes), remoteSrcAddr=0x%lx, remoteSrcSize=%zu\n", rank, descSize,
            remoteSrcAddr, remoteSrcSize);

        // 4. Load remote agent via AgentDesc path
        //    This parses NIXL metadata blob + FabricMemInfo (POSIX FD + UDS path),
        //    connects to rank 1's UDS server, receives FDs, imports and maps remote memory
        std::string remoteAgentName = "nixl_fd_agent_1";
        AgentDesc remoteAgentDesc{descStr};
        agent->loadRemoteAgent(remoteAgentName, remoteAgentDesc);
        printf("[Rank %d] loadRemoteAgent done for '%s'\n", rank, remoteAgentName.c_str());

        // 5. Submit transfer: read from rank 1's VMM memory into local buffer
        //    NIXL convention: srcDescs = LOCAL, dstDescs = REMOTE
        //    kREAD: pull dstDescs(remote) → srcDescs(local)
        TransferDescs srcDescs{
            MemoryType::kVRAM, {MemoryDesc{reinterpret_cast<uintptr_t>(localDst), DATA_SIZE, DEVICE_ID}}};
        TransferDescs dstDescs{MemoryType::kVRAM, {MemoryDesc{remoteSrcAddr, remoteSrcSize, DEVICE_ID}}};

        TransferRequest readReq{TransferOp::kREAD, srcDescs, dstDescs, remoteAgentName};
        auto status = agent->submitTransferRequests(readReq);
        auto result = status->wait();

        printf("[Rank %d] submitTransferRequests result: %s\n", rank,
            result == TransferState::kSUCCESS ? "SUCCESS"
                                              : (result == TransferState::kFAILURE ? "FAILURE" : "IN_PROGRESS"));

        if (result != TransferState::kSUCCESS)
        {
            printf("[Rank %d] FAIL: transfer did not succeed\n", rank);
            ok = false;
        }
        else
        {
            // 6. Verify transferred data
            std::vector<uint8_t> hostBuf(DATA_SIZE);
            TLLM_CUDA_CHECK(cudaMemcpy(hostBuf.data(), localDst, DATA_SIZE, cudaMemcpyDeviceToHost));

            bool dataOk = true;
            for (size_t i = 0; i < DATA_SIZE; ++i)
            {
                if (hostBuf[i] != FILL_PATTERN)
                {
                    printf("[Rank %d] FAIL: data mismatch at offset %zu: expected 0x%02X, got 0x%02X\n", rank, i,
                        FILL_PATTERN, hostBuf[i]);
                    dataOk = false;
                    break;
                }
            }
            if (dataOk)
            {
                printf("[Rank %d] NixlAgent transfer verification PASSED (all %zu bytes == 0x%02X)\n", rank, DATA_SIZE,
                    FILL_PATTERN);
            }
            ok = ok && dataOk;
        }

        // 7. Cleanup
        agent->deregisterMemory(localDescs);
        agent->invalidateRemoteAgent(remoteAgentName);
        TLLM_CUDA_CHECK(cudaFree(localDst));

        MPI_Barrier(MPI_COMM_WORLD);
        printf("[Rank %d] Cleanup done\n", rank);
    }

    printf("[Rank %d] === NixlAgentPosixFdTransfer: %s ===\n", rank, ok ? "PASSED" : "FAILED");
    return ok;
}

// ============================================================================
// Test: NixlTransferAgent with multiple small segments (batch transfer)
// Simulates the real KV cache scenario: many small discontinuous data blocks
// ============================================================================

bool testNixlAgentPosixFdBatchTransfer(int rank, int worldSize)
{
    printf("[Rank %d] === Test: NixlAgentPosixFdBatchTransfer ===\n", rank);

    if (worldSize < 2)
    {
        printf("[Rank %d] Need at least 2 ranks, skipping\n", rank);
        return true;
    }

    int deviceId = 0;
    TLLM_CUDA_CHECK(cudaSetDevice(deviceId));

    // Check POSIX FD support
    CUdevice device;
    CU_CHECK(cuDeviceGet(&device, deviceId));
    int posixFdSupported = 0;
    CU_CHECK(cuDeviceGetAttribute(
        &posixFdSupported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, device));
    MPI_Bcast(&posixFdSupported, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (!posixFdSupported)
    {
        printf("[Rank %d] POSIX FD not supported, skipping batch test\n", rank);
        return true;
    }

    bool ok = true;
    uint32_t const DEVICE_ID = 0;
    // Simulate KV cache scenario: multiple small segments within one VMM allocation
    size_t const NUM_SEGMENTS = 16;
    size_t const SEGMENT_SIZE = 8 * 1024;                    // 8KB per segment (small, like real KV heads)
    size_t const TOTAL_SIZE = 2 * 1024 * 1024;               // 2MB VMM allocation (much larger than segments)
    size_t const SEGMENT_STRIDE = TOTAL_SIZE / NUM_SEGMENTS; // Spread segments across allocation

    std::string agentName = "nixl_batch_agent_" + std::to_string(rank);
    BaseAgentConfig config{agentName, true, false, false, false};

    std::unique_ptr<BaseTransferAgent> agent;
    try
    {
        agent = std::unique_ptr<BaseTransferAgent>(createNixlTransferAgent(&config));
    }
    catch (std::exception const& e)
    {
        printf("[Rank %d] SKIP: could not create NixlTransferAgent: %s\n", rank, e.what());
        MPI_Barrier(MPI_COMM_WORLD);
        return true;
    }

    if (!agent)
    {
        printf("[Rank %d] SKIP: createNixlTransferAgent returned nullptr\n", rank);
        MPI_Barrier(MPI_COMM_WORLD);
        return true;
    }

    printf("[Rank %d] NixlTransferAgent created: '%s'\n", rank, agentName.c_str());

    if (rank == 1)
    {
        // ===== EXPORTER =====
        auto alloc = allocateVmmPosixFd(TOTAL_SIZE, deviceId);
        printf("[Rank %d] VMM allocated: ptr=0x%lx, size=%zu\n", rank, alloc.ptr, alloc.allocSize);

        // Fill each segment with a distinct pattern
        for (size_t seg = 0; seg < NUM_SEGMENTS; ++seg)
        {
            uint8_t pattern = static_cast<uint8_t>(0x10 + seg);
            TLLM_CUDA_CHECK(
                cudaMemset(reinterpret_cast<void*>(alloc.ptr + seg * SEGMENT_STRIDE), pattern, SEGMENT_SIZE));
        }
        TLLM_CUDA_CHECK(cudaDeviceSynchronize());
        printf("[Rank %d] Filled %zu segments (each %zu bytes, stride %zu)\n", rank, NUM_SEGMENTS, SEGMENT_SIZE,
            SEGMENT_STRIDE);

        // Register the entire VMM allocation
        MemoryDescs vmmDescs{
            MemoryType::kVRAM, {MemoryDesc{reinterpret_cast<void*>(alloc.ptr), alloc.allocSize, DEVICE_ID}}};
        agent->registerMemory(vmmDescs);

        // Get and send AgentDesc
        auto agentDesc = agent->getLocalAgentDesc();
        auto const& descStr = agentDesc.getBackendAgentDesc();
        int descSize = static_cast<int>(descStr.size());
        MPI_Send(&descSize, 1, MPI_INT, 0, 20, MPI_COMM_WORLD);
        MPI_Send(descStr.data(), descSize, MPI_BYTE, 0, 21, MPI_COMM_WORLD);

        // Send base address
        uintptr_t baseAddr = static_cast<uintptr_t>(alloc.ptr);
        MPI_Send(&baseAddr, sizeof(baseAddr), MPI_BYTE, 0, 22, MPI_COMM_WORLD);
        printf("[Rank %d] Sent AgentDesc and base address\n", rank);

        MPI_Barrier(MPI_COMM_WORLD);

        agent->deregisterMemory(vmmDescs);
        alloc.cleanup();
        printf("[Rank %d] Cleanup done\n", rank);
    }
    else if (rank == 0)
    {
        // ===== IMPORTER =====
        // Allocate local destination for each segment
        char* localDst = nullptr;
        size_t localTotalSize = NUM_SEGMENTS * SEGMENT_SIZE;
        TLLM_CUDA_CHECK(cudaMalloc(&localDst, localTotalSize));
        TLLM_CUDA_CHECK(cudaMemset(localDst, 0, localTotalSize));

        // Register local memory
        MemoryDescs localDescs{MemoryType::kVRAM, {MemoryDesc{localDst, localTotalSize, DEVICE_ID}}};
        agent->registerMemory(localDescs);

        // Receive AgentDesc
        int descSize = 0;
        MPI_Recv(&descSize, 1, MPI_INT, 1, 20, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::string descStr(descSize, '\0');
        MPI_Recv(descStr.data(), descSize, MPI_BYTE, 1, 21, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        uintptr_t remoteBaseAddr = 0;
        MPI_Recv(&remoteBaseAddr, sizeof(remoteBaseAddr), MPI_BYTE, 1, 22, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("[Rank %d] Received AgentDesc (%d bytes), remoteBase=0x%lx\n", rank, descSize, remoteBaseAddr);

        // Load remote agent
        std::string remoteAgentName = "nixl_batch_agent_1";
        AgentDesc remoteAgentDesc{descStr};
        agent->loadRemoteAgent(remoteAgentName, remoteAgentDesc);
        printf("[Rank %d] loadRemoteAgent done\n", rank);

        // Build multi-segment TransferRequest
        // NIXL convention: srcDescs = LOCAL, dstDescs = REMOTE
        // kREAD: pull dstDescs(remote) → srcDescs(local)
        std::vector<MemoryDesc> localSegments;  // srcDescs = LOCAL
        std::vector<MemoryDesc> remoteSegments; // dstDescs = REMOTE
        for (size_t seg = 0; seg < NUM_SEGMENTS; ++seg)
        {
            uintptr_t localAddr = reinterpret_cast<uintptr_t>(localDst) + seg * SEGMENT_SIZE;
            uintptr_t remoteAddr = remoteBaseAddr + seg * SEGMENT_STRIDE;
            localSegments.emplace_back(localAddr, SEGMENT_SIZE, DEVICE_ID);
            remoteSegments.emplace_back(remoteAddr, SEGMENT_SIZE, DEVICE_ID);
        }

        TransferDescs srcDescs{MemoryType::kVRAM, std::move(localSegments)};
        TransferDescs dstDescs{MemoryType::kVRAM, std::move(remoteSegments)};

        printf("[Rank %d] Submitting batch transfer: %zu segments x %zu bytes\n", rank, NUM_SEGMENTS, SEGMENT_SIZE);
        TransferRequest readReq{TransferOp::kREAD, srcDescs, dstDescs, remoteAgentName};
        auto status = agent->submitTransferRequests(readReq);
        auto result = status->wait();

        printf(
            "[Rank %d] Batch transfer result: %s\n", rank, result == TransferState::kSUCCESS ? "SUCCESS" : "FAILURE");

        if (result != TransferState::kSUCCESS)
        {
            printf("[Rank %d] FAIL: batch transfer did not succeed\n", rank);
            ok = false;
        }
        else
        {
            // Verify each segment has its unique pattern
            std::vector<uint8_t> hostBuf(localTotalSize);
            TLLM_CUDA_CHECK(cudaMemcpy(hostBuf.data(), localDst, localTotalSize, cudaMemcpyDeviceToHost));

            bool dataOk = true;
            for (size_t seg = 0; seg < NUM_SEGMENTS && dataOk; ++seg)
            {
                uint8_t expectedPattern = static_cast<uint8_t>(0x10 + seg);
                for (size_t i = 0; i < SEGMENT_SIZE; ++i)
                {
                    if (hostBuf[seg * SEGMENT_SIZE + i] != expectedPattern)
                    {
                        printf("[Rank %d] FAIL: seg=%zu offset=%zu: expected 0x%02X, got 0x%02X\n", rank, seg, i,
                            expectedPattern, hostBuf[seg * SEGMENT_SIZE + i]);
                        dataOk = false;
                        break;
                    }
                }
            }
            if (dataOk)
            {
                printf("[Rank %d] Batch transfer verification PASSED (%zu segments, each with unique pattern)\n", rank,
                    NUM_SEGMENTS);
            }
            ok = ok && dataOk;
        }

        agent->deregisterMemory(localDescs);
        agent->invalidateRemoteAgent(remoteAgentName);
        TLLM_CUDA_CHECK(cudaFree(localDst));

        MPI_Barrier(MPI_COMM_WORLD);
        printf("[Rank %d] Cleanup done\n", rank);
    }

    printf("[Rank %d] === NixlAgentPosixFdBatchTransfer: %s ===\n", rank, ok ? "PASSED" : "FAILED");
    return ok;
}

// ============================================================================
// Test: NixlTransferAgent WRITE to remote POSIX FD VMM memory
// Rank 0 pushes local data to Rank 1's VMM memory via kWRITE
// ============================================================================

bool testNixlAgentPosixFdWrite(int rank, int worldSize)
{
    printf("[Rank %d] === Test: NixlAgentPosixFdWrite ===\n", rank);

    if (worldSize < 2)
    {
        printf("[Rank %d] Need at least 2 ranks, skipping\n", rank);
        return true;
    }

    int deviceId = 0;
    TLLM_CUDA_CHECK(cudaSetDevice(deviceId));

    // Check POSIX FD support
    CUdevice device;
    CU_CHECK(cuDeviceGet(&device, deviceId));
    int posixFdSupported = 0;
    CU_CHECK(cuDeviceGetAttribute(
        &posixFdSupported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, device));
    MPI_Bcast(&posixFdSupported, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (!posixFdSupported)
    {
        printf("[Rank %d] POSIX FD not supported, skipping write test\n", rank);
        return true;
    }

    bool ok = true;
    size_t const DATA_SIZE = 4 * 1024 * 1024; // 4MB
    uint8_t const WRITE_PATTERN = 0xBE;
    uint32_t const DEVICE_ID = 0;

    // Create NixlTransferAgent
    std::string agentName = "nixl_write_agent_" + std::to_string(rank);
    BaseAgentConfig config{agentName, true, false, false, false};

    std::unique_ptr<BaseTransferAgent> agent;
    try
    {
        agent = std::unique_ptr<BaseTransferAgent>(createNixlTransferAgent(&config));
    }
    catch (std::exception const& e)
    {
        printf("[Rank %d] SKIP: could not create NixlTransferAgent: %s\n", rank, e.what());
        MPI_Barrier(MPI_COMM_WORLD);
        return true;
    }

    if (!agent)
    {
        printf("[Rank %d] SKIP: createNixlTransferAgent returned nullptr\n", rank);
        MPI_Barrier(MPI_COMM_WORLD);
        return true;
    }

    printf("[Rank %d] NixlTransferAgent created: '%s'\n", rank, agentName.c_str());

    if (rank == 1)
    {
        // ===== RECEIVER (Rank 1) =====
        // 1. Allocate VMM POSIX FD memory, zero-initialize
        auto alloc = allocateVmmPosixFd(DATA_SIZE, deviceId);
        TLLM_CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(alloc.ptr), 0x00, DATA_SIZE));
        TLLM_CUDA_CHECK(cudaDeviceSynchronize());
        printf("[Rank %d] VMM allocated and zeroed: ptr=0x%llx, size=%zu\n", rank,
            static_cast<unsigned long long>(alloc.ptr), alloc.allocSize);

        // 2. Register VMM memory
        MemoryDescs vmmDescs{
            MemoryType::kVRAM, {MemoryDesc{reinterpret_cast<void*>(alloc.ptr), alloc.allocSize, DEVICE_ID}}};
        agent->registerMemory(vmmDescs);
        printf("[Rank %d] registerMemory done\n", rank);

        // 3. Get AgentDesc and send to Rank 0
        auto agentDesc = agent->getLocalAgentDesc();
        auto const& descStr = agentDesc.getBackendAgentDesc();
        int descSize = static_cast<int>(descStr.size());
        MPI_Send(&descSize, 1, MPI_INT, 0, 30, MPI_COMM_WORLD);
        MPI_Send(descStr.data(), descSize, MPI_BYTE, 0, 31, MPI_COMM_WORLD);

        // Send remote VMM address
        uintptr_t remoteVmmAddr = static_cast<uintptr_t>(alloc.ptr);
        size_t remoteVmmSize = alloc.allocSize;
        MPI_Send(&remoteVmmAddr, sizeof(remoteVmmAddr), MPI_BYTE, 0, 32, MPI_COMM_WORLD);
        MPI_Send(&remoteVmmSize, sizeof(remoteVmmSize), MPI_BYTE, 0, 33, MPI_COMM_WORLD);
        printf("[Rank %d] Sent AgentDesc and VMM address to rank 0\n", rank);

        // 4. Wait for Rank 0 to finish writing
        MPI_Barrier(MPI_COMM_WORLD);

        // 5. Verify that VMM memory now contains the WRITE_PATTERN
        TLLM_CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<uint8_t> hostBuf(DATA_SIZE);
        TLLM_CUDA_CHECK(
            cudaMemcpy(hostBuf.data(), reinterpret_cast<void*>(alloc.ptr), DATA_SIZE, cudaMemcpyDeviceToHost));

        bool dataOk = true;
        for (size_t i = 0; i < DATA_SIZE; ++i)
        {
            if (hostBuf[i] != WRITE_PATTERN)
            {
                printf("[Rank %d] FAIL: data mismatch at offset %zu: expected 0x%02X, got 0x%02X\n", rank, i,
                    WRITE_PATTERN, hostBuf[i]);
                dataOk = false;
                break;
            }
        }
        if (dataOk)
        {
            printf("[Rank %d] WRITE verification PASSED (all %zu bytes == 0x%02X)\n", rank, DATA_SIZE, WRITE_PATTERN);
        }
        ok = ok && dataOk;

        // 6. Cleanup
        agent->deregisterMemory(vmmDescs);
        alloc.cleanup();
        printf("[Rank %d] Cleanup done\n", rank);
    }
    else if (rank == 0)
    {
        // ===== WRITER (Rank 0) =====
        // 1. Allocate local GPU source buffer and fill with pattern
        char* localSrc = nullptr;
        TLLM_CUDA_CHECK(cudaMalloc(&localSrc, DATA_SIZE));
        TLLM_CUDA_CHECK(cudaMemset(localSrc, WRITE_PATTERN, DATA_SIZE));
        TLLM_CUDA_CHECK(cudaDeviceSynchronize());
        printf("[Rank %d] Local src allocated and filled with 0x%02X\n", rank, WRITE_PATTERN);

        // 2. Register local memory
        MemoryDescs localDescs{MemoryType::kVRAM, {MemoryDesc{localSrc, DATA_SIZE, DEVICE_ID}}};
        agent->registerMemory(localDescs);
        printf("[Rank %d] registerMemory done for local src buffer\n", rank);

        // 3. Receive AgentDesc from Rank 1
        int descSize = 0;
        MPI_Recv(&descSize, 1, MPI_INT, 1, 30, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::string descStr(descSize, '\0');
        MPI_Recv(descStr.data(), descSize, MPI_BYTE, 1, 31, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        uintptr_t remoteVmmAddr = 0;
        size_t remoteVmmSize = 0;
        MPI_Recv(&remoteVmmAddr, sizeof(remoteVmmAddr), MPI_BYTE, 1, 32, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&remoteVmmSize, sizeof(remoteVmmSize), MPI_BYTE, 1, 33, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("[Rank %d] Received AgentDesc (%d bytes), remoteVmmAddr=0x%lx, remoteVmmSize=%zu\n", rank, descSize,
            remoteVmmAddr, remoteVmmSize);

        // 4. Load remote agent
        std::string remoteAgentName = "nixl_write_agent_1";
        AgentDesc remoteAgentDesc{descStr};
        agent->loadRemoteAgent(remoteAgentName, remoteAgentDesc);
        printf("[Rank %d] loadRemoteAgent done for '%s'\n", rank, remoteAgentName.c_str());

        // 5. Submit WRITE transfer: push local data to remote VMM memory
        //    NIXL convention: srcDescs = LOCAL, dstDescs = REMOTE
        //    kWRITE: push srcDescs(local) → dstDescs(remote)
        TransferDescs srcDescs{
            MemoryType::kVRAM, {MemoryDesc{reinterpret_cast<uintptr_t>(localSrc), DATA_SIZE, DEVICE_ID}}};
        TransferDescs dstDescs{MemoryType::kVRAM, {MemoryDesc{remoteVmmAddr, remoteVmmSize, DEVICE_ID}}};

        TransferRequest writeReq{TransferOp::kWRITE, srcDescs, dstDescs, remoteAgentName};
        auto status = agent->submitTransferRequests(writeReq);
        auto result = status->wait();

        printf("[Rank %d] WRITE submitTransferRequests result: %s\n", rank,
            result == TransferState::kSUCCESS ? "SUCCESS"
                                              : (result == TransferState::kFAILURE ? "FAILURE" : "IN_PROGRESS"));

        if (result != TransferState::kSUCCESS)
        {
            printf("[Rank %d] FAIL: WRITE transfer did not succeed\n", rank);
            ok = false;
        }
        else
        {
            printf("[Rank %d] WRITE transfer completed successfully\n", rank);
        }

        // 6. Signal Rank 1 that write is done
        TLLM_CUDA_CHECK(cudaDeviceSynchronize());
        MPI_Barrier(MPI_COMM_WORLD);

        // 7. Cleanup
        agent->deregisterMemory(localDescs);
        agent->invalidateRemoteAgent(remoteAgentName);
        TLLM_CUDA_CHECK(cudaFree(localSrc));
        printf("[Rank %d] Cleanup done\n", rank);
    }

    printf("[Rank %d] === NixlAgentPosixFdWrite: %s ===\n", rank, ok ? "PASSED" : "FAILED");
    return ok;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    printf("[Rank %d] Process started (worldSize=%d)\n", rank, worldSize);

    // Initialize CUDA
    CU_CHECK(cuInit(0));
    TLLM_CUDA_CHECK(cudaSetDevice(0));

    bool allPassed = true;

    // Run tests
    MPI_Barrier(MPI_COMM_WORLD);
    allPassed = testPosixFdDetection(rank) && allPassed;

    MPI_Barrier(MPI_COMM_WORLD);
    allPassed = testPosixFdSerialization(rank) && allPassed;

    MPI_Barrier(MPI_COMM_WORLD);
    allPassed = testPosixFdTransfer(rank, worldSize) && allPassed;

    MPI_Barrier(MPI_COMM_WORLD);
    allPassed = testPosixFdMultiChunk(rank, worldSize) && allPassed;

    MPI_Barrier(MPI_COMM_WORLD);
    allPassed = testNixlAgentPosixFdTransfer(rank, worldSize) && allPassed;

    MPI_Barrier(MPI_COMM_WORLD);
    allPassed = testNixlAgentPosixFdBatchTransfer(rank, worldSize) && allPassed;

    MPI_Barrier(MPI_COMM_WORLD);
    allPassed = testNixlAgentPosixFdWrite(rank, worldSize) && allPassed;

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("\n=== All tests %s ===\n", allPassed ? "PASSED" : "FAILED");
    }

    MPI_Finalize();
    return allPassed ? 0 : 1;
}

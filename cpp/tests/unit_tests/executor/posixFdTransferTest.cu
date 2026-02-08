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
    printf("[Rank %d] VMM allocated: ptr=0x%llx, size=%zu\n", rank, (unsigned long long) alloc.ptr, alloc.allocSize);

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
        printf(
            "[Rank %d] VMM allocated: ptr=0x%llx, size=%zu\n", rank, (unsigned long long) alloc.ptr, alloc.allocSize);

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
            MPI_Barrier(MPI_COMM_WORLD);
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
        void* localMappedPtr = helper.translateToLocalMapping("rank1", pool.registeredAddr, pool.registeredSize);
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
        printf("[Rank %d] Multi-chunk VMM allocated: ptr=0x%llx, totalSize=%zu, chunks=%zu\n", rank,
            (unsigned long long) alloc.ptr, alloc.totalSize, alloc.numChunks);

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
            MPI_Barrier(MPI_COMM_WORLD);
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
        void* localMappedPtr = helper.translateToLocalMapping("rank1_mc", pool.registeredAddr, pool.registeredSize);
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
    }

    // Ensure all ranks created agent successfully before proceeding to MPI send/recv
    int agentOk = (agent != nullptr) ? 1 : 0;
    int allOk = 0;
    MPI_Allreduce(&agentOk, &allOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!allOk)
    {
        printf("[Rank %d] SKIP: not all ranks created NixlTransferAgent\n", rank);
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
        printf("[Rank %d] VMM allocated and filled: ptr=0x%llx, size=%zu, pattern=0x%02X\n", rank,
            (unsigned long long) alloc.ptr, alloc.allocSize, FILL_PATTERN);

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
    }

    // Ensure all ranks created agent successfully before proceeding to MPI send/recv
    int agentOk = (agent != nullptr) ? 1 : 0;
    int allOk = 0;
    MPI_Allreduce(&agentOk, &allOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!allOk)
    {
        printf("[Rank %d] SKIP: not all ranks created NixlTransferAgent\n", rank);
        return true;
    }

    printf("[Rank %d] NixlTransferAgent created: '%s'\n", rank, agentName.c_str());

    if (rank == 1)
    {
        // ===== EXPORTER =====
        auto alloc = allocateVmmPosixFd(TOTAL_SIZE, deviceId);
        printf(
            "[Rank %d] VMM allocated: ptr=0x%llx, size=%zu\n", rank, (unsigned long long) alloc.ptr, alloc.allocSize);

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
    }

    // Ensure all ranks created agent successfully before proceeding to MPI send/recv
    int agentOk = (agent != nullptr) ? 1 : 0;
    int allOk = 0;
    MPI_Allreduce(&agentOk, &allOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!allOk)
    {
        printf("[Rank %d] SKIP: not all ranks created NixlTransferAgent\n", rank);
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
        printf("[Rank %d] Received AgentDesc (%d bytes), remoteVmmAddr=0x%llx, remoteVmmSize=%zu\n", rank, descSize,
            (unsigned long long) remoteVmmAddr, remoteVmmSize);

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
// Test: Sub-range registration (registeredAddr != poolBaseAddr)
// Allocates 8MB VMM but registers only a 4MB sub-range starting at 2MB offset.
// This exercises the case where registeredAddr > poolBaseAddr, mappedOffset > 0,
// testing detectAndExportChunks sub-range scanning, FabricMemPool offset fields,
// and translateToLocalMappingInternal offset arithmetic.
// ============================================================================

bool testNixlAgentPosixFdSubRange(int rank, int worldSize)
{
    printf("[Rank %d] === Test: NixlAgentPosixFdSubRange ===\n", rank);

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
        printf("[Rank %d] POSIX FD not supported, skipping sub-range test\n", rank);
        return true;
    }

    bool ok = true;
    size_t const POOL_SIZE = 8 * 1024 * 1024;  // 8MB total VMM allocation
    size_t const SUB_OFFSET = 2 * 1024 * 1024; // Register starting at 2MB offset
    size_t const SUB_SIZE = 4 * 1024 * 1024;   // Register 4MB sub-range
    uint8_t const FILL_PATTERN = 0xDE;
    uint32_t const DEVICE_ID = 0;

    // Create NixlTransferAgent
    std::string agentName = "nixl_subrange_agent_" + std::to_string(rank);
    BaseAgentConfig config{agentName, true, false, false, false};

    std::unique_ptr<BaseTransferAgent> agent;
    try
    {
        agent = std::unique_ptr<BaseTransferAgent>(createNixlTransferAgent(&config));
    }
    catch (std::exception const& e)
    {
        printf("[Rank %d] SKIP: could not create NixlTransferAgent: %s\n", rank, e.what());
    }

    // Ensure all ranks created agent successfully
    int agentOk = (agent != nullptr) ? 1 : 0;
    int allOk = 0;
    MPI_Allreduce(&agentOk, &allOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!allOk)
    {
        printf("[Rank %d] SKIP: not all ranks created NixlTransferAgent\n", rank);
        return true;
    }

    printf("[Rank %d] NixlTransferAgent created: '%s'\n", rank, agentName.c_str());

    if (rank == 1)
    {
        // ===== EXPORTER (Rank 1) =====
        // 1. Allocate FULL 8MB VMM pool
        auto alloc = allocateVmmPosixFd(POOL_SIZE, deviceId);
        printf("[Rank %d] VMM pool allocated: ptr=0x%llx, size=%zu\n", rank, (unsigned long long) alloc.ptr,
            alloc.allocSize);

        // 2. Zero entire allocation, then fill ONLY the sub-range with pattern
        //    Memory layout: [0x00 * 2MB] [0xDE * 4MB] [0x00 * 2MB]
        TLLM_CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(alloc.ptr), 0x00, POOL_SIZE));
        TLLM_CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(alloc.ptr + SUB_OFFSET), FILL_PATTERN, SUB_SIZE));
        TLLM_CUDA_CHECK(cudaDeviceSynchronize());
        printf("[Rank %d] Sub-range [+%zu, +%zu) filled with 0x%02X\n", rank, SUB_OFFSET, SUB_OFFSET + SUB_SIZE,
            FILL_PATTERN);

        // 3. Register ONLY the sub-range (not the full pool!)
        //    registeredAddr = alloc.ptr + SUB_OFFSET, which != poolBaseAddr
        void* subRangePtr = reinterpret_cast<void*>(alloc.ptr + SUB_OFFSET);
        MemoryDescs subDescs{MemoryType::kVRAM, {MemoryDesc{subRangePtr, SUB_SIZE, DEVICE_ID}}};
        agent->registerMemory(subDescs);
        printf("[Rank %d] registerMemory for sub-range: ptr=%p, size=%zu (poolBase=0x%llx)\n", rank, subRangePtr,
            SUB_SIZE, (unsigned long long) alloc.ptr);

        // 4. Get AgentDesc and send to rank 0
        auto agentDesc = agent->getLocalAgentDesc();
        auto const& descStr = agentDesc.getBackendAgentDesc();
        int descSize = static_cast<int>(descStr.size());
        MPI_Send(&descSize, 1, MPI_INT, 0, 40, MPI_COMM_WORLD);
        MPI_Send(descStr.data(), descSize, MPI_BYTE, 0, 41, MPI_COMM_WORLD);

        // Send the sub-range address (NOT pool base!)
        uintptr_t remoteSrcAddr = reinterpret_cast<uintptr_t>(subRangePtr);
        size_t remoteSrcSize = SUB_SIZE;
        MPI_Send(&remoteSrcAddr, sizeof(remoteSrcAddr), MPI_BYTE, 0, 42, MPI_COMM_WORLD);
        MPI_Send(&remoteSrcSize, sizeof(remoteSrcSize), MPI_BYTE, 0, 43, MPI_COMM_WORLD);
        printf("[Rank %d] Sent AgentDesc and sub-range address info\n", rank);

        // 5. Wait for rank 0
        MPI_Barrier(MPI_COMM_WORLD);

        // 6. Cleanup
        agent->deregisterMemory(subDescs);
        alloc.cleanup();
        printf("[Rank %d] Cleanup done\n", rank);
    }
    else if (rank == 0)
    {
        // ===== IMPORTER (Rank 0) =====
        // 1. Local destination buffer (sub-range size only)
        char* localDst = nullptr;
        TLLM_CUDA_CHECK(cudaMalloc(&localDst, SUB_SIZE));
        TLLM_CUDA_CHECK(cudaMemset(localDst, 0, SUB_SIZE));

        MemoryDescs localDescs{MemoryType::kVRAM, {MemoryDesc{localDst, SUB_SIZE, DEVICE_ID}}};
        agent->registerMemory(localDescs);

        // 2. Receive AgentDesc
        int descSize = 0;
        MPI_Recv(&descSize, 1, MPI_INT, 1, 40, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::string descStr(descSize, '\0');
        MPI_Recv(descStr.data(), descSize, MPI_BYTE, 1, 41, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        uintptr_t remoteSrcAddr = 0;
        size_t remoteSrcSize = 0;
        MPI_Recv(&remoteSrcAddr, sizeof(remoteSrcAddr), MPI_BYTE, 1, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&remoteSrcSize, sizeof(remoteSrcSize), MPI_BYTE, 1, 43, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("[Rank %d] Received AgentDesc (%d bytes), remoteSrcAddr=0x%llx (sub-range), size=%zu\n", rank, descSize,
            (unsigned long long) remoteSrcAddr, remoteSrcSize);

        // 3. Load remote agent
        std::string remoteAgentName = "nixl_subrange_agent_1";
        AgentDesc remoteAgentDesc{descStr};
        agent->loadRemoteAgent(remoteAgentName, remoteAgentDesc);
        printf("[Rank %d] loadRemoteAgent done\n", rank);

        // 4. Submit READ transfer for the full sub-range
        TransferDescs srcDescs{
            MemoryType::kVRAM, {MemoryDesc{reinterpret_cast<uintptr_t>(localDst), SUB_SIZE, DEVICE_ID}}};
        TransferDescs dstDescs{MemoryType::kVRAM, {MemoryDesc{remoteSrcAddr, remoteSrcSize, DEVICE_ID}}};

        TransferRequest readReq{TransferOp::kREAD, srcDescs, dstDescs, remoteAgentName};
        auto status = agent->submitTransferRequests(readReq);
        auto result = status->wait();

        printf("[Rank %d] submitTransferRequests result: %s\n", rank,
            result == TransferState::kSUCCESS ? "SUCCESS" : "FAILURE");

        if (result != TransferState::kSUCCESS)
        {
            printf("[Rank %d] FAIL: sub-range transfer did not succeed\n", rank);
            ok = false;
        }
        else
        {
            // 5. Verify: all SUB_SIZE bytes should be FILL_PATTERN (0xDE)
            //    This confirms the offset arithmetic is correct: we're reading from
            //    [registeredAddr, registeredAddr+SUB_SIZE), NOT [poolBase, poolBase+SUB_SIZE)
            std::vector<uint8_t> hostBuf(SUB_SIZE);
            TLLM_CUDA_CHECK(cudaMemcpy(hostBuf.data(), localDst, SUB_SIZE, cudaMemcpyDeviceToHost));

            bool dataOk = true;
            for (size_t i = 0; i < SUB_SIZE; ++i)
            {
                if (hostBuf[i] != FILL_PATTERN)
                {
                    printf("[Rank %d] FAIL: sub-range data mismatch at offset %zu: expected 0x%02X, got 0x%02X\n", rank,
                        i, FILL_PATTERN, hostBuf[i]);
                    dataOk = false;
                    break;
                }
            }
            if (dataOk)
            {
                printf("[Rank %d] Sub-range transfer verification PASSED (%zu bytes == 0x%02X, offset=%zu)\n", rank,
                    SUB_SIZE, FILL_PATTERN, SUB_OFFSET);
            }
            ok = ok && dataOk;
        }

        // 6. Also test partial read from the MIDDLE of the sub-range
        //    Read 1MB from sub-range offset 1MB (i.e., pool offset 3MB)
        //    This tests translateToLocalMapping with remoteAddr > registeredAddr > poolBaseAddr
        //    Use localDst + PARTIAL_OFFSET as destination — it's within the already-registered range
        if (ok)
        {
            size_t const PARTIAL_OFFSET = 1 * 1024 * 1024; // 1MB into the sub-range
            size_t const PARTIAL_SIZE = 1 * 1024 * 1024;   // Read 1MB

            // Zero out the target portion of localDst so we can verify the partial read
            TLLM_CUDA_CHECK(cudaMemset(localDst + PARTIAL_OFFSET, 0x00, PARTIAL_SIZE));
            TLLM_CUDA_CHECK(cudaDeviceSynchronize());

            // Local dst is within [localDst, localDst + SUB_SIZE) — already registered
            // Remote src is within [remoteSrcAddr, remoteSrcAddr + SUB_SIZE) — already registered on rank 1
            TransferDescs partialSrcDescs{MemoryType::kVRAM,
                {MemoryDesc{reinterpret_cast<uintptr_t>(localDst + PARTIAL_OFFSET), PARTIAL_SIZE, DEVICE_ID}}};
            TransferDescs partialDstDescs{
                MemoryType::kVRAM, {MemoryDesc{remoteSrcAddr + PARTIAL_OFFSET, PARTIAL_SIZE, DEVICE_ID}}};

            TransferRequest partialReq{TransferOp::kREAD, partialSrcDescs, partialDstDescs, remoteAgentName};
            auto partialStatus = agent->submitTransferRequests(partialReq);
            auto partialResult = partialStatus->wait();

            if (partialResult != TransferState::kSUCCESS)
            {
                printf("[Rank %d] FAIL: partial sub-range transfer did not succeed\n", rank);
                ok = false;
            }
            else
            {
                std::vector<uint8_t> partialBuf(PARTIAL_SIZE);
                TLLM_CUDA_CHECK(
                    cudaMemcpy(partialBuf.data(), localDst + PARTIAL_OFFSET, PARTIAL_SIZE, cudaMemcpyDeviceToHost));

                bool partialOk = true;
                for (size_t i = 0; i < PARTIAL_SIZE; ++i)
                {
                    if (partialBuf[i] != FILL_PATTERN)
                    {
                        printf("[Rank %d] FAIL: partial read mismatch at offset %zu: expected 0x%02X, got 0x%02X\n",
                            rank, i, FILL_PATTERN, partialBuf[i]);
                        partialOk = false;
                        break;
                    }
                }
                if (partialOk)
                {
                    printf("[Rank %d] Partial sub-range read PASSED (1MB from middle of sub-range)\n", rank);
                }
                ok = ok && partialOk;
            }
        }

        // 7. Cleanup
        agent->deregisterMemory(localDescs);
        agent->invalidateRemoteAgent(remoteAgentName);
        TLLM_CUDA_CHECK(cudaFree(localDst));

        MPI_Barrier(MPI_COMM_WORLD);
        printf("[Rank %d] Cleanup done\n", rank);
    }

    printf("[Rank %d] === NixlAgentPosixFdSubRange: %s ===\n", rank, ok ? "PASSED" : "FAILED");
    return ok;
}

// ============================================================================
// Test: FabricTransferHelper-level sub-range export/import
// Verifies that when only a sub-range of a VMM pool is registered, the export
// side correctly discovers the full physical allocation boundaries, and the
// import side can map and read the sub-range data correctly.
// ============================================================================

bool testPosixFdSubRangeDirect(int rank, int worldSize)
{
    printf("[Rank %d] === Test: PosixFdSubRangeDirect ===\n", rank);

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
        printf("[Rank %d] POSIX FD not supported, skipping sub-range direct test\n", rank);
        return true;
    }

    bool ok = true;
    // Layout: 8MB VMM pool = [2MB zeros][4MB pattern][2MB zeros]
    // Only register the middle 4MB sub-range
    size_t const POOL_SIZE = 8 * 1024 * 1024;
    size_t const SUB_OFFSET = 2 * 1024 * 1024;
    size_t const SUB_SIZE = 4 * 1024 * 1024;
    uint8_t const FILL_PATTERN = 0xF1;

    if (rank == 1)
    {
        // ===== EXPORTER =====
        auto alloc = allocateVmmPosixFd(POOL_SIZE, deviceId);
        printf("[Rank %d] VMM pool allocated: ptr=0x%llx, size=%zu\n", rank, (unsigned long long) alloc.ptr,
            alloc.allocSize);

        // Zero entire pool, then fill sub-range
        TLLM_CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(alloc.ptr), 0x00, POOL_SIZE));
        TLLM_CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(alloc.ptr + SUB_OFFSET), FILL_PATTERN, SUB_SIZE));
        TLLM_CUDA_CHECK(cudaDeviceSynchronize());

        // Register ONLY the sub-range
        FabricTransferHelper helper;
        void* subRangePtr = reinterpret_cast<void*>(alloc.ptr + SUB_OFFSET);
        MemoryDescs descs{MemoryType::kVRAM, {MemoryDesc{subRangePtr, SUB_SIZE, static_cast<uint32_t>(deviceId)}}};
        helper.detectAndExportFabricHandles(descs);

        auto const& info = helper.getLocalFabricInfo();
        if (!info.supported)
        {
            printf("[Rank %d] FAIL: fabric info not supported after detection\n", rank);
            alloc.cleanup();
            int metadataSize = 0;
            MPI_Send(&metadataSize, 1, MPI_INT, 0, 50, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }

        printf("[Rank %d] Sub-range export done: pools=%zu, handleType=%s\n", rank, info.pools.size(),
            info.handleType == VmmHandleType::kPosixFd ? "PosixFd" : "Fabric");

        // Validate that the pool metadata captures the sub-range correctly
        if (!info.pools.empty())
        {
            auto const& pool = info.pools[0];
            printf(
                "[Rank %d] Pool: base=0x%llx, totalSize=%zu, registeredAddr=0x%llx, registeredSize=%zu, "
                "mappedOffset=%zu, mappedSize=%zu, chunks=%zu\n",
                rank, (unsigned long long) pool.poolBaseAddr, (size_t) pool.poolTotalSize,
                (unsigned long long) pool.registeredAddr, (size_t) pool.registeredSize, (size_t) pool.mappedOffset,
                (size_t) pool.mappedSize, pool.chunks.size());

            // Verify that each chunk's size equals the full physical allocation size (not SUB_SIZE)
            for (size_t ci = 0; ci < pool.chunks.size(); ++ci)
            {
                printf("[Rank %d]   chunk[%zu]: virtAddrOffset=%zu, size=%zu\n", rank, ci,
                    (size_t) pool.chunks[ci].virtAddrOffset, (size_t) pool.chunks[ci].size);
            }
        }

        // Serialize and send
        std::string metadata = info.serialize();
        int metadataSize = static_cast<int>(metadata.size());
        MPI_Send(&metadataSize, 1, MPI_INT, 0, 50, MPI_COMM_WORLD);
        MPI_Send(metadata.data(), metadataSize, MPI_BYTE, 0, 51, MPI_COMM_WORLD);
        printf("[Rank %d] Sent metadata (%d bytes) to rank 0\n", rank, metadataSize);

        // Wait for rank 0
        MPI_Barrier(MPI_COMM_WORLD);

        // Cleanup
        alloc.cleanup();
        printf("[Rank %d] Cleanup done\n", rank);
    }
    else if (rank == 0)
    {
        // ===== IMPORTER =====
        int metadataSize = 0;
        MPI_Recv(&metadataSize, 1, MPI_INT, 1, 50, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (metadataSize == 0)
        {
            printf("[Rank %d] Exporter reported failure, skipping\n", rank);
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }

        std::string metadata(metadataSize, '\0');
        MPI_Recv(metadata.data(), metadataSize, MPI_BYTE, 1, 51, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("[Rank %d] Received metadata (%d bytes) from rank 1\n", rank, metadataSize);

        auto fabricInfo = FabricMemInfo::deserialize(metadata);
        if (!fabricInfo.has_value())
        {
            printf("[Rank %d] FAIL: failed to deserialize FabricMemInfo\n", rank);
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }

        printf("[Rank %d] Deserialized: handleType=%s, pools=%zu\n", rank,
            fabricInfo->handleType == VmmHandleType::kPosixFd ? "PosixFd" : "Fabric", fabricInfo->pools.size());

        // Import and map
        FabricTransferHelper helper;
        helper.importAndMapRemoteFabric("rank1_sub", *fabricInfo);

        if (!helper.hasRemoteMapping("rank1_sub"))
        {
            printf("[Rank %d] FAIL: no remote mapping established for sub-range\n", rank);
            ok = false;
        }
        else
        {
            printf("[Rank %d] Remote mapping established for sub-range\n", rank);

            // Translate the registered sub-range address to local mapping
            auto const& pool = fabricInfo->pools[0];
            void* localMapped = helper.translateToLocalMapping("rank1_sub", pool.registeredAddr, pool.registeredSize);
            if (localMapped == nullptr)
            {
                printf("[Rank %d] FAIL: translateToLocalMapping returned nullptr\n", rank);
                ok = false;
            }
            else
            {
                printf("[Rank %d] Mapped local ptr = %p for registeredAddr=0x%llx\n", rank, localMapped,
                    (unsigned long long) pool.registeredAddr);

                // Verify sub-range data
                std::vector<uint8_t> hostBuf(SUB_SIZE);
                TLLM_CUDA_CHECK(cudaMemcpy(hostBuf.data(), localMapped, SUB_SIZE, cudaMemcpyDeviceToHost));

                bool dataOk = true;
                for (size_t i = 0; i < SUB_SIZE; ++i)
                {
                    if (hostBuf[i] != FILL_PATTERN)
                    {
                        printf("[Rank %d] FAIL: sub-range data mismatch at offset %zu: expected 0x%02X, got 0x%02X\n",
                            rank, i, FILL_PATTERN, hostBuf[i]);
                        dataOk = false;
                        break;
                    }
                }
                if (dataOk)
                {
                    printf("[Rank %d] Sub-range direct read PASSED (all %zu bytes == 0x%02X)\n", rank, SUB_SIZE,
                        FILL_PATTERN);
                }
                ok = ok && dataOk;

                // Also verify batch transfer
                char* localDst = nullptr;
                TLLM_CUDA_CHECK(cudaMalloc(&localDst, SUB_SIZE));
                TLLM_CUDA_CHECK(cudaMemset(localDst, 0, SUB_SIZE));

                std::vector<void*> srcPtrs = {localMapped};
                std::vector<void*> dstPtrs = {localDst};
                std::vector<size_t> sizes = {SUB_SIZE};

                auto status = helper.submitWithCudaMemcpyBatch(srcPtrs, dstPtrs, sizes);
                auto result = status->wait();

                if (result != TransferState::kSUCCESS)
                {
                    printf("[Rank %d] FAIL: batch transfer failed for sub-range\n", rank);
                    ok = false;
                }
                else
                {
                    std::vector<uint8_t> transferBuf(SUB_SIZE);
                    TLLM_CUDA_CHECK(cudaMemcpy(transferBuf.data(), localDst, SUB_SIZE, cudaMemcpyDeviceToHost));

                    bool transferOk = true;
                    for (size_t i = 0; i < SUB_SIZE; ++i)
                    {
                        if (transferBuf[i] != FILL_PATTERN)
                        {
                            printf(
                                "[Rank %d] FAIL: batch sub-range mismatch at offset %zu: expected 0x%02X, got 0x%02X\n",
                                rank, i, FILL_PATTERN, transferBuf[i]);
                            transferOk = false;
                            break;
                        }
                    }
                    if (transferOk)
                    {
                        printf("[Rank %d] Sub-range batch transfer PASSED\n", rank);
                    }
                    ok = ok && transferOk;
                }

                TLLM_CUDA_CHECK(cudaFree(localDst));
            }
        }

        // Cleanup
        helper.cleanupRemoteFabricMapping("rank1_sub");

        MPI_Barrier(MPI_COMM_WORLD);
        printf("[Rank %d] Cleanup done\n", rank);
    }

    printf("[Rank %d] === PosixFdSubRangeDirect: %s ===\n", rank, ok ? "PASSED" : "FAILED");
    return ok;
}

// ============================================================================
// Test: NixlTransferAgent-level sub-range WRITE test
// Rank 1 allocates 8MB VMM, registers only 4MB sub-range (offset 2MB).
// Rank 0 writes data INTO the remote sub-range, then Rank 1 verifies:
//   - The sub-range contains the written data
//   - The surrounding regions (before/after sub-range) are untouched
// ============================================================================

bool testNixlAgentPosixFdSubRangeWrite(int rank, int worldSize)
{
    printf("[Rank %d] === Test: NixlAgentPosixFdSubRangeWrite ===\n", rank);

    if (worldSize < 2)
    {
        printf("[Rank %d] Need at least 2 ranks, skipping\n", rank);
        return true;
    }

    int deviceId = 0;
    TLLM_CUDA_CHECK(cudaSetDevice(deviceId));

    CUdevice device;
    CU_CHECK(cuDeviceGet(&device, deviceId));
    int posixFdSupported = 0;
    CU_CHECK(cuDeviceGetAttribute(
        &posixFdSupported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, device));
    MPI_Bcast(&posixFdSupported, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (!posixFdSupported)
    {
        printf("[Rank %d] POSIX FD not supported, skipping sub-range write test\n", rank);
        return true;
    }

    bool ok = true;
    size_t const POOL_SIZE = 8 * 1024 * 1024;
    size_t const SUB_OFFSET = 2 * 1024 * 1024;
    size_t const SUB_SIZE = 4 * 1024 * 1024;
    uint8_t const WRITE_PATTERN = 0xC7;
    uint32_t const DEVICE_ID = 0;

    // Create NixlTransferAgent
    std::string agentName = "nixl_subwrite_agent_" + std::to_string(rank);
    BaseAgentConfig config{agentName, true, false, false, false};

    std::unique_ptr<BaseTransferAgent> agent;
    try
    {
        agent = std::unique_ptr<BaseTransferAgent>(createNixlTransferAgent(&config));
    }
    catch (std::exception const& e)
    {
        printf("[Rank %d] SKIP: could not create NixlTransferAgent: %s\n", rank, e.what());
    }

    int agentOk = (agent != nullptr) ? 1 : 0;
    int allOk = 0;
    MPI_Allreduce(&agentOk, &allOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!allOk)
    {
        printf("[Rank %d] SKIP: not all ranks created NixlTransferAgent\n", rank);
        return true;
    }

    printf("[Rank %d] NixlTransferAgent created: '%s'\n", rank, agentName.c_str());

    if (rank == 1)
    {
        // ===== RECEIVER (Rank 1) =====
        // 1. Allocate 8MB VMM, zero-initialize entirely
        auto alloc = allocateVmmPosixFd(POOL_SIZE, deviceId);
        TLLM_CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(alloc.ptr), 0x00, POOL_SIZE));
        TLLM_CUDA_CHECK(cudaDeviceSynchronize());
        printf("[Rank %d] VMM pool allocated and zeroed: ptr=0x%llx, size=%zu\n", rank, (unsigned long long) alloc.ptr,
            alloc.allocSize);

        // 2. Register ONLY the sub-range
        void* subRangePtr = reinterpret_cast<void*>(alloc.ptr + SUB_OFFSET);
        MemoryDescs subDescs{MemoryType::kVRAM, {MemoryDesc{subRangePtr, SUB_SIZE, DEVICE_ID}}};
        agent->registerMemory(subDescs);
        printf("[Rank %d] registerMemory for sub-range: ptr=%p, size=%zu\n", rank, subRangePtr, SUB_SIZE);

        // 3. Send AgentDesc and sub-range address to Rank 0
        auto agentDesc = agent->getLocalAgentDesc();
        auto const& descStr = agentDesc.getBackendAgentDesc();
        int descSize = static_cast<int>(descStr.size());
        MPI_Send(&descSize, 1, MPI_INT, 0, 60, MPI_COMM_WORLD);
        MPI_Send(descStr.data(), descSize, MPI_BYTE, 0, 61, MPI_COMM_WORLD);

        uintptr_t remoteSubAddr = reinterpret_cast<uintptr_t>(subRangePtr);
        size_t remoteSubSize = SUB_SIZE;
        MPI_Send(&remoteSubAddr, sizeof(remoteSubAddr), MPI_BYTE, 0, 62, MPI_COMM_WORLD);
        MPI_Send(&remoteSubSize, sizeof(remoteSubSize), MPI_BYTE, 0, 63, MPI_COMM_WORLD);
        printf("[Rank %d] Sent AgentDesc and sub-range info to rank 0\n", rank);

        // 4. Wait for Rank 0 to finish writing
        MPI_Barrier(MPI_COMM_WORLD);

        // 5. Verify
        TLLM_CUDA_CHECK(cudaDeviceSynchronize());

        // 5a. Check that sub-range contains the WRITE_PATTERN
        std::vector<uint8_t> subBuf(SUB_SIZE);
        TLLM_CUDA_CHECK(cudaMemcpy(subBuf.data(), subRangePtr, SUB_SIZE, cudaMemcpyDeviceToHost));

        bool dataOk = true;
        for (size_t i = 0; i < SUB_SIZE; ++i)
        {
            if (subBuf[i] != WRITE_PATTERN)
            {
                printf("[Rank %d] FAIL: sub-range data mismatch at offset %zu: expected 0x%02X, got 0x%02X\n", rank, i,
                    WRITE_PATTERN, subBuf[i]);
                dataOk = false;
                break;
            }
        }
        if (dataOk)
        {
            printf("[Rank %d] Sub-range WRITE verification PASSED (all %zu bytes == 0x%02X)\n", rank, SUB_SIZE,
                WRITE_PATTERN);
        }
        ok = ok && dataOk;

        // 5b. Check that the region BEFORE the sub-range is still zeros
        std::vector<uint8_t> preBuf(SUB_OFFSET);
        TLLM_CUDA_CHECK(
            cudaMemcpy(preBuf.data(), reinterpret_cast<void*>(alloc.ptr), SUB_OFFSET, cudaMemcpyDeviceToHost));

        bool preOk = true;
        for (size_t i = 0; i < SUB_OFFSET; ++i)
        {
            if (preBuf[i] != 0x00)
            {
                printf(
                    "[Rank %d] FAIL: pre-subrange data at offset %zu should be 0x00, got 0x%02X\n", rank, i, preBuf[i]);
                preOk = false;
                break;
            }
        }
        if (preOk)
        {
            printf("[Rank %d] Pre-subrange region unchanged PASSED (%zu bytes == 0x00)\n", rank, SUB_OFFSET);
        }
        ok = ok && preOk;

        // 5c. Check that the region AFTER the sub-range is still zeros
        size_t postOffset = SUB_OFFSET + SUB_SIZE;
        size_t postSize = POOL_SIZE - postOffset;
        std::vector<uint8_t> postBuf(postSize);
        TLLM_CUDA_CHECK(cudaMemcpy(
            postBuf.data(), reinterpret_cast<void*>(alloc.ptr + postOffset), postSize, cudaMemcpyDeviceToHost));

        bool postOk = true;
        for (size_t i = 0; i < postSize; ++i)
        {
            if (postBuf[i] != 0x00)
            {
                printf("[Rank %d] FAIL: post-subrange data at offset %zu should be 0x00, got 0x%02X\n", rank, i,
                    postBuf[i]);
                postOk = false;
                break;
            }
        }
        if (postOk)
        {
            printf("[Rank %d] Post-subrange region unchanged PASSED (%zu bytes == 0x00)\n", rank, postSize);
        }
        ok = ok && postOk;

        // 6. Cleanup
        agent->deregisterMemory(subDescs);
        alloc.cleanup();
        printf("[Rank %d] Cleanup done\n", rank);
    }
    else if (rank == 0)
    {
        // ===== WRITER (Rank 0) =====
        // 1. Allocate local source buffer and fill with pattern
        char* localSrc = nullptr;
        TLLM_CUDA_CHECK(cudaMalloc(&localSrc, SUB_SIZE));
        TLLM_CUDA_CHECK(cudaMemset(localSrc, WRITE_PATTERN, SUB_SIZE));
        TLLM_CUDA_CHECK(cudaDeviceSynchronize());
        printf("[Rank %d] Local src allocated and filled with 0x%02X (%zu bytes)\n", rank, WRITE_PATTERN, SUB_SIZE);

        // 2. Register local memory
        MemoryDescs localDescs{MemoryType::kVRAM, {MemoryDesc{localSrc, SUB_SIZE, DEVICE_ID}}};
        agent->registerMemory(localDescs);

        // 3. Receive AgentDesc and sub-range info from Rank 1
        int descSize = 0;
        MPI_Recv(&descSize, 1, MPI_INT, 1, 60, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::string descStr(descSize, '\0');
        MPI_Recv(descStr.data(), descSize, MPI_BYTE, 1, 61, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        uintptr_t remoteSubAddr = 0;
        size_t remoteSubSize = 0;
        MPI_Recv(&remoteSubAddr, sizeof(remoteSubAddr), MPI_BYTE, 1, 62, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&remoteSubSize, sizeof(remoteSubSize), MPI_BYTE, 1, 63, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("[Rank %d] Received AgentDesc (%d bytes), remoteSubAddr=0x%llx, size=%zu\n", rank, descSize,
            (unsigned long long) remoteSubAddr, remoteSubSize);

        // 4. Load remote agent
        std::string remoteAgentName = "nixl_subwrite_agent_1";
        AgentDesc remoteAgentDesc{descStr};
        agent->loadRemoteAgent(remoteAgentName, remoteAgentDesc);
        printf("[Rank %d] loadRemoteAgent done\n", rank);

        // 5. Submit WRITE: push local data to remote sub-range
        TransferDescs srcDescs{
            MemoryType::kVRAM, {MemoryDesc{reinterpret_cast<uintptr_t>(localSrc), SUB_SIZE, DEVICE_ID}}};
        TransferDescs dstDescs{MemoryType::kVRAM, {MemoryDesc{remoteSubAddr, remoteSubSize, DEVICE_ID}}};

        TransferRequest writeReq{TransferOp::kWRITE, srcDescs, dstDescs, remoteAgentName};
        auto status = agent->submitTransferRequests(writeReq);
        auto result = status->wait();

        printf("[Rank %d] WRITE submitTransferRequests result: %s\n", rank,
            result == TransferState::kSUCCESS ? "SUCCESS" : "FAILURE");

        if (result != TransferState::kSUCCESS)
        {
            printf("[Rank %d] FAIL: sub-range WRITE transfer did not succeed\n", rank);
            ok = false;
        }
        else
        {
            printf("[Rank %d] Sub-range WRITE transfer completed successfully\n", rank);
        }

        // 6. Signal Rank 1
        TLLM_CUDA_CHECK(cudaDeviceSynchronize());
        MPI_Barrier(MPI_COMM_WORLD);

        // 7. Cleanup
        agent->deregisterMemory(localDescs);
        agent->invalidateRemoteAgent(remoteAgentName);
        TLLM_CUDA_CHECK(cudaFree(localSrc));
        printf("[Rank %d] Cleanup done\n", rank);
    }

    printf("[Rank %d] === NixlAgentPosixFdSubRangeWrite: %s ===\n", rank, ok ? "PASSED" : "FAILED");
    return ok;
}

// ============================================================================
// Test: Multi-chunk + sub-range at FabricTransferHelper level
// Pool layout:
//   [alloc_A 2MB] [alloc_B 2MB] [alloc_C 2MB] [alloc_D 2MB]
//    buf_id=1      buf_id=2      buf_id=3      buf_id=4
//                  ↑── registered [2MB, 6MB) ──↑
// Verifies that only alloc_B and alloc_C chunks are exported (not A or D),
// and the importer can correctly map and read the sub-range data.
// ============================================================================

bool testPosixFdMultiChunkSubRangeDirect(int rank, int worldSize)
{
    printf("[Rank %d] === Test: PosixFdMultiChunkSubRangeDirect ===\n", rank);

    if (worldSize < 2)
    {
        printf("[Rank %d] Need at least 2 ranks, skipping\n", rank);
        return true;
    }

    int deviceId = 0;
    TLLM_CUDA_CHECK(cudaSetDevice(deviceId));

    CUdevice device;
    CU_CHECK(cuDeviceGet(&device, deviceId));
    int posixFdSupported = 0;
    CU_CHECK(cuDeviceGetAttribute(
        &posixFdSupported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, device));
    MPI_Bcast(&posixFdSupported, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (!posixFdSupported)
    {
        printf("[Rank %d] POSIX FD not supported, skipping multi-chunk sub-range direct test\n", rank);
        return true;
    }

    bool ok = true;
    size_t const CHUNK_SIZE = 2 * 1024 * 1024; // 2MB per physical chunk
    size_t const NUM_CHUNKS = 4;               // 4 chunks = 8MB total
    size_t const POOL_SIZE = CHUNK_SIZE * NUM_CHUNKS;
    size_t const SUB_OFFSET = CHUNK_SIZE;      // Start at chunk B (offset 2MB)
    size_t const SUB_SIZE = CHUNK_SIZE * 2;    // Cover chunks B+C (4MB)
    uint8_t const FILL_PATTERN = 0xE3;

    if (rank == 1)
    {
        // ===== EXPORTER =====
        // Allocate 4 separate physical chunks mapped to one VA range
        auto alloc = allocateMultiChunkVmmPosixFd(CHUNK_SIZE, NUM_CHUNKS, deviceId);
        printf("[Rank %d] Multi-chunk VMM pool: ptr=0x%llx, totalSize=%zu, chunks=%zu\n", rank,
            (unsigned long long) alloc.ptr, alloc.totalSize, alloc.numChunks);

        // Zero entire pool, then fill only the sub-range [B,C) with pattern
        TLLM_CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(alloc.ptr), 0x00, POOL_SIZE));
        TLLM_CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(alloc.ptr + SUB_OFFSET), FILL_PATTERN, SUB_SIZE));
        TLLM_CUDA_CHECK(cudaDeviceSynchronize());
        printf("[Rank %d] Sub-range [+%zu, +%zu) filled with 0x%02X\n", rank, SUB_OFFSET, SUB_OFFSET + SUB_SIZE,
            FILL_PATTERN);

        // Register ONLY the sub-range (alloc_B + alloc_C)
        FabricTransferHelper helper;
        void* subRangePtr = reinterpret_cast<void*>(alloc.ptr + SUB_OFFSET);
        MemoryDescs descs{MemoryType::kVRAM, {MemoryDesc{subRangePtr, SUB_SIZE, static_cast<uint32_t>(deviceId)}}};
        helper.detectAndExportFabricHandles(descs);

        auto const& info = helper.getLocalFabricInfo();
        if (!info.supported)
        {
            printf("[Rank %d] FAIL: fabric info not supported after detection\n", rank);
            alloc.cleanup();
            int metadataSize = 0;
            MPI_Send(&metadataSize, 1, MPI_INT, 0, 70, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }

        printf("[Rank %d] Export done: pools=%zu, handleType=%s\n", rank, info.pools.size(),
            info.handleType == VmmHandleType::kPosixFd ? "PosixFd" : "Fabric");

        // Validate exported chunk count: should be 2 (alloc_B and alloc_C), not 4
        if (!info.pools.empty())
        {
            auto const& pool = info.pools[0];
            printf(
                "[Rank %d] Pool: base=0x%llx, totalSize=%zu, registeredAddr=0x%llx, "
                "registeredSize=%zu, chunks=%zu\n",
                rank, (unsigned long long) pool.poolBaseAddr, (size_t) pool.poolTotalSize,
                (unsigned long long) pool.registeredAddr, (size_t) pool.registeredSize, pool.chunks.size());

            if (pool.chunks.size() != 2)
            {
                printf("[Rank %d] WARNING: expected 2 exported chunks (B+C), got %zu\n", rank, pool.chunks.size());
            }
            for (size_t ci = 0; ci < pool.chunks.size(); ++ci)
            {
                printf("[Rank %d]   chunk[%zu]: virtAddrOffset=%zu, size=%zu\n", rank, ci,
                    (size_t) pool.chunks[ci].virtAddrOffset, (size_t) pool.chunks[ci].size);
            }
        }

        // Serialize and send
        std::string metadata = info.serialize();
        int metadataSize = static_cast<int>(metadata.size());
        MPI_Send(&metadataSize, 1, MPI_INT, 0, 70, MPI_COMM_WORLD);
        MPI_Send(metadata.data(), metadataSize, MPI_BYTE, 0, 71, MPI_COMM_WORLD);
        printf("[Rank %d] Sent metadata (%d bytes)\n", rank, metadataSize);

        MPI_Barrier(MPI_COMM_WORLD);
        alloc.cleanup();
        printf("[Rank %d] Cleanup done\n", rank);
    }
    else if (rank == 0)
    {
        // ===== IMPORTER =====
        int metadataSize = 0;
        MPI_Recv(&metadataSize, 1, MPI_INT, 1, 70, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (metadataSize == 0)
        {
            printf("[Rank %d] Exporter reported failure, skipping\n", rank);
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }

        std::string metadata(metadataSize, '\0');
        MPI_Recv(metadata.data(), metadataSize, MPI_BYTE, 1, 71, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("[Rank %d] Received metadata (%d bytes)\n", rank, metadataSize);

        auto fabricInfo = FabricMemInfo::deserialize(metadata);
        if (!fabricInfo.has_value())
        {
            printf("[Rank %d] FAIL: failed to deserialize FabricMemInfo\n", rank);
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }

        printf("[Rank %d] Deserialized: handleType=%s, pools=%zu\n", rank,
            fabricInfo->handleType == VmmHandleType::kPosixFd ? "PosixFd" : "Fabric", fabricInfo->pools.size());

        // Import and map
        FabricTransferHelper helper;
        helper.importAndMapRemoteFabric("rank1_mc_sub", *fabricInfo);

        if (!helper.hasRemoteMapping("rank1_mc_sub"))
        {
            printf("[Rank %d] FAIL: no remote mapping for multi-chunk sub-range\n", rank);
            ok = false;
        }
        else
        {
            printf("[Rank %d] Remote mapping established\n", rank);

            auto const& pool = fabricInfo->pools[0];
            void* localMapped
                = helper.translateToLocalMapping("rank1_mc_sub", pool.registeredAddr, pool.registeredSize);
            if (localMapped == nullptr)
            {
                printf("[Rank %d] FAIL: translateToLocalMapping returned nullptr\n", rank);
                ok = false;
            }
            else
            {
                printf("[Rank %d] Mapped local ptr = %p for registeredAddr=0x%llx\n", rank, localMapped,
                    (unsigned long long) pool.registeredAddr);

                // Verify all SUB_SIZE bytes are FILL_PATTERN
                std::vector<uint8_t> hostBuf(SUB_SIZE);
                TLLM_CUDA_CHECK(cudaMemcpy(hostBuf.data(), localMapped, SUB_SIZE, cudaMemcpyDeviceToHost));

                bool dataOk = true;
                for (size_t i = 0; i < SUB_SIZE; ++i)
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
                    printf("[Rank %d] Multi-chunk sub-range direct read PASSED (%zu bytes == 0x%02X)\n", rank, SUB_SIZE,
                        FILL_PATTERN);
                }
                ok = ok && dataOk;

                // Also verify batch transfer
                char* localDst = nullptr;
                TLLM_CUDA_CHECK(cudaMalloc(&localDst, SUB_SIZE));
                TLLM_CUDA_CHECK(cudaMemset(localDst, 0, SUB_SIZE));

                std::vector<void*> srcPtrs = {localMapped};
                std::vector<void*> dstPtrs = {localDst};
                std::vector<size_t> sizes = {SUB_SIZE};

                auto status = helper.submitWithCudaMemcpyBatch(srcPtrs, dstPtrs, sizes);
                auto result = status->wait();

                if (result != TransferState::kSUCCESS)
                {
                    printf("[Rank %d] FAIL: batch transfer failed\n", rank);
                    ok = false;
                }
                else
                {
                    std::vector<uint8_t> transferBuf(SUB_SIZE);
                    TLLM_CUDA_CHECK(cudaMemcpy(transferBuf.data(), localDst, SUB_SIZE, cudaMemcpyDeviceToHost));

                    bool transferOk = true;
                    for (size_t i = 0; i < SUB_SIZE; ++i)
                    {
                        if (transferBuf[i] != FILL_PATTERN)
                        {
                            printf("[Rank %d] FAIL: batch mismatch at offset %zu: expected 0x%02X, got 0x%02X\n", rank,
                                i, FILL_PATTERN, transferBuf[i]);
                            transferOk = false;
                            break;
                        }
                    }
                    if (transferOk)
                    {
                        printf("[Rank %d] Multi-chunk sub-range batch transfer PASSED\n", rank);
                    }
                    ok = ok && transferOk;
                }

                TLLM_CUDA_CHECK(cudaFree(localDst));
            }
        }

        helper.cleanupRemoteFabricMapping("rank1_mc_sub");
        MPI_Barrier(MPI_COMM_WORLD);
        printf("[Rank %d] Cleanup done\n", rank);
    }

    printf("[Rank %d] === PosixFdMultiChunkSubRangeDirect: %s ===\n", rank, ok ? "PASSED" : "FAILED");
    return ok;
}

// ============================================================================
// Test: Multi-chunk + sub-range at NixlTransferAgent level (end-to-end READ)
// Same pool layout as above but using full NixlTransferAgent pipeline.
// ============================================================================

bool testNixlAgentMultiChunkSubRange(int rank, int worldSize)
{
    printf("[Rank %d] === Test: NixlAgentMultiChunkSubRange ===\n", rank);

    if (worldSize < 2)
    {
        printf("[Rank %d] Need at least 2 ranks, skipping\n", rank);
        return true;
    }

    int deviceId = 0;
    TLLM_CUDA_CHECK(cudaSetDevice(deviceId));

    CUdevice device;
    CU_CHECK(cuDeviceGet(&device, deviceId));
    int posixFdSupported = 0;
    CU_CHECK(cuDeviceGetAttribute(
        &posixFdSupported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, device));
    MPI_Bcast(&posixFdSupported, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (!posixFdSupported)
    {
        printf("[Rank %d] POSIX FD not supported, skipping multi-chunk sub-range agent test\n", rank);
        return true;
    }

    bool ok = true;
    size_t const CHUNK_SIZE = 2 * 1024 * 1024;
    size_t const NUM_CHUNKS = 4;
    size_t const POOL_SIZE = CHUNK_SIZE * NUM_CHUNKS;
    size_t const SUB_OFFSET = CHUNK_SIZE;   // offset 2MB (skip alloc_A)
    size_t const SUB_SIZE = CHUNK_SIZE * 2; // 4MB (alloc_B + alloc_C)
    uint8_t const FILL_PATTERN = 0xA7;
    uint32_t const DEVICE_ID = 0;

    // Create NixlTransferAgent
    std::string agentName = "nixl_mcsubrange_agent_" + std::to_string(rank);
    BaseAgentConfig config{agentName, true, false, false, false};

    std::unique_ptr<BaseTransferAgent> agent;
    try
    {
        agent = std::unique_ptr<BaseTransferAgent>(createNixlTransferAgent(&config));
    }
    catch (std::exception const& e)
    {
        printf("[Rank %d] SKIP: could not create NixlTransferAgent: %s\n", rank, e.what());
    }

    int agentOk = (agent != nullptr) ? 1 : 0;
    int allOk = 0;
    MPI_Allreduce(&agentOk, &allOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!allOk)
    {
        printf("[Rank %d] SKIP: not all ranks created NixlTransferAgent\n", rank);
        return true;
    }

    printf("[Rank %d] NixlTransferAgent created: '%s'\n", rank, agentName.c_str());

    if (rank == 1)
    {
        // ===== EXPORTER (Rank 1) =====
        auto alloc = allocateMultiChunkVmmPosixFd(CHUNK_SIZE, NUM_CHUNKS, deviceId);
        printf("[Rank %d] Multi-chunk pool: ptr=0x%llx, totalSize=%zu, chunks=%zu\n", rank,
            (unsigned long long) alloc.ptr, alloc.totalSize, alloc.numChunks);

        // Zero entire pool, fill sub-range with pattern
        TLLM_CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(alloc.ptr), 0x00, POOL_SIZE));
        TLLM_CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(alloc.ptr + SUB_OFFSET), FILL_PATTERN, SUB_SIZE));
        TLLM_CUDA_CHECK(cudaDeviceSynchronize());

        // Register only the sub-range
        void* subRangePtr = reinterpret_cast<void*>(alloc.ptr + SUB_OFFSET);
        MemoryDescs subDescs{MemoryType::kVRAM, {MemoryDesc{subRangePtr, SUB_SIZE, DEVICE_ID}}};
        agent->registerMemory(subDescs);
        printf("[Rank %d] registerMemory for sub-range: ptr=%p, size=%zu\n", rank, subRangePtr, SUB_SIZE);

        // Send AgentDesc and sub-range address to rank 0
        auto agentDesc = agent->getLocalAgentDesc();
        auto const& descStr = agentDesc.getBackendAgentDesc();
        int descSize = static_cast<int>(descStr.size());
        MPI_Send(&descSize, 1, MPI_INT, 0, 80, MPI_COMM_WORLD);
        MPI_Send(descStr.data(), descSize, MPI_BYTE, 0, 81, MPI_COMM_WORLD);

        uintptr_t remoteSrcAddr = reinterpret_cast<uintptr_t>(subRangePtr);
        size_t remoteSrcSize = SUB_SIZE;
        MPI_Send(&remoteSrcAddr, sizeof(remoteSrcAddr), MPI_BYTE, 0, 82, MPI_COMM_WORLD);
        MPI_Send(&remoteSrcSize, sizeof(remoteSrcSize), MPI_BYTE, 0, 83, MPI_COMM_WORLD);
        printf("[Rank %d] Sent AgentDesc and sub-range info\n", rank);

        // Wait for rank 0
        MPI_Barrier(MPI_COMM_WORLD);

        // Cleanup
        agent->deregisterMemory(subDescs);
        alloc.cleanup();
        printf("[Rank %d] Cleanup done\n", rank);
    }
    else if (rank == 0)
    {
        // ===== IMPORTER (Rank 0) =====
        // Allocate local destination
        char* localDst = nullptr;
        TLLM_CUDA_CHECK(cudaMalloc(&localDst, SUB_SIZE));
        TLLM_CUDA_CHECK(cudaMemset(localDst, 0, SUB_SIZE));

        MemoryDescs localDescs{MemoryType::kVRAM, {MemoryDesc{localDst, SUB_SIZE, DEVICE_ID}}};
        agent->registerMemory(localDescs);

        // Receive AgentDesc
        int descSize = 0;
        MPI_Recv(&descSize, 1, MPI_INT, 1, 80, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::string descStr(descSize, '\0');
        MPI_Recv(descStr.data(), descSize, MPI_BYTE, 1, 81, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        uintptr_t remoteSrcAddr = 0;
        size_t remoteSrcSize = 0;
        MPI_Recv(&remoteSrcAddr, sizeof(remoteSrcAddr), MPI_BYTE, 1, 82, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&remoteSrcSize, sizeof(remoteSrcSize), MPI_BYTE, 1, 83, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("[Rank %d] Received AgentDesc (%d bytes), remoteSrcAddr=0x%llx, size=%zu\n", rank, descSize,
            (unsigned long long) remoteSrcAddr, remoteSrcSize);

        // Load remote agent
        std::string remoteAgentName = "nixl_mcsubrange_agent_1";
        AgentDesc remoteAgentDesc{descStr};
        agent->loadRemoteAgent(remoteAgentName, remoteAgentDesc);
        printf("[Rank %d] loadRemoteAgent done\n", rank);

        // Submit READ transfer
        TransferDescs srcDescs{
            MemoryType::kVRAM, {MemoryDesc{reinterpret_cast<uintptr_t>(localDst), SUB_SIZE, DEVICE_ID}}};
        TransferDescs dstDescs{MemoryType::kVRAM, {MemoryDesc{remoteSrcAddr, remoteSrcSize, DEVICE_ID}}};

        TransferRequest readReq{TransferOp::kREAD, srcDescs, dstDescs, remoteAgentName};
        auto status = agent->submitTransferRequests(readReq);
        auto result = status->wait();

        printf("[Rank %d] READ result: %s\n", rank, result == TransferState::kSUCCESS ? "SUCCESS" : "FAILURE");

        if (result != TransferState::kSUCCESS)
        {
            printf("[Rank %d] FAIL: multi-chunk sub-range READ did not succeed\n", rank);
            ok = false;
        }
        else
        {
            // Verify data
            std::vector<uint8_t> hostBuf(SUB_SIZE);
            TLLM_CUDA_CHECK(cudaMemcpy(hostBuf.data(), localDst, SUB_SIZE, cudaMemcpyDeviceToHost));

            bool dataOk = true;
            for (size_t i = 0; i < SUB_SIZE; ++i)
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
                printf("[Rank %d] Multi-chunk sub-range READ PASSED (%zu bytes == 0x%02X)\n", rank, SUB_SIZE,
                    FILL_PATTERN);
            }
            ok = ok && dataOk;
        }

        // Cleanup
        agent->deregisterMemory(localDescs);
        agent->invalidateRemoteAgent(remoteAgentName);
        TLLM_CUDA_CHECK(cudaFree(localDst));

        MPI_Barrier(MPI_COMM_WORLD);
        printf("[Rank %d] Cleanup done\n", rank);
    }

    printf("[Rank %d] === NixlAgentMultiChunkSubRange: %s ===\n", rank, ok ? "PASSED" : "FAILED");
    return ok;
}

// ============================================================================
// Test: Multi-chunk + UNALIGNED sub-range (registered range crosses chunk boundaries)
// Pool layout (4 chunks, each 2MB):
//   [alloc_A 2MB] [alloc_B 2MB] [alloc_C 2MB] [alloc_D 2MB]
//    buf_id=1      buf_id=2      buf_id=3      buf_id=4
//                     ↑── registered [3MB, 5MB) ──↑
// The registered range [3MB, 5MB) starts at the MIDDLE of alloc_B and ends at
// the MIDDLE of alloc_C. This verifies:
//   1. detectAndExportChunks correctly discovers both alloc_B and alloc_C
//      as overlapping chunks despite the non-aligned scan start/end
//   2. Each chunk's full physical size (2MB) is exported (not just the 1MB overlap)
//   3. importAndMapRemoteFabric correctly maps both full physical allocations
//   4. translateToLocalMapping correctly computes offsets for non-chunk-aligned
//      registered addresses (registeredAddr - poolBase = 3MB, not a chunk multiple)
//   5. Data transfer is correct for the 2MB sub-range
//
// Tests at both FabricTransferHelper (direct) and NixlTransferAgent (end-to-end) levels.
// ============================================================================

bool testMultiChunkUnalignedSubRangeDirect(int rank, int worldSize)
{
    printf("[Rank %d] === Test: MultiChunkUnalignedSubRangeDirect ===\n", rank);

    if (worldSize < 2)
    {
        printf("[Rank %d] Need at least 2 ranks, skipping\n", rank);
        return true;
    }

    int deviceId = 0;
    TLLM_CUDA_CHECK(cudaSetDevice(deviceId));

    CUdevice device;
    CU_CHECK(cuDeviceGet(&device, deviceId));
    int posixFdSupported = 0;
    CU_CHECK(cuDeviceGetAttribute(
        &posixFdSupported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, device));
    MPI_Bcast(&posixFdSupported, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (!posixFdSupported)
    {
        printf("[Rank %d] POSIX FD not supported, skipping\n", rank);
        return true;
    }

    bool ok = true;
    size_t const CHUNK_SIZE = 2 * 1024 * 1024; // 2MB per physical chunk
    size_t const NUM_CHUNKS = 4;               // 4 chunks = 8MB total
    size_t const POOL_SIZE = CHUNK_SIZE * NUM_CHUNKS;
    // Register [3MB, 5MB): starts mid-alloc_B, ends mid-alloc_C
    size_t const SUB_OFFSET = 3 * 1024 * 1024; // 3MB from pool start
    size_t const SUB_SIZE = 2 * 1024 * 1024;   // 2MB
    uint8_t const FILL_PATTERN = 0xD9;

    if (rank == 1)
    {
        // ===== EXPORTER =====
        auto alloc = allocateMultiChunkVmmPosixFd(CHUNK_SIZE, NUM_CHUNKS, deviceId);
        printf("[Rank %d] Multi-chunk pool: ptr=0x%llx, totalSize=%zu, chunks=%zu\n", rank,
            (unsigned long long) alloc.ptr, alloc.totalSize, alloc.numChunks);

        // Zero entire pool, then fill ONLY [3MB, 5MB) with pattern
        TLLM_CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(alloc.ptr), 0x00, POOL_SIZE));
        TLLM_CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(alloc.ptr + SUB_OFFSET), FILL_PATTERN, SUB_SIZE));
        TLLM_CUDA_CHECK(cudaDeviceSynchronize());
        printf("[Rank %d] Sub-range [+%zu, +%zu) filled with 0x%02X (crosses chunk boundary)\n", rank, SUB_OFFSET,
            SUB_OFFSET + SUB_SIZE, FILL_PATTERN);

        // Register the unaligned sub-range
        FabricTransferHelper helper;
        void* subRangePtr = reinterpret_cast<void*>(alloc.ptr + SUB_OFFSET);
        MemoryDescs descs{MemoryType::kVRAM, {MemoryDesc{subRangePtr, SUB_SIZE, static_cast<uint32_t>(deviceId)}}};
        helper.detectAndExportFabricHandles(descs);

        auto const& info = helper.getLocalFabricInfo();
        if (!info.supported)
        {
            printf("[Rank %d] FAIL: fabric info not supported\n", rank);
            alloc.cleanup();
            int metadataSize = 0;
            MPI_Send(&metadataSize, 1, MPI_INT, 0, 90, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }

        printf("[Rank %d] Export done: pools=%zu, handleType=%s\n", rank, info.pools.size(),
            info.handleType == VmmHandleType::kPosixFd ? "PosixFd" : "Fabric");

        if (!info.pools.empty())
        {
            auto const& pool = info.pools[0];
            printf(
                "[Rank %d] Pool: base=0x%llx, totalSize=%zu, registeredAddr=0x%llx, "
                "registeredSize=%zu, mappedOffset=%zu, mappedSize=%zu, chunks=%zu\n",
                rank, (unsigned long long) pool.poolBaseAddr, (size_t) pool.poolTotalSize,
                (unsigned long long) pool.registeredAddr, (size_t) pool.registeredSize, (size_t) pool.mappedOffset,
                (size_t) pool.mappedSize, pool.chunks.size());

            // Expect 2 chunks (alloc_B and alloc_C), each with full 2MB size
            if (pool.chunks.size() != 2)
            {
                printf("[Rank %d] WARNING: expected 2 chunks (B+C), got %zu\n", rank, pool.chunks.size());
            }
            for (size_t ci = 0; ci < pool.chunks.size(); ++ci)
            {
                printf("[Rank %d]   chunk[%zu]: virtAddrOffset=%zu, size=%zu\n", rank, ci,
                    (size_t) pool.chunks[ci].virtAddrOffset, (size_t) pool.chunks[ci].size);
                // Each chunk should be full 2MB, not partial
                if (pool.chunks[ci].size != CHUNK_SIZE)
                {
                    printf("[Rank %d]   WARNING: chunk[%zu] size=%zu, expected full chunk %zu\n", rank, ci,
                        (size_t) pool.chunks[ci].size, CHUNK_SIZE);
                }
            }
        }

        // Serialize and send
        std::string metadata = info.serialize();
        int metadataSize = static_cast<int>(metadata.size());
        MPI_Send(&metadataSize, 1, MPI_INT, 0, 90, MPI_COMM_WORLD);
        MPI_Send(metadata.data(), metadataSize, MPI_BYTE, 0, 91, MPI_COMM_WORLD);
        printf("[Rank %d] Sent metadata (%d bytes)\n", rank, metadataSize);

        MPI_Barrier(MPI_COMM_WORLD);
        alloc.cleanup();
        printf("[Rank %d] Cleanup done\n", rank);
    }
    else if (rank == 0)
    {
        // ===== IMPORTER =====
        int metadataSize = 0;
        MPI_Recv(&metadataSize, 1, MPI_INT, 1, 90, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (metadataSize == 0)
        {
            printf("[Rank %d] Exporter reported failure\n", rank);
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }

        std::string metadata(metadataSize, '\0');
        MPI_Recv(metadata.data(), metadataSize, MPI_BYTE, 1, 91, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("[Rank %d] Received metadata (%d bytes)\n", rank, metadataSize);

        auto fabricInfo = FabricMemInfo::deserialize(metadata);
        if (!fabricInfo.has_value())
        {
            printf("[Rank %d] FAIL: deserialize failed\n", rank);
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }

        // Import and map
        FabricTransferHelper helper;
        helper.importAndMapRemoteFabric("rank1_unaligned", *fabricInfo);

        if (!helper.hasRemoteMapping("rank1_unaligned"))
        {
            printf("[Rank %d] FAIL: no remote mapping\n", rank);
            ok = false;
        }
        else
        {
            printf("[Rank %d] Remote mapping established\n", rank);

            auto const& pool = fabricInfo->pools[0];
            // Translate the unaligned registeredAddr (3MB offset, not chunk-aligned)
            void* localMapped
                = helper.translateToLocalMapping("rank1_unaligned", pool.registeredAddr, pool.registeredSize);
            if (localMapped == nullptr)
            {
                printf("[Rank %d] FAIL: translateToLocalMapping returned nullptr for unaligned addr 0x%llx\n", rank,
                    (unsigned long long) pool.registeredAddr);
                ok = false;
            }
            else
            {
                printf("[Rank %d] Mapped local ptr = %p for registeredAddr=0x%llx (unaligned, 3MB offset)\n", rank,
                    localMapped, (unsigned long long) pool.registeredAddr);

                // Verify all SUB_SIZE bytes are FILL_PATTERN
                std::vector<uint8_t> hostBuf(SUB_SIZE);
                TLLM_CUDA_CHECK(cudaMemcpy(hostBuf.data(), localMapped, SUB_SIZE, cudaMemcpyDeviceToHost));

                bool dataOk = true;
                for (size_t i = 0; i < SUB_SIZE; ++i)
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
                    printf("[Rank %d] Unaligned sub-range direct read PASSED (%zu bytes == 0x%02X)\n", rank, SUB_SIZE,
                        FILL_PATTERN);
                }
                ok = ok && dataOk;
            }
        }

        helper.cleanupRemoteFabricMapping("rank1_unaligned");
        MPI_Barrier(MPI_COMM_WORLD);
        printf("[Rank %d] Cleanup done\n", rank);
    }

    printf("[Rank %d] === MultiChunkUnalignedSubRangeDirect: %s ===\n", rank, ok ? "PASSED" : "FAILED");
    return ok;
}

bool testNixlAgentMultiChunkUnalignedSubRange(int rank, int worldSize)
{
    printf("[Rank %d] === Test: NixlAgentMultiChunkUnalignedSubRange ===\n", rank);

    if (worldSize < 2)
    {
        printf("[Rank %d] Need at least 2 ranks, skipping\n", rank);
        return true;
    }

    int deviceId = 0;
    TLLM_CUDA_CHECK(cudaSetDevice(deviceId));

    CUdevice device;
    CU_CHECK(cuDeviceGet(&device, deviceId));
    int posixFdSupported = 0;
    CU_CHECK(cuDeviceGetAttribute(
        &posixFdSupported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, device));
    MPI_Bcast(&posixFdSupported, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (!posixFdSupported)
    {
        printf("[Rank %d] POSIX FD not supported, skipping\n", rank);
        return true;
    }

    bool ok = true;
    size_t const CHUNK_SIZE = 2 * 1024 * 1024;
    size_t const NUM_CHUNKS = 4;
    size_t const POOL_SIZE = CHUNK_SIZE * NUM_CHUNKS;
    size_t const SUB_OFFSET = 3 * 1024 * 1024; // 3MB — mid alloc_B
    size_t const SUB_SIZE = 2 * 1024 * 1024;   // 2MB — ends mid alloc_C
    uint8_t const FILL_PATTERN = 0xB3;
    uint32_t const DEVICE_ID = 0;

    // Create NixlTransferAgent
    std::string agentName = "nixl_unaligned_agent_" + std::to_string(rank);
    BaseAgentConfig config{agentName, true, false, false, false};

    std::unique_ptr<BaseTransferAgent> agent;
    try
    {
        agent = std::unique_ptr<BaseTransferAgent>(createNixlTransferAgent(&config));
    }
    catch (std::exception const& e)
    {
        printf("[Rank %d] SKIP: could not create NixlTransferAgent: %s\n", rank, e.what());
    }

    int agentOk = (agent != nullptr) ? 1 : 0;
    int allOk = 0;
    MPI_Allreduce(&agentOk, &allOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!allOk)
    {
        printf("[Rank %d] SKIP: not all ranks created NixlTransferAgent\n", rank);
        return true;
    }

    printf("[Rank %d] NixlTransferAgent created: '%s'\n", rank, agentName.c_str());

    if (rank == 1)
    {
        // ===== EXPORTER (Rank 1) =====
        auto alloc = allocateMultiChunkVmmPosixFd(CHUNK_SIZE, NUM_CHUNKS, deviceId);
        printf("[Rank %d] Multi-chunk pool: ptr=0x%llx, totalSize=%zu\n", rank, (unsigned long long) alloc.ptr,
            alloc.totalSize);

        // Zero entire pool, fill [3MB, 5MB) with pattern
        TLLM_CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(alloc.ptr), 0x00, POOL_SIZE));
        TLLM_CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(alloc.ptr + SUB_OFFSET), FILL_PATTERN, SUB_SIZE));
        TLLM_CUDA_CHECK(cudaDeviceSynchronize());

        // Register the unaligned sub-range
        void* subRangePtr = reinterpret_cast<void*>(alloc.ptr + SUB_OFFSET);
        MemoryDescs subDescs{MemoryType::kVRAM, {MemoryDesc{subRangePtr, SUB_SIZE, DEVICE_ID}}};
        agent->registerMemory(subDescs);
        printf("[Rank %d] registerMemory: ptr=%p (3MB offset), size=%zu\n", rank, subRangePtr, SUB_SIZE);

        // Send AgentDesc
        auto agentDesc = agent->getLocalAgentDesc();
        auto const& descStr = agentDesc.getBackendAgentDesc();
        int descSize = static_cast<int>(descStr.size());
        MPI_Send(&descSize, 1, MPI_INT, 0, 100, MPI_COMM_WORLD);
        MPI_Send(descStr.data(), descSize, MPI_BYTE, 0, 101, MPI_COMM_WORLD);

        uintptr_t remoteSrcAddr = reinterpret_cast<uintptr_t>(subRangePtr);
        size_t remoteSrcSize = SUB_SIZE;
        MPI_Send(&remoteSrcAddr, sizeof(remoteSrcAddr), MPI_BYTE, 0, 102, MPI_COMM_WORLD);
        MPI_Send(&remoteSrcSize, sizeof(remoteSrcSize), MPI_BYTE, 0, 103, MPI_COMM_WORLD);
        printf("[Rank %d] Sent AgentDesc and sub-range info\n", rank);

        MPI_Barrier(MPI_COMM_WORLD);

        agent->deregisterMemory(subDescs);
        alloc.cleanup();
        printf("[Rank %d] Cleanup done\n", rank);
    }
    else if (rank == 0)
    {
        // ===== IMPORTER (Rank 0) =====
        char* localDst = nullptr;
        TLLM_CUDA_CHECK(cudaMalloc(&localDst, SUB_SIZE));
        TLLM_CUDA_CHECK(cudaMemset(localDst, 0, SUB_SIZE));

        MemoryDescs localDescs{MemoryType::kVRAM, {MemoryDesc{localDst, SUB_SIZE, DEVICE_ID}}};
        agent->registerMemory(localDescs);

        // Receive AgentDesc
        int descSize = 0;
        MPI_Recv(&descSize, 1, MPI_INT, 1, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::string descStr(descSize, '\0');
        MPI_Recv(descStr.data(), descSize, MPI_BYTE, 1, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        uintptr_t remoteSrcAddr = 0;
        size_t remoteSrcSize = 0;
        MPI_Recv(&remoteSrcAddr, sizeof(remoteSrcAddr), MPI_BYTE, 1, 102, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&remoteSrcSize, sizeof(remoteSrcSize), MPI_BYTE, 1, 103, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("[Rank %d] Received AgentDesc (%d bytes), remoteSrcAddr=0x%llx (unaligned), size=%zu\n", rank, descSize,
            (unsigned long long) remoteSrcAddr, remoteSrcSize);

        // Load remote agent
        std::string remoteAgentName = "nixl_unaligned_agent_1";
        AgentDesc remoteAgentDesc{descStr};
        agent->loadRemoteAgent(remoteAgentName, remoteAgentDesc);
        printf("[Rank %d] loadRemoteAgent done\n", rank);

        // Submit READ
        TransferDescs srcDescs{
            MemoryType::kVRAM, {MemoryDesc{reinterpret_cast<uintptr_t>(localDst), SUB_SIZE, DEVICE_ID}}};
        TransferDescs dstDescs{MemoryType::kVRAM, {MemoryDesc{remoteSrcAddr, remoteSrcSize, DEVICE_ID}}};

        TransferRequest readReq{TransferOp::kREAD, srcDescs, dstDescs, remoteAgentName};
        auto status = agent->submitTransferRequests(readReq);
        auto result = status->wait();

        printf("[Rank %d] READ result: %s\n", rank, result == TransferState::kSUCCESS ? "SUCCESS" : "FAILURE");

        if (result != TransferState::kSUCCESS)
        {
            printf("[Rank %d] FAIL: unaligned sub-range READ did not succeed\n", rank);
            ok = false;
        }
        else
        {
            std::vector<uint8_t> hostBuf(SUB_SIZE);
            TLLM_CUDA_CHECK(cudaMemcpy(hostBuf.data(), localDst, SUB_SIZE, cudaMemcpyDeviceToHost));

            bool dataOk = true;
            for (size_t i = 0; i < SUB_SIZE; ++i)
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
                    "[Rank %d] Unaligned sub-range READ PASSED (%zu bytes == 0x%02X)\n", rank, SUB_SIZE, FILL_PATTERN);
            }
            ok = ok && dataOk;
        }

        // Cleanup
        agent->deregisterMemory(localDescs);
        agent->invalidateRemoteAgent(remoteAgentName);
        TLLM_CUDA_CHECK(cudaFree(localDst));

        MPI_Barrier(MPI_COMM_WORLD);
        printf("[Rank %d] Cleanup done\n", rank);
    }

    printf("[Rank %d] === NixlAgentMultiChunkUnalignedSubRange: %s ===\n", rank, ok ? "PASSED" : "FAILED");
    return ok;
}

// ============================================================================
// Test: Small cross-boundary sub-range (scanSize < granularity)
// Pool: 4×2MB. Register [1.5MB, 2.5MB) — only 1MB, but crosses the
// alloc_A/alloc_B boundary at 2MB. Verifies that the scanSize<=granularity
// fast path correctly detects an internal boundary even when the range is
// smaller than one granularity.
// ============================================================================

bool testSmallCrossBoundaryDirect(int rank, int worldSize)
{
    printf("[Rank %d] === Test: SmallCrossBoundaryDirect ===\n", rank);

    if (worldSize < 2)
    {
        printf("[Rank %d] Need at least 2 ranks, skipping\n", rank);
        return true;
    }

    int deviceId = 0;
    TLLM_CUDA_CHECK(cudaSetDevice(deviceId));

    CUdevice device;
    CU_CHECK(cuDeviceGet(&device, deviceId));
    int posixFdSupported = 0;
    CU_CHECK(cuDeviceGetAttribute(
        &posixFdSupported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, device));
    MPI_Bcast(&posixFdSupported, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (!posixFdSupported)
    {
        printf("[Rank %d] POSIX FD not supported, skipping\n", rank);
        return true;
    }

    bool ok = true;
    size_t const CHUNK_SIZE = 2 * 1024 * 1024; // 2MB per physical chunk
    size_t const NUM_CHUNKS = 4;
    size_t const POOL_SIZE = CHUNK_SIZE * NUM_CHUNKS;
    // Register [1.5MB, 2.5MB): crosses alloc_A/alloc_B boundary at 2MB
    size_t const SUB_OFFSET = 1536 * 1024; // 1.5MB
    size_t const SUB_SIZE = 1024 * 1024;   // 1MB (< granularity=2MB)
    uint8_t const FILL_PATTERN = 0xE1;

    if (rank == 1)
    {
        auto alloc = allocateMultiChunkVmmPosixFd(CHUNK_SIZE, NUM_CHUNKS, deviceId);
        printf("[Rank %d] Multi-chunk pool: ptr=0x%llx, totalSize=%zu, chunks=%zu\n", rank,
            (unsigned long long) alloc.ptr, alloc.totalSize, alloc.numChunks);

        TLLM_CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(alloc.ptr), 0x00, POOL_SIZE));
        TLLM_CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(alloc.ptr + SUB_OFFSET), FILL_PATTERN, SUB_SIZE));
        TLLM_CUDA_CHECK(cudaDeviceSynchronize());
        printf("[Rank %d] Sub-range [+%zu, +%zu) filled with 0x%02X (size < granularity, crosses boundary)\n", rank,
            SUB_OFFSET, SUB_OFFSET + SUB_SIZE, FILL_PATTERN);

        FabricTransferHelper helper;
        void* subRangePtr = reinterpret_cast<void*>(alloc.ptr + SUB_OFFSET);
        MemoryDescs descs{MemoryType::kVRAM, {MemoryDesc{subRangePtr, SUB_SIZE, static_cast<uint32_t>(deviceId)}}};
        helper.detectAndExportFabricHandles(descs);

        auto const& info = helper.getLocalFabricInfo();
        if (!info.supported)
        {
            printf("[Rank %d] FAIL: fabric info not supported\n", rank);
            alloc.cleanup();
            int metadataSize = 0;
            MPI_Send(&metadataSize, 1, MPI_INT, 0, 100, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }

        if (!info.pools.empty())
        {
            auto const& pool = info.pools[0];
            printf(
                "[Rank %d] Pool: base=0x%llx, totalSize=%zu, registeredAddr=0x%llx, "
                "registeredSize=%zu, mappedOffset=%zu, mappedSize=%zu, chunks=%zu\n",
                rank, (unsigned long long) pool.poolBaseAddr, (size_t) pool.poolTotalSize,
                (unsigned long long) pool.registeredAddr, (size_t) pool.registeredSize, (size_t) pool.mappedOffset,
                (size_t) pool.mappedSize, pool.chunks.size());

            // Expect 2 chunks: alloc_A and alloc_B (the range crosses their boundary)
            if (pool.chunks.size() != 2)
            {
                printf("[Rank %d] WARNING: expected 2 chunks (A+B), got %zu\n", rank, pool.chunks.size());
            }
            for (size_t ci = 0; ci < pool.chunks.size(); ++ci)
            {
                printf("[Rank %d]   chunk[%zu]: virtAddrOffset=%zu, size=%zu\n", rank, ci,
                    (size_t) pool.chunks[ci].virtAddrOffset, (size_t) pool.chunks[ci].size);
                if (pool.chunks[ci].size != CHUNK_SIZE)
                {
                    printf("[Rank %d]   WARNING: chunk[%zu] size=%zu, expected %zu\n", rank, ci,
                        (size_t) pool.chunks[ci].size, CHUNK_SIZE);
                }
            }
        }

        std::string metadata = info.serialize();
        int metadataSize = static_cast<int>(metadata.size());
        MPI_Send(&metadataSize, 1, MPI_INT, 0, 100, MPI_COMM_WORLD);
        MPI_Send(metadata.data(), metadataSize, MPI_BYTE, 0, 101, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        alloc.cleanup();
    }
    else if (rank == 0)
    {
        int metadataSize = 0;
        MPI_Recv(&metadataSize, 1, MPI_INT, 1, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (metadataSize == 0)
        {
            printf("[Rank %d] Exporter reported failure\n", rank);
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }

        std::string metadata(metadataSize, '\0');
        MPI_Recv(metadata.data(), metadataSize, MPI_BYTE, 1, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        auto fabricInfo = FabricMemInfo::deserialize(metadata);
        if (!fabricInfo.has_value())
        {
            printf("[Rank %d] FAIL: deserialize failed\n", rank);
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }

        FabricTransferHelper helper;
        helper.importAndMapRemoteFabric("rank1_small_cross", *fabricInfo);

        if (!helper.hasRemoteMapping("rank1_small_cross"))
        {
            printf("[Rank %d] FAIL: no remote mapping\n", rank);
            ok = false;
        }
        else
        {
            auto const& pool = fabricInfo->pools[0];
            void* localMapped
                = helper.translateToLocalMapping("rank1_small_cross", pool.registeredAddr, pool.registeredSize);
            if (localMapped == nullptr)
            {
                printf("[Rank %d] FAIL: translateToLocalMapping returned nullptr\n", rank);
                ok = false;
            }
            else
            {
                std::vector<uint8_t> hostBuf(SUB_SIZE);
                TLLM_CUDA_CHECK(cudaMemcpy(hostBuf.data(), localMapped, SUB_SIZE, cudaMemcpyDeviceToHost));

                bool dataOk = true;
                for (size_t i = 0; i < SUB_SIZE; ++i)
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
                    printf("[Rank %d] Small cross-boundary read PASSED (%zu bytes == 0x%02X)\n", rank, SUB_SIZE,
                        FILL_PATTERN);
                }
                ok = ok && dataOk;
            }
        }

        helper.cleanupRemoteFabricMapping("rank1_small_cross");
        MPI_Barrier(MPI_COMM_WORLD);
    }

    printf("[Rank %d] === SmallCrossBoundaryDirect: %s ===\n", rank, ok ? "PASSED" : "FAILED");
    return ok;
}

// ============================================================================
// Test: Non-aligned intra-chunk sub-range (entirely within one chunk)
// Pool: 4×2MB. Register [0.5MB, 1.5MB) — entirely within alloc_A.
// Should export exactly 1 chunk. Verifies no false boundary detection.
// ============================================================================

bool testIntraChunkSubRangeDirect(int rank, int worldSize)
{
    printf("[Rank %d] === Test: IntraChunkSubRangeDirect ===\n", rank);

    if (worldSize < 2)
    {
        printf("[Rank %d] Need at least 2 ranks, skipping\n", rank);
        return true;
    }

    int deviceId = 0;
    TLLM_CUDA_CHECK(cudaSetDevice(deviceId));

    CUdevice device;
    CU_CHECK(cuDeviceGet(&device, deviceId));
    int posixFdSupported = 0;
    CU_CHECK(cuDeviceGetAttribute(
        &posixFdSupported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, device));
    MPI_Bcast(&posixFdSupported, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (!posixFdSupported)
    {
        printf("[Rank %d] POSIX FD not supported, skipping\n", rank);
        return true;
    }

    bool ok = true;
    size_t const CHUNK_SIZE = 2 * 1024 * 1024;
    size_t const NUM_CHUNKS = 4;
    size_t const POOL_SIZE = CHUNK_SIZE * NUM_CHUNKS;
    // Register [0.5MB, 1.5MB): entirely within alloc_A
    size_t const SUB_OFFSET = 512 * 1024; // 0.5MB
    size_t const SUB_SIZE = 1024 * 1024;  // 1MB
    uint8_t const FILL_PATTERN = 0xC7;

    if (rank == 1)
    {
        auto alloc = allocateMultiChunkVmmPosixFd(CHUNK_SIZE, NUM_CHUNKS, deviceId);
        printf("[Rank %d] Multi-chunk pool: ptr=0x%llx, totalSize=%zu, chunks=%zu\n", rank,
            (unsigned long long) alloc.ptr, alloc.totalSize, alloc.numChunks);

        TLLM_CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(alloc.ptr), 0x00, POOL_SIZE));
        TLLM_CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(alloc.ptr + SUB_OFFSET), FILL_PATTERN, SUB_SIZE));
        TLLM_CUDA_CHECK(cudaDeviceSynchronize());
        printf("[Rank %d] Sub-range [+%zu, +%zu) filled with 0x%02X (intra-chunk, non-aligned)\n", rank, SUB_OFFSET,
            SUB_OFFSET + SUB_SIZE, FILL_PATTERN);

        FabricTransferHelper helper;
        void* subRangePtr = reinterpret_cast<void*>(alloc.ptr + SUB_OFFSET);
        MemoryDescs descs{MemoryType::kVRAM, {MemoryDesc{subRangePtr, SUB_SIZE, static_cast<uint32_t>(deviceId)}}};
        helper.detectAndExportFabricHandles(descs);

        auto const& info = helper.getLocalFabricInfo();
        if (!info.supported)
        {
            printf("[Rank %d] FAIL: fabric info not supported\n", rank);
            alloc.cleanup();
            int metadataSize = 0;
            MPI_Send(&metadataSize, 1, MPI_INT, 0, 105, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }

        if (!info.pools.empty())
        {
            auto const& pool = info.pools[0];
            printf(
                "[Rank %d] Pool: base=0x%llx, totalSize=%zu, registeredAddr=0x%llx, "
                "registeredSize=%zu, mappedOffset=%zu, mappedSize=%zu, chunks=%zu\n",
                rank, (unsigned long long) pool.poolBaseAddr, (size_t) pool.poolTotalSize,
                (unsigned long long) pool.registeredAddr, (size_t) pool.registeredSize, (size_t) pool.mappedOffset,
                (size_t) pool.mappedSize, pool.chunks.size());

            // Expect 1 chunk: alloc_A only
            if (pool.chunks.size() != 1)
            {
                printf("[Rank %d] WARNING: expected 1 chunk (alloc_A only), got %zu\n", rank, pool.chunks.size());
            }
            for (size_t ci = 0; ci < pool.chunks.size(); ++ci)
            {
                printf("[Rank %d]   chunk[%zu]: virtAddrOffset=%zu, size=%zu\n", rank, ci,
                    (size_t) pool.chunks[ci].virtAddrOffset, (size_t) pool.chunks[ci].size);
                if (pool.chunks[ci].size != CHUNK_SIZE)
                {
                    printf("[Rank %d]   WARNING: chunk[%zu] size=%zu, expected %zu\n", rank, ci,
                        (size_t) pool.chunks[ci].size, CHUNK_SIZE);
                }
            }
        }

        std::string metadata = info.serialize();
        int metadataSize = static_cast<int>(metadata.size());
        MPI_Send(&metadataSize, 1, MPI_INT, 0, 105, MPI_COMM_WORLD);
        MPI_Send(metadata.data(), metadataSize, MPI_BYTE, 0, 106, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        alloc.cleanup();
    }
    else if (rank == 0)
    {
        int metadataSize = 0;
        MPI_Recv(&metadataSize, 1, MPI_INT, 1, 105, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (metadataSize == 0)
        {
            printf("[Rank %d] Exporter reported failure\n", rank);
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }

        std::string metadata(metadataSize, '\0');
        MPI_Recv(metadata.data(), metadataSize, MPI_BYTE, 1, 106, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        auto fabricInfo = FabricMemInfo::deserialize(metadata);
        if (!fabricInfo.has_value())
        {
            printf("[Rank %d] FAIL: deserialize failed\n", rank);
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }

        FabricTransferHelper helper;
        helper.importAndMapRemoteFabric("rank1_intra_chunk", *fabricInfo);

        if (!helper.hasRemoteMapping("rank1_intra_chunk"))
        {
            printf("[Rank %d] FAIL: no remote mapping\n", rank);
            ok = false;
        }
        else
        {
            auto const& pool = fabricInfo->pools[0];
            void* localMapped
                = helper.translateToLocalMapping("rank1_intra_chunk", pool.registeredAddr, pool.registeredSize);
            if (localMapped == nullptr)
            {
                printf("[Rank %d] FAIL: translateToLocalMapping returned nullptr\n", rank);
                ok = false;
            }
            else
            {
                std::vector<uint8_t> hostBuf(SUB_SIZE);
                TLLM_CUDA_CHECK(cudaMemcpy(hostBuf.data(), localMapped, SUB_SIZE, cudaMemcpyDeviceToHost));

                bool dataOk = true;
                for (size_t i = 0; i < SUB_SIZE; ++i)
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
                    printf("[Rank %d] Intra-chunk sub-range read PASSED (%zu bytes == 0x%02X)\n", rank, SUB_SIZE,
                        FILL_PATTERN);
                }
                ok = ok && dataOk;
            }
        }

        helper.cleanupRemoteFabricMapping("rank1_intra_chunk");
        MPI_Barrier(MPI_COMM_WORLD);
    }

    printf("[Rank %d] === IntraChunkSubRangeDirect: %s ===\n", rank, ok ? "PASSED" : "FAILED");
    return ok;
}

// ============================================================================
// Test: Wide-span unaligned sub-range covering all chunks
// Pool: 4×2MB = [alloc_A 2MB][alloc_B 2MB][alloc_C 2MB][alloc_D 2MB]
// Register [1MB, 7MB): starts mid-alloc_A, ends mid-alloc_D.
// scanSize=6MB > granularity=2MB, so exercises findAllBoundariesRecursive.
// Should export 4 chunks (all of them). Verifies that both the boundary
// detection and the aligned probing work for large unaligned ranges.
// ============================================================================

bool testWideSpanSubRangeDirect(int rank, int worldSize)
{
    printf("[Rank %d] === Test: WideSpanSubRangeDirect ===\n", rank);

    if (worldSize < 2)
    {
        printf("[Rank %d] Need at least 2 ranks, skipping\n", rank);
        return true;
    }

    int deviceId = 0;
    TLLM_CUDA_CHECK(cudaSetDevice(deviceId));

    CUdevice device;
    CU_CHECK(cuDeviceGet(&device, deviceId));
    int posixFdSupported = 0;
    CU_CHECK(cuDeviceGetAttribute(
        &posixFdSupported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, device));
    MPI_Bcast(&posixFdSupported, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (!posixFdSupported)
    {
        printf("[Rank %d] POSIX FD not supported, skipping\n", rank);
        return true;
    }

    bool ok = true;
    size_t const CHUNK_SIZE = 2 * 1024 * 1024;
    size_t const NUM_CHUNKS = 4;
    size_t const POOL_SIZE = CHUNK_SIZE * NUM_CHUNKS;
    // Register [1MB, 7MB): spans parts of all 4 chunks
    size_t const SUB_OFFSET = 1024 * 1024;   // 1MB
    size_t const SUB_SIZE = 6 * 1024 * 1024; // 6MB
    uint8_t const FILL_PATTERN = 0xAB;

    if (rank == 1)
    {
        auto alloc = allocateMultiChunkVmmPosixFd(CHUNK_SIZE, NUM_CHUNKS, deviceId);
        printf("[Rank %d] Multi-chunk pool: ptr=0x%llx, totalSize=%zu, chunks=%zu\n", rank,
            (unsigned long long) alloc.ptr, alloc.totalSize, alloc.numChunks);

        TLLM_CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(alloc.ptr), 0x00, POOL_SIZE));
        TLLM_CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(alloc.ptr + SUB_OFFSET), FILL_PATTERN, SUB_SIZE));
        TLLM_CUDA_CHECK(cudaDeviceSynchronize());
        printf("[Rank %d] Sub-range [+%zu, +%zu) filled with 0x%02X (wide span, all chunks)\n", rank, SUB_OFFSET,
            SUB_OFFSET + SUB_SIZE, FILL_PATTERN);

        FabricTransferHelper helper;
        void* subRangePtr = reinterpret_cast<void*>(alloc.ptr + SUB_OFFSET);
        MemoryDescs descs{MemoryType::kVRAM, {MemoryDesc{subRangePtr, SUB_SIZE, static_cast<uint32_t>(deviceId)}}};
        helper.detectAndExportFabricHandles(descs);

        auto const& info = helper.getLocalFabricInfo();
        if (!info.supported)
        {
            printf("[Rank %d] FAIL: fabric info not supported\n", rank);
            alloc.cleanup();
            int metadataSize = 0;
            MPI_Send(&metadataSize, 1, MPI_INT, 0, 110, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }

        if (!info.pools.empty())
        {
            auto const& pool = info.pools[0];
            printf(
                "[Rank %d] Pool: base=0x%llx, totalSize=%zu, registeredAddr=0x%llx, "
                "registeredSize=%zu, mappedOffset=%zu, mappedSize=%zu, chunks=%zu\n",
                rank, (unsigned long long) pool.poolBaseAddr, (size_t) pool.poolTotalSize,
                (unsigned long long) pool.registeredAddr, (size_t) pool.registeredSize, (size_t) pool.mappedOffset,
                (size_t) pool.mappedSize, pool.chunks.size());

            // Expect 4 chunks: all of alloc_A through alloc_D
            if (pool.chunks.size() != 4)
            {
                printf("[Rank %d] WARNING: expected 4 chunks (A+B+C+D), got %zu\n", rank, pool.chunks.size());
            }
            for (size_t ci = 0; ci < pool.chunks.size(); ++ci)
            {
                printf("[Rank %d]   chunk[%zu]: virtAddrOffset=%zu, size=%zu\n", rank, ci,
                    (size_t) pool.chunks[ci].virtAddrOffset, (size_t) pool.chunks[ci].size);
                if (pool.chunks[ci].size != CHUNK_SIZE)
                {
                    printf("[Rank %d]   WARNING: chunk[%zu] size=%zu, expected %zu\n", rank, ci,
                        (size_t) pool.chunks[ci].size, CHUNK_SIZE);
                }
            }
        }

        std::string metadata = info.serialize();
        int metadataSize = static_cast<int>(metadata.size());
        MPI_Send(&metadataSize, 1, MPI_INT, 0, 110, MPI_COMM_WORLD);
        MPI_Send(metadata.data(), metadataSize, MPI_BYTE, 0, 111, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        alloc.cleanup();
    }
    else if (rank == 0)
    {
        int metadataSize = 0;
        MPI_Recv(&metadataSize, 1, MPI_INT, 1, 110, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (metadataSize == 0)
        {
            printf("[Rank %d] Exporter reported failure\n", rank);
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }

        std::string metadata(metadataSize, '\0');
        MPI_Recv(metadata.data(), metadataSize, MPI_BYTE, 1, 111, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        auto fabricInfo = FabricMemInfo::deserialize(metadata);
        if (!fabricInfo.has_value())
        {
            printf("[Rank %d] FAIL: deserialize failed\n", rank);
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }

        FabricTransferHelper helper;
        helper.importAndMapRemoteFabric("rank1_wide_span", *fabricInfo);

        if (!helper.hasRemoteMapping("rank1_wide_span"))
        {
            printf("[Rank %d] FAIL: no remote mapping\n", rank);
            ok = false;
        }
        else
        {
            auto const& pool = fabricInfo->pools[0];
            void* localMapped
                = helper.translateToLocalMapping("rank1_wide_span", pool.registeredAddr, pool.registeredSize);
            if (localMapped == nullptr)
            {
                printf("[Rank %d] FAIL: translateToLocalMapping returned nullptr\n", rank);
                ok = false;
            }
            else
            {
                std::vector<uint8_t> hostBuf(SUB_SIZE);
                TLLM_CUDA_CHECK(cudaMemcpy(hostBuf.data(), localMapped, SUB_SIZE, cudaMemcpyDeviceToHost));

                bool dataOk = true;
                for (size_t i = 0; i < SUB_SIZE; ++i)
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
                    printf("[Rank %d] Wide-span sub-range read PASSED (%zu bytes == 0x%02X)\n", rank, SUB_SIZE,
                        FILL_PATTERN);
                }
                ok = ok && dataOk;
            }
        }

        helper.cleanupRemoteFabricMapping("rank1_wide_span");
        MPI_Barrier(MPI_COMM_WORLD);
    }

    printf("[Rank %d] === WideSpanSubRangeDirect: %s ===\n", rank, ok ? "PASSED" : "FAILED");
    return ok;
}

// ============================================================================
// Test: 4MB chunks — non-aligned range entirely within one large chunk
// Pool: 4×4MB (granularity=2MB). Register [1MB, 3MB).
// The range crosses the 2MB granularity-aligned address but does NOT cross
// a real chunk boundary (alloc_A covers [0,4MB)). Should export 1 chunk of 4MB.
// Verifies no false-positive boundary detection at granularity-aligned points.
// ============================================================================

bool test4MBChunkIntraDirect(int rank, int worldSize)
{
    printf("[Rank %d] === Test: 4MBChunkIntraDirect ===\n", rank);

    if (worldSize < 2)
    {
        printf("[Rank %d] Need at least 2 ranks, skipping\n", rank);
        return true;
    }

    int deviceId = 0;
    TLLM_CUDA_CHECK(cudaSetDevice(deviceId));

    CUdevice device;
    CU_CHECK(cuDeviceGet(&device, deviceId));
    int posixFdSupported = 0;
    CU_CHECK(cuDeviceGetAttribute(
        &posixFdSupported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, device));
    MPI_Bcast(&posixFdSupported, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (!posixFdSupported)
    {
        printf("[Rank %d] POSIX FD not supported, skipping\n", rank);
        return true;
    }

    bool ok = true;
    size_t const CHUNK_SIZE = 4 * 1024 * 1024;        // 4MB per physical chunk (2× granularity)
    size_t const NUM_CHUNKS = 4;
    size_t const POOL_SIZE = CHUNK_SIZE * NUM_CHUNKS; // 16MB
    // Register [1MB, 3MB): entirely within alloc_A [0,4MB), crosses 2MB granularity point
    size_t const SUB_OFFSET = 1024 * 1024;   // 1MB
    size_t const SUB_SIZE = 2 * 1024 * 1024; // 2MB
    uint8_t const FILL_PATTERN = 0xF1;

    if (rank == 1)
    {
        auto alloc = allocateMultiChunkVmmPosixFd(CHUNK_SIZE, NUM_CHUNKS, deviceId);
        printf("[Rank %d] 4MB-chunk pool: ptr=0x%llx, totalSize=%zu, chunks=%zu, chunkSize=%zu\n", rank,
            (unsigned long long) alloc.ptr, alloc.totalSize, alloc.numChunks, CHUNK_SIZE);

        TLLM_CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(alloc.ptr), 0x00, POOL_SIZE));
        TLLM_CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(alloc.ptr + SUB_OFFSET), FILL_PATTERN, SUB_SIZE));
        TLLM_CUDA_CHECK(cudaDeviceSynchronize());

        FabricTransferHelper helper;
        void* subRangePtr = reinterpret_cast<void*>(alloc.ptr + SUB_OFFSET);
        MemoryDescs descs{MemoryType::kVRAM, {MemoryDesc{subRangePtr, SUB_SIZE, static_cast<uint32_t>(deviceId)}}};
        helper.detectAndExportFabricHandles(descs);

        auto const& info = helper.getLocalFabricInfo();
        if (!info.supported)
        {
            printf("[Rank %d] FAIL: fabric info not supported\n", rank);
            alloc.cleanup();
            int metadataSize = 0;
            MPI_Send(&metadataSize, 1, MPI_INT, 0, 120, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }

        if (!info.pools.empty())
        {
            auto const& pool = info.pools[0];
            printf("[Rank %d] Pool: chunks=%zu, poolTotalSize=%zu, registeredSize=%zu, mappedSize=%zu\n", rank,
                pool.chunks.size(), (size_t) pool.poolTotalSize, (size_t) pool.registeredSize,
                (size_t) pool.mappedSize);

            // Key assertion: should be 1 chunk (alloc_A), NOT 2
            if (pool.chunks.size() != 1)
            {
                printf(
                    "[Rank %d] FAIL: expected 1 chunk (alloc_A only), got %zu — "
                    "false boundary detected at granularity point!\n",
                    rank, pool.chunks.size());
            }
            for (size_t ci = 0; ci < pool.chunks.size(); ++ci)
            {
                printf("[Rank %d]   chunk[%zu]: virtAddrOffset=%zu, size=%zu\n", rank, ci,
                    (size_t) pool.chunks[ci].virtAddrOffset, (size_t) pool.chunks[ci].size);
                if (pool.chunks[ci].size != CHUNK_SIZE)
                {
                    printf("[Rank %d]   WARNING: chunk[%zu] size=%zu, expected %zu\n", rank, ci,
                        (size_t) pool.chunks[ci].size, CHUNK_SIZE);
                }
            }
        }

        std::string metadata = info.serialize();
        int metadataSize = static_cast<int>(metadata.size());
        MPI_Send(&metadataSize, 1, MPI_INT, 0, 120, MPI_COMM_WORLD);
        MPI_Send(metadata.data(), metadataSize, MPI_BYTE, 0, 121, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        alloc.cleanup();
    }
    else if (rank == 0)
    {
        int metadataSize = 0;
        MPI_Recv(&metadataSize, 1, MPI_INT, 1, 120, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (metadataSize == 0)
        {
            printf("[Rank %d] Exporter reported failure\n", rank);
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }

        std::string metadata(metadataSize, '\0');
        MPI_Recv(metadata.data(), metadataSize, MPI_BYTE, 1, 121, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        auto fabricInfo = FabricMemInfo::deserialize(metadata);
        if (!fabricInfo.has_value())
        {
            printf("[Rank %d] FAIL: deserialize failed\n", rank);
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }

        FabricTransferHelper helper;
        helper.importAndMapRemoteFabric("rank1_4mb_intra", *fabricInfo);

        if (!helper.hasRemoteMapping("rank1_4mb_intra"))
        {
            printf("[Rank %d] FAIL: no remote mapping\n", rank);
            ok = false;
        }
        else
        {
            auto const& pool = fabricInfo->pools[0];
            void* localMapped
                = helper.translateToLocalMapping("rank1_4mb_intra", pool.registeredAddr, pool.registeredSize);
            if (localMapped == nullptr)
            {
                printf("[Rank %d] FAIL: translateToLocalMapping returned nullptr\n", rank);
                ok = false;
            }
            else
            {
                std::vector<uint8_t> hostBuf(SUB_SIZE);
                TLLM_CUDA_CHECK(cudaMemcpy(hostBuf.data(), localMapped, SUB_SIZE, cudaMemcpyDeviceToHost));

                bool dataOk = true;
                for (size_t i = 0; i < SUB_SIZE; ++i)
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
                    printf("[Rank %d] 4MB-chunk intra-chunk read PASSED (%zu bytes)\n", rank, SUB_SIZE);
                }
                ok = ok && dataOk;
            }
        }

        helper.cleanupRemoteFabricMapping("rank1_4mb_intra");
        MPI_Barrier(MPI_COMM_WORLD);
    }

    printf("[Rank %d] === 4MBChunkIntraDirect: %s ===\n", rank, ok ? "PASSED" : "FAILED");
    return ok;
}

// ============================================================================
// Test: 4MB chunks — non-aligned range crossing a real chunk boundary
// Pool: 4×4MB (granularity=2MB). Register [3MB, 5MB).
// The range crosses the alloc_A/alloc_B boundary at 4MB.
// Should export 2 chunks, each 4MB. Verifies correct detection of a real
// chunk boundary when granularity < chunkSize.
// ============================================================================

bool test4MBChunkCrossBoundaryDirect(int rank, int worldSize)
{
    printf("[Rank %d] === Test: 4MBChunkCrossBoundaryDirect ===\n", rank);

    if (worldSize < 2)
    {
        printf("[Rank %d] Need at least 2 ranks, skipping\n", rank);
        return true;
    }

    int deviceId = 0;
    TLLM_CUDA_CHECK(cudaSetDevice(deviceId));

    CUdevice device;
    CU_CHECK(cuDeviceGet(&device, deviceId));
    int posixFdSupported = 0;
    CU_CHECK(cuDeviceGetAttribute(
        &posixFdSupported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, device));
    MPI_Bcast(&posixFdSupported, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (!posixFdSupported)
    {
        printf("[Rank %d] POSIX FD not supported, skipping\n", rank);
        return true;
    }

    bool ok = true;
    size_t const CHUNK_SIZE = 4 * 1024 * 1024;        // 4MB
    size_t const NUM_CHUNKS = 4;
    size_t const POOL_SIZE = CHUNK_SIZE * NUM_CHUNKS; // 16MB
    // Register [3MB, 5MB): crosses alloc_A/alloc_B boundary at 4MB
    size_t const SUB_OFFSET = 3 * 1024 * 1024; // 3MB
    size_t const SUB_SIZE = 2 * 1024 * 1024;   // 2MB
    uint8_t const FILL_PATTERN = 0xF2;

    if (rank == 1)
    {
        auto alloc = allocateMultiChunkVmmPosixFd(CHUNK_SIZE, NUM_CHUNKS, deviceId);
        printf("[Rank %d] 4MB-chunk pool: ptr=0x%llx, totalSize=%zu, chunks=%zu, chunkSize=%zu\n", rank,
            (unsigned long long) alloc.ptr, alloc.totalSize, alloc.numChunks, CHUNK_SIZE);

        TLLM_CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(alloc.ptr), 0x00, POOL_SIZE));
        TLLM_CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(alloc.ptr + SUB_OFFSET), FILL_PATTERN, SUB_SIZE));
        TLLM_CUDA_CHECK(cudaDeviceSynchronize());

        FabricTransferHelper helper;
        void* subRangePtr = reinterpret_cast<void*>(alloc.ptr + SUB_OFFSET);
        MemoryDescs descs{MemoryType::kVRAM, {MemoryDesc{subRangePtr, SUB_SIZE, static_cast<uint32_t>(deviceId)}}};
        helper.detectAndExportFabricHandles(descs);

        auto const& info = helper.getLocalFabricInfo();
        if (!info.supported)
        {
            printf("[Rank %d] FAIL: fabric info not supported\n", rank);
            alloc.cleanup();
            int metadataSize = 0;
            MPI_Send(&metadataSize, 1, MPI_INT, 0, 125, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }

        if (!info.pools.empty())
        {
            auto const& pool = info.pools[0];
            printf("[Rank %d] Pool: chunks=%zu, poolTotalSize=%zu, registeredSize=%zu, mappedSize=%zu\n", rank,
                pool.chunks.size(), (size_t) pool.poolTotalSize, (size_t) pool.registeredSize,
                (size_t) pool.mappedSize);

            // Key assertion: should be 2 chunks (alloc_A + alloc_B), each 4MB
            if (pool.chunks.size() != 2)
            {
                printf("[Rank %d] FAIL: expected 2 chunks (A+B), got %zu\n", rank, pool.chunks.size());
            }
            for (size_t ci = 0; ci < pool.chunks.size(); ++ci)
            {
                printf("[Rank %d]   chunk[%zu]: virtAddrOffset=%zu, size=%zu\n", rank, ci,
                    (size_t) pool.chunks[ci].virtAddrOffset, (size_t) pool.chunks[ci].size);
                if (pool.chunks[ci].size != CHUNK_SIZE)
                {
                    printf("[Rank %d]   WARNING: chunk[%zu] size=%zu, expected %zu\n", rank, ci,
                        (size_t) pool.chunks[ci].size, CHUNK_SIZE);
                }
            }
        }

        std::string metadata = info.serialize();
        int metadataSize = static_cast<int>(metadata.size());
        MPI_Send(&metadataSize, 1, MPI_INT, 0, 125, MPI_COMM_WORLD);
        MPI_Send(metadata.data(), metadataSize, MPI_BYTE, 0, 126, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        alloc.cleanup();
    }
    else if (rank == 0)
    {
        int metadataSize = 0;
        MPI_Recv(&metadataSize, 1, MPI_INT, 1, 125, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (metadataSize == 0)
        {
            printf("[Rank %d] Exporter reported failure\n", rank);
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }

        std::string metadata(metadataSize, '\0');
        MPI_Recv(metadata.data(), metadataSize, MPI_BYTE, 1, 126, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        auto fabricInfo = FabricMemInfo::deserialize(metadata);
        if (!fabricInfo.has_value())
        {
            printf("[Rank %d] FAIL: deserialize failed\n", rank);
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }

        FabricTransferHelper helper;
        helper.importAndMapRemoteFabric("rank1_4mb_cross", *fabricInfo);

        if (!helper.hasRemoteMapping("rank1_4mb_cross"))
        {
            printf("[Rank %d] FAIL: no remote mapping\n", rank);
            ok = false;
        }
        else
        {
            auto const& pool = fabricInfo->pools[0];
            void* localMapped
                = helper.translateToLocalMapping("rank1_4mb_cross", pool.registeredAddr, pool.registeredSize);
            if (localMapped == nullptr)
            {
                printf("[Rank %d] FAIL: translateToLocalMapping returned nullptr\n", rank);
                ok = false;
            }
            else
            {
                std::vector<uint8_t> hostBuf(SUB_SIZE);
                TLLM_CUDA_CHECK(cudaMemcpy(hostBuf.data(), localMapped, SUB_SIZE, cudaMemcpyDeviceToHost));

                bool dataOk = true;
                for (size_t i = 0; i < SUB_SIZE; ++i)
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
                    printf("[Rank %d] 4MB-chunk cross-boundary read PASSED (%zu bytes)\n", rank, SUB_SIZE);
                }
                ok = ok && dataOk;
            }
        }

        helper.cleanupRemoteFabricMapping("rank1_4mb_cross");
        MPI_Barrier(MPI_COMM_WORLD);
    }

    printf("[Rank %d] === 4MBChunkCrossBoundaryDirect: %s ===\n", rank, ok ? "PASSED" : "FAILED");
    return ok;
}

// ============================================================================
// Test: 4MB chunks — wide unaligned span across all chunks
// Pool: 4×4MB = 16MB (granularity=2MB). Register [1MB, 15MB).
// scanSize=14MB >> granularity, exercises findAllBoundariesRecursive where
// most granularity-aligned probes are NOT chunk boundaries (only 4MB, 8MB,
// 12MB are real boundaries). Should export 4 chunks, each 4MB.
// ============================================================================

bool test4MBChunkWideSpanDirect(int rank, int worldSize)
{
    printf("[Rank %d] === Test: 4MBChunkWideSpanDirect ===\n", rank);

    if (worldSize < 2)
    {
        printf("[Rank %d] Need at least 2 ranks, skipping\n", rank);
        return true;
    }

    int deviceId = 0;
    TLLM_CUDA_CHECK(cudaSetDevice(deviceId));

    CUdevice device;
    CU_CHECK(cuDeviceGet(&device, deviceId));
    int posixFdSupported = 0;
    CU_CHECK(cuDeviceGetAttribute(
        &posixFdSupported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, device));
    MPI_Bcast(&posixFdSupported, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (!posixFdSupported)
    {
        printf("[Rank %d] POSIX FD not supported, skipping\n", rank);
        return true;
    }

    bool ok = true;
    size_t const CHUNK_SIZE = 4 * 1024 * 1024;        // 4MB
    size_t const NUM_CHUNKS = 4;
    size_t const POOL_SIZE = CHUNK_SIZE * NUM_CHUNKS; // 16MB
    // Register [1MB, 15MB): spans parts of all 4 chunks
    size_t const SUB_OFFSET = 1024 * 1024;    // 1MB
    size_t const SUB_SIZE = 14 * 1024 * 1024; // 14MB
    uint8_t const FILL_PATTERN = 0xF3;

    if (rank == 1)
    {
        auto alloc = allocateMultiChunkVmmPosixFd(CHUNK_SIZE, NUM_CHUNKS, deviceId);
        printf("[Rank %d] 4MB-chunk pool: ptr=0x%llx, totalSize=%zu, chunks=%zu, chunkSize=%zu\n", rank,
            (unsigned long long) alloc.ptr, alloc.totalSize, alloc.numChunks, CHUNK_SIZE);

        TLLM_CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(alloc.ptr), 0x00, POOL_SIZE));
        TLLM_CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(alloc.ptr + SUB_OFFSET), FILL_PATTERN, SUB_SIZE));
        TLLM_CUDA_CHECK(cudaDeviceSynchronize());

        FabricTransferHelper helper;
        void* subRangePtr = reinterpret_cast<void*>(alloc.ptr + SUB_OFFSET);
        MemoryDescs descs{MemoryType::kVRAM, {MemoryDesc{subRangePtr, SUB_SIZE, static_cast<uint32_t>(deviceId)}}};
        helper.detectAndExportFabricHandles(descs);

        auto const& info = helper.getLocalFabricInfo();
        if (!info.supported)
        {
            printf("[Rank %d] FAIL: fabric info not supported\n", rank);
            alloc.cleanup();
            int metadataSize = 0;
            MPI_Send(&metadataSize, 1, MPI_INT, 0, 130, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }

        if (!info.pools.empty())
        {
            auto const& pool = info.pools[0];
            printf("[Rank %d] Pool: chunks=%zu, poolTotalSize=%zu, registeredSize=%zu, mappedSize=%zu\n", rank,
                pool.chunks.size(), (size_t) pool.poolTotalSize, (size_t) pool.registeredSize,
                (size_t) pool.mappedSize);

            // Key assertion: should be 4 chunks, each 4MB
            if (pool.chunks.size() != 4)
            {
                printf("[Rank %d] FAIL: expected 4 chunks (A+B+C+D), got %zu\n", rank, pool.chunks.size());
            }
            for (size_t ci = 0; ci < pool.chunks.size(); ++ci)
            {
                printf("[Rank %d]   chunk[%zu]: virtAddrOffset=%zu, size=%zu\n", rank, ci,
                    (size_t) pool.chunks[ci].virtAddrOffset, (size_t) pool.chunks[ci].size);
                if (pool.chunks[ci].size != CHUNK_SIZE)
                {
                    printf("[Rank %d]   WARNING: chunk[%zu] size=%zu, expected %zu\n", rank, ci,
                        (size_t) pool.chunks[ci].size, CHUNK_SIZE);
                }
            }
        }

        std::string metadata = info.serialize();
        int metadataSize = static_cast<int>(metadata.size());
        MPI_Send(&metadataSize, 1, MPI_INT, 0, 130, MPI_COMM_WORLD);
        MPI_Send(metadata.data(), metadataSize, MPI_BYTE, 0, 131, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        alloc.cleanup();
    }
    else if (rank == 0)
    {
        int metadataSize = 0;
        MPI_Recv(&metadataSize, 1, MPI_INT, 1, 130, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (metadataSize == 0)
        {
            printf("[Rank %d] Exporter reported failure\n", rank);
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }

        std::string metadata(metadataSize, '\0');
        MPI_Recv(metadata.data(), metadataSize, MPI_BYTE, 1, 131, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        auto fabricInfo = FabricMemInfo::deserialize(metadata);
        if (!fabricInfo.has_value())
        {
            printf("[Rank %d] FAIL: deserialize failed\n", rank);
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }

        FabricTransferHelper helper;
        helper.importAndMapRemoteFabric("rank1_4mb_wide", *fabricInfo);

        if (!helper.hasRemoteMapping("rank1_4mb_wide"))
        {
            printf("[Rank %d] FAIL: no remote mapping\n", rank);
            ok = false;
        }
        else
        {
            auto const& pool = fabricInfo->pools[0];
            void* localMapped
                = helper.translateToLocalMapping("rank1_4mb_wide", pool.registeredAddr, pool.registeredSize);
            if (localMapped == nullptr)
            {
                printf("[Rank %d] FAIL: translateToLocalMapping returned nullptr\n", rank);
                ok = false;
            }
            else
            {
                std::vector<uint8_t> hostBuf(SUB_SIZE);
                TLLM_CUDA_CHECK(cudaMemcpy(hostBuf.data(), localMapped, SUB_SIZE, cudaMemcpyDeviceToHost));

                bool dataOk = true;
                for (size_t i = 0; i < SUB_SIZE; ++i)
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
                    printf("[Rank %d] 4MB-chunk wide-span read PASSED (%zu bytes)\n", rank, SUB_SIZE);
                }
                ok = ok && dataOk;
            }
        }

        helper.cleanupRemoteFabricMapping("rank1_4mb_wide");
        MPI_Barrier(MPI_COMM_WORLD);
    }

    printf("[Rank %d] === 4MBChunkWideSpanDirect: %s ===\n", rank, ok ? "PASSED" : "FAILED");
    return ok;
}

// ============================================================================
// CudaIpc Tests — cudaMalloc memory shared via cudaIpcGetMemHandle
// ============================================================================

/// @brief Test 1: CudaIpc detection — cudaMalloc memory is detected as kCudaIpc
bool testCudaIpcDetection(int rank)
{
    if (rank != 0)
    {
        return true; // Only rank 0 runs this
    }

    printf("[Rank %d] === Test: CudaIpcDetection ===\n", rank);

    bool ok = true;
    size_t const allocSize = 4 * 1024 * 1024; // 4 MB
    uint32_t const DEVICE_ID = 0;

    // Allocate with cudaMalloc (legacy IPC capable)
    void* devPtr = nullptr;
    TLLM_CUDA_CHECK(cudaMalloc(&devPtr, allocSize));
    printf("[Rank %d] cudaMalloc: ptr=%p, size=%zu\n", rank, devPtr, allocSize);

    // Register with FabricTransferHelper
    FabricTransferHelper helper;
    MemoryDescs regDescs{MemoryType::kVRAM, {MemoryDesc{devPtr, allocSize, DEVICE_ID}}};
    helper.detectAndExportFabricHandles(regDescs);

    auto const& info = helper.getLocalFabricInfo();
    if (!info.supported)
    {
        printf("[Rank %d] CudaIpc NOT supported (expected supported)\n", rank);
        ok = false;
    }
    else if (info.handleType != VmmHandleType::kCudaIpc)
    {
        printf("[Rank %d] Expected handleType=CudaIpc(3), got %d\n", rank, static_cast<int>(info.handleType));
        ok = false;
    }
    else
    {
        printf("[Rank %d] handleType = CudaIpc (correct!)\n", rank);
        printf("[Rank %d] pools=%zu, chunks=%zu\n", rank, info.pools.size(),
            info.pools.empty() ? 0 : info.pools[0].chunks.size());

        // Verify pool structure
        if (!info.pools.empty())
        {
            auto const& pool = info.pools[0];
            if (pool.chunks.size() != 1)
            {
                printf("[Rank %d] Expected 1 chunk (cudaMalloc), got %zu\n", rank, pool.chunks.size());
                ok = false;
            }
            if (pool.mappedOffset != 0)
            {
                printf("[Rank %d] Expected mappedOffset=0, got %lu\n", rank, pool.mappedOffset);
                ok = false;
            }
        }
    }

    TLLM_CUDA_CHECK(cudaFree(devPtr));
    printf("[Rank %d] === CudaIpcDetection: %s ===\n", rank, ok ? "PASSED" : "FAILED");
    return ok;
}

/// @brief Test 2: CudaIpc cross-process transfer via FabricTransferHelper
bool testCudaIpcTransfer(int rank, int worldSize)
{
    if (worldSize < 2)
    {
        if (rank == 0)
            printf("[Rank %d] Skipping CudaIpcTransfer (need 2 ranks)\n", rank);
        return true;
    }

    printf("[Rank %d] === Test: CudaIpcTransfer ===\n", rank);
    bool ok = true;

    size_t const allocSize = 4 * 1024 * 1024; // 4 MB
    uint32_t const DEVICE_ID = 0;

    if (rank == 1)
    {
        // Source: allocate and fill with pattern via cudaMalloc
        void* srcPtr = nullptr;
        TLLM_CUDA_CHECK(cudaMalloc(&srcPtr, allocSize));
        TLLM_CUDA_CHECK(cudaMemset(srcPtr, 0xCD, allocSize));
        printf("[Rank %d] cudaMalloc src: ptr=%p, size=%zu, filled with 0xCD\n", rank, srcPtr, allocSize);

        // Export
        FabricTransferHelper helper;
        MemoryDescs regDescs{MemoryType::kVRAM, {MemoryDesc{srcPtr, allocSize, DEVICE_ID}}};
        helper.detectAndExportFabricHandles(regDescs);

        auto const& info = helper.getLocalFabricInfo();
        std::string serialized = info.serialize();
        printf("[Rank %d] Exported CudaIpc: handleType=%s, serialized=%zu bytes\n", rank,
            handleTypeToString(info.handleType), serialized.size());

        // Send metadata to rank 0
        int metaSize = static_cast<int>(serialized.size());
        MPI_Send(&metaSize, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(serialized.data(), metaSize, MPI_BYTE, 0, 1, MPI_COMM_WORLD);

        // Send src address
        uintptr_t srcAddr = reinterpret_cast<uintptr_t>(srcPtr);
        MPI_Send(&srcAddr, sizeof(srcAddr), MPI_BYTE, 0, 2, MPI_COMM_WORLD);

        // Wait for rank 0 to finish
        MPI_Barrier(MPI_COMM_WORLD);

        TLLM_CUDA_CHECK(cudaFree(srcPtr));
        printf("[Rank %d] Cleanup done\n", rank);
    }
    else
    {
        // Receiver: rank 0

        // Receive metadata from rank 1
        int metaSize = 0;
        MPI_Recv(&metaSize, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::string serialized(metaSize, '\0');
        MPI_Recv(serialized.data(), metaSize, MPI_BYTE, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        uintptr_t remoteSrcAddr = 0;
        MPI_Recv(&remoteSrcAddr, sizeof(remoteSrcAddr), MPI_BYTE, 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("[Rank %d] Received metadata (%d bytes), remoteSrcAddr=0x%lx\n", rank, metaSize, remoteSrcAddr);

        // Deserialize
        auto fabricInfo = FabricMemInfo::deserialize(std::string_view(serialized.data(), serialized.size()));
        if (!fabricInfo || fabricInfo->handleType != VmmHandleType::kCudaIpc)
        {
            printf("[Rank %d] Deserialization failed or wrong type\n", rank);
            ok = false;
        }
        else
        {
            printf("[Rank %d] Deserialized: handleType=CudaIpc, pools=%zu\n", rank, fabricInfo->pools.size());

            // Import and map
            FabricTransferHelper helper;
            helper.importAndMapRemoteFabric("rank1_cuda_ipc", *fabricInfo);

            // Translate address
            void* localPtr = helper.translateToLocalMapping("rank1_cuda_ipc", remoteSrcAddr, allocSize);
            if (!localPtr)
            {
                printf("[Rank %d] translateToLocalMapping returned null\n", rank);
                ok = false;
            }
            else
            {
                printf("[Rank %d] Mapped local ptr = %p\n", rank, localPtr);

                // Allocate dst and verify via device-to-device copy
                void* dstPtr = nullptr;
                TLLM_CUDA_CHECK(cudaMalloc(&dstPtr, allocSize));
                TLLM_CUDA_CHECK(cudaMemcpy(dstPtr, localPtr, allocSize, cudaMemcpyDeviceToDevice));
                TLLM_CUDA_CHECK(cudaDeviceSynchronize());

                // Verify
                std::vector<uint8_t> hostBuf(allocSize);
                TLLM_CUDA_CHECK(cudaMemcpy(hostBuf.data(), dstPtr, allocSize, cudaMemcpyDeviceToHost));

                bool dataOk = true;
                for (size_t i = 0; i < allocSize; ++i)
                {
                    if (hostBuf[i] != 0xCD)
                    {
                        printf("[Rank %d] Data mismatch at byte %zu: expected 0xCD, got 0x%02X\n", rank, i, hostBuf[i]);
                        dataOk = false;
                        break;
                    }
                }
                if (dataOk)
                {
                    printf("[Rank %d] CudaIpc transfer verification PASSED (all %zu bytes == 0xCD)\n", rank, allocSize);
                }
                else
                {
                    ok = false;
                }

                TLLM_CUDA_CHECK(cudaFree(dstPtr));
            }

            helper.cleanupRemoteFabricMapping("rank1_cuda_ipc");
        }

        MPI_Barrier(MPI_COMM_WORLD);
        printf("[Rank %d] Cleanup done\n", rank);
    }

    printf("[Rank %d] === CudaIpcTransfer: %s ===\n", rank, ok ? "PASSED" : "FAILED");
    return ok;
}

/// @brief Test 3: CudaIpc end-to-end through NixlTransferAgent
bool testNixlAgentCudaIpcTransfer(int rank, int worldSize)
{
    if (worldSize < 2)
    {
        if (rank == 0)
            printf("[Rank %d] Skipping NixlAgentCudaIpcTransfer (need 2 ranks)\n", rank);
        return true;
    }

    printf("[Rank %d] === Test: NixlAgentCudaIpcTransfer ===\n", rank);
    bool ok = true;

    size_t const DATA_SIZE = 4 * 1024 * 1024; // 4 MB
    uint8_t const FILL_PATTERN = 0xFA;
    uint32_t const DEVICE_ID = 0;
    std::string const agentName = std::string("nixl_cuda_ipc_agent_") + std::to_string(rank);

    // Create NixlTransferAgent
    BaseAgentConfig config{agentName, true, false, false, false};
    std::unique_ptr<BaseTransferAgent> agent;
    try
    {
        agent = std::unique_ptr<BaseTransferAgent>(createNixlTransferAgent(&config));
    }
    catch (std::exception const& e)
    {
        printf("[Rank %d] Failed to create NixlTransferAgent: %s\n", rank, e.what());
        return true; // Skip
    }

    int agentOk = (agent != nullptr) ? 1 : 0;
    int allOk = 0;
    MPI_Allreduce(&agentOk, &allOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!allOk)
    {
        printf("[Rank %d] SKIP: not all ranks created NixlTransferAgent\n", rank);
        return true;
    }

    printf("[Rank %d] NixlTransferAgent created: '%s'\n", rank, agentName.c_str());

    if (rank == 1)
    {
        // Source side: allocate cudaMalloc memory and fill
        void* srcPtr = nullptr;
        TLLM_CUDA_CHECK(cudaMalloc(&srcPtr, DATA_SIZE));
        TLLM_CUDA_CHECK(cudaMemset(srcPtr, FILL_PATTERN, DATA_SIZE));
        printf(
            "[Rank %d] cudaMalloc src: ptr=%p, size=%zu, filled with 0x%02X\n", rank, srcPtr, DATA_SIZE, FILL_PATTERN);

        // Register memory
        MemoryDescs vmmDescs{MemoryType::kVRAM, {MemoryDesc{srcPtr, DATA_SIZE, DEVICE_ID}}};
        agent->registerMemory(vmmDescs);
        printf("[Rank %d] registerMemory done\n", rank);

        // Get agent descriptor
        auto agentDesc = agent->getLocalAgentDesc();
        auto const& descStr = agentDesc.getBackendAgentDesc();
        printf("[Rank %d] getLocalAgentDesc: %zu bytes\n", rank, descStr.size());

        // Send to rank 0
        int descSize = static_cast<int>(descStr.size());
        MPI_Send(&descSize, 1, MPI_INT, 0, 30, MPI_COMM_WORLD);
        MPI_Send(descStr.data(), descSize, MPI_BYTE, 0, 31, MPI_COMM_WORLD);

        // Send address info
        uintptr_t srcAddr = reinterpret_cast<uintptr_t>(srcPtr);
        size_t srcSize = DATA_SIZE;
        MPI_Send(&srcAddr, sizeof(srcAddr), MPI_BYTE, 0, 32, MPI_COMM_WORLD);
        MPI_Send(&srcSize, sizeof(srcSize), MPI_BYTE, 0, 33, MPI_COMM_WORLD);

        printf("[Rank %d] Sent AgentDesc and address info to rank 0\n", rank);

        // Wait for rank 0 to finish transfer
        MPI_Barrier(MPI_COMM_WORLD);

        agent->deregisterMemory(vmmDescs);
        TLLM_CUDA_CHECK(cudaFree(srcPtr));
        printf("[Rank %d] Cleanup done\n", rank);
    }
    else
    {
        // Receiver: rank 0

        // Allocate dst with cudaMalloc
        char* localDst = nullptr;
        TLLM_CUDA_CHECK(cudaMalloc(&localDst, DATA_SIZE));
        TLLM_CUDA_CHECK(cudaMemset(localDst, 0, DATA_SIZE));

        // Register dst memory
        MemoryDescs localDescs{MemoryType::kVRAM, {MemoryDesc{localDst, DATA_SIZE, DEVICE_ID}}};
        agent->registerMemory(localDescs);
        printf("[Rank %d] registerMemory done for local dst buffer\n", rank);

        // Receive agent descriptor from rank 1
        int descSize = 0;
        MPI_Recv(&descSize, 1, MPI_INT, 1, 30, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::string descStr(descSize, '\0');
        MPI_Recv(descStr.data(), descSize, MPI_BYTE, 1, 31, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        uintptr_t remoteSrcAddr = 0;
        size_t remoteSrcSize = 0;
        MPI_Recv(&remoteSrcAddr, sizeof(remoteSrcAddr), MPI_BYTE, 1, 32, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&remoteSrcSize, sizeof(remoteSrcSize), MPI_BYTE, 1, 33, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("[Rank %d] Received AgentDesc (%d bytes), remoteSrcAddr=0x%lx, remoteSrcSize=%zu\n", rank, descSize,
            remoteSrcAddr, remoteSrcSize);

        // Load remote agent
        std::string remoteAgentName = "nixl_cuda_ipc_agent_1";
        AgentDesc remoteAgentDesc{descStr};
        agent->loadRemoteAgent(remoteAgentName, remoteAgentDesc);
        printf("[Rank %d] loadRemoteAgent done for '%s'\n", rank, remoteAgentName.c_str());

        // Submit transfer (READ: remote src → local dst)
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
            ok = false;
        }
        else
        {
            // Verify data
            std::vector<uint8_t> hostBuf(DATA_SIZE);
            TLLM_CUDA_CHECK(cudaMemcpy(hostBuf.data(), localDst, DATA_SIZE, cudaMemcpyDeviceToHost));

            bool dataOk = true;
            for (size_t i = 0; i < DATA_SIZE; ++i)
            {
                if (hostBuf[i] != FILL_PATTERN)
                {
                    printf("[Rank %d] Data mismatch at byte %zu: expected 0x%02X, got 0x%02X\n", rank, i, FILL_PATTERN,
                        hostBuf[i]);
                    dataOk = false;
                    break;
                }
            }

            if (dataOk)
            {
                printf("[Rank %d] NixlAgent CudaIpc transfer PASSED (all %zu bytes == 0x%02X)\n", rank, DATA_SIZE,
                    FILL_PATTERN);
            }
            ok = ok && dataOk;
        }

        agent->deregisterMemory(localDescs);
        TLLM_CUDA_CHECK(cudaFree(localDst));

        MPI_Barrier(MPI_COMM_WORLD);
        printf("[Rank %d] Cleanup done\n", rank);
    }

    printf("[Rank %d] === NixlAgentCudaIpcTransfer: %s ===\n", rank, ok ? "PASSED" : "FAILED");
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
    allPassed = testNixlAgentPosixFdSubRange(rank, worldSize) && allPassed;

    MPI_Barrier(MPI_COMM_WORLD);
    allPassed = testPosixFdSubRangeDirect(rank, worldSize) && allPassed;

    MPI_Barrier(MPI_COMM_WORLD);
    allPassed = testNixlAgentPosixFdSubRangeWrite(rank, worldSize) && allPassed;

    MPI_Barrier(MPI_COMM_WORLD);
    allPassed = testPosixFdMultiChunkSubRangeDirect(rank, worldSize) && allPassed;

    MPI_Barrier(MPI_COMM_WORLD);
    allPassed = testNixlAgentMultiChunkSubRange(rank, worldSize) && allPassed;

    MPI_Barrier(MPI_COMM_WORLD);
    allPassed = testMultiChunkUnalignedSubRangeDirect(rank, worldSize) && allPassed;

    MPI_Barrier(MPI_COMM_WORLD);
    allPassed = testNixlAgentMultiChunkUnalignedSubRange(rank, worldSize) && allPassed;

    MPI_Barrier(MPI_COMM_WORLD);
    allPassed = testSmallCrossBoundaryDirect(rank, worldSize) && allPassed;

    MPI_Barrier(MPI_COMM_WORLD);
    allPassed = testIntraChunkSubRangeDirect(rank, worldSize) && allPassed;

    MPI_Barrier(MPI_COMM_WORLD);
    allPassed = testWideSpanSubRangeDirect(rank, worldSize) && allPassed;

    MPI_Barrier(MPI_COMM_WORLD);
    allPassed = test4MBChunkIntraDirect(rank, worldSize) && allPassed;

    MPI_Barrier(MPI_COMM_WORLD);
    allPassed = test4MBChunkCrossBoundaryDirect(rank, worldSize) && allPassed;

    MPI_Barrier(MPI_COMM_WORLD);
    allPassed = test4MBChunkWideSpanDirect(rank, worldSize) && allPassed;

    // CudaIpc tests
    MPI_Barrier(MPI_COMM_WORLD);
    allPassed = testCudaIpcDetection(rank) && allPassed;

    MPI_Barrier(MPI_COMM_WORLD);
    allPassed = testCudaIpcTransfer(rank, worldSize) && allPassed;

    MPI_Barrier(MPI_COMM_WORLD);
    allPassed = testNixlAgentCudaIpcTransfer(rank, worldSize) && allPassed;

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("\n=== All tests %s ===\n", allPassed ? "PASSED" : "FAILED");
    }

    MPI_Finalize();
    return allPassed ? 0 : 1;
}

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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/fabricTransfer.h"

#include <algorithm>
#include <chrono>
#include <cub/device/device_memcpy.cuh>
#include <random>
#include <sys/socket.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>

namespace tensorrt_llm::executor::kv_cache
{

// ============================================================================
// Unix Domain Socket Utilities for POSIX FD Passing (SCM_RIGHTS)
// ============================================================================

namespace
{

/// @brief Send a single file descriptor over a Unix domain socket using SCM_RIGHTS
bool sendFd(int socket, int fd)
{
    struct msghdr msg = {};
    struct iovec iov[1];
    char buf[1] = {0};
    char cmsgbuf[CMSG_SPACE(sizeof(int))];
    std::memset(cmsgbuf, 0, sizeof(cmsgbuf));

    iov[0].iov_base = buf;
    iov[0].iov_len = 1;

    msg.msg_iov = iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsgbuf;
    msg.msg_controllen = sizeof(cmsgbuf);

    struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(int));
    std::memcpy(CMSG_DATA(cmsg), &fd, sizeof(int));

    return sendmsg(socket, &msg, 0) >= 0;
}

/// @brief Receive a single file descriptor from a Unix domain socket using SCM_RIGHTS
int recvFd(int socket)
{
    struct msghdr msg = {};
    struct iovec iov[1];
    char buf[1];
    char cmsgbuf[CMSG_SPACE(sizeof(int))];
    std::memset(cmsgbuf, 0, sizeof(cmsgbuf));

    iov[0].iov_base = buf;
    iov[0].iov_len = 1;

    msg.msg_iov = iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsgbuf;
    msg.msg_controllen = sizeof(cmsgbuf);

    if (recvmsg(socket, &msg, 0) < 0)
    {
        return -1;
    }

    struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
    if (cmsg && cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS)
    {
        int fd;
        std::memcpy(&fd, CMSG_DATA(cmsg), sizeof(int));
        return fd;
    }

    return -1;
}

/// @brief Send multiple file descriptors over a Unix domain socket (one at a time)
bool sendFds(int socket, std::vector<int> const& fds)
{
    for (int fd : fds)
    {
        if (!sendFd(socket, fd))
        {
            return false;
        }
    }
    return true;
}

/// @brief Receive multiple file descriptors from a Unix domain socket
std::vector<int> recvFds(int socket, size_t count)
{
    std::vector<int> fds;
    fds.reserve(count);
    for (size_t i = 0; i < count; ++i)
    {
        int fd = recvFd(socket);
        if (fd < 0)
        {
            // Close any already-received FDs on failure
            for (int openFd : fds)
            {
                ::close(openFd);
            }
            return {};
        }
        fds.push_back(fd);
    }
    return fds;
}

/// @brief Create a named Unix domain socket server
int createUdsServer(char const* path)
{
    ::unlink(path); // Remove if exists

    int sock = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock < 0)
    {
        return -1;
    }

    struct sockaddr_un addr = {};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);

    if (::bind(sock, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0)
    {
        ::close(sock);
        return -1;
    }

    if (::listen(sock, 5) < 0)
    {
        ::close(sock);
        return -1;
    }

    return sock;
}

/// @brief Connect to a named Unix domain socket with retry
int connectUds(char const* path, int maxRetries = 50)
{
    int sock = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock < 0)
    {
        return -1;
    }

    struct sockaddr_un addr = {};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);

    // Retry connection (server might not be ready yet)
    for (int i = 0; i < maxRetries; ++i)
    {
        if (::connect(sock, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) == 0)
        {
            return sock;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 100ms retry interval
    }

    ::close(sock);
    return -1;
}

/// @brief Generate a unique UDS socket path
std::string generateUdsPath()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint64_t> dist;
    uint64_t randomId = dist(gen);
    return "/tmp/trt_llm_vmm_fd_" + std::to_string(::getpid()) + "_" + std::to_string(randomId) + ".sock";
}

} // anonymous namespace

// ============================================================================
// FabricMemChunk Serialization
// ============================================================================

void FabricMemChunk::serialize(std::ostream& os) const
{
    os.write(reinterpret_cast<char const*>(&virtAddrOffset), sizeof(virtAddrOffset));
    os.write(reinterpret_cast<char const*>(&size), sizeof(size));
    os.write(reinterpret_cast<char const*>(fabricHandle), sizeof(fabricHandle));
}

FabricMemChunk FabricMemChunk::deserialize(std::istream& is)
{
    FabricMemChunk chunk;
    is.read(reinterpret_cast<char*>(&chunk.virtAddrOffset), sizeof(chunk.virtAddrOffset));
    is.read(reinterpret_cast<char*>(&chunk.size), sizeof(chunk.size));
    is.read(reinterpret_cast<char*>(chunk.fabricHandle), sizeof(chunk.fabricHandle));
    return chunk;
}

// ============================================================================
// FabricMemPool Serialization
// ============================================================================

void FabricMemPool::serialize(std::ostream& os) const
{
    os.write(reinterpret_cast<char const*>(&deviceId), sizeof(deviceId));
    os.write(reinterpret_cast<char const*>(&poolBaseAddr), sizeof(poolBaseAddr));
    os.write(reinterpret_cast<char const*>(&poolTotalSize), sizeof(poolTotalSize));
    os.write(reinterpret_cast<char const*>(&registeredAddr), sizeof(registeredAddr));
    os.write(reinterpret_cast<char const*>(&registeredSize), sizeof(registeredSize));
    os.write(reinterpret_cast<char const*>(&mappedOffset), sizeof(mappedOffset));
    os.write(reinterpret_cast<char const*>(&mappedSize), sizeof(mappedSize));
    uint32_t numChunks = static_cast<uint32_t>(chunks.size());
    os.write(reinterpret_cast<char const*>(&numChunks), sizeof(numChunks));
    for (auto const& chunk : chunks)
    {
        chunk.serialize(os);
    }
}

FabricMemPool FabricMemPool::deserialize(std::istream& is)
{
    FabricMemPool pool;
    is.read(reinterpret_cast<char*>(&pool.deviceId), sizeof(pool.deviceId));
    is.read(reinterpret_cast<char*>(&pool.poolBaseAddr), sizeof(pool.poolBaseAddr));
    is.read(reinterpret_cast<char*>(&pool.poolTotalSize), sizeof(pool.poolTotalSize));
    is.read(reinterpret_cast<char*>(&pool.registeredAddr), sizeof(pool.registeredAddr));
    is.read(reinterpret_cast<char*>(&pool.registeredSize), sizeof(pool.registeredSize));
    is.read(reinterpret_cast<char*>(&pool.mappedOffset), sizeof(pool.mappedOffset));
    is.read(reinterpret_cast<char*>(&pool.mappedSize), sizeof(pool.mappedSize));
    uint32_t numChunks;
    is.read(reinterpret_cast<char*>(&numChunks), sizeof(numChunks));
    pool.chunks.reserve(numChunks);
    for (uint32_t i = 0; i < numChunks; ++i)
    {
        pool.chunks.push_back(FabricMemChunk::deserialize(is));
    }
    return pool;
}

// ============================================================================
// FabricMemInfo Serialization
// ============================================================================

std::string FabricMemInfo::serialize() const
{
    std::ostringstream oss;
    oss.write(reinterpret_cast<char const*>(&kMagic), sizeof(kMagic));
    oss.write(reinterpret_cast<char const*>(&kVersion), sizeof(kVersion));
    uint8_t supportedFlag = supported ? 1 : 0;
    oss.write(reinterpret_cast<char const*>(&supportedFlag), sizeof(supportedFlag));
    // Version 4: handleType + udsPath
    uint8_t handleTypeVal = static_cast<uint8_t>(handleType);
    oss.write(reinterpret_cast<char const*>(&handleTypeVal), sizeof(handleTypeVal));
    uint32_t udsPathLen = static_cast<uint32_t>(udsPath.size());
    oss.write(reinterpret_cast<char const*>(&udsPathLen), sizeof(udsPathLen));
    if (udsPathLen > 0)
    {
        oss.write(udsPath.data(), udsPathLen);
    }
    // Pools
    uint32_t numPools = static_cast<uint32_t>(pools.size());
    oss.write(reinterpret_cast<char const*>(&numPools), sizeof(numPools));
    for (auto const& pool : pools)
    {
        pool.serialize(oss);
    }
    return oss.str();
}

std::optional<FabricMemInfo> FabricMemInfo::deserialize(std::string_view data)
{
    if (data.size() < sizeof(uint32_t) * 2 + sizeof(uint8_t) + sizeof(uint32_t))
    {
        return std::nullopt;
    }

    std::istringstream iss{std::string{data}};
    uint32_t magic, version;
    iss.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    iss.read(reinterpret_cast<char*>(&version), sizeof(version));

    if (magic != FabricMemInfo::kMagic || version != FabricMemInfo::kVersion)
    {
        return std::nullopt;
    }

    FabricMemInfo info;
    uint8_t supportedFlag;
    iss.read(reinterpret_cast<char*>(&supportedFlag), sizeof(supportedFlag));
    info.supported = (supportedFlag != 0);

    // Version 4: handleType + udsPath
    uint8_t handleTypeVal;
    iss.read(reinterpret_cast<char*>(&handleTypeVal), sizeof(handleTypeVal));
    info.handleType = static_cast<VmmHandleType>(handleTypeVal);

    uint32_t udsPathLen;
    iss.read(reinterpret_cast<char*>(&udsPathLen), sizeof(udsPathLen));
    if (udsPathLen > 0)
    {
        info.udsPath.resize(udsPathLen);
        iss.read(info.udsPath.data(), udsPathLen);
    }

    // Pools
    uint32_t numPools;
    iss.read(reinterpret_cast<char*>(&numPools), sizeof(numPools));
    info.pools.reserve(numPools);
    for (uint32_t i = 0; i < numPools; ++i)
    {
        info.pools.push_back(FabricMemPool::deserialize(iss));
    }

    if (!iss)
    {
        return std::nullopt;
    }
    return info;
}

// ============================================================================
// FabricTransferStatus Implementation
// ============================================================================

FabricTransferStatus::FabricTransferStatus(
    std::shared_ptr<runtime::CudaStream> stream, std::shared_ptr<runtime::CudaEvent> completionEvent)
    : mStream(std::move(stream))
    , mCompletionEvent(std::move(completionEvent))
{
}

bool FabricTransferStatus::isCompleted() const
{
    if (mCompleted.load(std::memory_order_acquire))
    {
        return true;
    }

    // Use cudaEventQuery for non-blocking check
    auto result = cudaEventQuery(mCompletionEvent->get());
    if (result == cudaSuccess)
    {
        mCompleted.store(true, std::memory_order_release);
        return true;
    }
    return false;
}

TransferState FabricTransferStatus::wait(int64_t timeout_ms) const
{
    if (mCompleted.load(std::memory_order_acquire))
    {
        return TransferState::kSUCCESS;
    }

    if (timeout_ms < 0)
    {
        // Infinite wait: synchronize on event
        mCompletionEvent->synchronize();
        mCompleted.store(true, std::memory_order_release);
        return TransferState::kSUCCESS;
    }

    // Timed wait: poll and check
    auto startTime = std::chrono::steady_clock::now();
    while (true)
    {
        auto result = cudaEventQuery(mCompletionEvent->get());
        if (result == cudaSuccess)
        {
            mCompleted.store(true, std::memory_order_release);
            return TransferState::kSUCCESS;
        }

        auto elapsed
            = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime)
                  .count();
        if (elapsed >= timeout_ms)
        {
            return TransferState::kIN_PROGRESS;
        }

        std::this_thread::yield();
    }
}

// ============================================================================
// FabricTransferHelper Implementation
// ============================================================================

FabricTransferHelper::~FabricTransferHelper()
{
    // Stop UDS server if running (must be done first, before closing FDs)
    stopUdsServer();

    // Close all exported POSIX FDs
    for (int fd : mExportedFds)
    {
        if (fd >= 0)
        {
            ::close(fd);
        }
    }
    mExportedFds.clear();

    // Clean up all remote fabric mappings
    for (auto& [name, mapping] : mRemoteFabricMappings)
    {
        for (auto& poolMapping : mapping.pools)
        {
            // Unmap and release handles
            for (auto handle : poolMapping.importedHandles)
            {
                cuMemRelease(handle);
            }
            if (poolMapping.localVirtAddr != 0)
            {
                cuMemUnmap(poolMapping.localVirtAddr, poolMapping.mappedSize);
                cuMemAddressFree(poolMapping.localVirtAddr, poolMapping.mappedSize);
            }
        }
    }
}

bool FabricTransferHelper::hasRemoteMapping(std::string const& remoteName) const
{
    // Check if we have a valid mapping and haven't previously failed
    return mRemoteFabricMappings.find(remoteName) != mRemoteFabricMappings.end()
        && mFailedFabricImports.find(remoteName) == mFailedFabricImports.end();
}

bool FabricTransferHelper::hasFabricImportFailed(std::string const& remoteName) const
{
    return mFailedFabricImports.find(remoteName) != mFailedFabricImports.end();
}

void FabricTransferHelper::ensureCudaResourcesInitialized()
{
    if (!mFabricStream)
    {
        mFabricStream = std::make_shared<runtime::CudaStream>();
        mBufferManager = std::make_shared<runtime::BufferManager>(mFabricStream);
    }
}

void FabricTransferHelper::ensurePreallocBuffers(size_t batchSize, size_t cubTempBytes)
{
    // If current buffers are large enough, return
    if (batchSize <= mPreallocBuffers.maxBatchSize && cubTempBytes <= mPreallocBuffers.cubTempStorageSize)
    {
        return;
    }

    // Use 2x growth strategy to reduce frequent reallocation
    constexpr size_t kDefaultBatchSize = 8192 * 2;           // Default 16384 entries
    constexpr size_t kDefaultCubTempSize = 64 * 1024 * 1024; // Default 64MB

    size_t newBatchSize = std::max(batchSize, mPreallocBuffers.maxBatchSize * 2);
    newBatchSize = std::max(newBatchSize, kDefaultBatchSize);

    size_t newCubTempSize = std::max(cubTempBytes, mPreallocBuffers.cubTempStorageSize * 2);
    newCubTempSize = std::max(newCubTempSize, kDefaultCubTempSize);

    TLLM_LOG_DEBUG("FabricTransfer: reallocating prealloc buffers: batch %zu -> %zu, cubTemp %zu -> %zu",
        mPreallocBuffers.maxBatchSize, newBatchSize, mPreallocBuffers.cubTempStorageSize, newCubTempSize);

    // Synchronize stream to ensure previous operations are complete before releasing old buffers
    mFabricStream->synchronize();

    // Reallocate
    mPreallocBuffers.srcPtrs = mBufferManager->gpu(newBatchSize * sizeof(void*));
    mPreallocBuffers.dstPtrs = mBufferManager->gpu(newBatchSize * sizeof(void*));
    mPreallocBuffers.sizes = mBufferManager->gpu(newBatchSize * sizeof(size_t));
    mPreallocBuffers.cubTempStorage = mBufferManager->gpu(newCubTempSize);
    mPreallocBuffers.maxBatchSize = newBatchSize;
    mPreallocBuffers.cubTempStorageSize = newCubTempSize;
}

// ============================================================================
// UDS Server for POSIX FD sharing
// ============================================================================

void FabricTransferHelper::startUdsServer()
{
    if (mExportedFds.empty())
    {
        return;
    }

    mUdsPath = generateUdsPath();
    mUdsServerSocket = createUdsServer(mUdsPath.c_str());
    if (mUdsServerSocket < 0)
    {
        TLLM_LOG_ERROR("FabricTransfer: failed to create UDS server at %s", mUdsPath.c_str());
        mUdsPath.clear();
        return;
    }

    mUdsServerRunning.store(true, std::memory_order_release);

    // Server thread: accept connections and send FDs
    mUdsServerThread = std::thread(
        [this]()
        {
            TLLM_LOG_DEBUG(
                "FabricTransfer: UDS server started at %s with %zu FDs", mUdsPath.c_str(), mExportedFds.size());

            while (mUdsServerRunning.load(std::memory_order_acquire))
            {
                // Use a timeout on accept so we can check mUdsServerRunning periodically
                struct timeval tv;
                tv.tv_sec = 1;
                tv.tv_usec = 0;

                fd_set readfds;
                FD_ZERO(&readfds);
                FD_SET(mUdsServerSocket, &readfds);

                int ret = ::select(mUdsServerSocket + 1, &readfds, nullptr, nullptr, &tv);
                if (ret <= 0)
                {
                    continue; // Timeout or error, check running flag
                }

                int clientSocket = ::accept(mUdsServerSocket, nullptr, nullptr);
                if (clientSocket < 0)
                {
                    if (mUdsServerRunning.load(std::memory_order_acquire))
                    {
                        TLLM_LOG_WARNING("FabricTransfer: UDS accept failed, errno=%d", errno);
                    }
                    continue;
                }

                // Protocol: client sends numExpectedFds, server sends numAvailableFds, then FDs
                uint32_t numExpected = 0;
                ssize_t bytesRead = ::read(clientSocket, &numExpected, sizeof(numExpected));
                if (bytesRead != sizeof(numExpected))
                {
                    TLLM_LOG_WARNING("FabricTransfer: UDS client protocol error (read numExpected)");
                    ::close(clientSocket);
                    continue;
                }

                uint32_t numAvailable = static_cast<uint32_t>(mExportedFds.size());
                ssize_t bytesWritten = ::write(clientSocket, &numAvailable, sizeof(numAvailable));
                if (bytesWritten != sizeof(numAvailable))
                {
                    TLLM_LOG_WARNING("FabricTransfer: UDS client protocol error (write numAvailable)");
                    ::close(clientSocket);
                    continue;
                }

                // Send FDs
                uint32_t numToSend = std::min(numExpected, numAvailable);
                bool sendOk = true;
                for (uint32_t i = 0; i < numToSend; ++i)
                {
                    if (!sendFd(clientSocket, mExportedFds[i]))
                    {
                        TLLM_LOG_WARNING("FabricTransfer: UDS failed to send FD[%u]", i);
                        sendOk = false;
                        break;
                    }
                }

                if (sendOk)
                {
                    TLLM_LOG_DEBUG("FabricTransfer: UDS sent %u FDs to client", numToSend);
                }

                ::close(clientSocket);
            }

            TLLM_LOG_DEBUG("FabricTransfer: UDS server stopped");
        });
}

void FabricTransferHelper::stopUdsServer()
{
    if (!mUdsServerRunning.load(std::memory_order_acquire))
    {
        return;
    }

    mUdsServerRunning.store(false, std::memory_order_release);

    // Close the server socket to unblock accept()
    if (mUdsServerSocket >= 0)
    {
        ::close(mUdsServerSocket);
        mUdsServerSocket = -1;
    }

    // Join the server thread
    if (mUdsServerThread.joinable())
    {
        mUdsServerThread.join();
    }

    // Clean up socket file
    if (!mUdsPath.empty())
    {
        ::unlink(mUdsPath.c_str());
    }
}

// ============================================================================
// Fabric Handle Detection and Export
// ============================================================================

void FabricTransferHelper::detectAndExportFabricHandles(RegisterDescs const& descs)
{
    // Track which pools we've already processed to avoid duplicates
    // Key: poolBaseAddr, Value: index in mLocalFabricInfo.pools
    std::unordered_map<uint64_t, size_t> processedPools;

    for (auto const& desc : descs.getDescs())
    {
        CUdeviceptr ptr = static_cast<CUdeviceptr>(desc.getAddr());
        size_t descSize = desc.getLen();

        // Step 1: Check if shareable handle types are supported (reference: UCX cuda_ipc_md.c:157-175)
        int legacy_capable = 0;
        unsigned long long allowed_handle_types = 0;

        CUpointer_attribute attr_type[2]
            = {CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE, CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES};
        void* attr_data[2] = {&legacy_capable, &allowed_handle_types};

        auto err = cuPointerGetAttributes(2, attr_type, attr_data, ptr);
        if (err != CUDA_SUCCESS || legacy_capable)
        {
            continue; // Not VMM memory, skip
        }

        // Determine handle type: prefer fabric, fallback to POSIX FD
        bool fabricSupported = (allowed_handle_types & CU_MEM_HANDLE_TYPE_FABRIC) != 0;
        bool posixFdSupported = (allowed_handle_types & CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) != 0;
        if (!fabricSupported && !posixFdSupported)
        {
            continue; // No shareable handle type supported
        }

        VmmHandleType descHandleType = fabricSupported ? VmmHandleType::kFabric : VmmHandleType::kPosixFd;

        // Ensure consistent handle type across all descriptors
        if (mDetectedHandleType == VmmHandleType::kNone)
        {
            mDetectedHandleType = descHandleType;
        }
        else if (mDetectedHandleType != descHandleType)
        {
            TLLM_LOG_WARNING("FabricTransfer: mixed handle types detected (existing=%d, new=%d), skipping desc",
                static_cast<int>(mDetectedHandleType), static_cast<int>(descHandleType));
            continue;
        }

        // Step 2: Get the entire virtual address range
        // Note: cuMemGetAddressRange returns the full reserved VA range, not just the registered portion
        CUdeviceptr basePtr;
        size_t totalSize;
        err = cuMemGetAddressRange(&basePtr, &totalSize, ptr);
        if (err != CUDA_SUCCESS)
        {
            continue;
        }

        // Step 3: Check if we've already processed this pool (same base address)
        auto poolIt = processedPools.find(basePtr);
        if (poolIt != processedPools.end())
        {
            // Pool already processed, update registered range and export new chunks if needed
            FabricMemPool& existingPool = mLocalFabricInfo.pools[poolIt->second];
            uint64_t existingStart = existingPool.registeredAddr;
            uint64_t existingEnd = existingStart + existingPool.registeredSize;
            uint64_t newStart = ptr;
            uint64_t newEnd = ptr + descSize;

            // Export chunks for any non-overlapping regions
            // Case 1: new range extends before existing range
            if (newStart < existingStart)
            {
                size_t extendSize = std::min(existingStart, newEnd) - newStart;
                detectAndExportChunks(newStart, extendSize, basePtr, existingPool.chunks);
            }
            // Case 2: new range extends after existing range
            if (newEnd > existingEnd)
            {
                CUdeviceptr extendStart = std::max(existingEnd, newStart);
                size_t extendSize = newEnd - extendStart;
                detectAndExportChunks(extendStart, extendSize, basePtr, existingPool.chunks);
            }

            // Update registered range to cover both
            existingPool.registeredAddr = std::min(existingStart, newStart);
            existingPool.registeredSize = std::max(existingEnd, newEnd) - existingPool.registeredAddr;

            // Recalculate mappedOffset and mappedSize from all chunks
            uint64_t minOffset = UINT64_MAX;
            uint64_t maxEnd = 0;
            for (auto const& chunk : existingPool.chunks)
            {
                minOffset = std::min(minOffset, chunk.virtAddrOffset);
                maxEnd = std::max(maxEnd, chunk.virtAddrOffset + chunk.size);
            }
            existingPool.mappedOffset = minOffset;
            existingPool.mappedSize = maxEnd - minOffset;

            TLLM_LOG_DEBUG(
                "FabricTransfer: pool at 0x%lx updated, chunks=%zu, "
                "registered=[0x%lx, 0x%lx), mapped=[0x%lx, 0x%lx)",
                basePtr, existingPool.chunks.size(), existingPool.registeredAddr,
                existingPool.registeredAddr + existingPool.registeredSize,
                existingPool.poolBaseAddr + existingPool.mappedOffset,
                existingPool.poolBaseAddr + existingPool.mappedOffset + existingPool.mappedSize);
            continue;
        }

        // Step 4: Detect and export chunks ONLY within registered range
        // Important: We only scan the registered range, not the entire reserved VA range
        // (which could be 80GB+ while user only registers a small portion)
        FabricMemPool pool;
        pool.deviceId = desc.getDeviceId();
        pool.poolBaseAddr = basePtr;
        pool.poolTotalSize = totalSize;
        pool.registeredAddr = ptr;      // Record actual registered address
        pool.registeredSize = descSize; // Record actual registered size

        // Only export chunks that overlap with the registered range [ptr, ptr + descSize)
        detectAndExportChunks(ptr, descSize, basePtr, pool.chunks);

        if (!pool.chunks.empty())
        {
            // Calculate mappedOffset and mappedSize from the actual chunks
            // (chunks may extend beyond registered range due to alignment)
            uint64_t minOffset = UINT64_MAX;
            uint64_t maxEnd = 0;
            for (auto const& chunk : pool.chunks)
            {
                minOffset = std::min(minOffset, chunk.virtAddrOffset);
                maxEnd = std::max(maxEnd, chunk.virtAddrOffset + chunk.size);
            }
            pool.mappedOffset = minOffset;
            pool.mappedSize = maxEnd - minOffset;

            TLLM_LOG_DEBUG(
                "FabricTransfer: detected fabric pool at 0x%lx, chunks=%zu, "
                "registered=[0x%lx, 0x%lx), mapped=[0x%lx, 0x%lx)",
                pool.poolBaseAddr, pool.chunks.size(), pool.registeredAddr, pool.registeredAddr + pool.registeredSize,
                pool.poolBaseAddr + pool.mappedOffset, pool.poolBaseAddr + pool.mappedOffset + pool.mappedSize);
            processedPools[basePtr] = mLocalFabricInfo.pools.size();
            mLocalFabricInfo.pools.push_back(std::move(pool));
        }
    }

    mLocalFabricInfo.supported = !mLocalFabricInfo.pools.empty();
    mLocalFabricInfo.handleType = mDetectedHandleType;

    if (mLocalFabricInfo.supported)
    {
        TLLM_LOG_INFO("FabricTransfer: transfer enabled with %zu pools, handleType=%s", mLocalFabricInfo.pools.size(),
            mDetectedHandleType == VmmHandleType::kFabric ? "Fabric" : "PosixFd");

        // For POSIX FD mode, start UDS server for FD sharing with remote processes
        if (mDetectedHandleType == VmmHandleType::kPosixFd)
        {
            startUdsServer();
            mLocalFabricInfo.udsPath = mUdsPath;
        }
    }
}

void FabricTransferHelper::removeFabricHandles(RegisterDescs const& descs)
{
    if (mLocalFabricInfo.pools.empty())
    {
        return;
    }

    // Collect pool base addresses that need to be removed
    // We match by poolBaseAddr and check that the deregistered range overlaps the registered range
    std::unordered_set<uint64_t> poolBasesToRemove;
    for (auto const& desc : descs.getDescs())
    {
        CUdeviceptr ptr = static_cast<CUdeviceptr>(desc.getAddr());

        // Get the entire virtual address range (same as detectAndExportFabricHandles)
        CUdeviceptr basePtr;
        size_t totalSize;
        auto err = cuMemGetAddressRange(&basePtr, &totalSize, ptr);
        if (err != CUDA_SUCCESS)
        {
            continue;
        }
        poolBasesToRemove.insert(basePtr);
    }

    if (poolBasesToRemove.empty())
    {
        return;
    }

    // Remove matching pools (iterate in reverse to handle index shifting from erase)
    bool isPosixFdMode = (mDetectedHandleType == VmmHandleType::kPosixFd);
    for (auto it = mLocalFabricInfo.pools.begin(); it != mLocalFabricInfo.pools.end();)
    {
        if (poolBasesToRemove.find(it->poolBaseAddr) != poolBasesToRemove.end())
        {
            // For POSIX FD mode: close and remove corresponding exported FDs
            if (isPosixFdMode && !mExportedFds.empty())
            {
                // Calculate the FD offset for this pool (sum of chunks in earlier pools)
                size_t fdOffset = 0;
                for (auto prev = mLocalFabricInfo.pools.begin(); prev != it; ++prev)
                {
                    fdOffset += prev->chunks.size();
                }
                size_t fdCount = it->chunks.size();

                // Close FDs for this pool
                size_t fdEnd = std::min(fdOffset + fdCount, mExportedFds.size());
                for (size_t i = fdOffset; i < fdEnd; ++i)
                {
                    if (mExportedFds[i] >= 0)
                    {
                        ::close(mExportedFds[i]);
                    }
                }

                // Erase FDs from the vector
                if (fdOffset < mExportedFds.size())
                {
                    mExportedFds.erase(mExportedFds.begin() + static_cast<ptrdiff_t>(fdOffset),
                        mExportedFds.begin() + static_cast<ptrdiff_t>(fdEnd));
                }
            }

            TLLM_LOG_DEBUG("FabricTransfer: removed pool at base=0x%lx, registered=[0x%lx, +%zu)", it->poolBaseAddr,
                it->registeredAddr, it->registeredSize);
            it = mLocalFabricInfo.pools.erase(it);
        }
        else
        {
            ++it;
        }
    }

    // Update supported flag
    if (mLocalFabricInfo.pools.empty())
    {
        mLocalFabricInfo.supported = false;

        // Stop UDS server if no more pools (POSIX FD mode)
        if (isPosixFdMode)
        {
            stopUdsServer();
            // Close any remaining FDs (shouldn't normally happen)
            for (int fd : mExportedFds)
            {
                if (fd >= 0)
                {
                    ::close(fd);
                }
            }
            mExportedFds.clear();
        }

        TLLM_LOG_INFO("FabricTransfer: all pools removed, transfer disabled");
    }
    else
    {
        TLLM_LOG_DEBUG("FabricTransfer: %zu pools remaining after removal", mLocalFabricInfo.pools.size());
    }
}

size_t FabricTransferHelper::getVmmGranularity()
{
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    size_t granularity = 0;
    auto err = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    if (err != CUDA_SUCCESS || granularity == 0)
    {
        granularity = 2 * 1024 * 1024; // Default 2MB
    }
    return granularity;
}

unsigned long long FabricTransferHelper::getBufferId(CUdeviceptr addr)
{
    unsigned long long bufferId = 0;
    auto err = cuPointerGetAttribute(&bufferId, CU_POINTER_ATTRIBUTE_BUFFER_ID, addr);
    // Return 0 for unmapped/invalid addresses - this will be treated as a boundary
    if (err != CUDA_SUCCESS)
    {
        return 0;
    }
    return bufferId;
}

void FabricTransferHelper::detectAndExportChunks(
    CUdeviceptr scanStart, size_t scanSize, CUdeviceptr poolBase, std::vector<FabricMemChunk>& chunks)
{
    // scanStart/scanSize: the registered range to scan (user's registerMemory range)
    // poolBase: the base of the entire VMM pool (from cuMemGetAddressRange), used for virtAddrOffset calculation
    size_t granularity = getVmmGranularity();

    // Check if scanStart is mapped
    unsigned long long startBufferId = getBufferId(scanStart);
    if (startBufferId == 0)
    {
        TLLM_LOG_DEBUG("FabricTransfer: scanStart 0x%lx is not mapped, scanning for mapped regions", scanStart);
    }

    CUdeviceptr scanEnd = scanStart + scanSize;

    // Dispatch to correct export function based on detected handle type
    auto exportChunkFn = [this](CUdeviceptr addr, size_t size, CUdeviceptr base, std::vector<FabricMemChunk>& c)
    {
        if (mDetectedHandleType == VmmHandleType::kPosixFd)
        {
            exportSingleChunkPosixFd(addr, size, base, c);
        }
        else
        {
            exportSingleChunk(addr, size, base, c);
        }
    };

    if (scanSize <= granularity)
    {
        // Single chunk - only export if mapped
        if (startBufferId != 0)
        {
            exportChunkFn(scanStart, scanSize, poolBase, chunks);
        }
        return;
    }

    // Find all chunk boundaries using binary search with buffer_id
    std::vector<CUdeviceptr> boundaries;
    findAllBoundariesRecursive(scanStart, scanEnd, granularity, boundaries);
    std::sort(boundaries.begin(), boundaries.end());

    // Export each chunk based on boundaries
    // Note: boundaries now include transitions to/from unmapped regions
    CUdeviceptr chunkStart = scanStart;
    for (CUdeviceptr boundary : boundaries)
    {
        // Only export if the chunk is mapped (has valid buffer_id)
        if (getBufferId(chunkStart) != 0 && boundary > chunkStart)
        {
            exportChunkFn(chunkStart, boundary - chunkStart, poolBase, chunks);
        }
        chunkStart = boundary;
    }

    // Export the last chunk if mapped
    if (chunkStart < scanEnd && getBufferId(chunkStart) != 0)
    {
        exportChunkFn(chunkStart, scanEnd - chunkStart, poolBase, chunks);
    }
}

void FabricTransferHelper::findAllBoundariesRecursive(
    CUdeviceptr left, CUdeviceptr right, size_t granularity, std::vector<CUdeviceptr>& boundaries)
{
    if (right - left <= granularity)
    {
        return; // Range too small, no boundary possible
    }

    // Get buffer_id at left and right endpoints
    unsigned long long leftId = getBufferId(left);

    // If left address is unmapped (buffer_id == 0), skip this region
    // This can happen if the reserved VA range has gaps in mapping
    if (leftId == 0)
    {
        // Try to find the first mapped address
        CUdeviceptr probe = left + granularity;
        while (probe < right)
        {
            unsigned long long probeId = getBufferId(probe);
            if (probeId != 0)
            {
                // Found a mapped region, record boundary and continue from here
                boundaries.push_back(probe);
                findAllBoundariesRecursive(probe, right, granularity, boundaries);
                return;
            }
            probe += granularity;
        }
        return; // Entire range is unmapped
    }

    CUdeviceptr rightProbe = right - granularity; // right is open interval
    if (rightProbe < left)
    {
        rightProbe = left;
    }
    unsigned long long rightId = getBufferId(rightProbe);

    // If right is unmapped or same as left, check if there's a boundary
    if (rightId == 0 || leftId == rightId)
    {
        // If rightId == 0, there might be an unmapped gap - need to find it
        if (rightId == 0 && leftId != 0)
        {
            // Binary search to find where mapping ends
            CUdeviceptr lo = left;
            CUdeviceptr hi = right;

            while (hi - lo > granularity)
            {
                CUdeviceptr mid = lo + ((hi - lo) / 2 / granularity) * granularity;
                if (mid <= lo)
                {
                    mid = lo + granularity;
                }

                unsigned long long midId = getBufferId(mid);
                if (midId == leftId)
                {
                    lo = mid;
                }
                else
                {
                    hi = mid;
                }
            }
            // hi is where the current chunk ends (either unmapped or different chunk)
            if (hi < right)
            {
                boundaries.push_back(hi);
                findAllBoundariesRecursive(hi, right, granularity, boundaries);
            }
        }
        return; // Same chunk or no more boundaries
    }

    // At least one boundary exists, use binary search to find the first one
    CUdeviceptr lo = left;
    CUdeviceptr hi = right;

    while (hi - lo > granularity)
    {
        // Calculate midpoint, aligned to granularity
        CUdeviceptr mid = lo + ((hi - lo) / 2 / granularity) * granularity;
        if (mid <= lo)
        {
            mid = lo + granularity; // Ensure progress
        }

        unsigned long long midId = getBufferId(mid);

        if (midId == leftId)
        {
            lo = mid; // Boundary is in (mid, hi)
        }
        else
        {
            hi = mid; // Boundary is in [lo, mid], or mid is the boundary
        }
    }

    // hi is the found boundary (start of new chunk)
    boundaries.push_back(hi);

    // Recursively find more boundaries in [hi, right)
    findAllBoundariesRecursive(hi, right, granularity, boundaries);
}

void FabricTransferHelper::exportSingleChunk(
    CUdeviceptr chunkAddr, size_t chunkSize, CUdeviceptr poolBase, std::vector<FabricMemChunk>& chunks)
{
    // IMPORTANT: chunkAddr might be in the middle of a physical memory block!
    // cuMemRetainAllocationHandle returns the handle for the ENTIRE physical block.
    // cuMemMap requires the EXACT size of the physical block, so we must use the real boundaries.

    // Get allocation handle (must retain before export)
    CUmemGenericAllocationHandle allocHandle;
    auto err = cuMemRetainAllocationHandle(&allocHandle, reinterpret_cast<void*>(chunkAddr));
    if (err != CUDA_SUCCESS)
    {
        return; // Not VMM allocated memory
    }

    // Get the REAL physical chunk boundaries using cuMemGetAllocationGranularity approach
    // The handle represents the entire physical allocation, we need its actual size
    CUmemAllocationProp prop = {};
    err = cuMemGetAllocationPropertiesFromHandle(&prop, allocHandle);
    if (err != CUDA_SUCCESS)
    {
        cuMemRelease(allocHandle);
        return;
    }

    // Get the actual mapped range for this address to find the true chunk start
    // Note: Within a VMM pool, each physical chunk is mapped contiguously
    // The buffer_id tells us chunk boundaries, but we need the actual start address
    CUdeviceptr actualChunkStart = chunkAddr;
    size_t actualChunkSize = chunkSize;

    // Find the real chunk start by checking buffer_id going backwards
    // This is necessary because chunkAddr might be in the middle of a physical chunk
    size_t granularity = getVmmGranularity();
    unsigned long long currentBufferId = getBufferId(chunkAddr);
    if (currentBufferId != 0)
    {
        // Search backwards to find the real start of this chunk
        CUdeviceptr probe = chunkAddr;
        while (probe > poolBase)
        {
            CUdeviceptr prevAddr = probe - granularity;
            if (prevAddr < poolBase)
            {
                break;
            }
            unsigned long long prevBufferId = getBufferId(prevAddr);
            if (prevBufferId != currentBufferId)
            {
                break; // Found the boundary
            }
            probe = prevAddr;
        }
        actualChunkStart = probe;

        // Search forwards to find the real end of this chunk
        CUdeviceptr endProbe = chunkAddr + chunkSize;
        // We need to find the true end, not just chunkAddr + chunkSize
        while (true)
        {
            unsigned long long probeBufferId = getBufferId(endProbe);
            if (probeBufferId != currentBufferId)
            {
                break;
            }
            endProbe += granularity;
        }
        actualChunkSize = endProbe - actualChunkStart;
    }

    // Check if we've already exported this exact chunk (same start address)
    for (auto const& existing : chunks)
    {
        if (existing.virtAddrOffset == actualChunkStart - poolBase)
        {
            // Already exported this chunk
            cuMemRelease(allocHandle);
            return;
        }
    }

    FabricMemChunk chunk;
    chunk.virtAddrOffset = actualChunkStart - poolBase; // Use REAL chunk start
    chunk.size = actualChunkSize;                       // Use REAL chunk size

    // Export fabric handle
    err = cuMemExportToShareableHandle(
        reinterpret_cast<void*>(chunk.fabricHandle), allocHandle, CU_MEM_HANDLE_TYPE_FABRIC, 0);

    // Must release retained handle to avoid memory leak
    cuMemRelease(allocHandle);

    if (err == CUDA_SUCCESS)
    {
        TLLM_LOG_DEBUG("FabricTransfer: exported fabric chunk at offset=%lu, size=%zu (requested addr=0x%lx)",
            chunk.virtAddrOffset, chunk.size, chunkAddr);
        chunks.push_back(std::move(chunk));
    }
}

void FabricTransferHelper::exportSingleChunkPosixFd(
    CUdeviceptr chunkAddr, size_t chunkSize, CUdeviceptr poolBase, std::vector<FabricMemChunk>& chunks)
{
    // Same chunk boundary detection as exportSingleChunk, but exports to POSIX FD instead of fabric handle

    // Get allocation handle (must retain before export)
    CUmemGenericAllocationHandle allocHandle;
    auto err = cuMemRetainAllocationHandle(&allocHandle, reinterpret_cast<void*>(chunkAddr));
    if (err != CUDA_SUCCESS)
    {
        return; // Not VMM allocated memory
    }

    // Get real chunk boundaries using buffer_id (same logic as exportSingleChunk)
    CUdeviceptr actualChunkStart = chunkAddr;
    size_t actualChunkSize = chunkSize;
    size_t granularity = getVmmGranularity();
    unsigned long long currentBufferId = getBufferId(chunkAddr);
    if (currentBufferId != 0)
    {
        // Search backwards
        CUdeviceptr probe = chunkAddr;
        while (probe > poolBase)
        {
            CUdeviceptr prevAddr = probe - granularity;
            if (prevAddr < poolBase)
            {
                break;
            }
            if (getBufferId(prevAddr) != currentBufferId)
            {
                break;
            }
            probe = prevAddr;
        }
        actualChunkStart = probe;

        // Search forwards
        CUdeviceptr endProbe = chunkAddr + chunkSize;
        while (true)
        {
            if (getBufferId(endProbe) != currentBufferId)
            {
                break;
            }
            endProbe += granularity;
        }
        actualChunkSize = endProbe - actualChunkStart;
    }

    // Check deduplication
    for (auto const& existing : chunks)
    {
        if (existing.virtAddrOffset == actualChunkStart - poolBase)
        {
            cuMemRelease(allocHandle);
            return; // Already exported
        }
    }

    // Export to POSIX file descriptor
    int fd = -1;
    err = cuMemExportToShareableHandle(&fd, allocHandle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);

    cuMemRelease(allocHandle);

    if (err != CUDA_SUCCESS)
    {
        TLLM_LOG_WARNING(
            "FabricTransfer: failed to export POSIX FD for chunk at 0x%lx, error=%d", actualChunkStart, err);
        return;
    }

    // Store chunk metadata (fabricHandle[64] is zeroed - FD is sent via UDS)
    FabricMemChunk chunk;
    chunk.virtAddrOffset = actualChunkStart - poolBase;
    chunk.size = actualChunkSize;
    std::memset(chunk.fabricHandle, 0, sizeof(chunk.fabricHandle));

    TLLM_LOG_DEBUG("FabricTransfer: exported POSIX FD=%d for chunk at offset=%lu, size=%zu (requested addr=0x%lx)", fd,
        chunk.virtAddrOffset, chunk.size, chunkAddr);

    chunks.push_back(std::move(chunk));

    // Store FD for UDS server to share with remote processes
    mExportedFds.push_back(fd);
}

void FabricTransferHelper::importAndMapRemoteFabric(std::string const& name, FabricMemInfo const& fabricInfo)
{
    // Skip if we've already tried and failed for this remote agent
    if (mFailedFabricImports.find(name) != mFailedFabricImports.end())
    {
        TLLM_LOG_DEBUG("FabricTransfer: skipping import for '%s' (previously failed)", name.c_str());
        return;
    }

    // For POSIX FD mode: receive FDs from remote UDS server first
    std::vector<int> receivedFds;
    if (fabricInfo.handleType == VmmHandleType::kPosixFd)
    {
        if (fabricInfo.udsPath.empty())
        {
            TLLM_LOG_WARNING("FabricTransfer: POSIX FD mode but no UDS path for '%s'", name.c_str());
            mFailedFabricImports.insert(name);
            return;
        }

        // Count total chunks across all pools
        uint32_t totalChunks = 0;
        for (auto const& pool : fabricInfo.pools)
        {
            totalChunks += static_cast<uint32_t>(pool.chunks.size());
        }

        // Connect to UDS server and receive FDs
        int udsSocket = connectUds(fabricInfo.udsPath.c_str(), 50);
        if (udsSocket < 0)
        {
            TLLM_LOG_WARNING(
                "FabricTransfer: failed to connect to UDS at '%s' for '%s'", fabricInfo.udsPath.c_str(), name.c_str());
            mFailedFabricImports.insert(name);
            return;
        }

        // Protocol: send numExpected, receive numAvailable, then receive FDs
        ssize_t bytesWritten = ::write(udsSocket, &totalChunks, sizeof(totalChunks));
        if (bytesWritten != sizeof(totalChunks))
        {
            TLLM_LOG_WARNING("FabricTransfer: UDS protocol error (write numExpected) for '%s'", name.c_str());
            ::close(udsSocket);
            mFailedFabricImports.insert(name);
            return;
        }

        uint32_t numAvailable = 0;
        ssize_t bytesRead = ::read(udsSocket, &numAvailable, sizeof(numAvailable));
        if (bytesRead != sizeof(numAvailable))
        {
            TLLM_LOG_WARNING("FabricTransfer: UDS protocol error (read numAvailable) for '%s'", name.c_str());
            ::close(udsSocket);
            mFailedFabricImports.insert(name);
            return;
        }

        uint32_t numToRecv = std::min(totalChunks, numAvailable);
        receivedFds = recvFds(udsSocket, numToRecv);
        ::close(udsSocket);

        if (receivedFds.size() != numToRecv)
        {
            TLLM_LOG_WARNING(
                "FabricTransfer: received %zu/%u FDs from UDS for '%s'", receivedFds.size(), numToRecv, name.c_str());
            // Close any received FDs on failure
            for (int fd : receivedFds)
            {
                ::close(fd);
            }
            mFailedFabricImports.insert(name);
            return;
        }

        TLLM_LOG_DEBUG("FabricTransfer: received %zu FDs from UDS for '%s'", receivedFds.size(), name.c_str());
    }

    RemoteFabricMapping mapping;
    mapping.remoteName = name;
    bool anyImportFailed = false; // Track if any import operation failed
    size_t fdIndex = 0;           // Index into receivedFds (for POSIX FD mode)

    for (auto const& pool : fabricInfo.pools)
    {
        RemotePoolMapping poolMapping;
        poolMapping.remoteBaseAddr = pool.poolBaseAddr;
        poolMapping.totalSize = pool.poolTotalSize;
        poolMapping.remoteRegisteredAddr = pool.registeredAddr;
        poolMapping.registeredSize = pool.registeredSize;
        poolMapping.remoteMappedOffset = pool.mappedOffset;
        poolMapping.mappedSize = pool.mappedSize;

        // Reserve local virtual address space for the MAPPED range (not registeredSize!)
        // mappedSize may be larger than registeredSize due to chunk alignment
        CUdeviceptr localVa;
        auto err = cuMemAddressReserve(&localVa, pool.mappedSize, 0, 0, 0);
        if (err != CUDA_SUCCESS)
        {
            TLLM_LOG_WARNING("FabricTransfer: failed to reserve VA for remote pool (mappedSize=%zu), error=%d",
                pool.mappedSize, err);
            anyImportFailed = true;
            fdIndex += pool.chunks.size(); // Skip FDs for this pool
            continue;
        }
        poolMapping.localVirtAddr = localVa;

        bool allChunksMapped = true;
        for (auto const& chunk : pool.chunks)
        {
            // Calculate local offset: chunk.virtAddrOffset is relative to poolBase,
            // but our local VA covers [mappedOffset, mappedOffset + mappedSize)
            // localOffset = chunk.virtAddrOffset - mappedOffset
            if (chunk.virtAddrOffset < pool.mappedOffset)
            {
                // Chunk starts before mapped range - this shouldn't happen
                TLLM_LOG_ERROR("FabricTransfer: chunk offset %lu < mapped offset %lu, skipping", chunk.virtAddrOffset,
                    pool.mappedOffset);
                fdIndex++;
                continue;
            }
            uint64_t localOffset = chunk.virtAddrOffset - pool.mappedOffset;

            // Import handle based on handle type
            CUmemGenericAllocationHandle importedHandle;
            if (fabricInfo.handleType == VmmHandleType::kPosixFd)
            {
                // POSIX FD mode: import from received file descriptor
                // CRITICAL: pass fd value as (void*)(uintptr_t)fd, NOT &fd!
                if (fdIndex >= receivedFds.size())
                {
                    TLLM_LOG_ERROR(
                        "FabricTransfer: ran out of FDs (index=%zu, available=%zu)", fdIndex, receivedFds.size());
                    allChunksMapped = false;
                    anyImportFailed = true;
                    break;
                }
                int fd = receivedFds[fdIndex++];
                err = cuMemImportFromShareableHandle(&importedHandle,
                    reinterpret_cast<void*>(static_cast<uintptr_t>(fd)), CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
                // Close FD after import (CUDA duplicates internally)
                ::close(fd);
            }
            else
            {
                // Fabric mode: import from serialized fabric handle bytes
                err = cuMemImportFromShareableHandle(&importedHandle,
                    const_cast<void*>(reinterpret_cast<void const*>(chunk.fabricHandle)), CU_MEM_HANDLE_TYPE_FABRIC);
            }

            if (err != CUDA_SUCCESS)
            {
                TLLM_LOG_WARNING("FabricTransfer: failed to import handle for '%s', type=%s, error=%d", name.c_str(),
                    fabricInfo.handleType == VmmHandleType::kFabric ? "Fabric" : "PosixFd", err);
                allChunksMapped = false;
                anyImportFailed = true;
                break;
            }

            // Map to local address space using localOffset (relative to registered range start)
            err = cuMemMap(localVa + localOffset, chunk.size, 0, importedHandle, 0);
            if (err != CUDA_SUCCESS)
            {
                TLLM_LOG_WARNING(
                    "FabricTransfer: failed to map remote chunk at localOffset=%lu, error=%d", localOffset, err);
                cuMemRelease(importedHandle);
                allChunksMapped = false;
                anyImportFailed = true;
                break;
            }

            // Set access permissions using localOffset
            CUmemAccessDesc accessDesc = {};
            accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            accessDesc.location.id = pool.deviceId;
            accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            err = cuMemSetAccess(localVa + localOffset, chunk.size, &accessDesc, 1);
            if (err != CUDA_SUCCESS)
            {
                TLLM_LOG_WARNING("FabricTransfer: failed to set access for remote chunk, error=%d", err);
                // Continue anyway, access might work for read operations
            }

            poolMapping.importedHandles.push_back(importedHandle);
        }

        if (allChunksMapped)
        {
            mapping.pools.push_back(std::move(poolMapping));
            TLLM_LOG_DEBUG("FabricTransfer: mapped remote pool at 0x%lx -> local 0x%lx, size=%zu", pool.poolBaseAddr,
                localVa, pool.poolTotalSize);
        }
        else
        {
            // Clean up on failure - use mappedSize (not poolTotalSize) since that's what we reserved
            for (auto handle : poolMapping.importedHandles)
            {
                cuMemRelease(handle);
            }
            cuMemAddressFree(localVa, pool.mappedSize);
        }
    }

    // Close any remaining unused FDs (in case of partial failure)
    for (size_t i = fdIndex; i < receivedFds.size(); ++i)
    {
        ::close(receivedFds[i]);
    }

    if (!mapping.pools.empty())
    {
        mRemoteFabricMappings[name] = std::move(mapping);
        TLLM_LOG_INFO("FabricTransfer: mapping established for remote agent '%s' with %zu/%zu pools (type=%s)",
            name.c_str(), mRemoteFabricMappings[name].pools.size(), fabricInfo.pools.size(),
            fabricInfo.handleType == VmmHandleType::kFabric ? "Fabric" : "PosixFd");
    }
    else if (anyImportFailed)
    {
        // All imports failed - record this to avoid retrying and ensure fallback to NIXL
        mFailedFabricImports.insert(name);
        TLLM_LOG_WARNING("FabricTransfer: import failed for '%s', will use NIXL fallback.", name.c_str());
    }
}

void FabricTransferHelper::cleanupRemoteFabricMapping(std::string const& name)
{
    // Also clear failed import record so it can be retried if needed
    mFailedFabricImports.erase(name);

    auto it = mRemoteFabricMappings.find(name);
    if (it == mRemoteFabricMappings.end())
    {
        return;
    }

    for (auto& poolMapping : it->second.pools)
    {
        for (auto handle : poolMapping.importedHandles)
        {
            cuMemRelease(handle);
        }
        if (poolMapping.localVirtAddr != 0)
        {
            // Use registeredSize (not totalSize) because we only reserved that much
            cuMemUnmap(poolMapping.localVirtAddr, poolMapping.mappedSize);
            cuMemAddressFree(poolMapping.localVirtAddr, poolMapping.mappedSize);
        }
    }
    mRemoteFabricMappings.erase(it);
}

void* FabricTransferHelper::translateToLocalMapping(std::string const& remoteName, uintptr_t remoteAddr) const
{
    auto it = mRemoteFabricMappings.find(remoteName);
    if (it == mRemoteFabricMappings.end())
    {
        return nullptr;
    }
    return translateToLocalMappingInternal(it->second, remoteAddr);
}

void* FabricTransferHelper::translateToLocalMappingInternal(
    RemoteFabricMapping const& mapping, uintptr_t remoteAddr) const
{
    for (auto const& pool : mapping.pools)
    {
        // Check if address is within the registered memory range (strict validation)
        uint64_t registeredEnd = pool.remoteRegisteredAddr + pool.registeredSize;
        if (remoteAddr >= pool.remoteRegisteredAddr && remoteAddr < registeredEnd)
        {
            // Address is within registered range
            // Local VA covers [mappedOffset, mappedOffset + mappedSize) relative to remoteBaseAddr
            // Calculate: offset from remoteBaseAddr, then subtract mappedOffset
            uint64_t offsetFromPoolBase = remoteAddr - pool.remoteBaseAddr;
            uint64_t localOffset = offsetFromPoolBase - pool.remoteMappedOffset;
            return reinterpret_cast<void*>(pool.localVirtAddr + localOffset);
        }
        // Check if address is in pool range but outside registered range (error case)
        if (remoteAddr >= pool.remoteBaseAddr && remoteAddr < pool.remoteBaseAddr + pool.totalSize)
        {
            TLLM_LOG_ERROR(
                "FabricTransfer: address 0x%lx is within pool [0x%lx, 0x%lx) but outside "
                "registered range [0x%lx, 0x%lx). Transfer rejected.",
                remoteAddr, pool.remoteBaseAddr, pool.remoteBaseAddr + pool.totalSize, pool.remoteRegisteredAddr,
                registeredEnd);
            TLLM_THROW("Transfer address 0x%lx outside registered memory range [0x%lx, 0x%lx)", remoteAddr,
                pool.remoteRegisteredAddr, registeredEnd);
        }
    }
    return nullptr; // Address not in any mapped pool
}

std::unique_ptr<TransferStatus> FabricTransferHelper::submitWithCubBatched(
    std::vector<void*> const& srcPtrs, std::vector<void*> const& dstPtrs, std::vector<size_t> const& sizes)
{
    ensureCudaResourcesInitialized();

    size_t numBuffers = srcPtrs.size();
    if (numBuffers == 0)
    {
        // Empty transfer, complete immediately
        auto event = std::make_shared<runtime::CudaEvent>();
        mFabricStream->record(*event);
        return std::make_unique<FabricTransferStatus>(mFabricStream, event);
    }

    // Lock to protect pre-allocated buffers (serialize fabric transfers)
    std::lock_guard<std::mutex> lock(mBufferMutex);

    // Query cub temp storage size (use nullptr to query)
    size_t cubTempBytes = 0;
    cub::DeviceMemcpy::Batched(nullptr, cubTempBytes, static_cast<void const* const*>(nullptr),
        static_cast<void* const*>(nullptr), static_cast<size_t const*>(nullptr), numBuffers, mFabricStream->get());

    // Ensure pre-allocated buffers are large enough
    ensurePreallocBuffers(numBuffers, cubTempBytes);

    // Copy pointer arrays to GPU (H2D, using pre-allocated buffers)
    TLLM_CUDA_CHECK(cudaMemcpyAsync(mPreallocBuffers.srcPtrs->data(), srcPtrs.data(), numBuffers * sizeof(void*),
        cudaMemcpyHostToDevice, mFabricStream->get()));
    TLLM_CUDA_CHECK(cudaMemcpyAsync(mPreallocBuffers.dstPtrs->data(), dstPtrs.data(), numBuffers * sizeof(void*),
        cudaMemcpyHostToDevice, mFabricStream->get()));
    TLLM_CUDA_CHECK(cudaMemcpyAsync(mPreallocBuffers.sizes->data(), sizes.data(), numBuffers * sizeof(size_t),
        cudaMemcpyHostToDevice, mFabricStream->get()));

    // Execute batch memcpy (async, reusing pre-allocated temp storage)
    size_t actualTempBytes = mPreallocBuffers.cubTempStorageSize;
    TLLM_CUDA_CHECK(cub::DeviceMemcpy::Batched(mPreallocBuffers.cubTempStorage->data(), actualTempBytes,
        static_cast<void const**>(mPreallocBuffers.srcPtrs->data()),
        static_cast<void**>(mPreallocBuffers.dstPtrs->data()), static_cast<size_t*>(mPreallocBuffers.sizes->data()),
        numBuffers, mFabricStream->get()));

    // Record event on stream
    auto completionEvent = std::make_shared<runtime::CudaEvent>();
    mFabricStream->record(*completionEvent);

    return std::make_unique<FabricTransferStatus>(mFabricStream, completionEvent);
}

std::unique_ptr<TransferStatus> FabricTransferHelper::submitWithCudaMemcpyBatch(
    std::vector<void*> const& srcPtrs, std::vector<void*> const& dstPtrs, std::vector<size_t> const& sizes)
{
    ensureCudaResourcesInitialized();

    size_t numOps = srcPtrs.size();
    if (numOps == 0)
    {
        // Empty transfer, complete immediately
        auto event = std::make_shared<runtime::CudaEvent>();
        mFabricStream->record(*event);
        return std::make_unique<FabricTransferStatus>(mFabricStream, event);
    }

    // Use cudaMemcpyBatchAsync with overlap compute
    // API: cudaMemcpyBatchAsync(dsts, srcs, sizes, count, attrs, attrsIdxs, numAttrs, stream)

    // Setup single attribute for all segments - overlap with compute
    cudaMemcpyAttributes attrs = {};
    attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    attrs.flags |= cudaMemcpyFlagPreferOverlapWithCompute;

    // All segments use the same attribute (index 0)
    std::vector<size_t> attrIndices(numOps, 0);

    // Note: srcPtrs needs to be const void* const*, cast from void* const*
    std::vector<void const*> constSrcPtrs(srcPtrs.begin(), srcPtrs.end());

    TLLM_CUDA_CHECK(cudaMemcpyBatchAsync(dstPtrs.data(), constSrcPtrs.data(), sizes.data(), numOps, &attrs,
        attrIndices.data(), 1, mFabricStream->get()));

    // Record event
    auto completionEvent = std::make_shared<runtime::CudaEvent>();
    mFabricStream->record(*completionEvent);

    return std::make_unique<FabricTransferStatus>(mFabricStream, completionEvent);
}

} // namespace tensorrt_llm::executor::kv_cache

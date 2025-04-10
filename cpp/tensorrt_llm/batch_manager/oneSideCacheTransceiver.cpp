

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

#include "tensorrt_llm/executor/types.h"
#include <cstdint>
#include <limits>
#include <sstream>
#define UCX_WRAPPER_LIB_NAME "tensorrt_llm_ucx_wrapper"

#if defined(_WIN32)
#include <windows.h>
#define dllOpen(name) LoadLibrary(name ".dll")
#define dllClose(handle) FreeLibrary(static_cast<HMODULE>(handle))
#define dllGetSym(handle, name) static_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle), name))
#else // For non-Windows platforms
#include <dlfcn.h>
#define dllOpen(name) dlopen("lib" name ".so", RTLD_LAZY)
#define dllClose(handle) dlclose(handle)
#define dllGetSym(handle, name) dlsym(handle, name)
#endif // defined(_WIN32)

#include "tensorrt_llm/batch_manager/cacheFormatter.h"
#include "tensorrt_llm/batch_manager/cacheTransceiver.h"
#include "tensorrt_llm/batch_manager/contextProgress.h"
#include "tensorrt_llm/batch_manager/dataTransceiverImpl.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/batch_manager/mlaCacheFormatter.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/cache_transmission/mpi_utils/connection.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/executor/serializeUtils.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include <algorithm>
#include <cstddef>
#include <memory>
#include <numeric>
#include <unordered_set>

namespace tensorrt_llm
{
namespace batch_manager
{

enum class RegisterMemoryType
{
    kHost,
    kDevice,
    kPinned,
};

class TransceiverEngineInfo
{

    // ip:port address
};

class TransceiverConnectionSideChannel
{

public:
    void send(void* ptr, size_t size) {}

    void recv(void* ptr, size_t size) {}
};

class TransferRequest
{
public:
    virtual ~TransferRequest() = default;
    virtual bool isCompleted() = 0;

    virtual void wait() = 0;
};

class RemoteBlockInfos
{
    class RemoteBlockEntry
    {
    public:
        size_t address;
        size_t blockSize;
    };

private:
    std::vector<RemoteBlockEntry> entries;

    uint64_t remoteBlockIds; // getFrom Remote side;

    std::string syncInfo;
};

class LocalBlockInfos
{
    class LocalBlockEntry
    {
    public:
        size_t address;
        size_t blockSize;
    };
};

class TransceiverConnection
{
public:
    std::shared_ptr<TransferRequest> writeBlocksAsync(
        std::vector<runtime::ITensor::SharedPtr> const& srcTensors, RemoteBlockInfos const& remoteBlockInfos){

    };

    std::shared_ptr<TransferRequest> readBlocksAsync(
        std::vector<runtime::ITensor::SharedPtr> const& dstTensors, RemoteBlockInfos const& localBlockInfos){

    };

private:
    TransceiverConnectionSideChannel sideChannel;
};

class TransceiverEngine
{

public:
    void registerMemory(void* ptr, size_t size, RegisterMemoryType memoryType) {}

    void unregisterMemory(void* ptr, size_t size, RegisterMemoryType memoryType) {}

    TransceiverEngineInfo getConnectionInfo() {}

    TransceiverConnection getTransceiverConnection(TransceiverEngineInfo& info) {}

    bool isRegistered(void* ptr, size_t size, RegisterMemoryType memoryType) {}

    std::string uniqueEngineId; // uuid? // hostname+pid+atomic_increment
};

} // namespace batch_manager
}; // namespace tensorrt_llm

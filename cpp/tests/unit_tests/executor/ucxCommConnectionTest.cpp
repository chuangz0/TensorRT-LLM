/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#define UCX_WRAPPER_LIB_NAME "tensorrt_llm_ucx_wrapper"

#if defined(_WIN32)
#include <windows.h>
#define dllOpen(name) LoadLibrary(name ".dll")
#define dllClose(handle) FreeLibrary(static_cast<HMODU`LE>(handle))
#define dllGetSym(handle, name) static_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle), name))
#else // For non-Windows platforms
#include <dlfcn.h>
#define dllOpen(name) dlopen("lib" name ".so", RTLD_LAZY)
#define dllClose(handle) dlclose(handle)
#define dllGetSym(handle, name) dlsym(handle, name)
#endif // defined(_WIN32)

#include "tensorrt_llm/batch_manager/cacheFormatter.h"
#include "tensorrt_llm/batch_manager/cacheTransceiver.h"
#include "tensorrt_llm/batch_manager/dataTransceiverImpl.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/executor/cache_transmission/mpi_utils/connection.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "gtest/gtest.h"
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <gmock/gmock.h>
#include <memory>
#include <random>
#include <tensorrt_llm/batch_manager/dataTransceiverImpl.h>
#include <tensorrt_llm/batch_manager/mlaCacheFormatter.h>
#include <tensorrt_llm/executor/cache_transmission/cacheConcatenate.h>

using SizeType32 = tensorrt_llm::runtime::SizeType32;
using LlmRequest = tensorrt_llm::batch_manager::LlmRequest;
using namespace tensorrt_llm::batch_manager::kv_cache_manager;
using namespace tensorrt_llm::batch_manager;
namespace texec = tensorrt_llm::executor;

#include "cxxopts.hpp"

namespace
{
std::mutex mDllMutex;

std::unique_ptr<texec::kv_cache::ConnectionManager> makeOneUcxConnectionManager()
{
    std::lock_guard<std::mutex> lock(mDllMutex);
    void* WrapperLibHandle{nullptr};
    WrapperLibHandle = dllOpen(UCX_WRAPPER_LIB_NAME);
    TLLM_CHECK_WITH_INFO(WrapperLibHandle != nullptr, "UCX wrapper library is not open correctly.");
    auto load_sym = [](void* handle, char const* name)
    {
        void* ret = dllGetSym(handle, name);

        TLLM_CHECK_WITH_INFO(ret != nullptr,
            "Unable to load UCX wrapper library symbol, possible cause is that TensorRT-LLM library is not "
            "built with UCX support, please rebuild in UCX-enabled environment.");
        return ret;
    };
    std::unique_ptr<tensorrt_llm::executor::kv_cache::ConnectionManager> (*makeUcxConnectionManager)();
    *(void**) (&makeUcxConnectionManager) = load_sym(WrapperLibHandle, "makeUcxConnectionManager");
    return makeUcxConnectionManager();
}

} // namespace

int main(int argc, char* argv[])
{
    cxxopts::Options options("ucxCommConnectionTest", "Test UCX communication connection");
    options.add_options()("h,help", "Print help");

    options.add_options()("is_server", "Whether to run as server", cxxopts::value<bool>()->default_value("false"));
    options.add_options()("server_ip", "Server IP", cxxopts::value<std::string>()->default_value("127.0.0.1"));
    options.add_options()("server_port", "Server port", cxxopts::value<int>()->default_value("12345"));

    auto result = options.parse(argc, argv);
    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        return 0;
    }

    auto connectionManager = makeOneUcxConnectionManager();

    bool const isServer = result["is_server"].as<bool>();

    tensorrt_llm::runtime::BufferManager bufferManager{std::make_shared<tensorrt_llm::runtime::CudaStream>()};
    constexpr size_t bufferSize = 1024;
    tensorrt_llm::executor::kv_cache::DataContext dataContext{0x75};
    if (isServer)
    {
        auto CommState = connectionManager->getCommState();
        TransceiverTag::Id id1Peer;

        TLLM_LOG_INFO("Server is running on %s", CommState.toString().c_str());
        auto connection1Peer = connectionManager->recvConnect(
            tensorrt_llm::executor::kv_cache::DataContext{TransceiverTag::kID_TAG}, &id1Peer, sizeof(id1Peer));

        auto srcBuffer1 = bufferManager.gpuSync(bufferSize, nvinfer1::DataType::kINT8);
        connection1Peer->send(dataContext, srcBuffer1->data(), srcBuffer1->getSizeInBytes());
        TLLM_LOG_INFO("Sent data to client");
    }
    else
    {
        auto serverIp = result["server_ip"].as<std::string>();
        auto serverPort = result["server_port"].as<int>();
        tensorrt_llm::executor::kv_cache::CommState CommState{static_cast<uint16_t>(serverPort), serverIp};

        TLLM_LOG_INFO("Client is connecting to server on %s", CommState.toString().c_str());

        auto connection = connectionManager->getConnections(CommState)[0];
        TransceiverTag::Id id1 = TransceiverTag::Id::REQUEST_SEND;
        connection->send(tensorrt_llm::executor::kv_cache::DataContext{TransceiverTag::kID_TAG}, &id1, sizeof(id1));

        auto dstBuffer1 = bufferManager.gpuSync(bufferSize, nvinfer1::DataType::kINT8);
        connection->recv(dataContext, dstBuffer1->data(), dstBuffer1->getSizeInBytes());
        // bufferManager.getStream().synchronize();
        TLLM_LOG_INFO("Received data from server");

        // TLLM_LOG_INFO("Client is connecting to server on port %d", CommState.getSocketState()[0].getPort());
    }
}

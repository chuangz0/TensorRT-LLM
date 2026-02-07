/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/executor/transferAgent.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/transferAgent.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/executor/serialization.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

using namespace tensorrt_llm::executor::kv_cache;

std::vector<std::string> getAvailableBackends()
{
    std::vector<std::string> backends;

#ifdef TEST_NIXL_BACKEND
    backends.push_back("nixl");
#endif

#ifdef TEST_MOONCAKE_BACKEND
    backends.push_back("mooncake");
#endif

    return backends;
}

class RegisteredHostMemory
{
public:
    RegisteredHostMemory(MemoryDescs mems, BaseTransferAgent* agent)
        : mDescs{std::move(mems)}
        , mAgentPtr{agent}
    {
        TLLM_CHECK(mAgentPtr);
        mAgentPtr->registerMemory(mDescs);
    }

    ~RegisteredHostMemory()
    {
        TLLM_CHECK(mAgentPtr);
        mAgentPtr->deregisterMemory(mDescs);
    }

    [[nodiscard]] MemoryDescs const& getDescs() const noexcept
    {
        return mDescs;
    }

private:
    MemoryDescs mDescs;
    BaseTransferAgent* mAgentPtr{};
};

class TransferAgentTest : public ::testing::TestWithParam<std::string> // NOLINT(cppcoreguidelines-pro-type-member-init)
{
public:
    void SetUp() override
    {
        backend = GetParam();
    }

    void TearDown() override {}

    [[nodiscard]] std::unique_ptr<BaseTransferAgent> makeTransferAgent(BaseAgentConfig const& config)
    {
        return tensorrt_llm::executor::kv_cache::makeTransferAgent(backend, &config);
    }

    std::string backend;
};

TEST_P(TransferAgentTest, Basic)
{

    std::string const agent0{"agent0"}, agent1{"agent1"};
    BaseAgentConfig config0{agent0, true, false, true}, config1{agent1, true, false, true};
    auto xferAgent0 = makeTransferAgent(config0);
    auto xferAgent1 = makeTransferAgent(config1);

    TLLM_CHECK(xferAgent0);
    TLLM_CHECK(xferAgent1);

    std::vector<char> memory0(100, 10);
    std::vector<char> memory1(100, 1);

    RegisteredHostMemory regMem0(MemoryDescs{MemoryType::kDRAM, {MemoryDesc{memory0}}}, xferAgent0.get());
    RegisteredHostMemory regMem1(MemoryDescs{MemoryType::kDRAM, {MemoryDesc{memory1}}}, xferAgent1.get());

    // xferAgent0->loadRemoteAgent(agent1);
    auto connectionInfo = xferAgent1->getLocalConnectionInfo();
    xferAgent0->loadRemoteAgent(agent1, connectionInfo);
    bool checked = false;
    do
    {
        checked = xferAgent0->checkRemoteDescs(agent1, regMem1.getDescs());
        // wait for regMem is unpacked by xferAgent0
    } while (!checked);

    TransferRequest writeReq{TransferOp::kWRITE, regMem0.getDescs(), regMem1.getDescs(), agent1};
    auto status = xferAgent0->submitTransferRequests(writeReq);
    TLLM_CHECK(status->wait() == TransferState::kSUCCESS);

    TLLM_CHECK(memory0 == memory1);
    xferAgent0->invalidateRemoteAgent(agent1);
}

TEST_P(TransferAgentTest, Basic2)
{

    std::string const agent0{"agent0"}, agent1{"agent1"};
    BaseAgentConfig config0{agent0, true, false, true}, config1{agent1, true, false, true};
    auto xferAgent0 = makeTransferAgent(config0);
    auto xferAgent1 = makeTransferAgent(config1);

    TLLM_CHECK(xferAgent0);
    TLLM_CHECK(xferAgent1);

    std::vector<char> memory0(100, 10);
    std::vector<char> memory1(100, 1);

    RegisteredHostMemory regMem0(MemoryDescs{MemoryType::kDRAM, {MemoryDesc{memory0}}}, xferAgent0.get());
    RegisteredHostMemory regMem1(MemoryDescs{MemoryType::kDRAM, {MemoryDesc{memory1}}}, xferAgent1.get());

    // xferAgent0->loadRemoteAgent(agent1);
    auto connectionInfo = xferAgent1->getLocalConnectionInfo();
    xferAgent0->loadRemoteAgent(agent1, connectionInfo);
    bool checked = false;
    do
    {
        checked = xferAgent0->checkRemoteDescs(agent1, regMem1.getDescs());
    } while (!checked);

    TransferRequest readReq{TransferOp::kREAD, regMem0.getDescs(), regMem1.getDescs(), agent1};
    auto status = xferAgent0->submitTransferRequests(readReq);
    TLLM_CHECK(status->wait() == TransferState::kSUCCESS);

    TLLM_CHECK(memory0 == memory1);

    xferAgent0->invalidateRemoteAgent(agent1);
}

TEST_P(TransferAgentTest, DeviceMemory)
{

    std::string const agent0{"agent0"}, agent1{"agent1"};
    BaseAgentConfig config0{agent0, true, false, true}, config1{agent1, true, false, true};
    auto xferAgent0 = makeTransferAgent(config0);
    auto xferAgent1 = makeTransferAgent(config1);

    TLLM_CHECK(xferAgent0);
    TLLM_CHECK(xferAgent1);
    char* dev_ptr0;
    char* dev_ptr1;
    size_t size = 100;
    uint32_t deviceId = 0;
    cudaMalloc(&dev_ptr0, size);
    cudaMalloc(&dev_ptr1, size);
    std::vector<char> memory0(size, 10);
    std::vector<char> memory1(size, 1);
    TLLM_CUDA_CHECK(cudaMemcpy(dev_ptr0, memory0.data(), size, cudaMemcpyHostToDevice));
    TLLM_CUDA_CHECK(cudaMemcpy(dev_ptr1, memory1.data(), size, cudaMemcpyHostToDevice));
    RegisteredHostMemory regMem0(
        MemoryDescs{MemoryType::kVRAM, {MemoryDesc{dev_ptr0, size, deviceId}}}, xferAgent0.get());
    RegisteredHostMemory regMem1(
        MemoryDescs{MemoryType::kVRAM, {MemoryDesc{dev_ptr1, size, deviceId}}}, xferAgent1.get());

    // xferAgent0->loadRemoteAgent(agent1);
    auto connectionInfo = xferAgent1->getLocalConnectionInfo();
    xferAgent0->loadRemoteAgent(agent1, connectionInfo);
    bool checked = false;
    do
    {
        checked = xferAgent0->checkRemoteDescs(agent1, regMem1.getDescs());
    } while (!checked);
    TransferRequest writeReq{TransferOp::kWRITE, regMem0.getDescs(), regMem1.getDescs(), agent1};
    auto status = xferAgent0->submitTransferRequests(writeReq);
    TLLM_CHECK(status->wait() == TransferState::kSUCCESS);

    TLLM_CUDA_CHECK(cudaMemcpy(memory0.data(), dev_ptr0, size, cudaMemcpyDeviceToHost));
    TLLM_CUDA_CHECK(cudaMemcpy(memory1.data(), dev_ptr1, size, cudaMemcpyDeviceToHost));

    TLLM_CHECK(memory0 == memory1);

    TLLM_CUDA_CHECK(cudaFree(dev_ptr0));
    TLLM_CUDA_CHECK(cudaFree(dev_ptr1));
    xferAgent0->invalidateRemoteAgent(agent1);
}

TEST_P(TransferAgentTest, Connect)
{

    std::string const agent0{"agent0"}, agent1{"agent1"}, agent2{"agent2"};
    BaseAgentConfig config0{agent0, true, false, true}, config1{agent1, true, false, true},
        config2{agent2, true, false, true};
    auto xferAgent0 = makeTransferAgent(config0);
    auto xferAgent1 = makeTransferAgent(config1);
    auto xferAgent2 = makeTransferAgent(config2);

    TLLM_CHECK(xferAgent0);
    TLLM_CHECK(xferAgent1);

    std::vector<char> memory0(100, 10);
    std::vector<char> memory1(100, 1);
    MemoryDescs memDescs0{MemoryType::kDRAM, {MemoryDesc{memory0}}};
    MemoryDescs memDescs1{MemoryType::kDRAM, {MemoryDesc{memory1}}};

    xferAgent0->registerMemory(memDescs0);
    xferAgent1->registerMemory(memDescs1);
    xferAgent2->registerMemory(memDescs0);

    // xferAgent0->loadRemoteAgent(agent1);
    auto connectionInfo = xferAgent1->getLocalConnectionInfo();
    xferAgent0->loadRemoteAgent(agent1, connectionInfo);
    bool checked = false;
    do
    {
        checked = xferAgent0->checkRemoteDescs(agent1, memDescs1);
    } while (!checked);
    TransferRequest writeReq{TransferOp::kWRITE, memDescs0, memDescs1, agent1};
    auto status = xferAgent0->submitTransferRequests(writeReq);
    TLLM_CHECK(status->wait() == TransferState::kSUCCESS);

    TLLM_CHECK(memory0 == memory1);
    xferAgent2->loadRemoteAgent(agent1, connectionInfo);
    checked = false;
    do
    {
        checked = xferAgent2->checkRemoteDescs(agent1, memDescs1);
    } while (!checked);
    TransferRequest writeReq2{TransferOp::kWRITE, memDescs0, memDescs1, agent1};
    auto status2 = xferAgent2->submitTransferRequests(writeReq2);
    TLLM_CHECK(status2->wait() == TransferState::kSUCCESS);
    TLLM_CHECK(memory0 == memory1);
    xferAgent0->invalidateRemoteAgent(agent1);
    xferAgent2->invalidateRemoteAgent(agent1);
    xferAgent0->deregisterMemory(memDescs0);
    xferAgent1->deregisterMemory(memDescs1);
    xferAgent2->deregisterMemory(memDescs0);
}

TEST_P(TransferAgentTest, SyncMessage)
{
    constexpr std::size_t MAX_QUERY_TIMES = std::numeric_limits<size_t>::max();
    std::string const agent0{"agent0"}, agent1{"agent1"};
    BaseAgentConfig config0{agent0, true, false, true}, config1{agent1, true, false, true};
    auto xferAgent0 = makeTransferAgent(config0);
    auto xferAgent1 = makeTransferAgent(config1);

    TLLM_CHECK(xferAgent0);
    TLLM_CHECK(xferAgent1);

    std::vector<char> memory0(100, 10);
    std::vector<char> memory1(100, 1);

    RegisteredHostMemory regMem0(MemoryDescs{MemoryType::kDRAM, {MemoryDesc{memory0}}}, xferAgent0.get());
    RegisteredHostMemory regMem1(MemoryDescs{MemoryType::kDRAM, {MemoryDesc{memory1}}}, xferAgent0.get());

    RegisteredHostMemory regMem2(MemoryDescs{MemoryType::kDRAM, {MemoryDesc{memory0}}}, xferAgent1.get());
    RegisteredHostMemory regMem3(MemoryDescs{MemoryType::kDRAM, {MemoryDesc{memory1}}}, xferAgent1.get());

    // xferAgent0->loadRemoteAgent(agent1);
    auto connectionInfo = xferAgent1->getLocalConnectionInfo();
    xferAgent0->loadRemoteAgent(agent1, connectionInfo);
    bool checked = false;
    do
    {
        checked = xferAgent0->checkRemoteDescs(agent1, regMem3.getDescs());
    } while (!checked);
    auto syncMessage = std::string("agent_sync_message");
    TransferRequest writeReq{TransferOp::kWRITE, regMem0.getDescs(), regMem3.getDescs(), agent1};
    auto status = xferAgent0->submitTransferRequests(writeReq);
    xferAgent0->notifySyncMessage(agent1, syncMessage);

    auto notif = xferAgent1->getNotifiedSyncMessages();
    for (std::size_t i = 0; i < MAX_QUERY_TIMES && notif.size() == 0; i++)
    {
        notif = xferAgent1->getNotifiedSyncMessages();
    }
    TLLM_CHECK(status->wait() == TransferState::kSUCCESS);
    TLLM_CHECK(status->isCompleted());
    TLLM_CHECK(notif.size() == 1);
    TLLM_CHECK(notif[agent0].size() == 1);
    TLLM_CHECK(notif[agent0][0] == syncMessage);

    TLLM_CHECK(memory0 == memory1);

    std::string syncMessage2 = "two_agent_sync_message";
    xferAgent0->notifySyncMessage(agent1, syncMessage2);
    auto notif2 = xferAgent1->getNotifiedSyncMessages();
    for (std::size_t i = 0; i < MAX_QUERY_TIMES && notif2.size() == 0; i++)
    {
        notif2 = xferAgent1->getNotifiedSyncMessages();
    }
    TLLM_CHECK(notif2.size() == 1);
    TLLM_CHECK(notif2[agent0].size() == 1);
    TLLM_CHECK(notif2[agent0][0] == syncMessage2);

    // xferAgent1->loadRemoteAgent(agent0);
    auto connectionInfo2 = xferAgent0->getLocalConnectionInfo();
    xferAgent1->loadRemoteAgent(agent0, connectionInfo2);
    std::string syncMessage3 = "three_agent_sync_message";
    xferAgent1->notifySyncMessage(agent0, syncMessage3);
    auto notif3 = xferAgent0->getNotifiedSyncMessages();
    for (std::size_t i = 0; i < MAX_QUERY_TIMES && notif3.size() == 0; i++)
    {
        notif3 = xferAgent0->getNotifiedSyncMessages();
    }
    TLLM_CHECK(notif3.size() == 1);
    TLLM_CHECK(notif3[agent1].size() == 1);
    TLLM_CHECK(notif3[agent1][0] == syncMessage3);

    bool checked2 = false;
    do
    {
        checked2 = xferAgent0->checkRemoteDescs(agent1, regMem1.getDescs());
    } while (!checked2);

    std::string syncMessage4 = "four_agent_sync_message";
    TransferRequest writeReq1{TransferOp::kWRITE, regMem2.getDescs(), regMem1.getDescs(), agent0};
    auto status1 = xferAgent1->submitTransferRequests(writeReq1);
    xferAgent1->notifySyncMessage(agent0, syncMessage4);

    auto notif4 = xferAgent0->getNotifiedSyncMessages();
    for (std::size_t i = 0; i < MAX_QUERY_TIMES && notif4.size() == 0; i++)
    {
        notif4 = xferAgent0->getNotifiedSyncMessages();
    }
    TLLM_CHECK(status1->wait() == TransferState::kSUCCESS);
    TLLM_CHECK(status1->isCompleted());
    TLLM_CHECK(notif4.size() == 1);
    TLLM_CHECK(notif4[agent1].size() == 1);
    TLLM_CHECK(notif4[agent1][0] == syncMessage4);

    TLLM_CHECK(memory0 == memory1);

    // serialization

    CommState state{std::vector<SocketState>{SocketState{1234, "127.0.0.1"}}, 0};
    using namespace tensorrt_llm::executor;
    std::stringstream ss;
    Serialization::serialize(state, ss);
    std::string serializedState = ss.str();
    xferAgent0->notifySyncMessage(agent1, serializedState);
    auto notif5 = xferAgent1->getNotifiedSyncMessages();
    for (size_t i = 0; i < MAX_QUERY_TIMES && notif5.size() == 0; i++)
    {
        notif5 = xferAgent1->getNotifiedSyncMessages();
    }
    TLLM_CHECK(notif5.size() == 1);
    TLLM_CHECK(notif5[agent0].size() == 1);
    TLLM_CHECK(notif5[agent0][0] == serializedState);
    std::stringstream ss2(notif5[agent0][0]);
    auto state2 = Serialization::deserializeCommState(ss2);
    TLLM_CHECK(state2 == state);

    xferAgent0->invalidateRemoteAgent(agent1);
    xferAgent1->invalidateRemoteAgent(agent0);
}

INSTANTIATE_TEST_SUITE_P(AvailableBackends, TransferAgentTest, ::testing::ValuesIn(getAvailableBackends()),
    [](::testing::TestParamInfo<TransferAgentTest::ParamType> const& info) { return info.param; });

// Skip LoopbackAgentTest for mooncake backend for now
#ifdef TEST_NIXL_BACKEND

class LoopbackAgentTest : public ::testing::Test,
                          public ::testing::WithParamInterface<bool> // NOLINT(cppcoreguidelines-pro-type-member-init)
{
public:
    void SetUp() override
    {
        static int file_num = 0;
        std::string filename = std::string("test_agent") + std::to_string(file_num++);
        auto dirPath = fs::absolute(filename);
        std::error_code ec;
        fs::create_directories(dirPath, ec);
        TLLM_CHECK_WITH_INFO(!ec, "Failed to create test directory: %s", ec.message().c_str());
        mDirectory = dirPath.string();
    }

    void TearDown() override
    {
        std::error_code ec;
        fs::remove_all(mDirectory, ec);
        if (ec)
            std::cerr << "Warning: Failed to clean up test directory: " << ec.message() << std::endl;
    }

    [[nodiscard]] std::shared_ptr<BaseLoopbackAgent> makeLoopbackAgent(BaseAgentConfig const& config)
    {
        return tensorrt_llm::executor::kv_cache::makeLoopbackAgent("nixl", &config);
    }

    [[nodiscard]] std::string getDirectory() const
    {
        return mDirectory;
    }

private:
    std::string mDirectory;
};

TEST_P(LoopbackAgentTest, FileToGpu)
{
    std::string const agentName{"loopbackAgent"};
    BaseAgentConfig config{agentName, true, GetParam()};
    auto loopbackAgent = makeLoopbackAgent(config);

    TLLM_CHECK(loopbackAgent);

    std::vector<char> memory(100, 1);
    char* cuda_mem;
    TLLM_CUDA_CHECK(cudaMalloc(&cuda_mem, 100));
    TLLM_CUDA_CHECK(cudaMemcpy(cuda_mem, memory.data(), 100, cudaMemcpyHostToDevice));
    std::string filename = getDirectory() + std::string("/file2gpu.bin");

    int fd = ::open(filename.c_str(), O_CREAT | O_WRONLY, 0664);
    TLLM_CHECK_WITH_INFO(fd >= 0, "Failed to open '%s' for writing", filename.c_str());

    std::vector<char> fileData(100, 10);
    ssize_t bytesWritten = ::write(fd, fileData.data(), fileData.size());
    TLLM_CHECK_WITH_INFO(bytesWritten == static_cast<ssize_t>(fileData.size()), "Failed to write to file");
    ::close(fd);

    {
        MemoryDesc mem_desc(cuda_mem, 100, 0);
        MemoryDescs memDescs{MemoryType::kVRAM, {mem_desc}};

        std::vector<FileDesc> fileDescVec;
        fileDescVec.emplace_back(filename, O_RDONLY, 0664, 100);
        FileDescs fileDescs{std::move(fileDescVec)};

        loopbackAgent->executeLoopbackRequest(memDescs, fileDescs, false);
    }

    TLLM_CUDA_CHECK(cudaMemcpy(memory.data(), cuda_mem, 100, cudaMemcpyDeviceToHost));

    TLLM_CHECK(memory == fileData);
    TLLM_CUDA_CHECK(cudaFree(cuda_mem));
}

TEST_P(LoopbackAgentTest, GpuToFile)
{
    std::string const agentName{"loopbackAgent"};
    BaseAgentConfig config{agentName, true, GetParam()};
    auto loopbackAgent = makeLoopbackAgent(config);

    TLLM_CHECK(loopbackAgent);

    std::vector<char> memory(100, 1);
    char* cuda_mem;
    TLLM_CUDA_CHECK(cudaMalloc(&cuda_mem, 100));
    TLLM_CUDA_CHECK(cudaMemcpy(cuda_mem, memory.data(), 100, cudaMemcpyHostToDevice));
    std::string filename = getDirectory() + std::string("/gpu2file.bin");

    {
        MemoryDesc mem_desc(cuda_mem, 100, 0);
        MemoryDescs memDescs{MemoryType::kVRAM, {mem_desc}};

        std::vector<FileDesc> fileDescVec;
        fileDescVec.emplace_back(filename, O_CREAT | O_WRONLY, 0664, 100);
        FileDescs fileDescs{std::move(fileDescVec)};

        loopbackAgent->executeLoopbackRequest(memDescs, fileDescs, true);
    }

    int fd = ::open(filename.c_str(), O_RDONLY, 0664);
    TLLM_CHECK_WITH_INFO(fd >= 0, "Failed to open '%s' for reading", filename.c_str());

    std::vector<char> fileData(100);
    ssize_t bytesRead = ::read(fd, fileData.data(), fileData.size());
    TLLM_CHECK_WITH_INFO(bytesRead == static_cast<ssize_t>(fileData.size()), "Failed to read from file");
    ::close(fd);

    TLLM_CHECK(fileData == memory);
    TLLM_CUDA_CHECK(cudaFree(cuda_mem));
}

INSTANTIATE_TEST_SUITE_P(, LoopbackAgentTest, ::testing::Values(true, false));

// ============================================================================
// FabricMemInfo Serialization Tests
// ============================================================================

TEST(FabricMemInfoTest, SerializeDeserializeEmpty)
{
    FabricMemInfo info;
    info.supported = false;

    std::string serialized = info.serialize();
    auto deserialized = FabricMemInfo::deserialize(serialized);

    ASSERT_TRUE(deserialized.has_value());
    EXPECT_FALSE(deserialized->supported);
    EXPECT_TRUE(deserialized->pools.empty());
}

TEST(FabricMemInfoTest, SerializeDeserializeSinglePool)
{
    FabricMemInfo info;
    info.supported = true;

    FabricMemPool pool;
    pool.deviceId = 0;
    pool.poolBaseAddr = 0x7f0000000000ULL;
    pool.poolTotalSize = 1024 * 1024 * 1024; // 1GB

    FabricMemChunk chunk1;
    chunk1.virtAddrOffset = 0;
    chunk1.size = 512 * 1024 * 1024;
    std::memset(chunk1.fabricHandle, 0xAB, sizeof(chunk1.fabricHandle));

    FabricMemChunk chunk2;
    chunk2.virtAddrOffset = 512 * 1024 * 1024;
    chunk2.size = 512 * 1024 * 1024;
    std::memset(chunk2.fabricHandle, 0xCD, sizeof(chunk2.fabricHandle));

    pool.chunks.push_back(chunk1);
    pool.chunks.push_back(chunk2);
    info.pools.push_back(pool);

    std::string serialized = info.serialize();
    auto deserialized = FabricMemInfo::deserialize(serialized);

    ASSERT_TRUE(deserialized.has_value());
    EXPECT_TRUE(deserialized->supported);
    ASSERT_EQ(deserialized->pools.size(), 1);
    EXPECT_EQ(deserialized->pools[0].deviceId, 0);
    EXPECT_EQ(deserialized->pools[0].poolBaseAddr, 0x7f0000000000ULL);
    EXPECT_EQ(deserialized->pools[0].poolTotalSize, 1024 * 1024 * 1024);
    ASSERT_EQ(deserialized->pools[0].chunks.size(), 2);
    EXPECT_EQ(deserialized->pools[0].chunks[0].virtAddrOffset, 0);
    EXPECT_EQ(deserialized->pools[0].chunks[0].size, 512 * 1024 * 1024);
    EXPECT_EQ(deserialized->pools[0].chunks[1].virtAddrOffset, 512 * 1024 * 1024);
    EXPECT_EQ(deserialized->pools[0].chunks[1].size, 512 * 1024 * 1024);
}

TEST(FabricMemInfoTest, SerializeDeserializeMultiplePools)
{
    FabricMemInfo info;
    info.supported = true;

    // Pool 1
    FabricMemPool pool1;
    pool1.deviceId = 0;
    pool1.poolBaseAddr = 0x7f0000000000ULL;
    pool1.poolTotalSize = 1024 * 1024 * 1024;
    FabricMemChunk chunk1;
    chunk1.virtAddrOffset = 0;
    chunk1.size = 1024 * 1024 * 1024;
    std::memset(chunk1.fabricHandle, 0x11, sizeof(chunk1.fabricHandle));
    pool1.chunks.push_back(chunk1);

    // Pool 2
    FabricMemPool pool2;
    pool2.deviceId = 1;
    pool2.poolBaseAddr = 0x8f0000000000ULL;
    pool2.poolTotalSize = 2 * 1024 * 1024 * 1024ULL;
    FabricMemChunk chunk2;
    chunk2.virtAddrOffset = 0;
    chunk2.size = 2 * 1024 * 1024 * 1024ULL;
    std::memset(chunk2.fabricHandle, 0x22, sizeof(chunk2.fabricHandle));
    pool2.chunks.push_back(chunk2);

    info.pools.push_back(pool1);
    info.pools.push_back(pool2);

    std::string serialized = info.serialize();
    auto deserialized = FabricMemInfo::deserialize(serialized);

    ASSERT_TRUE(deserialized.has_value());
    EXPECT_TRUE(deserialized->supported);
    ASSERT_EQ(deserialized->pools.size(), 2);

    EXPECT_EQ(deserialized->pools[0].deviceId, 0);
    EXPECT_EQ(deserialized->pools[1].deviceId, 1);
}

TEST(FabricMemInfoTest, DeserializeInvalidData)
{
    // Empty data
    auto result1 = FabricMemInfo::deserialize("");
    EXPECT_FALSE(result1.has_value());

    // Wrong magic
    std::string wrongMagic(20, '\0');
    auto result2 = FabricMemInfo::deserialize(wrongMagic);
    EXPECT_FALSE(result2.has_value());

    // Too short
    std::string tooShort(4, '\0');
    auto result3 = FabricMemInfo::deserialize(tooShort);
    EXPECT_FALSE(result3.has_value());
}

TEST(FabricMemChunkTest, SerializedSize)
{
    // FabricMemChunk should be 64 bytes for handle + 8 bytes for offset + 8 bytes for size = 80 bytes
    EXPECT_EQ(FabricMemChunk::serializedSize(), 80);
}

// ============================================================================
// POSIX FD Serialization Tests (Version 4)
// ============================================================================

TEST(FabricMemInfoTest, SerializeDeserializePosixFd)
{
    FabricMemInfo info;
    info.supported = true;
    info.handleType = VmmHandleType::kPosixFd;
    info.udsPath = "/tmp/test_posix_fd_12345.sock";

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
    std::memset(chunk.fabricHandle, 0, sizeof(chunk.fabricHandle)); // Zeroed for POSIX FD
    pool.chunks.push_back(chunk);
    info.pools.push_back(pool);

    std::string serialized = info.serialize();
    auto deserialized = FabricMemInfo::deserialize(serialized);

    ASSERT_TRUE(deserialized.has_value());
    EXPECT_TRUE(deserialized->supported);
    EXPECT_EQ(deserialized->handleType, VmmHandleType::kPosixFd);
    EXPECT_EQ(deserialized->udsPath, "/tmp/test_posix_fd_12345.sock");
    ASSERT_EQ(deserialized->pools.size(), 1);
    EXPECT_EQ(deserialized->pools[0].deviceId, 0);
    ASSERT_EQ(deserialized->pools[0].chunks.size(), 1);
    EXPECT_EQ(deserialized->pools[0].chunks[0].size, 4 * 1024 * 1024);
}

TEST(FabricMemInfoTest, SerializeDeserializeFabricHandleType)
{
    FabricMemInfo info;
    info.supported = true;
    info.handleType = VmmHandleType::kFabric;
    // udsPath empty for fabric mode

    FabricMemPool pool;
    pool.deviceId = 0;
    pool.poolBaseAddr = 0x7f0000000000ULL;
    pool.poolTotalSize = 1024 * 1024 * 1024;
    pool.registeredAddr = 0x7f0000000000ULL;
    pool.registeredSize = 512 * 1024 * 1024;
    pool.mappedOffset = 0;
    pool.mappedSize = 512 * 1024 * 1024;

    FabricMemChunk chunk;
    chunk.virtAddrOffset = 0;
    chunk.size = 512 * 1024 * 1024;
    std::memset(chunk.fabricHandle, 0xAA, sizeof(chunk.fabricHandle));
    pool.chunks.push_back(chunk);
    info.pools.push_back(pool);

    std::string serialized = info.serialize();
    auto deserialized = FabricMemInfo::deserialize(serialized);

    ASSERT_TRUE(deserialized.has_value());
    EXPECT_TRUE(deserialized->supported);
    EXPECT_EQ(deserialized->handleType, VmmHandleType::kFabric);
    EXPECT_TRUE(deserialized->udsPath.empty());
}

#endif // TEST_NIXL_BACKEND

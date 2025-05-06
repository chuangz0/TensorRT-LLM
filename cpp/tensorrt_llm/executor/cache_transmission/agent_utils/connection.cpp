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

#include "connection.h"
#include "tensorrt_llm/common/envUtils.h"
#include <unistd.h>

namespace tensorrt_llm::executor::kv_cache
{

std::string genUniqueAgentName()
{
    static std::atomic<uint64_t> counter{0};

    // hostname+pid+counter++
    char hostname[1024];
    gethostname(hostname, sizeof(hostname));
    auto pid = static_cast<uint64_t>(::getpid());
    return std::string(hostname) + "_" + std::to_string(pid) + "_" + std::to_string(counter++);
}

AgentConnection::AgentConnection(AgentDesc mAgentDesc, AgentDesc mRemoteAgentDesc,
    batch_manager::kv_cache_manager::CacheTransBufferManager* mCacheTransBufferManager)
    : mAgentDesc(mAgentDesc)
    , mRemoteAgentDesc(mRemoteAgentDesc)
    , mCacheTransBufferManager(mCacheTransBufferManager)
    , mSenderState()
{
}

std::optional<size_t> AgentConnection::getCacheBufferId() const
{
    return mCacheBufferId;
}

void MemoryDesc::serialize(MemoryDesc const& memoryDesc, std::ostream& os)
{
    namespace su = executor::serialize_utils;
    su::serialize(memoryDesc.mAddr, os);
    su::serialize(memoryDesc.mLen, os);
    su::serialize(memoryDesc.mDeviceId, os);
    // su::serialize(memoryDesc.mType, os);
}

MemoryDesc MemoryDesc::deserialize(std::istream& is)
{
    namespace su = executor::serialize_utils;
    auto addr = su::deserialize<decltype(mAddr)>(is);
    auto len = su::deserialize<decltype(mLen)>(is);
    auto deviceId = su::deserialize<decltype(mDeviceId)>(is);
    return MemoryDesc{addr, len, deviceId};
}

size_t MemoryDesc::serializedSize(MemoryDesc const& memoryDesc)
{
    namespace su = executor::serialize_utils;
    return su::serializedSize(memoryDesc.mAddr) + su::serializedSize(memoryDesc.mLen)
        + su::serializedSize(memoryDesc.mDeviceId);
}

struct RequestAndBufferInfo
{
    AgentDesc mAgentDesc;
    std::string mAddress;
    batch_manager::RequestInfo mRequestInfo;
    MemoryDesc mBufferDesc;
    int mValidConnectionIdx;

    static void serialize(RequestAndBufferInfo const& requestAndBufferInfo, std::ostream& os)
    {
        namespace su = executor::serialize_utils;
        su::serialize(requestAndBufferInfo.mAgentDesc, os);
        su::serialize(requestAndBufferInfo.mAddress, os);
        batch_manager::RequestInfo::serialize(requestAndBufferInfo.mRequestInfo, os);
        MemoryDesc::serialize(requestAndBufferInfo.mBufferDesc, os);
        su::serialize(requestAndBufferInfo.mValidConnectionIdx, os);
    }

    static RequestAndBufferInfo deserialize(std::istream& is)
    {
        namespace su = executor::serialize_utils;
        auto agentDesc = su::deserialize<decltype(mAgentDesc)>(is);
        auto address = su::deserialize<decltype(mAddress)>(is);
        auto requestInfo = batch_manager::RequestInfo::deserialize(is);
        auto bufferDesc = MemoryDesc::deserialize(is);
        auto validConnectionIdx = su::deserialize<decltype(mValidConnectionIdx)>(is);
        return RequestAndBufferInfo{agentDesc, address, requestInfo, bufferDesc, validConnectionIdx};
    }

    static size_t serializedSize(RequestAndBufferInfo const& requestAndBufferInfo)
    {
        namespace su = executor::serialize_utils;
        return su::serializedSize(requestAndBufferInfo.mAgentDesc) + su::serializedSize(requestAndBufferInfo.mAddress)
            + batch_manager::RequestInfo::serializedSize(requestAndBufferInfo.mRequestInfo)
            + MemoryDesc::serializedSize(requestAndBufferInfo.mBufferDesc)
            + su::serializedSize(requestAndBufferInfo.mValidConnectionIdx);
    }
};

struct NotificationSyncInfo
{

    AgentDesc mAgentDesc;
    DataContext mContext;

    static void serialize(NotificationSyncInfo const& notificationSyncInfo, std::ostream& os)
    {
        namespace su = executor::serialize_utils;
        su::serialize(notificationSyncInfo.mAgentDesc, os);
        su::serialize(notificationSyncInfo.mContext.getTag(), os);
    }

    static NotificationSyncInfo deserialize(std::istream& is)
    {
        namespace su = executor::serialize_utils;
        auto agentDesc = su::deserialize<decltype(mAgentDesc)>(is);
        auto contextTag = su::deserialize<decltype(mContext.getTag())>(is);
        DataContext context{contextTag};
        return NotificationSyncInfo{agentDesc, context};
    }

    static size_t serializedSize(NotificationSyncInfo const& notificationSyncInfo)
    {
        namespace su = executor::serialize_utils;
        return su::serializedSize(notificationSyncInfo.mAgentDesc)
            + su::serializedSize(notificationSyncInfo.mContext.getTag());
    }
};

struct NotificationInfo
{

    std::variant<RequestAndBufferInfo, NotificationSyncInfo> mInfo;

    static void serialize(NotificationInfo const& notificationInfo, std::ostream& os)
    {
        namespace su = executor::serialize_utils;
        su::serialize(notificationInfo.mInfo.index(), os);
        if (std::holds_alternative<RequestAndBufferInfo>(notificationInfo.mInfo))
        {
            RequestAndBufferInfo::serialize(std::get<RequestAndBufferInfo>(notificationInfo.mInfo), os);
        }
        else if (std::holds_alternative<NotificationSyncInfo>(notificationInfo.mInfo))
        {
            NotificationSyncInfo::serialize(std::get<NotificationSyncInfo>(notificationInfo.mInfo), os);
        }
        else
        {
            TLLM_THROW("Unknown variant type");
        }
    }

    static NotificationInfo deserialize(std::istream& is)
    {
        namespace su = executor::serialize_utils;
        auto variantIdx = su::deserialize<std::size_t>(is);
        constexpr std::size_t requestAndBufferInfoIdx{0};
        constexpr std::size_t notificationSyncInfoIdx{1};
        if (variantIdx == requestAndBufferInfoIdx)
        {
            return NotificationInfo{RequestAndBufferInfo::deserialize(is)};
        }
        else if (variantIdx == notificationSyncInfoIdx)
        {
            return NotificationInfo{NotificationSyncInfo::deserialize(is)};
        }
        else
        {
            TLLM_THROW("Unknown variant type");
        }
    }

    static size_t serializedSize(NotificationInfo const& notificationInfo)
    {
        namespace su = executor::serialize_utils;
        size_t totalSize = 0;
        totalSize += su::serializedSize(notificationInfo.mInfo.index());
        if (std::holds_alternative<RequestAndBufferInfo>(notificationInfo.mInfo))
        {
            totalSize += RequestAndBufferInfo::serializedSize(std::get<RequestAndBufferInfo>(notificationInfo.mInfo));
        }
        else if (std::holds_alternative<NotificationSyncInfo>(notificationInfo.mInfo))
        {
            totalSize += NotificationSyncInfo::serializedSize(std::get<NotificationSyncInfo>(notificationInfo.mInfo));
        }
        else
        {
            TLLM_THROW("Unknown variant type");
        }
        return totalSize;
    }
};

void AgentConnection::send(DataContext const& ctx, void const* data, size_t size) const
{

    MemoryDesc srcDesc{
        reinterpret_cast<uintptr_t>(data), size, static_cast<uint32_t>(mAgentConnectionManager->getDeviceId())};
    MemoryDescs srcDescs{MemoryType::kVRAM, {srcDesc}};
    auto dstBaseDesc = mSenderState.mReceiverBufferDesc;
    MemoryDesc dstDesc{dstBaseDesc.getAddr() + (mSenderState.valideSegmentIdx * size), size, dstBaseDesc.getDeviceId()};
    MemoryDescs dstDescs{MemoryType::kVRAM, {dstDesc}};
    std::stringstream ss;
    NotificationSyncInfo syncInfo{mRemoteAgentDesc, ctx};
    NotificationInfo notificationInfo{syncInfo};
    NotificationInfo::serialize(notificationInfo, ss);
    TransferRequest request{TransferOp::kWRITE, srcDescs, dstDescs, mRemoteAgentDesc, ss.str()};
    auto status = mAgentConnectionManager->getAgent()->submitTransferRequests(request);
    status->wait();
}

void AgentConnection::recv(DataContext const& ctx, void* data, size_t size) const
{

    NotificationSyncInfo syncInfo{mAgentDesc, ctx};
    mAgentConnectionManager->waitForSyncInfo(mRemoteAgentDesc, syncInfo);
}

void AgentConnection::sendRequestAndBufferInfo(
    batch_manager::RequestInfo& requestInfo, std::optional<size_t> cacheBufferId, int validConnectionIdx)
{
    TLLM_CHECK(!common::getEnvTryZCopyForKVCacheTransfer());

    TLLM_CHECK(cacheBufferId.has_value());
    auto preAllocateBuffer = mCacheTransBufferManager->getRecvBuffer(cacheBufferId.value());
    // memory Desp , validSegmentIdx send
    mCacheBufferId = cacheBufferId;
    // TODO: deviceID;
    int deviceId = -1;
    TLLM_CUDA_CHECK(cudaGetDevice(&deviceId));
    TLLM_CHECK(deviceId != -1);
    MemoryDesc bufferDesc(
        reinterpret_cast<uintptr_t>(preAllocateBuffer->data()), preAllocateBuffer->getSize(), deviceId);
    std::string address;
    RequestAndBufferInfo requestAndBufferInfo{mAgentDesc, address, requestInfo, bufferDesc, validConnectionIdx};
    std::stringstream ss;
    RequestAndBufferInfo::serialize(requestAndBufferInfo, ss);
    mAgentConnectionManager->getAgent()->notifySyncInfo(mRemoteAgentDesc, ss.str());
}

void AgentConnection::setSenderState(MemoryDesc mReceiverBufferDesc, int valideSegmentIdx)
{
    mSenderState = SenderState{mReceiverBufferDesc, valideSegmentIdx};
}

AgentConnectionManager::AgentConnectionManager(
    batch_manager::kv_cache_manager::CacheTransBufferManager* cacheTransBufferManager, BaseTransferAgent* agent)
{
    // TODO: register memory
    TLLM_CUDA_CHECK(cudaGetDevice(&mDeviceId));
    TLLM_CHECK(mDeviceId != -1);

    // TODO:
    // Create Agent
    m_Agent = agent;
    mCacheTransBufferManager = cacheTransBufferManager;
    auto recvBufferCount = mCacheTransBufferManager->getRecvBufferCount();
    auto sendBufferCount = mCacheTransBufferManager->getSendBufferCount();
    std::vector<MemoryDesc> MemDescs;
    for (size_t i = 0; i < recvBufferCount; i++)
    {
        auto recvBuffer = mCacheTransBufferManager->getRecvBuffer(i);
        MemDescs.emplace_back(recvBuffer->data(), recvBuffer->getSizeInBytes(), mDeviceId);
    }
    for (size_t i = 0; i < sendBufferCount; i++)
    {
        auto sendBuffer = mCacheTransBufferManager->getSendBuffer(i);
        MemDescs.emplace_back(sendBuffer->data(), sendBuffer->getSizeInBytes(), mDeviceId);
    }
    MemoryDescs descs{MemoryType::kVRAM, MemDescs};
    m_Agent->registerMemory(descs);

    AgentState localAgentState{m_Agent->getAgentDesc(), m_Agent->getMetaData().getBackendMetaData()};
    std::vector<AgentState> agentStates(mpi::MpiComm::session().getSize());
    if (mpi::MpiComm::session().getSize() > 1)
    {

        mpi::MpiComm::session().barrier();
        namespace su = executor::serialize_utils;

        std::ostringstream oStream;
        su::serialize(localAgentState, oStream);
        auto str = oStream.str();
        std::vector<char> buffer(str.begin(), str.end());
        std::vector<SizeType32> sizeofBuffer(mpi::MpiComm::session().getSize());
        SizeType32 bufferSize = buffer.size();
        mpi::MpiComm::session().allgather(&bufferSize, sizeofBuffer.data(), 1, mpi::MpiType::kINT32);
        SizeType32 recvBufferSize = std::accumulate(sizeofBuffer.begin(), sizeofBuffer.end(), 0);
        std::vector<char> recvBuffer(recvBufferSize);
        std::vector<int> displs(mpi::MpiComm::session().getSize());
        for (int r = 0; r < mpi::MpiComm::session().getSize(); r++)
        {
            displs[r] = (r == 0) ? 0 : (displs[r - 1] + sizeofBuffer[r - 1]);
        }
        mpi::MpiComm::session().allgatherv(buffer.data(), bufferSize, mpi::MpiType::kCHAR, recvBuffer.data(),
            sizeofBuffer, displs, mpi::MpiType::kCHAR);

        // deserialize
        for (int i = 0; i < mpi::MpiComm::session().getSize(); i++)
        {
            std::vector<char> serBuffer(
                recvBuffer.begin() + displs[i], recvBuffer.begin() + (displs[i] + sizeofBuffer[i]));
            su::VectorWrapBuf<char> strbuf(serBuffer);
            std::istream is(&strbuf);
            agentStates[i] = su::deserialize<executor::kv_cache::AgentState>(is);
            TLLM_LOG_DEBUG(
                mpi::MpiComm::world().getRank(), " recv  agentStates[%d]: %s", i, agentStates[i].toString().c_str());
        }
    }
    else
    {
        agentStates[0] = localAgentState;
    }
    mCommState = CommState(agentStates, mpi::MpiComm::session().getRank());
    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
        " ***** AgentConnectionManager::AgentConnectionManager    mCommState: %s", mCommState.toString().c_str());
}

AgentConnection const* AgentConnectionManager::recvConnectionAndRequestInfo(batch_manager::RequestInfo& requestInfo)
{
    // recv remoteAgentDesc, and bufferDesc , and validSegmentIdx ,

    while (true)
    {

        updateUnhandledNotifications();
        std::scoped_lock lock(mNotificationMutex);
        auto it = mUnhandledNotifications.begin();
        while (it != mUnhandledNotifications.end())
        {
            auto& [agent, notifs] = *it;
            auto it2 = notifs.begin();
            while (it2 != notifs.end())
            {
                std::stringstream ss(*it2);
                NotificationInfo notificationInfo = NotificationInfo::deserialize(ss);
                bool erase = false;
                if (std::holds_alternative<RequestAndBufferInfo>(notificationInfo.mInfo))
                {
                    auto requestAndBufferInfo = std::get<RequestAndBufferInfo>(notificationInfo.mInfo);

                    erase = true;
                    // TODO: recv buffer and requestInfo
                    requestInfo = requestAndBufferInfo.mRequestInfo;
                    auto address = requestAndBufferInfo.mAddress;
                    auto bufferDesc = requestAndBufferInfo.mBufferDesc;
                    auto validConnectionIdx = requestAndBufferInfo.mValidConnectionIdx;
                    auto remoteAgentDesc = requestAndBufferInfo.mAgentDesc;
                    auto connection = connect(remoteAgentDesc, address);
                    connection->setSenderState(bufferDesc, validConnectionIdx);
                    it2 = notifs.erase(it2);
                    // TODO: if notifs.empty(), erase it
                    if (notifs.empty())
                    {
                        it = mUnhandledNotifications.erase(it);
                    }
                    return connection;
                }

                if (!erase)
                {
                    it2++;
                }
            }
            if (notifs.empty())
            {
                it = mUnhandledNotifications.erase(it);
            }
            else
            {
                it++;
            }
        }
    }
    return nullptr;
}

void AgentConnectionManager::updateUnhandledNotifications()
{

    auto notif_map = m_Agent->getSyncInfo();
    std::lock_guard<std::mutex> lock(mNotificationMutex);

    // Merge new notifications with existing ones
    for (auto const& [agent, notifs] : notif_map)
    {
        auto& existing_notifs = mUnhandledNotifications[agent];
        existing_notifs.insert(existing_notifs.end(), notifs);
    }
}

[[nodiscard]] std::vector<Connection const*> AgentConnectionManager::getConnections(CommState const& state)
{
    //  agentDesc +ip
    // get metaData from ip;
    TLLM_CHECK(state.isAgentState());
    // TODO:  AgentCommState
    auto ret = std::vector<Connection const*>();
    for (auto&& agentState : state.getAgentState())
    {
        std::string agentName = agentState.mAgentName;
        std::string connectionInfo = agentState.mConnectionInfo;
        ret.emplace_back(connect(agentName, connectionInfo));
    }
    return ret;
}

BaseTransferAgent* AgentConnectionManager::getAgent() const
{
    return m_Agent;
}

batch_manager::kv_cache_manager::CacheTransBufferManager* AgentConnectionManager::getCacheTransBufferManager()
{
    return mCacheTransBufferManager;
}

AgentConnection* AgentConnectionManager::connect(std::string const& remoteAgentName, std::string const& connecitonInfo)
{

    std::scoped_lock lock(mConnectionsMutex);
    auto it = mConnections.find(remoteAgentName);
    if (it != mConnections.end())
    {
        return it->second.get();
    }

    m_Agent->connectRemoteAgent(remoteAgentName, connecitonInfo);
    auto connection
        = std::make_shared<AgentConnection>(m_Agent->getAgentDesc(), remoteAgentName, mCacheTransBufferManager);
    mConnections[remoteAgentName] = connection;
    return connection.get();
}

CommState const& AgentConnectionManager::getCommState() const
{

    return mCommState;
}

AgentConnection* AgentConnectionManager::recvConnect(DataContext const& ctx, void* data, size_t size)
{

    TLLM_THROW("Not implemented");
    return nullptr;
}

int AgentConnectionManager::getDeviceId() const
{
    return mDeviceId;
}

void AgentConnectionManager::waitForSyncInfo(AgentDesc const& remoteAgentDesc, NotificationSyncInfo const& syncInfo)
{
    while (true)
    {

        updateUnhandledNotifications();
        std::scoped_lock lock(mNotificationMutex);
        auto it = mUnhandledNotifications.begin();
        while (it != mUnhandledNotifications.end())
        {
            auto& [agent, notifs] = *it;
            if (agent != remoteAgentDesc)
            {
                it++;
                continue;
            }
            auto it2 = notifs.begin();
            while (it2 != notifs.end())
            {
                std::stringstream ss(*it2);
                NotificationInfo notificationInfo = NotificationInfo::deserialize(ss);
                bool erase = false;
                if (std::holds_alternative<NotificationSyncInfo>(notificationInfo.mInfo))
                {
                    auto notificationSyncInfo = std::get<NotificationSyncInfo>(notificationInfo.mInfo);
                    if (notificationSyncInfo.mContext.getTag() == syncInfo.mContext.getTag()
                        && notificationSyncInfo.mAgentDesc == syncInfo.mAgentDesc)
                    {
                        erase = true;
                        it2 = notifs.erase(it2);
                        if (notifs.empty())
                        {
                            it = mUnhandledNotifications.erase(it);
                        }
                        return;
                    }
                }
                if (!erase)
                {
                    it2++;
                }
            }
            if (notifs.empty())
            {
                it = mUnhandledNotifications.erase(it);
            }
            else
            {
                it++;
            }
        }
    }
}

} // namespace tensorrt_llm::executor::kv_cache

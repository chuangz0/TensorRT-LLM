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

namespace tensorrt_llm::executor::kv_cache
{

AgentConnection::AgentConnection(AgentDesc mAgentDesc, AgentDesc mRemoteAgentDesc,
    batch_manager::kv_cache_manager::CacheTransBufferManager* mCacheTransBufferManager)
    : mAgentDesc(mAgentDesc)
    , mRemoteAgentDesc(mRemoteAgentDesc)
    , mCacheTransBufferManager(mCacheTransBufferManager)
    , mSenderState()
{
}

void AgentConnection::send(DataContext const& ctx, void const* data, size_t size) const {}

void AgentConnection::recv(DataContext const& ctx, void* data, size_t size) const {}

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
    ContextPhaseParams::RequestIdType mRequestId;

    static void serialize(NotificationSyncInfo const& notificationSyncInfo, std::ostream& os)
    {
        namespace su = executor::serialize_utils;
        su::serialize(notificationSyncInfo.mAgentDesc, os);
        su::serialize(notificationSyncInfo.mRequestId, os);
    }

    static NotificationSyncInfo deserialize(std::istream& is)
    {
        namespace su = executor::serialize_utils;
        auto agentDesc = su::deserialize<decltype(mAgentDesc)>(is);
        auto requestId = su::deserialize<decltype(mRequestId)>(is);
        return NotificationSyncInfo{agentDesc, requestId};
    }

    static size_t serializedSize(NotificationSyncInfo const& notificationSyncInfo)
    {
        namespace su = executor::serialize_utils;
        return su::serializedSize(notificationSyncInfo.mAgentDesc)
            + su::serializedSize(notificationSyncInfo.mRequestId);
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

void AgentConnection::sendRequestAndBufferInfo(
    batch_manager::RequestInfo& requestInfo, std::optional<size_t> cacheBufferId, int validConnectionIdx)
{
    TLLM_CHECK(!common::getEnvTryZCopyForKVCacheTransfer());

    TLLM_CHECK(cacheBufferId.has_value());
    auto preAllocateBuffer = mCacheTransBufferManager->getSendBuffer(cacheBufferId.value());
    // memory Desp , validSegmentIdx send
    mCacheBufferId = cacheBufferId;
    // TODO: deviceID;
    int deviceId = 0;
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
    // mSenderState = SenderState{mReceiverBufferDesc, valideSegmentIdx};
    mSenderState = SenderState{mReceiverBufferDesc, valideSegmentIdx};
}

AgentConnectionManager::AgentConnectionManager(
    batch_manager::kv_cache_manager::CacheTransBufferManager const* cacheTransBufferManager)
{
}

AgentConnection const* AgentConnectionManager::recvConnectionAndRequestInfo(batch_manager::RequestInfo& requestInfo)
{
    // recv remoteAgentDesc, and bufferDesc , and validSegmentIdx ,

    while (true)
    {

        updateUnhandledNotifications();

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
                    return connection;
                }
                if (erase)
                {
                    it2 = notifs.erase(it2);
                }
                else
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
    std::lock_guard<std::mutex> lock(mMutex);

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

    //
    auto ret = std::vector<Connection const*>();
    for (auto&& scoketState : state.getSocketState())
    {
        std::string agentName;
        std::string address = scoketState.mIp + ":" + std::to_string(scoketState.mPort);
        bool found = false;
        {
            std::scoped_lock lock(mConnectionsMutex);
            found = mConnections.find(agentName) != mConnections.end();
        }
        if (!found)
        {
            mConnections[agentName] = connect(agentName, address);
        }
        ret.emplace_back(mConnections[agentName]);
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

AgentConnection* AgentConnectionManager::connect(std::string const& agentName, std::string const& address)
{

    return nullptr;
}

CommState const& AgentConnectionManager::getCommState() const
{
    return mCommState;
}

AgentConnection* AgentConnectionManager::recvConnect(DataContext const& ctx, void* data, size_t size)
{
    return nullptr;
}
} // namespace tensorrt_llm::executor::kv_cache

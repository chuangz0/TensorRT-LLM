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

#include "tensorrt_llm/batch_manager/cacheTransBuffer.h"
#include "tensorrt_llm/batch_manager/dataTransceiver.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/executor/cacheCommunicator.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include <map>

namespace tensorrt_llm::executor::kv_cache
{
class AgentConnectionManager;
class NotificationSyncInfo;

class AgentConnection : public Connection
{
public:
    AgentConnection(AgentDesc mAgentDesc, AgentDesc mRemoteAgentDesc,
        batch_manager::kv_cache_manager::CacheTransBufferManager* mCacheTransBufferManager);
    void send(DataContext const& ctx, void const* data, size_t size) const override;
    void recv(DataContext const& ctx, void* data, size_t size) const override;
    void sendRequestAndBufferInfo(
        batch_manager::RequestInfo& requestInfo, std::optional<size_t> cacheBufferId, int validConnectionIdx);
    void setSenderState(MemoryDesc mReceiverBufferDesc, int valideSegmentIdx);
    [[nodiscard]] std::optional<size_t> getCacheBufferId() const;

private:
    AgentDesc mAgentDesc;
    AgentDesc mRemoteAgentDesc;

    struct SenderState
    {
        MemoryDesc mReceiverBufferDesc;
        int valideSegmentIdx;
        SenderState() = default;
    };

    batch_manager::kv_cache_manager::CacheTransBufferManager* mCacheTransBufferManager;
    std::optional<size_t> mCacheBufferId;
    AgentConnectionManager* mAgentConnectionManager;
    SenderState mSenderState;
};

class AgentConnectionManager : public ConnectionManager
{
public:
    AgentConnectionManager(
        batch_manager::kv_cache_manager::CacheTransBufferManager* cacheTransBufferManager, BaseTransferAgent* agent);
    AgentConnection* recvConnect(DataContext const& ctx, void* data, size_t size) override;
    [[nodiscard]] std::vector<Connection const*> getConnections(CommState const& state) override;
    [[nodiscard]] CommState const& getCommState() const override;
    AgentConnection const* recvConnectionAndRequestInfo(batch_manager::RequestInfo& requestInfo);
    [[nodiscard]] batch_manager::kv_cache_manager::CacheTransBufferManager* getCacheTransBufferManager();
    void updateUnhandledNotifications();
    [[nodiscard]] BaseTransferAgent* getAgent() const;
    AgentConnection* connect(std::string const& remoteAgentName, std::string const& address);
    int getDeviceId() const;
    void waitForSyncInfo(AgentDesc const& remoteAgentDesc, NotificationSyncInfo const& syncInfo);

private:
    std::map<AgentDesc, std::shared_ptr<AgentConnection>> mConnections;
    std::mutex mConnectionsMutex;
    CommState mCommState;
    batch_manager::kv_cache_manager::CacheTransBufferManager* mCacheTransBufferManager;
    std::mutex mNotificationMutex;
    std::unordered_map<AgentDesc, std::list<std::string>> mUnhandledNotifications;
    BaseTransferAgent* m_Agent;
    int mDeviceId;
};

} // namespace tensorrt_llm::executor::kv_cache

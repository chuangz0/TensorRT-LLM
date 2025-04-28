/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/executor/serialization.h"
#include <memory>
#include <string>
#include <vector>

namespace tensorrt_llm::executor::kv_cache
{

class CommState;

struct DataContext
{
public:
    explicit DataContext(int tag)
        : mTag{tag}
    {
    }

    [[nodiscard]] int getTag() const noexcept
    {
        return mTag;
    }

private:
    int const mTag;
};

class Connection
{
public:
    virtual ~Connection() = default;

    virtual void send(DataContext const& ctx, void const* data, size_t size) const = 0;

    virtual void recv(DataContext const& ctx, void* data, size_t size) const = 0;

    [[nodiscard]] virtual bool isThreadSafe() const noexcept
    {
        return false;
    }
};

class ConnectionManager
{
public:
    virtual ~ConnectionManager() = default;

    virtual Connection const* recvConnect(DataContext const& ctx, void* data, size_t size) = 0;

    [[nodiscard]] virtual std::vector<Connection const*> getConnections(CommState const& state) = 0;

    [[nodiscard]] virtual CommState const& getCommState() const = 0;
};

// -----

enum class MemoryType : uint8_t
{
    kDRAM,
    kVRAM,
    kBLK,
    kOBJ,
    kFILE
};

class MemoryDesc
{
public:
    MemoryDesc(uintptr_t addr, size_t len, uint32_t deviceId)
        : mAddr{addr}
        , mLen{len}
        , mDeviceId{deviceId}
    {
    }

    MemoryDesc() = default;

    [[nodiscard]] uintptr_t getAddr() const noexcept
    {
        return mAddr;
    }

    [[nodiscard]] size_t getLen() const noexcept
    {
        return mLen;
    }

    [[nodiscard]] uint32_t getDeviceId() const noexcept
    {
        return mDeviceId;
    }

    static void serialize(MemoryDesc const& memoryDesc, std::ostream& os);
    [[nodiscard]] static MemoryDesc deserialize(std::istream& is);
    [[nodiscard]] static size_t serializedSize(MemoryDesc const& memoryDesc);

private:
    uintptr_t mAddr;
    size_t mLen;
    uint32_t mDeviceId;
};

class MemoryDescs
{
public:
    MemoryDescs(MemoryType type, std::vector<MemoryDesc> descs)
        : mType{type}
        , mDescs{std::move(descs)}
    {
    }

    [[nodiscard]] MemoryType getType() const noexcept
    {
        return mType;
    }

    std::vector<MemoryDesc> const& getDescs() const noexcept
    {
        return mDescs;
    }

private:
    MemoryType mType;
    std::vector<MemoryDesc> mDescs;
};

using TransferDescs = MemoryDescs;
using RegisterDescs = MemoryDescs;
using SyncMessage = std::string;
using AgentDesc = std::string;

// class AgentDesc
// {
// public:
//     AgentDesc(std::string backendAgentDesc)
//         : mBackendAgentDesc{std::move(backendAgentDesc)}
//     {
//     }

//     virtual ~AgentDesc() = default;

//     [[nodiscard]] std::string const& getBackendAgentDesc() const noexcept
//     {
//         return mBackendAgentDesc;
//     }

// private:
//     std::string mBackendAgentDesc;
// };
class AgentMetaData
{
public:
    AgentMetaData(std::string metaData)
        : mMetaData{std::move(metaData)}
    {
    }

    [[nodiscard]] std::string const& getBackendMetaData() const noexcept
    {
        return mMetaData;
    }

private:
    std::string mMetaData;
};
enum class TransferOp : uint8_t
{
    kREAD,
    kWRITE,
};

class TransferRequest
{
public:
    TransferRequest(TransferOp op, TransferDescs srcDescs, TransferDescs dstDescs, AgentDesc const& remoteAgent)
        : mOp{op}
        , mSrcDescs{std::move(srcDescs)}
        , mDstDescs{std::move(dstDescs)}
        , mRemoteAgent{std::addressof(remoteAgent)}
    {
    }

    TransferOp getOp() const noexcept
    {
        return mOp;
    }

    TransferDescs const& getSrcDescs() const noexcept
    {
        return mSrcDescs;
    }

    TransferDescs const& getDstDescs() const noexcept
    {
        return mDstDescs;
    }

    AgentDesc const& getRemoteAgent() const noexcept
    {
        return *mRemoteAgent;
    }

private:
    TransferOp mOp;
    TransferDescs mSrcDescs;
    TransferDescs mDstDescs;
    AgentDesc const* mRemoteAgent;
};

class TransferStatus
{
public:
    virtual ~TransferStatus() = default;
    [[nodiscard]] virtual bool isCompleted() const = 0;
    virtual void wait() const = 0;
};

class BaseTransferAgent
{
public:
    virtual ~BaseTransferAgent() = default;

    virtual void registerMemory(RegisterDescs const& descs) = 0;

    virtual void deregisterMemory(RegisterDescs const& descs) = 0;

    [[nodiscard]] virtual std::unique_ptr<TransferStatus> submitTransferRequests(TransferRequest const& request) = 0;

    // for MLA , some rank didn't need to be read.
    virtual void notifySyncInfo(AgentDesc const& agent, SyncMessage const& syncMessage) = 0;

    virtual std::vector<std::pair<AgentDesc, SyncMessage>> getSyncInfo();
    virtual AgentMetaData getMetaData() = 0;
    virtual AgentDesc getAgentDesc() = 0;
    virtual void connectRemoteAgent(AgentDesc const& agent, AgentMetaData const& metaData) = 0;
};

} // namespace tensorrt_llm::executor::kv_cache

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

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/transferAgent.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <netinet/in.h>
#include <unistd.h>

namespace tensorrt_llm::executor::kv_cache
{

static std::string getAvailableIP()
{
    struct ifaddrs *ifaddr, *ifa;
    void* addr_ptr;
    std::string ip("UNKNOWN IP");

    // Get the list of network interfaces
    if (getifaddrs(&ifaddr) == -1)
    {
        perror("getifaddrs");
        return ip;
    }

    // Loop through the linked list of interfaces
    for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next)
    {
        // Check if the interface is an IP interface
        if (ifa->ifa_addr == nullptr)
            continue;

        std::string ucxInterface = common::getEnvUCXInterface();
        if (!ucxInterface.empty() && strcmp(ifa->ifa_name, ucxInterface.c_str()) != 0)
        {
            continue;
        }

        // Skip the loopback interface
        if (ucxInterface.empty() && (strncmp(ifa->ifa_name, "docker", 6) == 0 || strcmp(ifa->ifa_name, "lo") == 0))
        {
            continue;
        }

        // Check if the address family is AF_INET (IPv4)
        // TODO: USER CAN SPECIFY THE IP ADDRESS
        if (ifa->ifa_addr->sa_family == AF_INET)
        {
            addr_ptr = &((struct sockaddr_in*) ifa->ifa_addr)->sin_addr;
            char address_buffer[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, addr_ptr, address_buffer, sizeof(address_buffer));

            TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), " ***** UCX    Interface: %s IP Address: %s", ifa->ifa_name,
                address_buffer);
            ip = address_buffer;
            break;
        }
    }
    if (ifa == nullptr)
    {
        TLLM_LOG_ERROR(mpi::MpiComm::world().getRank(),
            "UCX   No valid IP address found please set correct UCX interface with env variable TRTLLM_UCX_INTERFACE");
    }

    freeifaddrs(ifaddr);
    return ip;
}

uint16_t getAvailablePort(std::string const& ip = "0.0.0.0")
{
    struct addrinfo hints
    {
    };

    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;

    struct addrinfo* res;
    int ret = getaddrinfo(ip.c_str(), "0", &hints, &res);
    TLLM_CHECK(ret == 0);

    int sockfd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    TLLM_CHECK(sockfd != -1);

    ret = bind(sockfd, res->ai_addr, res->ai_addrlen);
    TLLM_CHECK(ret == 0);

    struct sockaddr_in addr;
    socklen_t addrlen = sizeof(addr);
    ret = getsockname(sockfd, (struct sockaddr*) &addr, &addrlen);
    TLLM_CHECK(ret == 0);

    uint16_t port = ntohs(addr.sin_port);
    close(sockfd);
    freeaddrinfo(res);

    return port;
}

uint16_t getIncrmentPort(uint16_t basePort)
{
    static uint16_t times = 0;
    return basePort + (times++) * mpi::MpiComm::world().getSize();
}

[[nodiscard]] nixl_mem_t NixlHelper::convert(MemoryType type)
{
    switch (type)
    {
    case MemoryType::kDRAM: return DRAM_SEG;
    case MemoryType::kVRAM: return VRAM_SEG;
    case MemoryType::kBLK: return BLK_SEG;
    case MemoryType::kOBJ: return OBJ_SEG;
    case MemoryType::kFILE: return FILE_SEG;
    default: TLLM_THROW("Unknown MemoryType value");
    }
}

[[nodiscard]] nixlBasicDesc NixlHelper::convert(MemoryDesc const& desc)
{
    return nixlBasicDesc{desc.getAddr(), desc.getLen(), desc.getDeviceId()};
}

[[nodiscard]] nixl_reg_dlist_t NixlHelper::convertRegDlist(RegisterDescs const& descs)
{
    nixl_reg_dlist_t list{convert(descs.getType())};
    for (auto const& desc : descs.getDescs())
    {
        list.addDesc(nixlBlobDesc{desc.getAddr(), desc.getLen(), desc.getDeviceId()});
    }
    return list;
}

[[nodiscard]] nixl_xfer_op_t NixlHelper::convert(TransferOp const& op)
{
    switch (op)
    {
    case TransferOp::kREAD: return NIXL_READ;
    case TransferOp::kWRITE: return NIXL_WRITE;
    default: TLLM_THROW("Unknown TransferOp value");
    }
}

[[nodiscard]] nixl_xfer_dlist_t NixlHelper::convertXferDist(TransferDescs const& descs)
{
    nixl_xfer_dlist_t list{convert(descs.getType())};
    for (auto const& desc : descs.getDescs())
    {
        list.addDesc(nixlBasicDesc{desc.getAddr(), desc.getLen(), desc.getDeviceId()});
    }
    return list;
}

NixlTransferStatus::NixlTransferStatus(nixlAgent* agent, nixlXferReqH* handle)
    : mRawAgent{agent}
    , mHandle{handle}
{
    TLLM_CHECK(mRawAgent);
    TLLM_CHECK(mHandle);
}

void NixlTransferStatus::wait() const
{
    while (!isCompleted())
        ;
}

[[nodiscard]] bool NixlTransferStatus::isCompleted() const
{
    return mRawAgent->getXferStatus(mHandle) == NIXL_SUCCESS;
}

NixlTransferAgent::NixlTransferAgent(BaseAgentConfig const& config)
    : mName{config.mName}
{
    nixl_status_t status;
    nixlAgentConfig nixlConfig{config.useProgThread};
    mRawAgent = std::make_unique<nixlAgent>(config.mName, std::move(nixlConfig));

    nixl_b_params_t init1;
    nixl_mem_list_t mems1;
    status = mRawAgent->getPluginParams("UCX", mems1, init1);
    TLLM_CHECK(status == NIXL_SUCCESS);

    status = mRawAgent->createBackend("UCX", init1, mRawBackend);
    if (status != NIXL_SUCCESS || !mRawBackend)
    {
        TLLM_THROW("Failed to create NIXL backend");
    }
    mExtraParams.backends.push_back(mRawBackend);

    mZmqRepSocket = zmq::socket_t(mZmqContext, zmq::socket_type::rep);
    mZmqRepSocket.set(zmq::sockopt::sndhwm, 2);
    std::string ip = getAvailableIP();
    mZmqRepSocket.bind("tcp://" + ip + ":*");
    mZmqRepEndpoint = mZmqRepSocket.get(zmq::sockopt::last_endpoint);
    TLLM_LOG_INFO("NixlTransferAgent::NixlTransferAgent mZmqRepEndpoint: %s", mZmqRepEndpoint.c_str());
    mZmqRepThread = std::thread(
        [this]()
        {
            while (true)
            {
                zmq::message_t message;
                auto ret = mZmqRepSocket.recv(message);
                TLLM_CHECK_WITH_INFO(ret, "mZmqRepSocket.recv failed");
                std::string recvMessage(static_cast<char*>(message.data()), message.size());
                if (recvMessage == std::string("get_md"))
                {
                    std::string localMD;
                    auto status = mRawAgent->getLocalMD(localMD);
                    TLLM_CHECK_WITH_INFO(status == NIXL_SUCCESS, "getLocalMD failed with status: %s",
                        nixlEnumStrings::statusStr(status).c_str());
                    mZmqRepSocket.send(zmq::buffer(localMD), zmq::send_flags::none);
                }
                else if (recvMessage == std::string("stop"))
                {
                    std::string stopMessage = "stop";
                    mZmqRepSocket.send(zmq::buffer(stopMessage), zmq::send_flags::none);
                    break;
                }
                else
                {
                    TLLM_THROW("Unknown message: %s", recvMessage.c_str());
                }
            }
        });
}

void NixlTransferAgent::registerMemory(RegisterDescs const& descs)
{
    nixl_status_t status;
    status = mRawAgent->registerMem(NixlHelper::convertRegDlist(descs), &mExtraParams);
    TLLM_CHECK(status == NIXL_SUCCESS);

    std::string localMD;
    status = mRawAgent->getLocalMD(localMD);
    TLLM_CHECK(status == NIXL_SUCCESS);
}

void NixlTransferAgent::deregisterMemory(RegisterDescs const& descs)
{
    nixl_status_t status;
    status = mRawAgent->deregisterMem(NixlHelper::convertRegDlist(descs), &mExtraParams);
    TLLM_CHECK(status == NIXL_SUCCESS);
}

void NixlTransferAgent::loadRemoteAgent(std::string const& name, AgentDesc const& agentDesc)
{
    nixl_status_t status;
    std::string remoteName;
    status = mRawAgent->loadRemoteMD(agentDesc.getBackendAgentDesc(), remoteName);
    TLLM_CHECK(status == NIXL_SUCCESS);
    TLLM_CHECK_WITH_INFO(
        name == remoteName, "loadRemoteAgent gets error agent name: %s != %s", name.c_str(), remoteName.c_str());
}

void NixlTransferAgent::invalidateRemoteAgent(std::string const& name)
{
    mRawAgent->invalidateRemoteMD(name);
}

[[nodiscard]] std::unique_ptr<TransferStatus> NixlTransferAgent::submitTransferRequests(TransferRequest const& request)
{
    nixl_status_t status;
    nixlXferReqH* handle;

    if (request.getSyncMessage().has_value())
    {
        mExtraParams.hasNotif = true;

        mExtraParams.notifMsg = request.getSyncMessage().value();
    }
    else
    {
        mExtraParams.hasNotif = false;
    }

    status = mRawAgent->createXferReq(NixlHelper::convert(request.getOp()),
        NixlHelper::convertXferDist(request.getSrcDescs()), NixlHelper::convertXferDist(request.getDstDescs()),
        request.getRemoteName(), handle, &mExtraParams);

    TLLM_CHECK_WITH_INFO(status == NIXL_SUCCESS, " rank: %d createXferReq failed with status: %s remoteAgent name: %s",
        mpi::MpiComm::world().getRank(), nixlEnumStrings::statusStr(status).c_str(), request.getRemoteName().c_str());

    status = mRawAgent->postXferReq(handle, &mExtraParams);
    return std::make_unique<NixlTransferStatus>(mRawAgent.get(), handle);
}

void NixlTransferAgent::notifySyncMessage(std::string const& name, SyncMessage const& syncMessage)
{
    auto status = mRawAgent->genNotif(name, syncMessage);
    TLLM_CHECK_WITH_INFO(
        status == NIXL_SUCCESS, "genNotif failed with status: %s", nixlEnumStrings::statusStr(status).c_str());
}

[[nodiscard]] std::unordered_map<std::string, std::vector<SyncMessage>> NixlTransferAgent::getNotifiedSyncMessages()
{

    nixl_notifs_t notifs;
    auto status = mRawAgent->getNotifs(notifs);
    TLLM_CHECK_WITH_INFO(
        status == NIXL_SUCCESS, "getNotifs failed with status: %s", nixlEnumStrings::statusStr(status).c_str());

    return notifs;
}

ConnectionInfoType NixlTransferAgent::getConnectionInfo()
{
    return mZmqRepEndpoint;
}

void NixlTransferAgent::connectRemoteAgent(std::string const& name, ConnectionInfoType const& connectionInfo)
{
    auto reqSocket = zmq::socket_t(mZmqContext, zmq::socket_type::req);
    TLLM_LOG_DEBUG("NixlTransferAgent::connectRemoteAgent connectionInfo: %s", connectionInfo.c_str());

    try
    {
        reqSocket.connect(connectionInfo);
        TLLM_LOG_DEBUG("NixlTransferAgent::connectRemoteAgent reqSocket.connect");
        std::string getMDMessage = "get_md";
        reqSocket.send(zmq::buffer(getMDMessage), zmq::send_flags::none);
        zmq::message_t message;
        if (!reqSocket.recv(message))
        {
            TLLM_THROW("Failed to receive message from remote agent");
        }
        std::string remoteMD(static_cast<char*>(message.data()), message.size());
        std::string remoteName;
        auto status = mRawAgent->loadRemoteMD(remoteMD, remoteName);

        TLLM_CHECK_WITH_INFO(
            status == NIXL_SUCCESS, "loadRemoteMD failed with status: %s", nixlEnumStrings::statusStr(status).c_str());
        TLLM_CHECK_WITH_INFO(
            name == remoteName, "connectRemoteAgent gets error agent name: %s != %s", name.c_str(), remoteName.c_str());
    }
    catch (zmq::error_t const& e)
    {
        TLLM_THROW("ZMQ error during connection: %s", e.what());
    }
    TLLM_LOG_DEBUG("NixlTransferAgent::connectRemoteAgent connectRemoteAgent success");
}

NixlTransferAgent::~NixlTransferAgent()
{
    TLLM_LOG_DEBUG("NixlTransferAgent::~NixlTransferAgent");

    if (mZmqRepThread.joinable())
    {
        zmq::socket_t socket(mZmqContext, zmq::socket_type::req);
        socket.connect(mZmqRepEndpoint);
        std::string stopMessage = "stop";
        socket.send(zmq::buffer(stopMessage), zmq::send_flags::none);
        zmq::message_t message;
        auto ret = socket.recv(message);
        TLLM_CHECK_WITH_INFO(ret, "socket.recv failed");
        std::string recvMessage(static_cast<char*>(message.data()), message.size());
        TLLM_CHECK_WITH_INFO(recvMessage == stopMessage, "recvMessage != stop , %s", recvMessage.c_str());
        socket.close();
        mZmqRepThread.join();
    }

    mZmqRepSocket.close();

    mZmqContext.close();
}

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#endif

extern "C"
{
    std::unique_ptr<BaseTransferAgent> createNixlTransferAgent(BaseAgentConfig const* config)
    {
        TLLM_CHECK(config);
        return std::make_unique<NixlTransferAgent>(*config);
    }
}

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

} // namespace tensorrt_llm::executor::kv_cache

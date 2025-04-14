

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

#include "tensorrt_llm/common/utils.h"
#include "tensorrt_llm/executor/types.h"
#include <cstdint>
#include <limits>
#include <sstream>

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
};

using SyncMessageType = std::string;

class LocalBlockInfos
{
    class LocalBlockEntry
    {
    public:
        size_t address;
        size_t blockSize;
    };

private:
    std::vector<LocalBlockEntry> entries;
};

class TransceiverConnection
{
public:
private:
    // TransceiverConnectionSideChannel sideChannel;
};

// 考虑etcd .
class TransferEngine
{

public:
    using TransferEngineIdType = std::string;
    using EngineMetaDataType = std::string;

    class TransferEngineInfo
    {

        TransferEngine::TransferEngineIdType mEngineId;
        EngineMetaDataType mEngineMetaData;
    };

    void registerMemory(void* ptr, size_t size, RegisterMemoryType memoryType) {}

    void unregisterMemory(void* ptr, size_t size, RegisterMemoryType memoryType) {}

    bool isRegistered(void* ptr, size_t size, RegisterMemoryType memoryType) {}

    std::shared_ptr<TransferRequest> writeBlocksAsync(LocalBlockInfos const& dstBlockInfos,
        RemoteBlockInfos const& srcBlockInfos, TransferEngineIdType const& remoteEngineId,
        std::optional<SyncMessageType> const& syncMessage){

    };

    std::shared_ptr<TransferRequest> readBlocksAsync(LocalBlockInfos const& dstBlockInfos,
        RemoteBlockInfos const& srcBlockInfos, TransferEngineIdType const& remoteEngineId,
        std::optional<SyncMessageType> const& syncMessage){

    };

    TransferEngineInfo getEngineConnectionInfo() {}

    std::vector<TransferEngineIdType> connectRemoteEngines(std::vector<TransferEngineInfo> const& remoteEngineInfos) {}

    void notifySyncInfo(SyncMessageType const& syncMessage)
    {
        // for MLA , some rank didn't need to be read.
    }

    std::optional<SyncMessageType> checkAndParseMatchedSyncInfo(SyncMessageType const& matchSyncInfo)
    {
        // check the sync info is matched.
        // return the entire matched sync info.
    }

private:
    bool connected(TransferEngineInfo const& remoteEngineInfo) {}

    TransferEngineIdType uniqueEngineId; // uuid? // hostname+pid+atomic_increment
    TransferEngineInfo mEngineInfo;
};

class MultiTransferRequest : public TransferRequest
{

    MultiTransferRequest(std::vector<std::shared_ptr<TransferRequest>>&& transferRequests)
        : mTransferRequests(std::move(transferRequests))
    {
    }

    MultiTransferRequest(std::vector<std::shared_ptr<TransferRequest>> const& transferRequests)
        : mTransferRequests(transferRequests)
    {
    }

    std::vector<std::shared_ptr<TransferRequest>> mTransferRequests;

    void wait() override
    {

        for (auto& request : mTransferRequests)
        {
            request->wait();
        }
    }

    bool isCompleted() override
    {

        return std::all_of(mTransferRequests.begin(), mTransferRequests.end(),
            [](std::shared_ptr<TransferRequest>& request) { return request->isCompleted(); });
    }
};

class OneSideContextParams
{

    std::vector<TransferEngine::TransferEngineInfo> mEngineInfos;
    std::vector<uint64_t> mBlockIds;
    std::uint64_t mKvCachePoolBaseAddress;
    std::uint64_t mBlockSIzeInBytes;
};

class OneSideFormatter
{

public:
    std::vector<RemoteBlockInfos> genRemoteBlockInfos(LlmRequest* llmRequest,
        executor::kv_cache::CacheState const& selfCacheState, executor::kv_cache::CacheState const& peerCacheState)
    {
        // to compute the remote block ptr need to be read
    }

    std::vector<LocalBlockInfos> genLocalBlockInfos(LlmRequest* llmRequest,
        executor::kv_cache::CacheState const& selfCacheState, executor::kv_cache::CacheState const& peerCacheState)
    {
        // to compute the local block ptr need to be filled.
    }

    std::vector<SyncMessageType> genSyncInfo(LlmRequest* llmRequest,
        executor::kv_cache::CacheState const& selfCacheState, executor::kv_cache::CacheState const& peerCacheState)
    {
        // generate the sync info  , which should be matched by the remote side., context requestId should be included.
    }

    std::shared_ptr<MultiTransferRequest> initiateTransfer(LlmRequest* llmRequest,
        std::vector<TransferEngine::TransferEngineIdType> const& remoteEngineIds,
        executor::kv_cache::CacheState const& selfCacheState, executor::kv_cache::CacheState const& peerCacheState,
        runtime::BufferManager& bufferManager)
    {

        auto localBlockInfos = genLocalBlockInfos(llmRequest, selfCacheState, peerCacheState);
        auto remoteBlockInfos = genRemoteBlockInfos(llmRequest, selfCacheState, peerCacheState);
        auto syncInfos = genSyncInfo(llmRequest, selfCacheState, peerCacheState);

        TLLM_CHECK_WITH_INFO(
            localBlockInfos.size() == remoteBlockInfos.size(), "localBlockInfos.size() != remoteBlockInfos.size()");

        TLLM_CHECK_WITH_INFO(
            remoteEngineIds.size() == localBlockInfos.size(), "remoteEngineIds.size() != localBlockInfos.size()");
        TLLM_CHECK_WITH_INFO(syncInfos.size() == localBlockInfos.size(), "syncInfos.size() != localBlockInfos.size()");

        int sz = remoteEngineIds.size();

        std::vector<std::shared_ptr<TransferRequest>> transferRequests;
        for (int i = 0; i < sz; ++i)
        {

            auto remoteEngineId = remoteEngineIds[i];
            auto localBlockInfo = localBlockInfos[i];
            auto remoteBlockInfo = remoteBlockInfos[i];

            auto transferRequest
                = mTransferEngine->readBlocksAsync(localBlockInfo, remoteBlockInfo, remoteEngineId, syncInfos[i]);

            transferRequests.push_back(transferRequest);
        }

        return std::make_shared<MultiTransferRequest>(std::move(transferRequests));
    }

    void postHandleFormatter(LlmRequest* llmRequest, std::shared_ptr<TransferRequest> const& transferRequest,
        runtime::BufferManager& bufferManager)
    {
    }

private:
    kv_cache_manager::KVCacheManager* mKvCacheManager;

    TransferEngine* mTransferEngine;
};

class OneSideDataInitiator
{

public:
    std::shared_ptr<TransferRequest> initiateTransfer(LlmRequest* llmRequest)
    {

        auto DataTransceiverState = llmRequest->getDataTransceiverState();

        // auto TransferEngineInfo = DataTransceiverState.getTransferEngineInfo();
        auto TransferEngineInfoList = getTransferEngineInfoList(llmRequest);
        auto peerCacheState = DataTransceiverState.getCacheState();

        auto remoteEngineIds = mTransferEngine->connectRemoteEngines(TransferEngineInfoList);
        auto transferRequest = mFormatter.initiateTransfer(
            llmRequest, remoteEngineIds, mSlefCacheState, peerCacheState.value(), mBufferManager);
    }

    void postHandle(LlmRequest* llmRequest, std::shared_ptr<TransferRequest> const& transferRequest)
    {
        mFormatter.postHandleFormatter(llmRequest, transferRequest, mBufferManager);
    }

private:
    std::vector<TransferEngine::TransferEngineInfo> getTransferEngineInfoList(LlmRequest* llmRequest) {}

    TransferEngine* mTransferEngine;
    executor::kv_cache::CacheState mSlefCacheState;
    runtime::BufferManager mBufferManager;
    OneSideFormatter mFormatter;
};

class OneSideInitiator
{

public:
    void enqueueRequest(LlmRequest* llmRequest)
    {

        try
        {
            auto promise
                = std::make_unique<std::promise<std::variant<std::shared_ptr<TransferRequest>, std::nullptr_t>>>();
            auto future = promise->get_future();

            if (isHomogenous(llmRequest))
            {
                {
                    std::unique_lock<std::mutex> lck(mHomoAsyncResource.mMtxForQueue);
                    mHomoAsyncResource.mRequestsQueue.emplace_back(llmRequest, std::move(promise));
                }
                mHomoAsyncResource.mCVforQueue.notify_all();
                mTransferHomoRequestFutures[llmRequest] = std::move(future);
            }
            else
            {
                {
                    std::unique_lock<std::mutex> lck(mHeteroAsyncResource.mMtxForQueue);
                    mHeteroAsyncResource.mRequestsQueue.emplace_back(llmRequest, std::move(promise));
                }
                mHeteroAsyncResource.mCVforQueue.notify_all();
                mTransferHeteroRequestFutures[llmRequest] = std::move(future);
            }
        }
        catch (std::exception& e)
        {
            TLLM_THROW("%s", e.what());
        }
    }

    void TransferSync(LlmRequest* llmRequest)
    {
        std::shared_ptr<TransferRequest> transferRequest = mDataInitiator.initiateTransfer(llmRequest);
        transferRequest->wait();
        mDataInitiator.postHandle(llmRequest, transferRequest);
        // CacheFormmater
        // TODO:
    }

    std::shared_ptr<TransferRequest> InitiateTransferAsync(LlmRequest* llmRequest)
    {
        // queue the request to the transfer engine.
        return mDataInitiator.initiateTransfer(llmRequest);
    }

    std::vector<LlmRequest*> getReadyRequests()
    {
        std::vector<LlmRequest*> readyRequests;
        for (auto it = mTransferHomoRequestFutures.begin(); it != mTransferHomoRequestFutures.end();)
        {
            if (it->second.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
            {
                mTransferHomoRequests[it->first] = std::get<std::shared_ptr<TransferRequest>>(it->second.get());
                it = mTransferHomoRequestFutures.erase(it);
            }
            else
            {
                ++it;
            }
        }
        for (auto& [request, transferRequest] : mTransferHomoRequests)
        {
            if (transferRequest->isCompleted())
            {
                readyRequests.push_back(request);
            }
        }
        for (auto& [request, transferRequest] : mTransferHeteroRequestFutures)
        {
            if (transferRequest.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
            {
                readyRequests.push_back(request);
            }
        }
        return readyRequests;
    }

    void releaseResources(LlmRequest* llmRequest)
    {
        // release the resources for the request.
        // after checkGenTransferStatus is done.
        if (isHomogenous(llmRequest))
        {
            mTransferHomoRequests.erase(llmRequest);
        }
        else
        {
            mTransferHeteroRequestFutures.erase(llmRequest);
        }
    }

    //
private:
    bool isHomogenous(LlmRequest* llmRequest)
    {

        return llmRequest->getDataTransceiverState().getCacheState().value() == mSelfCacheState;
    }

    void handleHomoRequestThreadFunc()
    {
        try
        {
            tensorrt_llm::common::setThreadName("oneSideInitiatorHomoRequest");
            TLLM_CUDA_CHECK(cudaSetDevice(mDeviceId));
            while (!mHomoAsyncResource.mTerminate)
            {

                RequestAndPromise requestAndPromise;
                {
                    std::unique_lock<std::mutex> lck(mHomoAsyncResource.mMtxForQueue);
                    mHomoAsyncResource.mCVforQueue.wait(lck,
                        [&] { return !mHomoAsyncResource.mRequestsQueue.empty() || mHomoAsyncResource.mTerminate; });
                    if (mHomoAsyncResource.mTerminate)
                    {
                        if (!mHomoAsyncResource.mRequestsQueue.empty())
                        {
                            TLLM_LOG_WARNING(
                                "There are still %zu requests in the mRequestsQueue, but encountered terminate.",
                                mHomoAsyncResource.mRequestsQueue.size());
                        }
                        break;
                    }
                    requestAndPromise = std::move(mHomoAsyncResource.mRequestsQueue.front());
                    mHomoAsyncResource.mRequestsQueue.pop_front();
                }
                {
                    TLLM_CHECK_WITH_INFO(requestAndPromise.mRequest != nullptr, "requestAndPromise.mRequest is null");
                    auto transferRequest = InitiateTransferAsync(requestAndPromise.mRequest);
                    auto variant = std::variant<std::shared_ptr<TransferRequest>, std::nullptr_t>(transferRequest);
                    requestAndPromise.mPromise->set_value(std::move(variant));
                }
            }
        }
        catch (std::exception& e)
        {
            TLLM_THROW("%s", e.what());
        }
    }

    void handleHeteroRequestThreadFunc()
    {
        try
        {
            tensorrt_llm::common::setThreadName("oneSideInitiatorHeteroRequest");
            TLLM_CUDA_CHECK(cudaSetDevice(mDeviceId));
            while (!mHeteroAsyncResource.mTerminate)
            {

                RequestAndPromise requestAndPromise;
                {
                    std::unique_lock<std::mutex> lck(mHeteroAsyncResource.mMtxForQueue);
                    mHeteroAsyncResource.mCVforQueue.wait(lck,
                        [&]
                        { return !mHeteroAsyncResource.mRequestsQueue.empty() || mHeteroAsyncResource.mTerminate; });
                    if (mHeteroAsyncResource.mTerminate)
                    {
                        if (!mHeteroAsyncResource.mRequestsQueue.empty())
                        {
                            TLLM_LOG_WARNING(
                                "There are still %zu requests in the mRequestsQueue, but encountered terminate.",
                                mHeteroAsyncResource.mRequestsQueue.size());
                        }
                        break;
                    }
                    requestAndPromise = std::move(mHeteroAsyncResource.mRequestsQueue.front());
                    mHeteroAsyncResource.mRequestsQueue.pop_front();
                }
                {
                    TLLM_CHECK_WITH_INFO(requestAndPromise.mRequest != nullptr, "requestAndPromise.mRequest is null");
                    TransferSync(requestAndPromise.mRequest);
                    auto variant = std::variant<std::shared_ptr<TransferRequest>, std::nullptr_t>(nullptr);
                    requestAndPromise.mPromise->set_value(std::move(variant));
                }
            }
        }
        catch (std::exception& e)
        {
            TLLM_THROW("%s", e.what());
        }
    }

    OneSideDataInitiator mDataInitiator;
    TransferEngine* mTransferEngine;
    kv_cache_manager::KVCacheManager* mKvCacheManager;

    std::map<LlmRequest*, std::future<std::variant<std::shared_ptr<TransferRequest>, std::nullptr_t>>>
        mTransferHomoRequestFutures;
    std::map<LlmRequest*, std::shared_ptr<TransferRequest>> mTransferHomoRequests;
    std::map<LlmRequest*, std::future<std::variant<std::shared_ptr<TransferRequest>, std::nullptr_t>>>
        mTransferHeteroRequestFutures;
    executor::kv_cache::CacheState mSelfCacheState;
    int mSelfRank = 0;
    int mDeviceId{-1};

    struct RequestAndPromise
    {
        LlmRequest* mRequest;
        std::unique_ptr<std::promise<std::variant<std::shared_ptr<TransferRequest>, std::nullptr_t>>> mPromise;
    };

    struct AsyncResource
    {
        std::deque<RequestAndPromise> mRequestsQueue;
        std::mutex mMtxForQueue;
        std::condition_variable mCVforQueue;
        std::atomic<bool> mTerminate{false};
    };

    AsyncResource mHomoAsyncResource;
    AsyncResource mHeteroAsyncResource;
};

class OneSideResponder
{
    // just to check the request is completed.
public:
    void pendingRequest(LlmRequest* llmRequest)
    {
        mRequestSet.insert(llmRequest);
    }

    bool isRequestTransferFinished(LlmRequest* llmRequest)
    {

        auto matchSyncMessage = genMatchSyncMessage(llmRequest);

        mTransferEngine->checkAndParseMatchedSyncInfo(matchSyncMessage);

        // need to check matchSyncMessage from multi  remote ranks (heterogeneous)
    }

    std::vector<LlmRequest*> getReadyRequests()
    {
        std::vector<LlmRequest*> readyRequests;
        for (auto& request : mRequestSet)
        {
            if (isRequestTransferFinished(request))
            {
                readyRequests.push_back(request);
            }
        }
        return readyRequests;
    }

    void releaseResources(LlmRequest* llmRequest)
    {
        mRequestSet.erase(llmRequest);
    }

private:
    SyncMessageType genMatchSyncMessage(LlmRequest* llmRequest) {}

    TransferEngine* mTransferEngine;
    std::set<LlmRequest*> mRequestSet;
    executor::kv_cache::CacheState mCacheState;
    int mSelfRank = 0;
};

class OneSideCacheTransceiver : public BaseCacheTransceiver
{

    void setContextState(LlmRequest* llmRequest)
    {

        // gather transfer Engine info from multi ranks in one instance
        // get block ids for llmRequest from KvCacheManager

        // let llmRequest attach the transferEngineInfo and blockIds
        // mResponseer.setContextState();
    }

    void respondAndSendAsync(LlmRequest* llmRequest) override
    {
        llmRequest->setState(LlmRequestState::kDISAGG_CONTEXT_TRANS_IN_PROGRESS);
        TLLM_CHECK(llmRequest && llmRequest->isContextOnlyRequest());

        setContextState(llmRequest);
        mResponder.pendingRequest(llmRequest);
    }

    void respondAndSendLayerWise(
        RequestVector const& requests, std::shared_ptr<ContextProgress> const& progress) override
    {
        TLLM_THROW("Not implemented");
    }

    void requestAndReceiveSync(LlmRequest* llmRequest) override {}

    void requestAndReceiveAsync(LlmRequest* llmRequest)
    {

        mInitiator.enqueueRequest(llmRequest);
    }

    void checkContextTransferStatus(bool blocking = false) override
    {

        // TODO:
    }

    void checkGenTransferStatus(int atLeastRequestNum = 0) override {}

    bool checkGenTransferComplete() const override {}

private:
    kv_cache_manager::KVCacheManager* mKvCacheManager;
    OneSideInitiator mInitiator;
    OneSideResponder mResponder;
    TransferEngine* mTransferEngine;
};
} // namespace batch_manager
}; // namespace tensorrt_llm

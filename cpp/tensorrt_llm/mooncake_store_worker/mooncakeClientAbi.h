/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// The Mooncake store client as the store worker sees it: the one place with unusual C++.
//
// The Mooncake wheel (mooncake-transfer-engine-cuda13 0.3.13) ships no C++ headers, but its Python
// store hands out its own C++ client: MooncakeDistributedStore._get_pyclient_capsule() returns a
// capsule holding a std::shared_ptr<mooncake::PyClient> (store_py.cpp:2097-2110, 2195-2199).
// PyClient is an abstract class, so the worker calls its methods through the object's method table
// (vtable) and links nothing of the wheel.
//
// mirror::PyClient declares the same virtual methods in the same order as class PyClient in
// mooncake-store/include/pyclient.h at v0.3.13 (e5598b0992, lines 212-436). Under the Itanium C++
// ABI the virtual methods of a class take the slots of its method table in declaration order, and the
// virtual destructor declared first takes two (slots 0 and 1). The wheel's client is a
// mooncake::RealClient, which derives from PyClient alone (real_client.h:77), so its slots hold
// RealClient's methods. A slot the worker calls must match the wheel in position, parameters and
// return type; the slots it never calls are placeholders that only keep the count.
//
// sharedClientOfCapsule() refuses a client, before any call, unless the symbol each called slot points to
// (dladdr) is RealClient's method of that name. Parameter types are not compared: the Python side
// requires the exact wheel version first. ReplicateConfig (replica.h:104-125), passed by reference to
// session start, is mirrored the same way and has no runtime check; the version pin and the
// conformance suite's put round trips cover it.
//
// The mirror's methods keep Mooncake's spelling, so each line can be compared with pyclient.h.
//
// TODO(mooncake): a stable C++ handle with a batched submit/poll API would make this mirror
// unnecessary.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(_GLIBCXX_USE_CXX11_ABI) || _GLIBCXX_USE_CXX11_ABI != 1
#error "the Mooncake wheel passes std::string and std::vector of the C++11 libstdc++ ABI"
#endif
#ifdef _GLIBCXX_DEBUG
#error "_GLIBCXX_DEBUG changes the std::vector and std::string layout the Mooncake client expects"
#endif

namespace tensorrt_llm::mooncake_store_worker
{

//! A client whose method table the mirror below does not match.
class ClientLayoutError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

namespace mirror
{

using Keys = std::vector<std::string>;
using RowBuffers = std::vector<std::vector<void*>>;
using RowSizes = std::vector<std::vector<size_t>>;

enum class SoftPinAction : uint8_t // replica.h:64
{
    kPreserve = 0,
};

enum class ObjectDataType : uint8_t // types.h:140
{
    kUnknown = 0,
};

// replica.h:104-125; the defaults are the SDK's, as Python's batch_put_session_start passes them.
struct ReplicateConfig
{
    size_t replicaNum{1};
    size_t nofReplicaNum{0};
    size_t dfsReplicaNum{0};
    SoftPinAction softPinAction{SoftPinAction::kPreserve};
    std::optional<uint64_t> softPinTtlMs{};
    bool withHardPin{false};
    std::vector<std::string> preferredSegments{};
    std::string preferredSegment{};
    std::vector<std::string> preferredNofSegments{};
    bool preferAllocInSameNode{false};
    ObjectDataType dataType{ObjectDataType::kUnknown};
    std::string hostId{};
    std::optional<std::vector<std::string>> groupIds{};
};

// The SDK answers the worker names (mooncake-store/include/types.h).
constexpr int kOk = 0;
constexpr int kNoAvailableHandle = -200;
constexpr int kInvalidParams = -600;
constexpr int kReplicaIsNotReady = -703;
constexpr int kObjectNotFound = -704;
constexpr int kObjectAlreadyExists = -705;
constexpr int kLeaseExpired = -707;
constexpr int kTransferFail = -800;
constexpr int kRpcFail = -900;
constexpr int kRpcTimeout = -901;
constexpr int kObjectExists = 1; // batchIsExist
constexpr int kObjectAbsent = 0;

// clang-format off
class PyClient
{
public:
    virtual ~PyClient() = default;                     // slots 0, 1   pyclient.h:220
    virtual void setup_real() {}                       // 2            :221
    virtual void setup_dummy() {}                      // 3            :233
    virtual void initAll() {}                          // 4            :237
    virtual void alloc_from_mem_pool() {}              // 5            :241
    virtual void put() {}                              // 6            :243
    virtual void register_buffer() {}                  // 7            :246
    virtual void unregister_buffer() {}                // 8            :248
    virtual void get_into() {}                         // 9            :250
    virtual void get_into_ranges() {}                  // 10           :256
    virtual void batch_query() {}                      // 11           :264
    virtual void batch_get_into() {}                   // 12           :267
    virtual std::vector<int> batch_get_into_multi_buffers(Keys const& keys, RowBuffers const& all_buffers,
        RowSizes const& all_sizes, bool prefer_same_node) = 0;        // 13  :272
    virtual void put_from() {}                         // 14           :278
    virtual void put_from_with_metadata() {}           // 15           :281
    virtual void batch_put_from() {}                   // 16           :286
    virtual void batch_put_from_multi_buffers() {}     // 17           :291
    virtual void batch_get_session_start() {}          // 18           :299
    virtual void batch_get_into_multi_buffer_ranges() {} // 19         :305
    virtual void batch_get_session_end() {}            // 20           :314
    virtual std::vector<int> batch_put_session_start(Keys const& keys, std::vector<size_t> const& sizes,
        ReplicateConfig const& config) = 0;                           // 21  :319
    virtual std::vector<int> batch_put_from_multi_buffer_ranges(Keys const& keys, RowBuffers const& all_buffers,
        RowSizes const& all_sizes, RowSizes const& all_dst_offsets) = 0; // 22  :327
    virtual std::vector<int> batch_put_session_end(Keys const& keys) = 0;    // 23  :336
    virtual std::vector<int> batch_put_session_revoke(Keys const& keys) = 0; // 24  :342
    virtual void get_buffer() {}                       // 25           :348
    virtual void batch_get_buffer() {}                 // 26           :351
    virtual void put_parts() {}                        // 27           :354
    virtual void put_batch() {}                        // 28           :358
    virtual void upsert() {}                           // 29           :363
    virtual void upsert_from() {}                      // 30           :366
    virtual void batch_upsert_from() {}                // 31           :370
    virtual void batch_upsert_from_multi_buffers() {}  // 32           :375
    virtual void upsert_parts() {}                     // 33           :381
    virtual void upsert_batch() {}                     // 34           :385
    virtual void get_hostname() {}                     // 35           :390
    virtual void remove() {}                           // 36           :392
    virtual void removeByRegex() {}                    // 37           :394
    virtual void removeAll() {}                        // 38           :396
    virtual void batchRemove() {}                      // 39           :398
    virtual void isExist() {}                          // 40           :401
    virtual std::vector<int> batchIsExist(Keys const& keys) = 0;             // 41  :403
};

// clang-format on

struct CalledSlot
{
    int index;
    char const* method;
};

// The slots the worker calls; a client is refused unless each holds RealClient's method of this name.
inline constexpr CalledSlot kCalledSlots[] = {
    {13, "batch_get_into_multi_buffers"},
    {21, "batch_put_session_start"},
    {22, "batch_put_from_multi_buffer_ranges"},
    {23, "batch_put_session_end"},
    {24, "batch_put_session_revoke"},
    {41, "batchIsExist"},
};

} // namespace mirror

//! The client a Mooncake store capsule holds, shared with the Python store. `capsulePointer` is the
//! pointer of a "mooncake.PyClient.shared_ptr" capsule. Throws ClientLayoutError naming the first called
//! slot that does not hold RealClient's method, before any call.
std::shared_ptr<mirror::PyClient> sharedClientOfCapsule(void const* capsulePointer);

} // namespace tensorrt_llm::mooncake_store_worker

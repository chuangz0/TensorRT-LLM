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

#include "sdkObjectClient.h"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace tensorrt_llm::mooncake_store_worker
{
namespace
{

std::vector<std::vector<uintptr_t>> addresses(mirror::RowBuffers const& buffers)
{
    std::vector<std::vector<uintptr_t>> rows;
    rows.reserve(buffers.size());
    for (auto const& row : buffers)
    {
        auto& out = rows.emplace_back();
        for (void* buffer : row)
        {
            out.push_back(reinterpret_cast<uintptr_t>(buffer));
        }
    }
    return rows;
}

class SdkObjectClient final : public mirror::PyClient
{
public:
    explicit SdkObjectClient(nb::object sdk)
        : mSdk(std::move(sdk))
    {
    }

    ~SdkObjectClient() override
    {
        nb::gil_scoped_acquire gil;
        mSdk.reset();
    }

    std::vector<int> batch_get_into_multi_buffers(mirror::Keys const& keys, mirror::RowBuffers const& all_buffers,
        mirror::RowSizes const& all_sizes, bool /*prefer_same_node*/) override
    {
        return call("batch_get_into_multi_buffers", keys, addresses(all_buffers), all_sizes);
    }

    // The driver passes no replicate config: the SDK's default.
    std::vector<int> batch_put_session_start(
        mirror::Keys const& keys, std::vector<size_t> const& sizes, mirror::ReplicateConfig const& /*config*/) override
    {
        return call("batch_put_session_start", keys, sizes);
    }

    std::vector<int> batch_put_from_multi_buffer_ranges(mirror::Keys const& keys, mirror::RowBuffers const& all_buffers,
        mirror::RowSizes const& all_sizes, mirror::RowSizes const& all_dst_offsets) override
    {
        return call("batch_put_from_multi_buffer_ranges", keys, addresses(all_buffers), all_sizes, all_dst_offsets);
    }

    std::vector<int> batch_put_session_end(mirror::Keys const& keys) override
    {
        return call("batch_put_session_end", keys);
    }

    std::vector<int> batch_put_session_revoke(mirror::Keys const& keys) override
    {
        return call("batch_put_session_revoke", keys);
    }

    std::vector<int> batchIsExist(mirror::Keys const& keys) override
    {
        return call("batch_is_exist", keys);
    }

private:
    //! The Python method's answers; a Python raise becomes a std::runtime_error carrying its repr, as
    //! the Python driver words it.
    template <typename... Args>
    std::vector<int> call(char const* method, Args const&... args)
    {
        nb::gil_scoped_acquire gil;
        try
        {
            return nb::cast<std::vector<int>>(mSdk.attr(method)(args...));
        }
        catch (nb::python_error const& error)
        {
            throw std::runtime_error(nb::repr(error.value()).c_str());
        }
    }

    nb::object mSdk;
};

} // namespace

std::shared_ptr<mirror::PyClient> clientOverSdkObject(nb::object sdk)
{
    return std::make_shared<SdkObjectClient>(std::move(sdk));
}

} // namespace tensorrt_llm::mooncake_store_worker

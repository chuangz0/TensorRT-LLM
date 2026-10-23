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

#include "mooncakeClientAbi.h"

#include <cxxabi.h>
#include <dlfcn.h>

#include <cstdlib>

namespace tensorrt_llm::mooncake_store_worker
{
namespace
{

//! The demangled name of the symbol at `address`, or "<no symbol>".
std::string symbolAt(void const* address)
{
    Dl_info info{};
    if (dladdr(address, &info) == 0 || info.dli_sname == nullptr)
    {
        return "<no symbol>";
    }
    int status = 0;
    char* demangled = abi::__cxa_demangle(info.dli_sname, nullptr, nullptr, &status);
    std::string name = status == 0 && demangled != nullptr ? demangled : info.dli_sname;
    std::free(demangled);
    return name;
}

void checkClientLayout(mirror::PyClient const& client)
{
    // An object of a class with virtual methods starts with a pointer to its method table.
    auto const* const* slots = *reinterpret_cast<void const* const* const*>(&client);
    for (auto const& slot : mirror::kCalledSlots)
    {
        std::string const expected = std::string("mooncake::RealClient::") + slot.method + "(";
        std::string const found = symbolAt(slots[slot.index]);
        if (found.rfind(expected, 0) != 0)
        {
            std::string const method = found.substr(0, found.find('(')); // without the parameter list
            throw ClientLayoutError("Mooncake client slot " + std::to_string(slot.index) + " holds " + method
                + ", where this store worker calls " + expected + "...)");
        }
    }
}

} // namespace

std::shared_ptr<mirror::PyClient> sharedClientOfCapsule(void const* capsulePointer)
{
    // The capsule holds a std::shared_ptr<mooncake::PyClient>. Any std::shared_ptr is a pointer and a
    // control block, so the copy shares ownership through Mooncake's control block: whoever drops the
    // last reference runs Mooncake's deleter, never the mirror's destructor.
    auto client = *static_cast<std::shared_ptr<mirror::PyClient> const*>(capsulePointer);
    checkClientLayout(*client);
    return client;
}

} // namespace tensorrt_llm::mooncake_store_worker

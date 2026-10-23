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

#include "mooncakeClientAbi.h"

#include <nanobind/nanobind.h>

#include <memory>

namespace tensorrt_llm::mooncake_store_worker
{

//! For tests: a client whose six calls go, under the GIL, to the methods of the same SDK name of a
//! Python object (a fake mooncake.store.MooncakeDistributedStore), so the worker's protocol runs over
//! scripted answers.
std::shared_ptr<mirror::PyClient> clientOverSdkObject(nanobind::object sdk);

} // namespace tensorrt_llm::mooncake_store_worker

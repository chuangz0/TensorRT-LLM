# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# The Mooncake store worker (tensorrt_llm.mooncake_store_worker): runs the store
# client of the Mooncake wheel on C++ threads that never hold the GIL. It needs
# no Mooncake headers or libraries (mooncakeClientAbi.h mirrors the client's
# method table), so it also builds on its own against the installed Python:
# cmake -S cpp/tensorrt_llm/mooncake_store_worker -B <dir>
# -DCMAKE_BUILD_TYPE=Release
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.27)
  project(mooncake_store_worker LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  find_package(
    Python 3.10
    COMPONENTS Interpreter Development.Module
    REQUIRED)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT)
  find_package(nanobind CONFIG REQUIRED)
  find_package(Threads REQUIRED)
endif()

set(MOONCAKE_STORE_WORKER_SOURCES bindings.cpp mooncakeClientAbi.cpp
                                  sdkObjectClient.cpp storeWorker.cpp)
# NB_DOMAIN: nanobind internals of its own, never shared with TRT-LLM's
# bindings. NOSTRIP keeps the module BOLT compatible, as the bindings module
# does.
if(ENABLE_BOLT_COMPATIBLE)
  nanobind_add_module(mooncake_store_worker NB_DOMAIN mooncake_store_worker
                      NOSTRIP ${MOONCAKE_STORE_WORKER_SOURCES})
else()
  nanobind_add_module(mooncake_store_worker NB_DOMAIN mooncake_store_worker
                      ${MOONCAKE_STORE_WORKER_SOURCES})
endif()
target_link_libraries(mooncake_store_worker PRIVATE ${CMAKE_DL_LIBS}
                                                    Threads::Threads)

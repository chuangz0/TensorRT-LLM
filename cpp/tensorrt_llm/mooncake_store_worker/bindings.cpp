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

// tensorrt_llm.mooncake_store_worker: the Mooncake store worker for blob/mooncake_native.py.
//
// GIL rule: every call holds the GIL and returns in microseconds, except drain and stop given time to
// wait, which wait without it. Under a busy Python thread, a short call that released the GIL would
// wait a whole switch interval (5 ms) to take it back. Nothing here blocks a Python thread in native
// code otherwise: a daemon thread taking the GIL back during interpreter finalization is ended by
// pthread_exit, whose unwinding through a native frame that may not throw ends the process.

#include "mooncakeClientAbi.h"
#include "sdkObjectClient.h"
#include "storeWorker.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <map>
#include <memory>
#include <tuple>

namespace nb = nanobind;
using namespace tensorrt_llm::mooncake_store_worker;

namespace
{

constexpr char const* kClientCapsuleName = "mooncake.PyClient.shared_ptr";

// blob/mooncake_setup.py STORE_WORKER_API_VERSION must equal this: bump it with any change Python sees,
// so that a stale module is refused at startup instead of failing on the executor thread.
constexpr int kApiVersion = 5;

template <typename T>
using Matrix = nb::ndarray<T const, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using Cuts = nb::ndarray<int64_t const, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

//! The client a set-up MooncakeDistributedStore holds, checked slot by slot.
std::shared_ptr<mirror::PyClient> clientOfStore(nb::handle store)
{
    nb::object const capsule = store.attr("_get_pyclient_capsule")();
    if (capsule.is_none())
    {
        throw std::runtime_error("the Mooncake store has no client");
    }
    void* held = PyCapsule_GetPointer(capsule.ptr(), kClientCapsuleName);
    if (held == nullptr)
    {
        throw nb::python_error();
    }
    return sharedClientOfCapsule(held);
}

template <typename T>
std::vector<T> copied(T const* data, size_t count)
{
    return std::vector<T>(data, data + count);
}

Rows rowsOf(std::string stem, Matrix<uint8_t> const& names, bool hexNames, Matrix<int64_t> const& starts,
    Matrix<int64_t> const& lengths, Cuts const& cuts)
{
    Rows rows;
    rows.stem = std::move(stem);
    rows.hexNames = hexNames;
    rows.count = names.shape(0);
    rows.nameWidth = names.shape(1);
    rows.spanColumns = starts.shape(1);
    bool const lengthsFit
        = (lengths.shape(0) == rows.count || lengths.shape(0) == 1) && lengths.shape(1) == rows.spanColumns;
    if (starts.shape(0) != rows.count || !lengthsFit)
    {
        throw std::invalid_argument("names, starts and lengths disagree in their row count or shape");
    }
    rows.names = copied(names.data(), names.size());
    rows.starts = copied(starts.data(), starts.size());
    rows.lengths = copied(lengths.data(), lengths.size());
    rows.cuts = copied(cuts.data(), cuts.size());
    return rows;
}

//! One finished operation as Python reads it: masks of the READY and the CANCELLED rows, and the few
//! other rows that are not MISS with what their fault text or skip reason is built from.
struct FinishedOpView
{
    uint64_t op;
    nb::object ready;                                                   //!< numpy bool [rows]
    nb::object cancelled;                                               //!< numpy bool [rows], or None: none
    std::vector<std::tuple<size_t, int8_t, int64_t, int8_t>> otherRows; //!< (row, RowState, code, StoreCall)
    bool accessEnded;
    std::map<int8_t, std::string> errors;                               //!< StoreCall -> its first raise or miscount
    std::vector<std::string> sessionsLeftForDrain;
    std::vector<std::string> sessionsLeftToMaster;
};

nb::object numpyBools(std::unique_ptr<bool[]> values, size_t count)
{
    bool* data = values.release();
    nb::capsule owner(data, [](void* pointer) noexcept { delete[] static_cast<bool*>(pointer); });
    return nb::cast(nb::ndarray<nb::numpy, bool, nb::ndim<1>>(data, {count}, owner));
}

FinishedOpView viewOf(FinishedOp&& finished)
{
    size_t const rows = finished.states.size();
    auto ready = std::make_unique<bool[]>(rows);
    std::unique_ptr<bool[]> cancelled; // made at the first cancelled row
    FinishedOpView view{.op = finished.op,
        .cancelled = nb::none(),
        .accessEnded = finished.accessEnded,
        .errors = std::move(finished.errors),
        .sessionsLeftForDrain = std::move(finished.sessionsLeftForDrain),
        .sessionsLeftToMaster = std::move(finished.sessionsLeftToMaster)};
    for (size_t row = 0; row < rows; ++row)
    {
        auto const state = static_cast<RowState>(finished.states[row]);
        ready[row] = state == RowState::kReady;
        if (state == RowState::kCancelled)
        {
            if (!cancelled)
            {
                cancelled = std::make_unique<bool[]>(rows); // all false
            }
            cancelled[row] = true;
        }
        else if (state != RowState::kReady && state != RowState::kMiss)
        {
            view.otherRows.emplace_back(row, finished.states[row], finished.codes[row], finished.calls[row]);
        }
    }
    view.ready = numpyBools(std::move(ready), rows);
    if (cancelled)
    {
        view.cancelled = numpyBools(std::move(cancelled), rows);
    }
    return view;
}

WorkerOptions optionsOf(int threads, int putThreads, int getsPerPut, size_t rowsPerCall, size_t bytesPerCall)
{
    return WorkerOptions{.threads = threads,
        .putThreads = putThreads,
        .getsPerPut = getsPerPut,
        .rowsPerCall = rowsPerCall,
        .bytesPerCall = bytesPerCall};
}

} // namespace

NB_MODULE(mooncake_store_worker, m)
{
    m.doc() = "Mooncake store calls on C++ threads that never hold the GIL";
    m.attr("API_VERSION") = kApiVersion;
    // NativeMooncakeDriver's weakref.finalize keeps each worker referenced until it runs, at the latest
    // at interpreter exit.
    nb::set_leak_warnings(false);
    nb::exception<ClientLayoutError>(m, "ClientLayoutError", PyExc_RuntimeError);

    nb::class_<FinishedOpView>(m, "FinishedOp")
        .def_ro("op", &FinishedOpView::op)
        .def_ro("ready", &FinishedOpView::ready)
        .def_ro("cancelled", &FinishedOpView::cancelled)
        .def_ro("other_rows", &FinishedOpView::otherRows)
        .def_ro("access_ended", &FinishedOpView::accessEnded)
        .def_ro("errors", &FinishedOpView::errors)
        .def_ro("sessions_left_for_drain", &FinishedOpView::sessionsLeftForDrain)
        .def_ro("sessions_left_to_master", &FinishedOpView::sessionsLeftToMaster);

    auto const submit = [](bool isPut)
    {
        return [isPut](StoreWorker& worker, std::string stem, Matrix<uint8_t> const& names, bool hexNames,
                   Matrix<int64_t> const& starts, Matrix<int64_t> const& lengths, Cuts const& cuts)
        { return worker.submit(isPut, rowsOf(std::move(stem), names, hexNames, starts, lengths, cuts)); };
    };

    nb::class_<StoreWorker>(m, "StoreWorker")
        .def(
            "__init__",
            [](StoreWorker* worker, nb::handle store, int threads, int putThreads, int getsPerPut, size_t rowsPerCall,
                size_t bytesPerCall)
            {
                new (worker) StoreWorker(
                    clientOfStore(store), optionsOf(threads, putThreads, getsPerPut, rowsPerCall, bytesPerCall));
            },
            nb::arg("store"), nb::kw_only(), nb::arg("threads"), nb::arg("put_threads"), nb::arg("gets_per_put"),
            nb::arg("rows_per_call"), nb::arg("bytes_per_call"))
        .def_static(
            "over_sdk_object",
            [](nb::object sdk, int threads, int putThreads, int getsPerPut, size_t rowsPerCall, size_t bytesPerCall)
            {
                return new StoreWorker(clientOverSdkObject(std::move(sdk)),
                    optionsOf(threads, putThreads, getsPerPut, rowsPerCall, bytesPerCall));
            },
            nb::arg("sdk"), nb::kw_only(), nb::arg("threads"), nb::arg("put_threads"), nb::arg("gets_per_put"),
            nb::arg("rows_per_call"), nb::arg("bytes_per_call"), nb::rv_policy::take_ownership,
            "For tests: a worker whose store calls go to `sdk`'s methods of the same names, without the slot check.")
        .def("get", submit(false), nb::arg("stem"), nb::arg("names"), nb::arg("hex_names"), nb::arg("starts"),
            nb::arg("lengths"), nb::arg("cuts"))
        .def("put", submit(true), nb::arg("stem"), nb::arg("names"), nb::arg("hex_names"), nb::arg("starts"),
            nb::arg("lengths"), nb::arg("cuts"))
        .def("cancel", &StoreWorker::cancel, nb::arg("op"))
        .def("take_finished",
            [](StoreWorker& worker)
            {
                std::vector<FinishedOpView> views;
                for (auto& finished : worker.takeFinished())
                {
                    views.push_back(viewOf(std::move(finished)));
                }
                return views;
            })
        .def("finished_waiting_s", &StoreWorker::finishedWaitingS)
        .def(
            "drain",
            [](StoreWorker& worker, double timeoutS)
            {
                if (timeoutS <= 0) // never waits: keeps the GIL
                {
                    return worker.drain(0.0);
                }
                nb::gil_scoped_release release;
                return worker.drain(timeoutS);
            },
            nb::arg("timeout_s"))
        .def(
            "stop",
            [](StoreWorker& worker, double timeoutS)
            {
                if (timeoutS <= 0) // never waits: keeps the GIL
                {
                    worker.stop(0.0);
                    return;
                }
                nb::gil_scoped_release release;
                worker.stop(timeoutS);
            },
            nb::arg("timeout_s"));
}

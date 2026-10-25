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

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace tensorrt_llm::mooncake_store_worker
{

//! A row's state; blob/mooncake_native.py _ROW_STATES reads these values.
enum class RowState : int8_t
{
    kPending,
    kReady,
    kMiss,
    kNotReady,
    kDeclined,
    kCancelled,
    kFault,
};

//! The store call whose answer gave a row its code. Python builds fault texts and skip reasons from
//! (call, code) with mooncake.py's helpers; blob/mooncake_native.py _StoreCall reads these values.
enum class StoreCall : int8_t
{
    kNone,
    kGet,
    kReread,
    kPutStart,
    kPutRanges,
    kPutEnd,
    kExistence,
    kRevoke,
    kLeftOpenExistence, //!< The existence check of a key whose put session this worker left open.
};

//! Not an SDK code: the call raised or answered for another number of keys. mooncake.py _Code.CALL_FAILED
//! reads this value.
constexpr int64_t kCallFailed = -100'000;
//! Not an SDK code: another put of this worker holds the key, so the row was not put (NOT_READY on
//! kPutStart, as another writer's open put). mooncake.py _Code.PUT_IN_FLIGHT_HERE reads this value.
constexpr int64_t kPutInFlightHere = -100'001;

struct WorkerOptions
{
    int threads;         //!< Store calls at once.
    int putThreads;      //!< At most this many of them puts, so a get always finds a thread.
    int getsPerPut;      //!< A waiting put goes after this many gets in a row.
    size_t rowsPerCall;  //!< Most rows in one store call.
    size_t bytesPerCall; //!< Most bytes in one store call; a call has at least one row.
};

//! The rows of one put or get, copied at submission. Row i is the object named by names[i]: `stem` and
//! the name's bytes in lowercase hex, or (without hexNames) the name's bytes up to the first NUL. Its
//! bytes are the spans (starts[i][j], length(i, j)) back to back, up to the first zero length. A span
//! is cut where a registration piece begins (`cuts`, sorted), since each transfer slice must lie in one
//! registration.
struct Rows
{
    std::string stem;
    bool hexNames{true};
    size_t count{0};
    size_t nameWidth{0};
    size_t spanColumns{0};
    std::vector<uint8_t> names;   //!< [count, nameWidth]
    std::vector<int64_t> starts;  //!< [count, spanColumns]
    std::vector<int64_t> lengths; //!< [count, spanColumns], or [1, spanColumns] for every row alike
    std::vector<int64_t> cuts;

    int64_t length(size_t row, size_t column) const
    {
        bool const everyRowAlike = lengths.size() == spanColumns;
        return lengths[(everyRowAlike ? 0 : row) * spanColumns + column];
    }
};

//! An operation whose every row concluded, handed to Python once.
struct FinishedOp
{
    uint64_t op;
    std::vector<int8_t> states;                    //!< RowState
    std::vector<int64_t> codes;                    //!< the SDK answer, or kCallFailed
    std::vector<int8_t> calls;                     //!< StoreCall
    bool accessEnded;                              //!< False: some call can never prove that its access ended.
    std::map<int8_t, std::string> errors;          //!< By StoreCall: the first raise or miscount of that call.
    std::vector<std::string> sessionsLeftForDrain; //!< Put sessions left open for MooncakeDriver.drain.
    std::vector<std::string> sessionsLeftToMaster; //!< Put sessions a target may still write into.
};

//! Runs Mooncake store calls on threads of its own, which never touch Python. Gets go before puts; puts
//! take at most `putThreads`. Each operation is cut into calls of `rowsPerCall` rows and `bytesPerCall`
//! bytes. A call follows mooncake.py's protocol: a get rereads once the rows whose read lease expired,
//! a put starts sessions, scatters, then ends or revokes them, except the sessions it leaves open, and
//! confirms keys that already existed. A call that faults a row drops the calls of its operation not
//! started yet (CANCELLED). A put call skips the keys another running put call of this worker holds
//! (kPutInFlightHere): the SDK keys put sessions per client, so a second session start would only fail.
//! It only checks the existence of a key whose session a call left open: such a key is never put again.
//! Every method but drain and stop returns in microseconds.
class StoreWorker
{
public:
    StoreWorker(std::shared_ptr<mirror::PyClient> client, WorkerOptions const& options);
    ~StoreWorker(); //!< stop(0): never waits on a store call.

    StoreWorker(StoreWorker const&) = delete;
    StoreWorker& operator=(StoreWorker const&) = delete;

    //! Queues the rows and returns the operation's id. Throws std::invalid_argument for bad rows, a name
    //! repeated included, and std::runtime_error after drain; nothing is queued then.
    uint64_t submit(bool isPut, Rows rows);
    //! Calls not started yet conclude their rows CANCELLED; running calls go on. Unknown ids are ignored.
    void cancel(uint64_t op);
    std::vector<FinishedOp> takeFinished();
    //! How long the oldest finished operation has waited to be taken, in seconds; 0 when none waits.
    double finishedWaitingS();
    //! Refuses submissions for good, cancels every queued call and waits up to `timeoutS` for the
    //! running ones; whether none runs.
    bool drain(double timeoutS);
    //! Drains, then joins the threads, or leaves them to end on their own when a call still runs.
    //! Idempotent, but not for two threads at once.
    void stop(double timeoutS);

private:
    struct State; // shared with the threads, which may outlive a stop that gave up waiting
    std::shared_ptr<State> mState;
    std::vector<std::thread> mThreads;
};

} // namespace tensorrt_llm::mooncake_store_worker

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

#include "storeWorker.h"

#include <pthread.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace tensorrt_llm::mooncake_store_worker
{
namespace
{

using Clock = std::chrono::steady_clock;

// A longer wait (inf included) is cut to this: the clock's integer ticks cannot hold it.
constexpr double kLongestWaitS = 24 * 3600.0;

// TODO(mooncake): the put scatter and the multi-buffer get answer each row's bytes as an int.
constexpr int64_t kMaxRowBytes = std::numeric_limits<int32_t>::max();

//! One put or get: its rows and their results. The calls of an operation write disjoint rows.
struct Op
{
    uint64_t id{0};
    bool isPut{false};
    Rows rows;
    std::vector<int64_t> rowBytes;
    std::vector<int8_t> states;
    std::vector<int64_t> codes;
    std::vector<int8_t> calls;
    // Under the worker's mutex.
    size_t callsLeft{0};
    bool accessEnded{true};
    std::map<int8_t, std::string> errors; // by StoreCall: the first raise or miscount of that call
    std::vector<std::string> sessionsLeftOpen;
    Clock::time_point finishedAt;

    void set(size_t row, RowState state, int64_t code, StoreCall call)
    {
        states[row] = static_cast<int8_t>(state);
        codes[row] = code;
        calls[row] = static_cast<int8_t>(call);
    }
};

//! Rows [begin, end) of an operation: one store call.
struct Call
{
    std::shared_ptr<Op> op;
    size_t begin{0};
    size_t end{0};
};

//! What a call learned beside its rows' results; merged into its operation under the mutex.
struct CallOutcome
{
    bool accessEnded{true};
    std::vector<std::pair<StoreCall, std::string>> errors;
    std::vector<std::string> sessionsLeftOpen;
};

//! The SDK's arguments for some rows: each row's key, and its spans cut where a registration piece
//! begins.
struct SdkArguments
{
    mirror::Keys keys;
    mirror::RowBuffers buffers;
    mirror::RowSizes sizes;
};

//! The answers, one per key; a call that answers for another number of keys has unknown state.
std::vector<int64_t> oneAnswerPerKey(std::vector<int> const& answers, size_t keyCount)
{
    if (answers.size() != keyCount)
    {
        // The words of mooncake.py's _WrongAnswerCount, whose fault text shows it as its repr.
        throw std::runtime_error(std::to_string(answers.size()) + " answers for " + std::to_string(keyCount) + " keys");
    }
    return {answers.begin(), answers.end()};
}

mirror::ReplicateConfig const& defaultReplicateConfig()
{
    static mirror::ReplicateConfig const config;
    return config;
}

std::string keyOf(Rows const& rows, size_t row)
{
    auto const* name = rows.names.data() + row * rows.nameWidth;
    std::string key = rows.stem;
    if (!rows.hexNames)
    {
        auto const* text = reinterpret_cast<char const*>(name);
        return key.append(text, strnlen(text, rows.nameWidth));
    }
    static constexpr char kHexDigits[] = "0123456789abcdef";
    key.reserve(key.size() + 2 * rows.nameWidth);
    for (size_t i = 0; i < rows.nameWidth; ++i)
    {
        key += kHexDigits[name[i] >> 4];
        key += kHexDigits[name[i] & 0xF];
    }
    return key;
}

//! Appends the buffers of `row`: its spans in order, each cut at the `cuts` inside it.
void appendBuffers(Rows const& rows, size_t row, std::vector<void*>& buffers, std::vector<size_t>& sizes)
{
    auto toBuffer = [](int64_t address) { return reinterpret_cast<void*>(static_cast<uintptr_t>(address)); };
    for (size_t column = 0; column < rows.spanColumns; ++column)
    {
        int64_t const length = rows.length(row, column);
        if (length == 0)
        {
            break;
        }
        int64_t start = rows.starts[row * rows.spanColumns + column];
        int64_t const end = start + length;
        for (auto cut = std::upper_bound(rows.cuts.begin(), rows.cuts.end(), start);
             cut != rows.cuts.end() && *cut < end; ++cut)
        {
            buffers.push_back(toBuffer(start));
            sizes.push_back(static_cast<size_t>(*cut - start));
            start = *cut;
        }
        buffers.push_back(toBuffer(start));
        sizes.push_back(static_cast<size_t>(end - start));
    }
}

//! One store call over rows [begin, end) of an operation, on a worker thread. It follows mooncake.py:
//! every SDK answer becomes a row's (state, code, call), and a call that raised or miscounted leaves
//! access unproven only where it could have moved bytes.
class CallRun
{
public:
    CallRun(mirror::PyClient& client, Op& op, size_t begin, size_t end)
        : mClient(client)
        , mOp(op)
        , mBegin(begin)
    {
        for (size_t row = begin; row < end; ++row)
        {
            mRows.push_back(row);
            mKeys.push_back(keyOf(op.rows, row));
        }
    }

    //! mooncake.py _get_rows: one batch_get_into_multi_buffers; rows whose read lease expired mid-read
    //! are read once more.
    CallOutcome get()
    {
        mOutcome.accessEnded = read(mRows, /*firstRead=*/true);
        return std::move(mOutcome);
    }

    //! mooncake.py _put_rows: a session per key, one scatter of the started rows, then the end of each
    //! row written whole and the revocation of the rest; a key that already exists is readable only
    //! once the master shows it committed.
    CallOutcome put()
    {
        std::vector<size_t> sizes;
        for (size_t row : mRows)
        {
            sizes.push_back(static_cast<size_t>(mOp.rowBytes[row]));
        }
        auto const starts = metadataCall(StoreCall::kPutStart, "batch_put_session_start", mRows,
            [&](mirror::Keys const& keys)
            { return mClient.batch_put_session_start(keys, sizes, defaultReplicateConfig()); });
        std::vector<size_t> started;
        std::vector<size_t> existing;
        for (size_t i = 0; i < mRows.size(); ++i)
        {
            size_t const row = mRows[i];
            switch (starts[i])
            {
            case mirror::kOk: started.push_back(row); break;
            case mirror::kObjectAlreadyExists: existing.push_back(row); break;
            case mirror::kNoAvailableHandle: mOp.set(row, RowState::kDeclined, starts[i], StoreCall::kPutStart); break;
            // This client already holds the key's session, or the master refused the row likewise.
            case mirror::kInvalidParams: mOp.set(row, RowState::kNotReady, starts[i], StoreCall::kPutStart); break;
            default: mOp.set(row, RowState::kFault, starts[i], StoreCall::kPutStart); break;
            }
        }
        mOutcome.accessEnded = started.empty() || writeAndFinishSessions(started);
        if (!existing.empty())
        {
            confirmExisting(existing);
        }
        return std::move(mOutcome);
    }

private:
    //! Returns whether access ended.
    bool read(std::vector<size_t> const& rows, bool firstRead)
    {
        StoreCall const call = firstRead ? StoreCall::kGet : StoreCall::kReread;
        SdkArguments const sdk = arguments(rows);
        std::vector<int64_t> answers;
        try
        {
            answers = oneAnswerPerKey(
                mClient.batch_get_into_multi_buffers(sdk.keys, sdk.buffers, sdk.sizes, /*prefer_same_node=*/false),
                rows.size());
        }
        catch (std::exception const& error)
        {
            // The transfers may still be writing the rows.
            fail(rows, call, "batch_get_into_multi_buffers", error);
            return false;
        }
        bool ended = true;
        std::vector<size_t> expired;
        for (size_t i = 0; i < rows.size(); ++i)
        {
            size_t const row = rows[i];
            int64_t const answer = answers[i];
            RowState state = RowState::kFault;
            if (answer == mOp.rowBytes[row])
            {
                state = RowState::kReady;
            }
            else if (firstRead && answer == mirror::kObjectNotFound)
            {
                state = RowState::kMiss;
            }
            else if (firstRead && answer == mirror::kReplicaIsNotReady)
            {
                state = RowState::kNotReady;
            }
            mOp.set(row, state, answer, call);
            // A TRANSFER_FAIL future may give up while its RDMA still runs (mooncake.py).
            ended = ended && answer != mirror::kTransferFail;
            if (firstRead && answer == mirror::kLeaseExpired)
            {
                expired.push_back(row);
            }
        }
        if (!expired.empty())
        {
            // The read lease expired before the transfer finished, so the master may have reused the
            // object's space mid-read: read once more. Anything but the whole object is a fault then.
            bool const rereadEnded = read(expired, /*firstRead=*/false); // whatever `ended` is
            ended = ended && rereadEnded;
        }
        return ended;
    }

    //! Returns whether access ended: the scatter waits, without a timeout, for every fragment it
    //! posted, so its return ends access whatever it answers.
    bool writeAndFinishSessions(std::vector<size_t> const& started)
    {
        SdkArguments const sdk = arguments(started);
        mirror::RowSizes dstOffsets; // a row's buffers fill its object back to back from offset 0
        for (auto const& sizes : sdk.sizes)
        {
            auto& offsets = dstOffsets.emplace_back();
            size_t offset = 0;
            for (size_t size : sizes)
            {
                offsets.push_back(offset);
                offset += size;
            }
        }
        std::vector<int64_t> answers;
        try
        {
            answers = oneAnswerPerKey(
                mClient.batch_put_from_multi_buffer_ranges(sdk.keys, sdk.buffers, sdk.sizes, dstOffsets),
                started.size());
        }
        catch (std::exception const& error)
        {
            // The scatter may still be reading the rows. Neither end (the object may be torn) nor revoke
            // (the master would give its space to another object while the scatter may still write).
            fail(started, StoreCall::kPutRanges, "batch_put_from_multi_buffer_ranges", error);
            mOutcome.sessionsLeftOpen.insert(mOutcome.sessionsLeftOpen.end(), sdk.keys.begin(), sdk.keys.end());
            return false;
        }
        std::vector<size_t> written;
        std::vector<size_t> unwritten;
        for (size_t i = 0; i < started.size(); ++i)
        {
            size_t const row = started[i];
            if (answers[i] == mOp.rowBytes[row])
            {
                written.push_back(row);
            }
            else // its object must never be marked complete
            {
                unwritten.push_back(row);
                mOp.set(row, RowState::kFault, answers[i], StoreCall::kPutRanges);
            }
        }
        std::vector<size_t> const notEnded = endSessions(written);
        unwritten.insert(unwritten.end(), notEnded.begin(), notEnded.end());
        revokeSessions(unwritten);
        return true;
    }

    //! Ends the sessions of rows written whole; returns the rows whose session did not end.
    std::vector<size_t> endSessions(std::vector<size_t> const& written)
    {
        if (written.empty())
        {
            return {};
        }
        auto end = [&](mirror::Keys const& keys) { return mClient.batch_put_session_end(keys); };
        std::vector<int64_t> codes = metadataCall(StoreCall::kPutEnd, "batch_put_session_end", written, end);
        std::vector<size_t> retried;
        std::vector<size_t> retriedAt;
        for (size_t i = 0; i < written.size(); ++i)
        {
            if (codes[i] == mirror::kRpcFail || codes[i] == mirror::kRpcTimeout)
            {
                retried.push_back(written[i]);
                retriedAt.push_back(i);
            }
        }
        if (!retried.empty()) // ending is idempotent on the master
        {
            auto const again = metadataCall(StoreCall::kPutEnd, "batch_put_session_end", retried, end);
            for (size_t j = 0; j < retried.size(); ++j)
            {
                codes[retriedAt[j]] = again[j];
            }
        }
        std::vector<size_t> notEnded;
        for (size_t i = 0; i < written.size(); ++i)
        {
            bool const ended = codes[i] == mirror::kOk;
            mOp.set(written[i], ended ? RowState::kReady : RowState::kFault, codes[i], StoreCall::kPutEnd);
            if (!ended)
            {
                notEnded.push_back(written[i]);
            }
        }
        return notEnded;
    }

    //! A session whose revocation neither succeeded nor found it gone (INVALID_PARAMS) is left open.
    void revokeSessions(std::vector<size_t> const& rows)
    {
        if (rows.empty())
        {
            return;
        }
        auto const codes = metadataCall(StoreCall::kRevoke, "batch_put_session_revoke", rows,
            [&](mirror::Keys const& keys) { return mClient.batch_put_session_revoke(keys); });
        for (size_t i = 0; i < rows.size(); ++i)
        {
            if (codes[i] != mirror::kOk && codes[i] != mirror::kInvalidParams)
            {
                mOutcome.sessionsLeftOpen.push_back(mKeys[rows[i] - mBegin]);
            }
        }
    }

    //! A key that already exists is readable only once the master shows it committed.
    void confirmExisting(std::vector<size_t> const& existing)
    {
        auto const answers = metadataCall(StoreCall::kExistence, "batch_is_exist", existing,
            [&](mirror::Keys const& keys) { return mClient.batchIsExist(keys); });
        for (size_t i = 0; i < existing.size(); ++i)
        {
            RowState state = RowState::kFault;
            if (answers[i] == mirror::kObjectExists)
            {
                state = RowState::kReady;
            }
            else if (answers[i] == mirror::kObjectAbsent)
            {
                state = RowState::kNotReady;
            }
            mOp.set(existing[i], state, answers[i], StoreCall::kExistence);
        }
    }

    //! A session or existence call touches no row memory, so its raise or miscount fails only its keys.
    template <typename SdkCall>
    std::vector<int64_t> metadataCall(
        StoreCall call, char const* name, std::vector<size_t> const& rows, SdkCall&& sdkCall)
    {
        try
        {
            return oneAnswerPerKey(sdkCall(keysOf(rows)), rows.size());
        }
        catch (std::exception const& error)
        {
            mOutcome.errors.emplace_back(call, std::string(name) + " raised " + error.what());
            return std::vector<int64_t>(rows.size(), kCallFailed);
        }
    }

    void fail(std::vector<size_t> const& rows, StoreCall call, char const* name, std::exception const& error)
    {
        for (size_t row : rows)
        {
            mOp.set(row, RowState::kFault, kCallFailed, call);
        }
        mOutcome.errors.emplace_back(call, std::string(name) + " raised " + error.what());
    }

    mirror::Keys keysOf(std::vector<size_t> const& rows) const
    {
        mirror::Keys keys;
        keys.reserve(rows.size());
        for (size_t row : rows)
        {
            keys.push_back(mKeys[row - mBegin]);
        }
        return keys;
    }

    SdkArguments arguments(std::vector<size_t> const& rows) const
    {
        SdkArguments sdk{keysOf(rows), {}, {}};
        for (size_t row : rows)
        {
            appendBuffers(mOp.rows, row, sdk.buffers.emplace_back(), sdk.sizes.emplace_back());
        }
        return sdk;
    }

    mirror::PyClient& mClient;
    Op& mOp;
    size_t mBegin;
    std::vector<size_t> mRows;
    mirror::Keys mKeys; // by row - mBegin
    CallOutcome mOutcome;
};

CallOutcome runCall(mirror::PyClient& client, Call const& call)
{
    try
    {
        CallRun run(client, *call.op, call.begin, call.end);
        return call.op->isPut ? run.put() : run.get();
    }
    catch (...) // a failure of the worker itself: a transfer may have started
    {
        StoreCall const dataCall = call.op->isPut ? StoreCall::kPutRanges : StoreCall::kGet;
        for (size_t row = call.begin; row < call.end; ++row)
        {
            call.op->set(row, RowState::kFault, kCallFailed, dataCall);
        }
        return CallOutcome{false, {{dataCall, "the store worker failed inside a store call"}}, {}};
    }
}

//! Each row's bytes: its spans up to the first zero length, as appendBuffers reads them.
std::vector<int64_t> checkedRowBytes(Rows const& rows)
{
    if (rows.count == 0 || rows.spanColumns == 0)
    {
        throw std::invalid_argument("an operation needs at least one row with a span");
    }
    if (!std::is_sorted(rows.cuts.begin(), rows.cuts.end()))
    {
        throw std::invalid_argument("cuts must be sorted");
    }
    std::vector<int64_t> rowBytes(rows.count);
    for (size_t row = 0; row < rows.count; ++row)
    {
        int64_t total = 0;
        for (size_t column = 0; column < rows.spanColumns && rows.length(row, column) != 0; ++column)
        {
            if (rows.length(row, column) < 0 || rows.starts[row * rows.spanColumns + column] < 0)
            {
                throw std::invalid_argument("row " + std::to_string(row) + " has a negative span");
            }
            total += rows.length(row, column);
        }
        if (total == 0 || total > kMaxRowBytes)
        {
            throw std::invalid_argument("row " + std::to_string(row) + " holds " + std::to_string(total)
                + " bytes; a row holds 1 to " + std::to_string(kMaxRowBytes));
        }
        rowBytes[row] = total;
    }
    return rowBytes;
}

//! Rows [begin, end) of each call: greedy, at most rowsPerCall rows and bytesPerCall bytes, at least
//! one row.
std::vector<std::pair<size_t, size_t>> callRanges(std::vector<int64_t> const& rowBytes, WorkerOptions const& options)
{
    std::vector<std::pair<size_t, size_t>> ranges;
    for (size_t begin = 0; begin < rowBytes.size();)
    {
        size_t end = begin + 1;
        auto bytes = static_cast<size_t>(rowBytes[begin]);
        while (end < rowBytes.size() && end - begin < options.rowsPerCall
            && bytes + static_cast<size_t>(rowBytes[end]) <= options.bytesPerCall)
        {
            bytes += static_cast<size_t>(rowBytes[end]);
            ++end;
        }
        ranges.emplace_back(begin, end);
        begin = end;
    }
    return ranges;
}

} // namespace

struct StoreWorker::State
{
    std::shared_ptr<mirror::PyClient> client;
    WorkerOptions options;
    std::mutex mutex;
    std::condition_variable workReady; // a call may be due, or the worker stops
    std::condition_variable callEnded; // a call returned: drain waits for none to run
    std::deque<Call> gets;
    std::deque<Call> puts;
    std::unordered_map<uint64_t, std::shared_ptr<Op>> open;
    std::deque<std::shared_ptr<Op>> finished; // oldest first, until taken
    uint64_t nextOp{1};
    int running{0};
    int putsRunning{0};
    int getsInARow{0};
    bool drained{false};
    bool stopping{false};

    // A put waits while gets are queued, unless this many gets went in a row, and never takes the last
    // thread for gets; as BlobBackend._pop_next does with batches.
    bool putDue() const
    {
        return !puts.empty() && putsRunning < options.putThreads && (gets.empty() || getsInARow >= options.getsPerPut);
    }

    bool callDue() const
    {
        return putDue() || !gets.empty();
    }

    std::optional<Call> popCall()
    {
        std::deque<Call>* queue = nullptr;
        if (putDue())
        {
            queue = &puts;
            ++putsRunning;
            getsInARow = 0;
        }
        else if (!gets.empty())
        {
            queue = &gets;
            ++getsInARow;
        }
        else
        {
            return std::nullopt;
        }
        Call call = std::move(queue->front());
        queue->pop_front();
        ++running;
        return call;
    }

    void runCalls()
    {
        for (;;)
        {
            Call call;
            std::shared_ptr<mirror::PyClient> callClient; // held for the call: a stop resets `client`
            {
                std::unique_lock lock(mutex);
                workReady.wait(lock, [&] { return stopping || callDue(); });
                std::optional<Call> next = popCall();
                if (!next) // only when stopping: otherwise the wait returns only with a call due
                {
                    return;
                }
                call = std::move(*next);
                callClient = client;
            }
            CallOutcome outcome = runCall(*callClient, call); // never under the mutex
            {
                std::lock_guard lock(mutex);
                finishCall(call, std::move(outcome));
            }
            callEnded.notify_all();
            workReady.notify_one(); // a put's thread may have come free
        }
    }

    void finishCall(Call const& call, CallOutcome outcome)
    {
        --running;
        if (call.op->isPut)
        {
            --putsRunning;
        }
        Op& op = *call.op;
        op.accessEnded = op.accessEnded && outcome.accessEnded;
        for (auto& [storeCall, message] : outcome.errors)
        {
            op.errors.try_emplace(static_cast<int8_t>(storeCall), std::move(message)); // the first one stays
        }
        op.sessionsLeftOpen.insert(op.sessionsLeftOpen.end(), std::make_move_iterator(outcome.sessionsLeftOpen.begin()),
            std::make_move_iterator(outcome.sessionsLeftOpen.end()));
        auto const callRows = op.states.begin();
        bool const faulted = std::any_of(callRows + call.begin, callRows + call.end,
            [](int8_t state) { return state == static_cast<int8_t>(RowState::kFault); });
        if (--op.callsLeft == 0)
        {
            finishOp(call.op);
        }
        else if (faulted)
        {
            // A fault fails the operation: its calls not started yet are dropped, as BlobBackend stops
            // an attempt at its first fault. The operation finishes once no call of it runs.
            cancelQueued(&op);
        }
    }

    void finishOp(std::shared_ptr<Op> const& op)
    {
        op->finishedAt = Clock::now();
        open.erase(op->id);
        finished.push_back(op);
    }

    //! Removes the queued calls of `onlyOp` (nullptr: of every operation), concluding their rows CANCELLED.
    void cancelQueued(Op const* onlyOp)
    {
        for (auto* queue : {&gets, &puts})
        {
            for (auto call = queue->begin(); call != queue->end();)
            {
                if (onlyOp != nullptr && call->op.get() != onlyOp)
                {
                    ++call;
                    continue;
                }
                for (size_t row = call->begin; row < call->end; ++row)
                {
                    call->op->set(row, RowState::kCancelled, /*code=*/0, StoreCall::kNone);
                }
                if (--call->op->callsLeft == 0)
                {
                    finishOp(call->op);
                }
                call = queue->erase(call);
            }
        }
    }
};

StoreWorker::StoreWorker(std::shared_ptr<mirror::PyClient> client, WorkerOptions const& options)
    : mState(std::make_shared<State>())
{
    if (options.threads < 1 || options.putThreads < 1 || options.getsPerPut < 1 || options.rowsPerCall < 1
        || options.bytesPerCall < 1)
    {
        throw std::invalid_argument("store worker options must be positive");
    }
    mState->client = std::move(client);
    mState->options = options;
    try
    {
        for (int i = 0; i < options.threads; ++i)
        {
            mThreads.emplace_back([state = mState] { state->runCalls(); });
            std::string const name = "mc-store-" + std::to_string(i);
            pthread_setname_np(mThreads.back().native_handle(), name.c_str());
        }
    }
    catch (...) // no destructor runs for a constructor that throws: end the threads already started
    {
        stop(0.0);
        throw;
    }
}

StoreWorker::~StoreWorker()
{
    stop(0.0);
}

uint64_t StoreWorker::submit(bool isPut, Rows rows)
{
    auto op = std::make_shared<Op>();
    op->isPut = isPut;
    op->rowBytes = checkedRowBytes(rows);
    op->states.assign(rows.count, static_cast<int8_t>(RowState::kPending));
    op->codes.assign(rows.count, 0);
    op->calls.assign(rows.count, static_cast<int8_t>(StoreCall::kNone));
    op->rows = std::move(rows);
    auto const ranges = callRanges(op->rowBytes, mState->options);
    op->callsLeft = ranges.size();
    {
        std::lock_guard lock(mState->mutex);
        if (mState->drained)
        {
            throw std::runtime_error("the store worker is drained");
        }
        op->id = mState->nextOp++;
        mState->open.emplace(op->id, op);
        auto& queue = isPut ? mState->puts : mState->gets;
        for (auto const& [begin, end] : ranges)
        {
            queue.push_back(Call{op, begin, end});
        }
    }
    mState->workReady.notify_all();
    return op->id;
}

void StoreWorker::cancel(uint64_t op)
{
    std::lock_guard lock(mState->mutex);
    auto const found = mState->open.find(op);
    if (found != mState->open.end())
    {
        mState->cancelQueued(found->second.get());
    }
}

std::vector<FinishedOp> StoreWorker::takeFinished()
{
    std::deque<std::shared_ptr<Op>> taken;
    {
        std::lock_guard lock(mState->mutex);
        taken.swap(mState->finished);
    }
    std::vector<FinishedOp> finished;
    finished.reserve(taken.size());
    for (auto& op : taken) // finished: no thread touches it any more
    {
        finished.push_back(FinishedOp{op->id, std::move(op->states), std::move(op->codes), std::move(op->calls),
            op->accessEnded, std::move(op->errors), std::move(op->sessionsLeftOpen)});
    }
    return finished;
}

double StoreWorker::finishedWaitingS()
{
    std::lock_guard lock(mState->mutex);
    if (mState->finished.empty())
    {
        return 0.0;
    }
    return std::chrono::duration<double>(Clock::now() - mState->finished.front()->finishedAt).count();
}

bool StoreWorker::drain(double timeoutS)
{
    State& state = *mState;
    std::unique_lock lock(state.mutex);
    state.drained = true;
    state.cancelQueued(/*onlyOp=*/nullptr);
    std::chrono::duration<double> const wait(timeoutS < kLongestWaitS ? timeoutS : kLongestWaitS); // NaN: the cap
    state.callEnded.wait_for(lock, wait, [&] { return state.running == 0; });
    return state.running == 0;
}

void StoreWorker::stop(double timeoutS)
{
    bool const quiet = drain(timeoutS);
    {
        std::lock_guard lock(mState->mutex);
        mState->stopping = true;
    }
    mState->workReady.notify_all();
    for (auto& thread : mThreads)
    {
        // A thread still in a store call holds its own client reference and the shared state; idle
        // threads leave at once.
        if (quiet)
        {
            thread.join();
        }
        else
        {
            thread.detach();
        }
    }
    mThreads.clear();
    // Dropped after the mutex is released: never take the GIL under the mutex, since a Python thread
    // holding the GIL may be waiting for it (submit, takeFinished). A test client's destructor takes it.
    std::shared_ptr<mirror::PyClient> lastReference;
    {
        std::lock_guard lock(mState->mutex);
        lastReference.swap(mState->client);
    }
}

} // namespace tensorrt_llm::mooncake_store_worker

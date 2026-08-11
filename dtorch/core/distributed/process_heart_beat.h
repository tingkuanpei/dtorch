/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "dtorch/common/debug.h"
#include "dtorch/common/utilities.h"
#include "dtorch/external/rpc/heart_beat_interface.h"

namespace dtorch {
namespace core {

class ProcessHeartBeatBase {
public:
    ProcessHeartBeatBase() : mMutex(), mCv(), mStopPoll(false) {}

    virtual ~ProcessHeartBeatBase() = default;

    DTORCH_FORCEINLINE void NotifyStopPoll() noexcept {
        {
            std::unique_lock<std::mutex> lock(mMutex);
            mStopPoll = true;
        }
        mCv.notify_all();
    }

    virtual void RegisterWorkerProcess(const std::string& workerAddress) = 0;

    virtual void UnregisterWorkerProcess(const std::string& workerAddress) = 0;

    DTORCH_FORCEINLINE void WaitForStopPoll(int64_t timeoutMillisecond = 4000) {
        std::unique_lock<std::mutex> lock(mMutex);
        mCv.wait_for(lock, std::chrono::milliseconds(timeoutMillisecond), [this] { return mStopPoll == true; });
    }

protected:
    std::mutex mMutex;
    std::condition_variable mCv;
    std::atomic_bool mStopPoll;
};

class MainProcessHeartBeat : public ProcessHeartBeatBase {
public:
    MainProcessHeartBeat(const std::string& mainAddress);

    ~MainProcessHeartBeat();

    DTORCH_FORCEINLINE const std::string& GetMainAddress() const noexcept { return mMainAddress; }

private:
    void PollWorkerHeartBeatAsyncMain();

    void StopWorkerPoll();

    void RegisterWorkerProcess(const std::string& workerAddress) override;

    void UnregisterWorkerProcess(const std::string& workerAddress) override;

private:
    std::string mMainAddress;
    external::rpc::HeartBeatServer mMainHeartBeatServer;
    std::thread mPollMainHeartBeatThread;
    std::unordered_map<std::string, external::rpc::HeartBeatClient> mWorkerHeartBeatClients;
};

class WorkerProcessHeartBeat : public ProcessHeartBeatBase {
public:
    // `onExit` — optional callback invoked when the polling thread detects
    // that Main has died (or when mStopPoll is set during normal shutdown).
    // The callback runs in the polling thread; the caller must ensure any
    // captured references remain valid until ~WorkerProcessHeartBeat returns.
    WorkerProcessHeartBeat(const std::string& mainAddress, const std::string& thisWorkerAddress,
                           std::function<void()> onExit = nullptr);

    ~WorkerProcessHeartBeat();

private:
    void PollMainHeartBeatAsyncMain();

    DTORCH_FORCEINLINE void RegisterWorkerProcess(const std::string& /*workerAddress*/) override { DUnsupportedImpl(); }

    DTORCH_FORCEINLINE void UnregisterWorkerProcess(const std::string& /*workerAddress*/) override {
        DUnsupportedImpl();
    }

private:
    std::string mMainAddress;
    std::string mThisWorkerAddress;
    external::rpc::HeartBeatServer mWorkerHeartBeatServer;
    std::thread mPollMainHeartBeatThread;
    external::rpc::HeartBeatClient mMainHeartBeatClient;
    std::function<void()> mOnExit;
};

}  // namespace core
}  // namespace dtorch

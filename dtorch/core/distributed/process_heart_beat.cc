/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "process_heart_beat.h"

#include <sstream>
#include <thread>

#include <unistd.h>

#include "dtorch/common/logging.h"

namespace dtorch {
namespace core {

MainProcessHeartBeat::MainProcessHeartBeat(const std::string& mainAddress)
    : mMainAddress(mainAddress),
      mMainHeartBeatServer(mainAddress, *this),
      mPollMainHeartBeatThread(),
      mWorkerHeartBeatClients() {
    mPollMainHeartBeatThread = std::thread(&MainProcessHeartBeat::PollWorkerHeartBeatAsyncMain, this);
}

MainProcessHeartBeat::~MainProcessHeartBeat() {
    NotifyStopPoll();
    if (mPollMainHeartBeatThread.joinable()) {
        mPollMainHeartBeatThread.join();
    }
}

void MainProcessHeartBeat::RegisterWorkerProcess(const std::string& workerAddress) {
    std::unique_lock<std::mutex> lock(mMutex);
    if (mWorkerHeartBeatClients.find(workerAddress) != mWorkerHeartBeatClients.end()) {
        std::stringstream ss;
        ss << "Worker process already registered: " << workerAddress;
        DLogFatal() << ss.str();
    }
    mWorkerHeartBeatClients.emplace(workerAddress, external::rpc::HeartBeatClient(workerAddress));
}

void MainProcessHeartBeat::UnregisterWorkerProcess(const std::string& workerAddress) {
    std::unique_lock<std::mutex> lock(mMutex);
    auto it = mWorkerHeartBeatClients.find(workerAddress);
    if (it == mWorkerHeartBeatClients.end()) {
        std::stringstream ss;
        ss << "Worker process not found when unregistering: " << workerAddress;
        DLogFatal() << ss.str();
    }
    mWorkerHeartBeatClients.erase(it);
}

void MainProcessHeartBeat::PollWorkerHeartBeatAsyncMain() {
    while (true) {
        if (mStopPoll) {
            break;
        }

        bool isAllWorkerBeat = true;
        {
            std::unique_lock<std::mutex> lock(mMutex);
            for (auto& it : mWorkerHeartBeatClients) {
                if (!it.second.IsBeat()) {
                    isAllWorkerBeat = false;
                    break;
                }
            }
        }

        if (!isAllWorkerBeat) {
            // If mStopPoll is already set, the destructor is performing a normal
            // shutdown.  Don't call std::exit(0) — just break and let the
            // destructor's join() complete cleanly.
            if (mStopPoll) {
                break;
            }
            std::stringstream ss;
            ss << "Worker heart beat client is not beat, stop main process now...";
            DLogError() << ss.str();
            StopWorkerPoll();
            std::abort();
        }

        WaitForStopPoll();
    }

    StopWorkerPoll();
}

void MainProcessHeartBeat::StopWorkerPoll() {
    // Collect clients under lock, then fire-and-forget NotifyStopPoll in
    // detached threads to avoid blocking the destructor when workers are dead.
    std::vector<external::rpc::HeartBeatClient> clients;
    {
        std::unique_lock<std::mutex> lock(mMutex);
        for (auto& it : mWorkerHeartBeatClients) {
            clients.push_back(it.second);
        }
    }
    for (auto& client : clients) {
        std::thread([client]() mutable { client.NotifyStopPoll(); }).detach();
    }
}

WorkerProcessHeartBeat::WorkerProcessHeartBeat(const std::string& mainAddress, const std::string& thisWorkerAddress,
                                               std::function<void()> onExit)
    : mMainAddress(mainAddress),
      mThisWorkerAddress(thisWorkerAddress),
      mWorkerHeartBeatServer(thisWorkerAddress, *this),
      mPollMainHeartBeatThread(),
      mMainHeartBeatClient(mainAddress),
      mOnExit(std::move(onExit)) {
    if (!mMainHeartBeatClient.RegisterWorker(thisWorkerAddress)) {
        throw std::runtime_error("Register WorkerProcessHeartBeat to main process failed, main address: " +
                                 mainAddress);
    }
    mPollMainHeartBeatThread = std::thread(&WorkerProcessHeartBeat::PollMainHeartBeatAsyncMain, this);
}

WorkerProcessHeartBeat::~WorkerProcessHeartBeat() {
    mMainHeartBeatClient.UnregisterWorker(mThisWorkerAddress);
    NotifyStopPoll();
    if (mPollMainHeartBeatThread.joinable()) {
        mPollMainHeartBeatThread.join();
    }
}

void WorkerProcessHeartBeat::PollMainHeartBeatAsyncMain() {
    while (true) {
        if (mStopPoll) {
            break;
        }

        if (!mMainHeartBeatClient.IsBeat()) {
            std::stringstream ss;
            ss << "Main server is not beat(" << mMainAddress << "). Stop this process now...";
            DLogError() << ss.str();
            break;
        }

        WaitForStopPoll();
    }

    if (mOnExit) {
        mOnExit();
    }
}

}  // namespace core
}  // namespace dtorch

/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <condition_variable>
#include <mutex>
#include <string>
#include <vector>

#include "dtorch/core/distributed/cluster_info.h"
#include "dtorch/core/distributed/process_heart_beat.h"
#include "dtorch/core/runner/remote/worker_node_multi_graph_node_runner.h"
#include "dtorch/external/rpc/main_node_interface.h"
#include "dtorch/external/rpc/worker_node_interface.h"

namespace dtorch {
namespace core {
namespace distributed {

class WorkerNode {
public:
    WorkerNode(const std::string& mainNodeAddress, const std::string& thisWorkerNodeAddress,
               double grpcInitTimeoutSecond);

    ~WorkerNode();

    DTORCH_FORCEINLINE const std::string& GetMainNodeAddress() const noexcept { return mMainNodeAddress; }

    DTORCH_FORCEINLINE const std::string& GetWorkerNodeAddress() const noexcept { return mWorkerNodeAddress; }

    // The MainNode's heartbeat address, received via SyncClusterConfig. Used by
    // Cluster::GetMainProcessHeartBeatAddress() on the worker so RemoteRunner subprocesses
    // spawned on this node can monitor the MainNode's liveness.
    DTORCH_FORCEINLINE const std::string& GetMainProcessHeartBeatAddress() const noexcept {
        return mMainProcessHeartBeatAddress;
    }

    void WaitUntilGetDestroySignal();

    // Spawn this WorkerNode's RemoteRunner subprocesses for graphId (joining the MainNode's
    // broadcast fabric at publisherAddress/pushPullAddress). The cluster-wide ClusterInfo is
    // installed into the global singleton first, so the subprocesses (spawned below) inherit it.
    // Delegates to mMultiGraphNodeRunner.
    void CreateGraph(uint64_t graphId, const api::cpp::GraphOption& graphOption,
                     const RunnerSupportedDevices& supportedDevices, const ClusterInfo& clusterInfo,
                     const std::string& publisherAddress, const std::string& pushPullAddress);

    void DestroyGraph(uint64_t graphId);

    void SendDestroySignal();

private:
    void SyncGlobalOptionFromMainNode();

    void SyncEnvVarsFromMainNode();

    void SyncClusterConfigFromMainNode();

private:
    using WorkerNodeServer = external::rpc::WorkerNodeServer;
    using MainNodeClient = external::rpc::MainNodeClient;

    std::string mMainNodeAddress;
    std::string mWorkerNodeAddress;
    std::string mMainProcessHeartBeatAddress;
    std::unique_ptr<WorkerNodeServer> mWorkerNodeServer;
    MainNodeClient mMainNodeClient;
    mutable std::mutex mMutex;
    std::condition_variable mCv;
    bool mGetDestroySignal;
    // Owns one PerDeviceProcessNodeRunner per graph (spawning this node's RemoteRunner subprocesses).
    WorkerNodeMultiGraphNodeRunner mMultiGraphNodeRunner;
    std::unique_ptr<WorkerProcessHeartBeat> mWorkerProcessHeartBeat;
};

}  // namespace distributed
}  // namespace core
}  // namespace dtorch

/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "worker_node.h"

#include <cstdlib>

#include "dtorch/api/cpp/global_option.h"
#include "dtorch/common/debug.h"
#include "dtorch/common/ip_address.h"
#include "dtorch/core/communication/global_instance_id.h"
#include "dtorch/external/python/python_gil.h"
#include "dtorch/external/torch/torch_util.h"

namespace dtorch {
namespace core {
namespace distributed {

WorkerNode::WorkerNode(const std::string& mainNodeAddress, const std::string& thisWorkerNodeAddress,
                       double grpcInitTimeoutSecond)
    : mMainNodeAddress(mainNodeAddress),
      mWorkerNodeAddress(thisWorkerNodeAddress),
      mMainProcessHeartBeatAddress(),
      mWorkerNodeServer(),
      mMainNodeClient(mainNodeAddress, grpcInitTimeoutSecond),
      mMutex(),
      mCv(),
      mGetDestroySignal(false),
      mMultiGraphNodeRunner(),
      mWorkerProcessHeartBeat() {
    // Sync config from MainNode FIRST, so mMainProcessHeartBeatAddress (and global option/env)
    // are populated before this node's gRPC server can receive CreateGraph (which spawns
    // RemoteRunner subprocesses that need the heartbeat address). Only then start the server and
    // register — the MainNode proceeds (WaitClusterReady) only after RegisterWorkerNode, by which
    // point this WorkerNode is fully ready to handle graph-lifecycle RPCs.
    SyncGlobalOptionFromMainNode();
    SyncEnvVarsFromMainNode();
    SyncClusterConfigFromMainNode();

    if (core::GlobalOption::GetSingleton().GetDTensorInSameDevice()) {
        throw std::invalid_argument(
            "Cannot start WorkerNode when DTensorInSameDevice is enabled. "
            "DTensorInSameDevice mode emulates multi-GPU on a single device and does not support worker nodes.");
    }

    mWorkerNodeServer = std::make_unique<WorkerNodeServer>(thisWorkerNodeAddress, *this);

    mMainNodeClient.RegisterWorkerNode(thisWorkerNodeAddress,
                                       static_cast<int64_t>(external::torch::TorchUtil::CudaDeviceCount()));
}

void WorkerNode::SyncGlobalOptionFromMainNode() {
    std::string data = mMainNodeClient.SyncGlobalOption();
    GlobalOption::GlobalOptionInitFromString(data);
}

void WorkerNode::SyncEnvVarsFromMainNode() {
    auto envVars = mMainNodeClient.SyncEnvVars();
    for (const auto& [key, optValue] : envVars) {
        if (optValue.has_value()) {
            setenv(key.c_str(), optValue->c_str(), 1);
        } else {
            unsetenv(key.c_str());
        }
    }
}

void WorkerNode::SyncClusterConfigFromMainNode() {
    auto [heartBeatAddress, instanceId] = mMainNodeClient.SyncClusterConfig();

    communication::GlobalCommInstanceId::GetSingleton().SetInstanceId(instanceId);

    // TODO:
    mMainProcessHeartBeatAddress = heartBeatAddress;
    // std::string workerHeartBeatAddress = dtorch::GetValidNodeAddress();
    // mWorkerProcessHeartBeat = std::make_unique<WorkerProcessHeartBeat>(heartBeatAddress, workerHeartBeatAddress,
    //                                                                    [this]() { SendDestroySignal(); });
}

WorkerNode::~WorkerNode() { SendDestroySignal(); }

void WorkerNode::WaitUntilGetDestroySignal() {
    std::unique_lock<std::mutex> lock(mMutex);
    if (this->mGetDestroySignal) {
        return;
    }

    auto scopedRelease = external::python::GetPythonGilScopedRelease();
    mCv.wait(lock, [this] { return this->mGetDestroySignal; });
}

void WorkerNode::SendDestroySignal() {
    std::unique_lock<std::mutex> lock(mMutex);
    this->mGetDestroySignal = true;
    mCv.notify_all();
}

void WorkerNode::CreateGraph(uint64_t graphId, const api::cpp::GraphOption& graphOption,
                             const RunnerSupportedDevices& supportedDevices, const ClusterInfo& clusterInfo,
                             const std::string& publisherAddress, const std::string& pushPullAddress) {
    // Install the cluster-wide ClusterInfo into the global singleton BEFORE spawning the
    // RemoteRunner subprocesses: the launch payload serializes ClusterInfo::GetSingleton(), and
    // the subprocesses use it to validate distributed specs against the true total GPU count.
    ClusterInfo::GetSingleton() = clusterInfo;
    mMultiGraphNodeRunner.CreateGraph(graphId, graphOption, supportedDevices, publisherAddress, pushPullAddress);
}

void WorkerNode::DestroyGraph(uint64_t graphId) { mMultiGraphNodeRunner.DestroyGraph(graphId); }

}  // namespace distributed
}  // namespace core
}  // namespace dtorch

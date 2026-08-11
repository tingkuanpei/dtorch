/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "main_node.h"

#include <cstdlib>
#include <future>

#include "dtorch/common/debug.h"
#include "dtorch/common/ip_address.h"
#include "dtorch/common/logging.h"

namespace dtorch {
namespace core {
namespace distributed {

std::string MainNode::kRefMainNodeAddress = "";

const std::vector<std::string>& MainNode::GetEnvVarsToSync() {
    // Environment variables listed here will be synced from MainNode to WorkerNode via SyncEnvVars RPC.
    // Whether defined or not, each variable is sent; WorkerNode calls setenv/unsetenv accordingly.
    static const std::vector<std::string> vars = {
        "CUDA_SCALE_LAUNCH_QUEUES",
    };
    return vars;
}

void MainNode::SetRefMainNodeAddress(const std::string& address) {
    if (!dtorch::CheckStringIsValidAddress(address)) {
        throw std::invalid_argument("Invalid main node address: " + address);
    }
    kRefMainNodeAddress = address;
}

std::string MainNode::GetRefMainNodeAddress() {
    // Priority 1: kRefMainNodeAddress (set programmatically via SetMainNodeAddress)
    if (!kRefMainNodeAddress.empty()) {
        if (!dtorch::CheckStringIsValidAddress(kRefMainNodeAddress)) {
            throw std::invalid_argument("Invalid kRefMainNodeAddress: " + kRefMainNodeAddress);
        }
        return kRefMainNodeAddress;
    }

    // Priority 2: DTORCH_MAIN_NODE_ADDRESS environment variable
    const char* env = std::getenv("DTORCH_MAIN_NODE_ADDRESS");
    if (env && env[0] != '\0') {
        std::string envAddress(env);
        if (!dtorch::CheckStringIsValidAddress(envAddress)) {
            throw std::invalid_argument("Invalid DTORCH_MAIN_NODE_ADDRESS: " + envAddress);
        }
        return envAddress;
    }

    // Priority 3: auto-detect a valid local address
    return dtorch::GetValidNodeAddress();
}

MainNode::MainNode(const std::string& mainNodeAddress)
    : mMainNodeServer(), mWorkerNodeClients(), mMainProcessHeartBeat(), mMutex(), mCv(), mGraphIdSet() {
    // Populate the global ClusterInfo singleton first, so any incoming RegisterWorkerNode RPC
    // (once the server starts below) observes an initialized cluster.
    ClusterInfo::GetSingleton() = ClusterInfo(mainNodeAddress);

    mMainNodeServer = std::make_unique<MainNodeServer>(mainNodeAddress, *this);

    std::string mainProcessHeartBeatAddress = api::cpp::distributed::Cluster::GetValidNodeAddress();
    mMainProcessHeartBeat = std::make_unique<MainProcessHeartBeat>(mainProcessHeartBeatAddress);
}

MainNode::~MainNode() {
    for (auto& client : mWorkerNodeClients) {
        client.SendDestroySignal();
    }
}

bool MainNode::WaitClusterReady(size_t numNodes, double timeoutSecond) {
    std::unique_lock<std::mutex> lock(mMutex);
    return mCv.wait_for(lock, std::chrono::milliseconds(static_cast<int>(timeoutSecond * 1000)),
                        [numNodes] { return ClusterInfo::GetSingleton().NodeSize() >= numNodes; });
}

int64_t MainNode::NumNodeInCluster() const {
    std::unique_lock<std::mutex> lock(mMutex);
    return ClusterInfo::GetSingleton().NodeSize();
}

const std::string& MainNode::GetMainNodeAddress() const {
    std::unique_lock<std::mutex> lock(mMutex);
    return ClusterInfo::GetSingleton().GetMainNodeRpcAddress();
}

const std::string& MainNode::GetMainProcessHeartBeatAddress() const {
    std::unique_lock<std::mutex> lock(mMutex);
    return mMainProcessHeartBeat->GetMainAddress();
}

void MainNode::CreateGraph(uint64_t graphId, const api::cpp::GraphOption& graphOption,
                           const std::vector<RunnerSupportedDevices>& supportedDevicesForNodes,
                           const std::string& publisherAddress, const std::string& pushPullAddress) {
    std::unique_lock<std::mutex> lock(mMutex);
    DDebugAssert(mGraphIdSet.count(graphId) == 0);
    const ClusterInfo& clusterInfo = ClusterInfo::GetSingleton();
    DDebugAssert(supportedDevicesForNodes.size() == clusterInfo.NodeSize());
    mGraphIdSet.insert(graphId);
    // Node 0 is the MainNode; worker client i corresponds to node i+1. Ship the cluster-wide
    // ClusterInfo to each WorkerNode so its subprocesses validate specs against the real GPU count.
    // Fan out concurrently and wait: each RPC blocks on the worker's subprocess spawn, so parallel
    // turns total latency from sum to ~max. mMutex is held throughout; the completion callbacks
    // only touch their own promise, so blocking on future.get() is safe.
    std::vector<std::future<void>> futures;
    futures.reserve(mWorkerNodeClients.size());
    for (size_t i = 0; i < mWorkerNodeClients.size(); ++i) {
        futures.push_back(mWorkerNodeClients[i].AsyncCreateGraph(graphId, graphOption, supportedDevicesForNodes[i + 1],
                                                                 clusterInfo, publisherAddress, pushPullAddress));
    }
    for (auto& future : futures) {
        future.get();
    }
}

void MainNode::DestroyGraph(uint64_t graphId) {
    std::unique_lock<std::mutex> lock(mMutex);
    DDebugAssert(mGraphIdSet.count(graphId) == 1);
    mGraphIdSet.erase(graphId);

    // Fan out DestroyGraph to all WorkerNodes concurrently, then wait for every one to finish.
    std::vector<std::future<void>> futures;
    futures.reserve(mWorkerNodeClients.size());
    for (auto& client : mWorkerNodeClients) {
        futures.push_back(client.AsyncDestroyGraph(graphId));
    }
    for (auto& future : futures) {
        future.get();
    }
}

void MainNode::AddWorkerNode(const std::string& workerNodeAddress, int gpuCount) {
    std::unique_lock<std::mutex> lock(mMutex);
    DDebugAssertMsg(mGraphIdSet.empty(),
                    "Cannot add worker node after graphs have been created. Workers must join before any Graph is "
                    "used (MainNode::CreateGraph).");
    if (ClusterInfo::GetSingleton().GetDTensorInSameDevice()) {
        throw std::invalid_argument(
            "Cannot add worker nodes when DTensorInSameDevice is enabled. "
            "DTensorInSameDevice mode emulates multi-GPU on a single device and does not support multiple nodes.");
    }
    mWorkerNodeClients.emplace_back(workerNodeAddress);
    AddNodeImp(NodeInfo::GetWorkerNodeInfo(gpuCount));
}

void MainNode::AddNodeImp(const NodeInfo& nodeInfo) {
    NodeInfo tmp = nodeInfo;
    DDebugAssert(nodeInfo.rank == -1 || nodeInfo.rank == static_cast<int64_t>(ClusterInfo::GetSingleton().NodeSize()));
    tmp.rank = ClusterInfo::GetSingleton().NodeSize();

    ClusterInfo::GetSingleton().PushBack(tmp);
    mCv.notify_all();
}

}  // namespace distributed
}  // namespace core
}  // namespace dtorch

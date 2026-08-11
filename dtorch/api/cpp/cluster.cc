/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "cluster.h"

#include <cstdlib>

#include "dtorch/common/argument_parser.h"
#include "dtorch/common/ip_address.h"
#include "dtorch/common/logging.h"
#include "dtorch/core/distributed/main_node.h"
#include "dtorch/core/distributed/worker_node.h"

namespace dtorch {
namespace api {
namespace cpp {
namespace distributed {

std::string NodeTypeToString(NodeType nodeType) {
    static const std::array<std::string, 2> kStringMap = {"main", "worker"};
    static_assert(static_cast<int>(kStringMap.size()) == EnumAsInteger(NodeType::kCount), "NodeType size not equal");

    if (nodeType == NodeType::kCount) {
        DLogError() << "NodeType invalid";
        return "";
    }

    return kStringMap[EnumAsInteger(nodeType)];
}

std::ostream& operator<<(std::ostream& os, NodeType nodeType) {
    os << NodeTypeToString(nodeType);
    return os;
}

std::string Cluster::GetValidNodeAddress() { return dtorch::GetValidNodeAddress(); }

// Cluster
struct Cluster::Impl {
    Impl(NodeType nodeType) : nodeType(nodeType), distributedClusterMainNode(), distributedClusterWorkerNode() {}

    ~Impl() = default;

    DTORCH_API_DISABLE_COPY_AND_MOVE(Impl);

    NodeType nodeType;
    std::unique_ptr<core::distributed::MainNode> distributedClusterMainNode;
    std::unique_ptr<core::distributed::WorkerNode> distributedClusterWorkerNode;
};

Cluster::Cluster() : mImplPtr() {}

Cluster::~Cluster() = default;

Cluster::Impl* Cluster::GetPtr() { return Cluster::GetSingleton().mImplPtr.get(); }

bool Cluster::IsCreate() const noexcept { return mImplPtr != nullptr; }

NodeType Cluster::GetNodeType() const noexcept { return mImplPtr->nodeType; }

const std::string& Cluster::GetMainNodeAddress() const noexcept {
    DDebugAssert(Cluster::GetPtr());
    if (Cluster::GetPtr()->nodeType == NodeType::kMain) {
        return MainNode::GetMainNodeAddress();
    } else {
        return WorkerNode::GetMainNodeAddress();
    }
}

const std::string& Cluster::GetMainProcessHeartBeatAddress() const noexcept {
    DDebugAssert(Cluster::GetPtr());
    if (Cluster::GetPtr()->nodeType == NodeType::kMain) {
        DDebugAssert(Cluster::GetPtr()->distributedClusterMainNode);
        return Cluster::GetPtr()->distributedClusterMainNode->GetMainProcessHeartBeatAddress();
    } else {
        // WorkerNode: the MainNode's heartbeat address was received via SyncClusterConfig.
        DDebugAssert(Cluster::GetPtr()->distributedClusterWorkerNode);
        return Cluster::GetPtr()->distributedClusterWorkerNode->GetMainProcessHeartBeatAddress();
    }
}

// MainNode
void MainNode::SetMainNodeAddress(const std::string& address) {
    if (Cluster::GetPtr()) {
        throw std::invalid_argument("Cluster already start, node type: " +
                                    NodeTypeToString(Cluster::GetPtr()->nodeType));
    }
    core::distributed::MainNode::SetRefMainNodeAddress(address);
}

bool MainNode::Start() {
    if (Cluster::GetPtr()) {
        throw std::invalid_argument("Cluster already start, node type: " +
                                    NodeTypeToString(Cluster::GetPtr()->nodeType));
    }

    std::string mainNodeAddress = core::distributed::MainNode::GetRefMainNodeAddress();

    Cluster::GetSingleton().mImplPtr = std::make_shared<Cluster::Impl>(NodeType::kMain);
    DDebugAssert(Cluster::GetPtr()->distributedClusterMainNode == nullptr);

    try {
        Cluster::GetPtr()->distributedClusterMainNode = std::make_unique<core::distributed::MainNode>(mainNodeAddress);
    } catch (std::exception& e) {
        Cluster::GetSingleton().mImplPtr.reset();
        std::stringstream ss;
        ss << "Main node start error, address: " << mainNodeAddress << ", error message: " << e.what();
        DLogError() << ss.str();
        return false;
    }

    return true;
}

void MainNode::Stop() {
    DDebugAssert(Cluster::GetPtr());
    DDebugAssert(Cluster::GetPtr()->distributedClusterMainNode);
    Cluster::GetSingleton().mImplPtr.reset();
}

bool MainNode::WaitClusterReady(size_t numNodes, double timeoutSecond) {
    DDebugAssert(Cluster::GetPtr());
    DDebugAssert(Cluster::GetPtr()->distributedClusterMainNode);
    return Cluster::GetPtr()->distributedClusterMainNode->WaitClusterReady(numNodes, timeoutSecond);
}

int64_t MainNode::NumNodeInCluster() {
    DDebugAssert(Cluster::GetPtr());
    DDebugAssert(Cluster::GetPtr()->distributedClusterMainNode);
    return Cluster::GetPtr()->distributedClusterMainNode->NumNodeInCluster();
}

const std::string& MainNode::GetMainNodeAddress() {
    DDebugAssert(Cluster::GetPtr());
    DDebugAssert(Cluster::GetPtr()->distributedClusterMainNode);
    return Cluster::GetPtr()->distributedClusterMainNode->GetMainNodeAddress();
}

core::distributed::MainNode* MainNode::Get() noexcept {
    DDebugAssert(Cluster::GetPtr() != nullptr);
    DDebugAssert(Cluster::GetPtr()->distributedClusterMainNode != nullptr);
    return Cluster::GetPtr()->distributedClusterMainNode.get();
}

// WorkerNode
void WorkerNode::Start(const std::string& mainNodeAddress, const std::string& workerNodeAddress, double timeoutSecond) {
    if (Cluster::GetPtr()) {
        throw std::invalid_argument("Cluster already start, node type: " +
                                    NodeTypeToString(Cluster::GetPtr()->nodeType));
    }

    Cluster::GetSingleton().mImplPtr = std::make_shared<Cluster::Impl>(NodeType::kWorker);
    DDebugAssert(Cluster::GetPtr()->distributedClusterWorkerNode == nullptr);

    Cluster::GetPtr()->distributedClusterWorkerNode =
        std::make_unique<core::distributed::WorkerNode>(mainNodeAddress, workerNodeAddress, timeoutSecond);
}

const std::string& WorkerNode::GetMainNodeAddress() {
    DDebugAssert(Cluster::GetPtr());
    DDebugAssert(Cluster::GetPtr()->distributedClusterWorkerNode);
    return Cluster::GetPtr()->distributedClusterWorkerNode->GetMainNodeAddress();
}

const std::string& WorkerNode::GetWorkerNodeAddress() {
    DDebugAssert(Cluster::GetPtr());
    DDebugAssert(Cluster::GetPtr()->distributedClusterWorkerNode);
    return Cluster::GetPtr()->distributedClusterWorkerNode->GetWorkerNodeAddress();
}

void WorkerNode::WaitUntilGetDestroySignal() {
    DDebugAssert(Cluster::GetPtr()->distributedClusterWorkerNode != nullptr);
    Cluster::GetPtr()->distributedClusterWorkerNode->WaitUntilGetDestroySignal();
}

int WorkerNode::ExecMain(const std::vector<std::string>& arguments) {
    // Parse arguments
    auto& parser = ArgumentParser::GetSingleton();
    parser.Init(arguments);

    // --main-node-address (required, fallback to env var)
    std::string mainNodeAddress;
    if (parser.HasOption("main-node-address")) {
        mainNodeAddress = parser.OptionValue("main-node-address");
    } else {
        const char* env = std::getenv("DTORCH_MAIN_NODE_ADDRESS");
        if (env) {
            mainNodeAddress = env;
        }
    }
    if (mainNodeAddress.empty()) {
        DLogError() << "ExecMain: --main-node-address is required. "
                    << "Set DTORCH_MAIN_NODE_ADDRESS or pass --main-node-address=<addr>";
        return 1;
    }

    // --this-node-address (optional, fallback to env var, then auto-detect)
    std::string thisNodeAddress;
    if (parser.HasOption("this-node-address")) {
        thisNodeAddress = parser.OptionValue("this-node-address");
    } else {
        const char* env = std::getenv("DTORCH_THIS_WORKER_NODE_ADDRESS");
        if (env) {
            thisNodeAddress = env;
        }
    }
    if (thisNodeAddress.empty()) {
        thisNodeAddress = GetValidNodeAddress();
    }

    // --timeout-second (optional, default 600)
    double timeoutSecond = 600.0;
    if (parser.HasOption("timeout-second")) {
        std::string val = parser.OptionValue("timeout-second");
        timeoutSecond = std::stod(val);
    }

    // Start worker node
    try {
        Start(mainNodeAddress, thisNodeAddress, timeoutSecond);
    } catch (std::exception& e) {
        DLogError() << "ExecMain: WorkerNode::Start failed (main: " << mainNodeAddress
                    << ", this worker: " << thisNodeAddress << ", timeout: " << timeoutSecond
                    << "), error message: " << e.what();
        return 1;
    }

    DLogInfo() << "WorkerNode started successfully, mainNodeAddress: " << mainNodeAddress
               << ", thisNodeAddress: " << thisNodeAddress;

    // Block until destroy signal
    WaitUntilGetDestroySignal();
    return 0;
}

core::distributed::WorkerNode* WorkerNode::Get() noexcept {
    DDebugAssert(Cluster::GetPtr() != nullptr);
    DDebugAssert(Cluster::GetPtr()->distributedClusterWorkerNode != nullptr);
    return Cluster::GetPtr()->distributedClusterWorkerNode.get();
}

}  // namespace distributed
}  // namespace cpp
}  // namespace api
}  // namespace dtorch

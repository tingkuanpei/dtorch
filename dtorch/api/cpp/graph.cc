/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "graph.h"

#include <torch/torch.h>

#include "dtorch/api/cpp/cluster.h"
#include "dtorch/api/cpp/distributed_spec.h"
#include "dtorch/api/cpp/functional/system_functional.h"
#include "dtorch/api/cpp/global_option.h"
#include "dtorch/common/debug.h"
#include "dtorch/core/distributed/main_node.h"
#include "dtorch/core/global_id_manager.h"
#include "dtorch/core/graph/graph_constructor.h"
#include "dtorch/external/boost/boost_serialization.h"
#include "dtorch/external/zmq/zmq.h"

namespace dtorch {
namespace api {
namespace cpp {

std::string GraphOption::ToString() const {
    std::stringstream ss;
    ss << "GraphOption(perDevicePerProcess: "
       << (perDevicePerProcess.has_value() ? std::to_string(perDevicePerProcess.value()) : "None");
    if (graphId.has_value()) {
        ss << ", graphId: " << graphId.value();
    }
    ss << ")";
    return ss.str();
}

bool GraphOption::operator==(const GraphOption& other) const {
    return perDevicePerProcess == other.perDevicePerProcess;
}

GraphOption GraphOption::UpdateOptionFromEnvironment(const GraphOption& referenceOption) {
    GraphOption option = referenceOption;

    // graphId
    if (option.graphId.has_value()) {
        DLogFatal() << "GraphOption with graphId is not supported in the constructor of Graph";
    } else {
        option.graphId = core::GraphIdManager::GetSingleton().GetUniqueId();
    }

    // perDevicePerProcess
    if (!option.perDevicePerProcess.has_value()) {
        option.perDevicePerProcess = core::GlobalOption::GetSingleton().GetPerDevicePerProcess();
    }

    return option;
}

std::string GraphOption::SerializeToString() const {
    std::ostringstream oss;
    boost::archive::text_oarchive oa(oss);
    oa << *this;
    return oss.str();
}

GraphOption GraphOption::FromString(const std::string& serializedData) {
    std::istringstream iss(serializedData);
    boost::archive::text_iarchive ia(iss);
    GraphOption option;
    ia >> option;
    return option;
}

struct Graph::Impl {
    Impl(const GraphOption& graphOption)
        : graphOption(graphOption),
          isInited(false),
          constructor(),
          defaultDeviceMesh(Device::GetDefaultCpuDevice()),
          defaultDataKind(DataKind::kFloat32) {}

    // Lazy initialization: defer heavy construction (MainNode::Start(), GraphConstructor creation, etc.)
    // to the first real use instead of at Python import time. This prevents Graph.default_graph() from
    // triggering distributed runtime initialization when used as a default argument value in function signatures.
    void EnsureInitialized() {
        if (isInited) {
            return;
        }
        graphOption = GraphOption::UpdateOptionFromEnvironment(graphOption);

        if (!distributed::Cluster::GetSingleton().IsCreate()) {
            distributed::MainNode::Start();
        }
        DAlwaysAssert(distributed::Cluster::GetSingleton().IsCreate());
        DAlwaysAssert(distributed::Cluster::GetSingleton().GetNodeType() == distributed::NodeType::kMain);

        const core::distributed::ClusterInfo& clusterInfo = core::distributed::ClusterInfo::GetSingleton();

        // Per-graph ZMQ broadcast addresses. Each graph owns its own RemoteRunnerPublisher (PUB) /
        // RemoteRunnerPuller (PULL), so the addresses are generated here and shared with WorkerNodes
        // via CreateGraph. They are only needed when there are per-device subprocesses or >1 node.
        std::string publisherAddress;
        std::string pushPullAddress;
        if (graphOption.perDevicePerProcess.value() || clusterInfo.NodeSize() > 1) {
            publisherAddress = external::zmq::GetRandomZmqIpcAddress();
            pushPullAddress = external::zmq::GetRandomZmqIpcAddress();
        }

        // Construct the GraphConstructor (which builds the EagerGraphExecutor) BEFORE
        // MainNode::CreateGraph. The executor BINDs the PUB/PULL fabric in its constructor, so it
        // must exist before WorkerNodes spawn: each worker's RemoteRunner blocks in its subscriber
        // until ZMQ_EVENT_CONNECTED fires against this PUB, then reports device-"ready" on the PULL.
        // WaitAllRunnerReady is deferred to after CreateGraph (below) so workers exist to report.
        constructor = std::make_unique<core::GraphConstructor>(graphOption, publisherAddress, pushPullAddress);

        // Notify WorkerNodes to spawn their RemoteRunner subprocesses for this graph. Their
        // subscribers connect to the already-bound PUB; their "ready" notifications land on the PULL.
        distributed::MainNode::Get()->CreateGraph(graphOption.graphId.value(), graphOption,
                                                  clusterInfo.GetRunnerSupportedDevicesForNode(), publisherAddress,
                                                  pushPullAddress);

        // Now that every WorkerNode has spawned its runners, drain their "ready" notifications.
        constructor->WaitAllRunnerReady();
        isInited = true;
    }

    ~Impl() {
        if (!isInited) {
            return;
        }
        // Shut down the EagerGraphExecutor (SendDestroy to this node's subprocesses via the
        // publisher + join the async thread). WorkerNode notification is done explicitly in
        // Graph::Destroy() (which must run while WorkerNodes are still alive); ~Impl() only runs
        // best-effort executor cleanup so it never reaches a possibly-dead WorkerNode over gRPC.
        constructor.reset();
        isInited = false;
    }

    DTORCH_API_DISABLE_COPY_AND_MOVE(Impl);

    GraphOption graphOption;
    bool isInited;
    std::unique_ptr<core::GraphConstructor> constructor;
    DeviceMesh defaultDeviceMesh;
    DataKind defaultDataKind;
};

Graph::Graph(const GraphOption& graphOption) : mImplPtr(std::make_shared<Impl>(graphOption)) {}

Graph::~Graph() = default;

void Graph::Destroy() {
    DDebugAssert(mImplPtr != nullptr);
    if (mImplPtr->isInited) {
        // Notify WorkerNodes to destroy their runners for this graph. Must be called while
        // WorkerNodes are still alive (before MainNode::Stop()): the gRPC aborts the process on
        // failure (DLogFatal), so guard against a torn-down cluster.
        if (distributed::Cluster::GetSingleton().IsCreate() &&
            distributed::Cluster::GetSingleton().GetNodeType() == distributed::NodeType::kMain) {
            distributed::MainNode::Get()->DestroyGraph(mImplPtr->graphOption.graphId.value());
        }
        mImplPtr->isInited = false;
        mImplPtr->constructor.reset();
    }
    mImplPtr.reset();
}

VoidFutureCollect Graph::SyncFuture() {
    DDebugAssert(mImplPtr != nullptr);
    mImplPtr->EnsureInitialized();
    DDebugAssert(mImplPtr->constructor != nullptr);
    return functional::_Sync(*this);
}

void Graph::Sync() { SyncFuture().Get(); }

core::GraphConstructor* Graph::GetGraphConstructor() const noexcept {
    DDebugAssert(mImplPtr != nullptr);
    mImplPtr->EnsureInitialized();
    DDebugAssert(mImplPtr->constructor != nullptr);
    return mImplPtr->constructor.get();
}

const core::distributed::ClusterInfo& Graph::GetClusterInfo() const noexcept {
    DDebugAssert(mImplPtr != nullptr);
    mImplPtr->EnsureInitialized();
    return core::distributed::ClusterInfo::GetSingleton();
}

uint64_t Graph::GetId() const noexcept {
    DDebugAssert(mImplPtr != nullptr);
    mImplPtr->EnsureInitialized();
    DDebugAssert(mImplPtr->graphOption.graphId.has_value());
    DDebugAssert(mImplPtr->graphOption.graphId.value() != core::GraphIdManager::kNoValue);
    return mImplPtr->graphOption.graphId.value();
}

void Graph::SetName(const std::string& graphName) {
    DDebugAssert(mImplPtr != nullptr);
    mImplPtr->EnsureInitialized();
    DDebugAssert(mImplPtr->constructor != nullptr);
    mImplPtr->constructor->SetGraphName(graphName);
}

const std::string& Graph::GetName() {
    DDebugAssert(mImplPtr != nullptr);
    mImplPtr->EnsureInitialized();
    DDebugAssert(mImplPtr->constructor != nullptr);
    return mImplPtr->constructor->GetGraphName();
}

Graph Graph::GetDefaultThreadLocalGraph() {
    static thread_local Graph defaultGraph;
    return defaultGraph;
}

void Graph::SetDefaultDeviceMesh(const DeviceMesh& deviceMesh) {
    DDebugAssert(mImplPtr != nullptr);
    mImplPtr->EnsureInitialized();
    DDebugAssert(mImplPtr->constructor != nullptr);
    mImplPtr->defaultDeviceMesh = deviceMesh;
}

const DeviceMesh& Graph::GetDefaultDeviceMesh() const noexcept {
    DDebugAssert(mImplPtr != nullptr);
    mImplPtr->EnsureInitialized();
    DDebugAssert(mImplPtr->constructor != nullptr);
    return mImplPtr->defaultDeviceMesh;
}

bool Graph::Satisfy(const DeviceMesh& deviceMesh) const noexcept {
    DDebugAssert(mImplPtr != nullptr);
    mImplPtr->EnsureInitialized();
    DDebugAssert(mImplPtr->constructor != nullptr);
    return DistributedSpec::CheckDeviceIdValid(
        deviceMesh, core::distributed::ClusterInfo::GetSingleton().GetTotalGpuCount(), false);
}

void Graph::SetDefaultDataKind(DataKind dataKind) {
    DDebugAssert(mImplPtr != nullptr);
    mImplPtr->EnsureInitialized();
    DDebugAssert(mImplPtr->constructor != nullptr);
    mImplPtr->defaultDataKind = dataKind;
}

const DataKind& Graph::GetDefaultDataKind() const noexcept {
    DDebugAssert(mImplPtr != nullptr);
    mImplPtr->EnsureInitialized();
    DDebugAssert(mImplPtr->constructor != nullptr);
    return mImplPtr->defaultDataKind;
}

const GraphOption& Graph::GetGraphOption() const noexcept {
    DDebugAssert(mImplPtr != nullptr);
    mImplPtr->EnsureInitialized();
    DDebugAssert(mImplPtr->constructor != nullptr);
    return mImplPtr->graphOption;
}

}  // namespace cpp
}  // namespace api
}  // namespace dtorch

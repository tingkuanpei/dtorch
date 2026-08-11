/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <memory>
#include <thread>
#include <utility>

#include "dtorch/core/distributed/cluster_info.h"
#include "dtorch/core/graph/graph_traversal_sequence.h"
#include "dtorch/core/graph/logical_graph.h"
#include "dtorch/core/runner/node_runner_base.h"
#include "dtorch/external/zmq/remote_runner_publisher.h"
#include "dtorch/external/zmq/remote_runner_puller.h"
#include "eager_graph_executor_message.h"

namespace dtorch {
namespace core {

// EagerGraphExecutor 会持有一个不断增加节点（Operator/Operand），同时不断删除节点的
// LogicalGraph(ComputeResource)。当用户调用 api::cpp::functional 或 api::cpp::AutoGradient
// 的接口时，会增加节点。每个新增节点只执行一次，执行完毕后会被销毁。节点的销毁

class EagerGraphExecutor {
public:
    EagerGraphExecutor(const GraphOption& graphOption, const std::string& publisherAddress,
                       const std::string& pushPullAddress);

    ~EagerGraphExecutor();

    // Block until every node's runners report device-"ready" on the PULL socket. MUST be called
    // AFTER MainNode::CreateGraph: the constructor only binds the PUB/PULL fabric and starts the
    // async loop + the MainNode's own runner, so WorkerNodes are not spawned (and cannot report
    // "ready") until CreateGraph fans out to them.
    void WaitAllRunnerReady();

    DTORCH_FORCEINLINE EGEMessageQueue& GetEGEMessageQueue() { return mEGEMessageQueue; }

    void CheckSupportOrThrow(const Operator& op);

    void AddOperator(std::unique_ptr<Operator> op);

    void AddApiTensorNoHoldOperand(const Operand* operand);

    void SetGraphName(const std::string& name);

    void SetOperandName(const Operand* operand, const std::string& name);

private:
    void InitCurrentNode(const GraphOption& graphOption, const distributed::NodeInfo& nodeInfo,
                         const RunnerSupportedDevices& supportedDevices, const std::string& publisherAddress,
                         const std::string& pushPullAddress);

    void AsyncMain();

    void GetEagerGraphExecutorMessage();

    void ExecuteOperatorsAndNoHoldOperands(const GraphTraversalSequence& traversalSeq,
                                           const std::vector<const Operand*>& noHoldOperands);

private:
    // async message
    std::thread mAsyncThread;
    std::atomic_bool mGetDestroySignal;
    EGEMessageQueue mEGEMessageQueue;
    std::vector<const Operator*> mNewAddOperatorSequence;
    std::vector<const Operand*> mNewAddNoHoldOperand;

    LogicalGraph mLogicalGraph;
    GraphOption mGraphOption;
    std::unique_ptr<external::zmq::RemoteRunnerPublisher> mRemoteRunnerPublisher;
    std::unique_ptr<external::zmq::RemoteRunnerPuller> mRemoteRunnerPuller;
    std::vector<std::unique_ptr<NodeRunnerBase>> mNodeRunners;
};

}  // namespace core
}  // namespace dtorch

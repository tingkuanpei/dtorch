/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "eager_graph_executor.h"

#include <cstdint>

#include "dtorch/api/cpp/global_option.h"
#include "dtorch/common/config.h"
#include "dtorch/common/debug.h"
#include "dtorch/common/logging.h"
#include "dtorch/core/runner/per_device_thread_node_runner.h"
#include "dtorch/core/runner/remote/per_device_process_node_runner.h"
#include "dtorch/external/cuda/nvtx_profiler.h"
#include "dtorch/external/zmq/remote_runner_pusher.h"
#include "eager_graph_executor_message_imp.h"

namespace dtorch {
namespace core {

EagerGraphExecutor::EagerGraphExecutor(const GraphOption& graphOption, const std::string& publisherAddress,
                                       const std::string& pushPullAddress)
    : mAsyncThread(),
      mGetDestroySignal(false),
      mEGEMessageQueue(),
      mNewAddOperatorSequence(),
      mNewAddNoHoldOperand(),
      mLogicalGraph(),
      mGraphOption(graphOption),
      mRemoteRunnerPublisher(),
      mRemoteRunnerPuller(),
      mNodeRunners() {
    mAsyncThread = std::thread(&EagerGraphExecutor::AsyncMain, this);

    const distributed::ClusterInfo& clusterInfo = distributed::ClusterInfo::GetSingleton();
    if (core::GlobalOption::GetSingleton().GetDTensorInSameDevice()) {
        DDebugAssert(clusterInfo.NodeSize() == 1);
        DDebugAssert(clusterInfo.GetTotalGpuCount() ==
                     core::GlobalOption::GetSingleton().GetNumGpuWhenEnableDtensorInSameDevice());
    }

    DDebugAssert(graphOption.perDevicePerProcess.has_value());
    // BIND the PUB/PULL fabric here. Graph::Impl constructs this executor (via GraphConstructor)
    // BEFORE MainNode::CreateGraph, so WorkerNode subprocesses spawned during CreateGraph can connect
    // to the already-bound PUB (their subscriber blocks until ZMQ_EVENT_CONNECTED) and report
    // device-"ready" on the PULL. WaitAllRunnerReady() is then called from Graph::Impl AFTER
    // CreateGraph to drain those "ready" notifications.
    if (graphOption.perDevicePerProcess.value() || clusterInfo.NodeSize() > 1) {
        DDebugAssert(!publisherAddress.empty());
        DDebugAssert(!pushPullAddress.empty());
    }
    if (!publisherAddress.empty()) {
        mRemoteRunnerPublisher = std::make_unique<external::zmq::RemoteRunnerPublisher>(publisherAddress);
        mRemoteRunnerPuller = std::make_unique<external::zmq::RemoteRunnerPuller>(pushPullAddress);
    }

    auto supportedDevicesForNodes = clusterInfo.GetRunnerSupportedDevicesForNode();
    DDebugAssert(clusterInfo.NodeSize() > 0);
    DDebugAssert(supportedDevicesForNodes.size() == clusterInfo.NodeSize());

    InitCurrentNode(mGraphOption, clusterInfo.Node(0), supportedDevicesForNodes[0], publisherAddress, pushPullAddress);
}

EagerGraphExecutor::~EagerGraphExecutor() {
    DDebugAssert(!mGetDestroySignal);
    if (mAsyncThread.joinable()) {
        mGetDestroySignal = true;
        mEGEMessageQueue.Destroy();
        mAsyncThread.join();
    }

    // Stop AsyncMain first to avoid concurrent sends on the same ZMQ publisher socket.
    // AsyncMain may call Execute()/Sync(), and SendDestroy() also sends via publisher.
    if (mRemoteRunnerPublisher) {
        mRemoteRunnerPublisher->SendDestroy();
    }
}

void EagerGraphExecutor::CheckSupportOrThrow(const Operator& op) {
    int64_t maxGpuId = op.GetOperatorAssignInfo().MaxGpuId();
    int64_t totalGpuCount = distributed::ClusterInfo::GetSingleton().GetTotalGpuCount();
    if (maxGpuId >= totalGpuCount) {
        std::stringstream ss;
        ss << "DeviceMesh of operator not supported. Max gpu id: " << maxGpuId
           << ". But cluster with number of gpu: " << totalGpuCount << ".";
        if (core::GlobalOption::GetSingleton().GetDTensorInSameDevice()) {
            ss << "You are use dtensorInSameDevice mode. Please set DTORCH_NUM_GPU_WHEN_ENABLE_DTENSOR_IN_SAME_DEVICE "
                  "to larger than "
               << maxGpuId << ".";
        }
        throw std::invalid_argument(ss.str());
    }
}

void EagerGraphExecutor::InitCurrentNode(const GraphOption& graphOption, const distributed::NodeInfo& nodeInfo,
                                         const RunnerSupportedDevices& supportedDevices,
                                         const std::string& publisherAddress, const std::string& pushPullAddress) {
    DDebugAssert(nodeInfo.nodeType == NodeType::kMain);
    DDebugAssert(nodeInfo.rank == 0);
    DDebugAssert(nodeInfo.gpuCount == static_cast<int64_t>(supportedDevices.DeviceSize()));

    std::unique_ptr<NodeRunnerBase> mainNodeRunner = nullptr;
    DDebugAssert(graphOption.perDevicePerProcess.has_value());
    if (graphOption.perDevicePerProcess.value()) {
        mainNodeRunner = std::make_unique<PerDeviceProcessNodeRunner>(graphOption, supportedDevices, publisherAddress,
                                                                      pushPullAddress);
    } else {
        mainNodeRunner = std::make_unique<PerDeviceThreadNodeRunner>(graphOption, supportedDevices);
    }

    DAlwaysAssert(mNodeRunners.size() == 0);
    mNodeRunners.push_back(std::move(mainNodeRunner));
}

void EagerGraphExecutor::WaitAllRunnerReady() {
    if (!mRemoteRunnerPuller) {
        return;
    }

    const std::vector<RunnerSupportedDevices> supportedDevicesForNodes =
        distributed::ClusterInfo::GetSingleton().GetRunnerSupportedDevicesForNode();

    // 1. 收集所有需要等待的设备，使用 DeviceKeySet (已有类型)
    DeviceKeySet expectedDevices;
    if (mGraphOption.perDevicePerProcess.value()) {
        for (const auto& device : supportedDevicesForNodes[0].AllDeviceList()) {
            expectedDevices.insert(DeviceKey::FromDevice(device));
        }
    }
    for (size_t i = 1; i < supportedDevicesForNodes.size(); ++i) {
        for (const auto& device : supportedDevicesForNodes[i].AllDeviceList()) {
            expectedDevices.insert(DeviceKey::FromDevice(device));
        }
    }

    // 2. 轮询直到每个 device 都已 ready
    while (!expectedDevices.empty()) {
        if (mRemoteRunnerPuller->Get()) {
            DAlwaysAssert(mRemoteRunnerPuller->GetMessageType() == external::zmq::RemoteRunnerPusher::kDevicesReadyStr);
            for (const auto& device : mRemoteRunnerPuller->GetReadyDevice()) {
                DeviceKey deviceKey = DeviceKey::FromDevice(device);
                DAlwaysAssert(expectedDevices.count(deviceKey) > 0);
                expectedDevices.erase(deviceKey);
            }
        }
    }
}

void EagerGraphExecutor::AddOperator(std::unique_ptr<Operator> op) {
    DDebugAssert(op != nullptr);

    for (auto& it : op->GetOutputOperands()) {
        mLogicalGraph.AddOperand(it);
    }
    mNewAddOperatorSequence.push_back(op.get());
    mLogicalGraph.AddOperator(std::move(op));
}

void EagerGraphExecutor::AddApiTensorNoHoldOperand(const Operand* operand) { mNewAddNoHoldOperand.push_back(operand); }

void EagerGraphExecutor::SetGraphName(const std::string& name) { mLogicalGraph.SetName(name); }

void EagerGraphExecutor::SetOperandName(const Operand* operand, const std::string& name) {
    DDebugAssert(mLogicalGraph.CountOperand(operand));
    const_cast<core::Operand*>(operand)->SetName(name);
}

void EagerGraphExecutor::AsyncMain() {
    UNvtxNameOsThread("DTorchEagerGraphExecutorThread");

    while (true) {
        // 1. 根据消息队列构建计算图
        // 2. 进行图改写
        // 3. 生成计算资源
        // 4. 执行计算

        GetEagerGraphExecutorMessage();

        if (mGetDestroySignal) {
            break;
        }

        GraphTraversalSequence traversalSeq;
        traversalSeq.FromVec(mNewAddOperatorSequence);
        mNewAddOperatorSequence.clear();

        ExecuteOperatorsAndNoHoldOperands(traversalSeq, mNewAddNoHoldOperand);
        mNewAddNoHoldOperand.clear();
    }
}

void EagerGraphExecutor::GetEagerGraphExecutorMessage() {
    size_t getMsgMaxSize = 100;
    mEGEMessageQueue.ProcessMessages(getMsgMaxSize, *this);
}

void EagerGraphExecutor::ExecuteOperatorsAndNoHoldOperands(const GraphTraversalSequence& traversalSeq,
                                                           const std::vector<const Operand*>& noHoldOperands) {
    if (traversalSeq.Size() + noHoldOperands.size() == 0) {
        return;
    }

    std::vector<std::shared_ptr<Operator>> ops;
    for (auto op : traversalSeq.ToVec()) {
        ops.push_back(mLogicalGraph.GetOperator(op));
        mLogicalGraph.DeleteOperator(op);
    }

    if (mRemoteRunnerPublisher) {
        mRemoteRunnerPublisher->Execute(ops, noHoldOperands);
    }

    for (auto& runner : mNodeRunners) {
        runner->Execute(ops, noHoldOperands);
    }

    for (auto operand : noHoldOperands) {
        mLogicalGraph.DeleteOperand(operand);
    }
}

}  // namespace core
}  // namespace dtorch

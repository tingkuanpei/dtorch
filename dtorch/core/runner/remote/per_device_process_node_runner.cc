/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "per_device_process_node_runner.h"

#include <memory>

#include "dtorch/common/debug.h"
#include "dtorch/common/logging.h"
#include "dtorch/external/rpc/rpc_common.h"

namespace dtorch {
namespace core {

PerDeviceProcessNodeRunner::PerDeviceProcessNodeRunner(const GraphOption& graphOption,
                                                       const RunnerSupportedDevices& supportedDevices,
                                                       const std::string& publisherAddress,
                                                       const std::string& pushPullAddress)
    : mGraphOption(graphOption), mCpuRunner(), mGpuRemoteRunnerMap() {
    bool supportCpu = true;
    mCpuRunner =
        std::make_unique<NaiveRunner>(graphOption, RunnerSupportedDevices(std::vector<DevicePair>(), supportCpu),
                                      communication::TensorStoreConfig(communication::TensorStoreType::kFile));

    for (const auto& devicePair : supportedDevices.AllDevices()) {
        supportCpu = false;
        RunnerSupportedDevices supportedDevices(devicePair, supportCpu);
        DeviceKey globalDeviceKey = DeviceKey::FromDevice(devicePair.globalDevice);
        DAlwaysAssert(mGpuRemoteRunnerMap.count(globalDeviceKey) == 0);
        mGpuRemoteRunnerMap.emplace(
            std::piecewise_construct, std::forward_as_tuple(globalDeviceKey),
            std::forward_as_tuple(graphOption, supportedDevices, publisherAddress, pushPullAddress));
    }

    // Parallel start servers, then wait all servers started to saving server startup time.
    for (auto& it : mGpuRemoteRunnerMap) {
        it.second.WaitSubProcessStarted();
    }
}

PerDeviceProcessNodeRunner::~PerDeviceProcessNodeRunner() {
    // Notify all subprocesses to exit concurrently before clearing the map.
    // Each NotifySubProcessExit() sets the exit flag in shared memory and
    // notifies the condition variable, allowing all GPU subprocesses to begin
    // shutdown in parallel. Without this explicit loop, each subprocess would
    // be notified one at a time during ~RemoteRunnerInProcess() inside clear(),
    // causing sequential (rather than parallel) child process exit.
    for (auto& it : mGpuRemoteRunnerMap) {
        it.second.NotifySubProcessExit();
    }
    // clear() triggers each ~RemoteRunnerInProcess(), whose destructor
    // also calls NotifySubProcessExit() (idempotent), then ~SubProcess()
    // blocks on process->wait() until the child has exited.
    mGpuRemoteRunnerMap.clear();
}

void PerDeviceProcessNodeRunner::Execute(const std::vector<std::shared_ptr<Operator>>& ops,
                                         const std::vector<const Operand*>& noHoldOperands) {
    mCpuRunner->Execute(ops, noHoldOperands);

    // GPU subprocess execution is NOT dispatched per-client here. Instead,
    // EagerGraphExecutor broadcasts Execute messages to all subprocess RemoteRunners
    // via RemoteRunnerPublisher (PUB socket). Each RemoteRunner receives the broadcast
    // through its RemoteRunnerSubscriber (SUB socket) and executes independently.
    // Therefore, we don't call it.second.Execute() for each GPU here.

    // for (auto& it : mGpuRemoteRunnerMap) {
    //     it.second.client.Execute(ops, noHoldOperands);
    // }
}

}  // namespace core
}  // namespace dtorch

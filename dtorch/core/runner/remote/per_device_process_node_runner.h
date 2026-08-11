/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <vector>

#include "dtorch/common/process/sub_process.h"
#include "dtorch/core/runner/naive_runner.h"
#include "dtorch/core/runner/remote/remote_runner_in_process.h"
#include "dtorch/core/runner/runner_supported_devices.h"

namespace dtorch {
namespace core {

// In a distributed cluster, each machine acts as a Node.
// The NodeRunner manages all CPU and GPU resources on a single machine.
class PerDeviceProcessNodeRunner : public NodeRunnerBase {
public:
    PerDeviceProcessNodeRunner(const GraphOption& graphOption, const RunnerSupportedDevices& supportedDevices,
                               const std::string& publisherAddress, const std::string& pushPullAddress);

    ~PerDeviceProcessNodeRunner();

    void Execute(const std::vector<std::shared_ptr<Operator>>& ops,
                 const std::vector<const Operand*>& noHoldOperands) override;

private:
    GraphOption mGraphOption;
    // All CPUs share mCpuRunner
    std::unique_ptr<NaiveRunner> mCpuRunner;
    // Each GPU has an independent runner
    DeviceKeyMap<RemoteRunnerInProcess> mGpuRemoteRunnerMap;
};

}  // namespace core
}  // namespace dtorch

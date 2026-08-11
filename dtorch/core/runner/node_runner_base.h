/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <memory>
#include <vector>

#include "dtorch/common/utilities.h"
#include "dtorch/core/operators/operator.h"

namespace dtorch {
namespace core {

// NodeRunnerBase holds all Runners for an api::cpp::Graph on a single machine (Node).
// It manages execution across all devices on that machine, including the CPU and all GPUs.
//
// There are three concrete subclasses, each targeting a different deployment scenario:
//
// 1. PerDeviceThreadNodeRunner
//      runs on the Main node, each device in a dedicated thread within the same process.
// 2. RemotePerDeviceThreadNodeRunner
//      like PerDeviceThreadNodeRunner, but runs on worker nodes with added inter-node communication.
// 3. PerDeviceProcessNodeRunner
//      each device in its own dedicated subprocess with inter-process communication, used on Main and worker nodes.
class NodeRunnerBase {
public:
    NodeRunnerBase() = default;

    virtual ~NodeRunnerBase() = default;

    DTORCH_DISABLE_COPY_AND_DEFAULT_MOVE(NodeRunnerBase);

    virtual void Execute(const std::vector<std::shared_ptr<Operator>>& ops,
                         const std::vector<const Operand*>& noHoldOperands) = 0;
};

}  // namespace core
}  // namespace dtorch

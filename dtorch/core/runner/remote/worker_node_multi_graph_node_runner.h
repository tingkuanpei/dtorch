/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

#include "dtorch/api/cpp/graph.h"
#include "dtorch/core/runner/remote/per_device_process_node_runner.h"
#include "dtorch/core/runner/runner_supported_devices.h"

namespace dtorch {
namespace core {

// WorkerNodeMultiGraphNodeRunner owns one NodeRunner per Graph on a WorkerNode.
//
// On a WorkerNode, graph execution is driven by the MainNode's ZMQ broadcast:
// the MainNode's EagerGraphExecutor publishes Execute/Destroy messages to all
// RemoteRunner subprocesses (on every node) via RemoteRunnerPublisher. Each
// RemoteRunner subscribes and executes the slices for its own devices.
//
// Therefore this runner's job is only to spawn and own the WorkerNode's
// RemoteRunner subprocesses (one per GPU) for each Graph, so they can join the
// MainNode's broadcast fabric. It is not called with Execute() directly.
class WorkerNodeMultiGraphNodeRunner {
public:
    WorkerNodeMultiGraphNodeRunner() : mRunnerForGraphMap() {}

    ~WorkerNodeMultiGraphNodeRunner() = default;

    DTORCH_DISABLE_COPY_AND_MOVE(WorkerNodeMultiGraphNodeRunner);

    // Create a PerDeviceProcessNodeRunner for graphId. This spawns the WorkerNode's
    // RemoteRunner subprocesses (subscribing to publisherAddress, pushing "ready" to
    // pushPullAddress) and blocks until they have started. Each subprocess obtains the MainNode
    // heartbeat address from Cluster::GetSingleton().GetMainProcessHeartBeatAddress().
    void CreateGraph(uint64_t graphId, const api::cpp::GraphOption& graphOption,
                     const RunnerSupportedDevices& supportedDevices, const std::string& publisherAddress,
                     const std::string& pushPullAddress);

    // Destroy the runner for graphId. Notifies the spawned subprocesses to exit.
    void DestroyGraph(uint64_t graphId);

private:
    // One Runner For One Graph
    std::unordered_map<uint64_t, std::unique_ptr<PerDeviceProcessNodeRunner>> mRunnerForGraphMap;
};

}  // namespace core
}  // namespace dtorch

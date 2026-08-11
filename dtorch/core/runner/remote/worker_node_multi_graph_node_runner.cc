/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "worker_node_multi_graph_node_runner.h"

#include "dtorch/common/debug.h"

namespace dtorch {
namespace core {

void WorkerNodeMultiGraphNodeRunner::CreateGraph(uint64_t graphId, const api::cpp::GraphOption& graphOption,
                                                 const RunnerSupportedDevices& supportedDevices,
                                                 const std::string& publisherAddress,
                                                 const std::string& pushPullAddress) {
    DDebugAssert(mRunnerForGraphMap.count(graphId) == 0);
    mRunnerForGraphMap.emplace(graphId, std::make_unique<PerDeviceProcessNodeRunner>(
                                            graphOption, supportedDevices, publisherAddress, pushPullAddress));
}

void WorkerNodeMultiGraphNodeRunner::DestroyGraph(uint64_t graphId) {
    DDebugAssert(mRunnerForGraphMap.count(graphId) == 1);
    mRunnerForGraphMap.erase(graphId);
}

}  // namespace core
}  // namespace dtorch

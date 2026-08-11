/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <optional>
#include <string>
#include <vector>

#include "dtorch/api/cpp/device.h"
#include "dtorch/api/cpp/graph.h"
#include "dtorch/api/cpp/memory_stats.h"
#include "dtorch/api/cpp/tensor.h"
#include "dtorch/api/cpp/void_future_collect.h"

namespace dtorch {
namespace api {
namespace cpp {
namespace functional {

void _NvtxRangePush(Graph graph, const std::string& message);

void _NvtxRangePop(Graph graph);

void _NvtxMark(Graph graph, const std::string& message);

void _EmptyCache(Graph graph);

MemoryStats _GetMemoryStats(Graph graph, const std::optional<DeviceMesh>& deviceMesh, bool reset_peak = false);

// Async sync all devices (or specified devices). Returns VoidFutureCollect immediately.
// Call VoidFutureCollect::Wait() to block until all devices are synchronized.
// If syncDevices is empty, syncs all CPU and CUDA devices in the graph's cluster.
VoidFutureCollect _Sync(const Graph& graph, const std::optional<std::vector<Device>>& syncDevices = std::nullopt);

}  // namespace functional
}  // namespace cpp
}  // namespace api
}  // namespace dtorch

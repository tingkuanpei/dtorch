/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "system_functional.h"

#include <torch/torch.h>

#include "dtorch/api/cpp/distributed_spec.h"
#include "dtorch/api/cpp/global_option.h"
#include "dtorch/common/debug.h"
#include "dtorch/core/communication/promise_future/void_promise_future.h"
#include "dtorch/core/distributed/cluster_info.h"
#include "dtorch/core/graph/graph_constructor.h"
#include "dtorch/core/operators/system/memory_op.h"
#include "dtorch/core/operators/system/nvtx_op.h"
#include "dtorch/core/operators/system/sync_op.h"

namespace dtorch {
namespace api {
namespace cpp {
namespace functional {

void NvtxOpImpl(Graph graph, core::NvtxType nvtxType, const std::string& message = "") {
    std::optional<Graph> optGraph(graph);

    // Get gpu device mesh
    std::vector<Device> devices = graph.GetClusterInfo().GetGpuDevices();
    std::vector<int64_t> meshVec;
    for (const auto& device : devices) {
        if (device.deviceKind == DeviceKind::kGpu) {
            meshVec.push_back(device.deviceId);
        }
    }
    std::sort(meshVec.begin(), meshVec.end());
    DeviceMesh deviceMesh(DeviceKind::kGpu, meshVec);

    auto param = std::make_unique<core::NvtxParam>(nvtxType, message, deviceMesh);
    auto tensorArray = core::GraphConstructor::AddOperator(std::move(param), api::cpp::TensorArray(), true, optGraph);
    DAlwaysAssert(tensorArray.size() == 0);
}

void _NvtxRangePush(Graph graph, const std::string& message) { NvtxOpImpl(graph, core::NvtxType::kRangePush, message); }

void _NvtxRangePop(Graph graph) { NvtxOpImpl(graph, core::NvtxType::kRangePop); }

void _NvtxMark(Graph graph, const std::string& message) { NvtxOpImpl(graph, core::NvtxType::kMark, message); }

api::cpp::TensorArray MemoryOpImpl(Graph graph, core::MemoryOperationType memoryOperationType,
                                   const std::vector<Device>& devices, bool reset_peak = false) {
    std::optional<Graph> optGraph(graph);
    auto param = std::make_unique<core::MemoryParam>(memoryOperationType, devices, reset_peak);
    return core::GraphConstructor::AddOperator(std::move(param), api::cpp::TensorArray(), true, optGraph);
}

void _EmptyCache(Graph graph) {
    // Torch only supports empty cache on GPU
    std::vector<Device> devices = graph.GetClusterInfo().GetGpuDevices();
    MemoryOpImpl(graph, core::MemoryOperationType::kEmptyCache, devices);
}

MemoryStats _GetMemoryStats(Graph graph, const std::optional<DeviceMesh>& deviceMesh, bool reset_peak) {
    // Torch only supports get memory stats on GPU
    std::vector<Device> devices;
    if (deviceMesh.has_value()) {
        DeviceMesh deviceMeshValue = deviceMesh.value();
        if (deviceMeshValue.GetDeviceKind() != DeviceKind::kGpu) {
            throw std::invalid_argument("Get memory stats is only supported on GPU, get device mesh: " +
                                        deviceMeshValue.ToString());
        }
        devices = deviceMeshValue.ToDeviceVec();
    } else {
        devices = graph.GetClusterInfo().GetGpuDevices();
    }

    auto tensorArray = MemoryOpImpl(graph, core::MemoryOperationType::kMemoryStats, devices, reset_peak);
    DDebugAssert(tensorArray.size() == 1);
    MemoryStats result = core::MemoryOp::ToMemoryStats(tensorArray[0].GetTorchTensor());
    if (core::GlobalOption::GetSingleton().GetDTensorInSameDevice()) {
        result.MergeAsDTensorInSameDevice();
    }
    return result;
}

VoidFutureCollect _Sync(const Graph& graph, const std::optional<std::vector<Device>>& syncDevices) {
    // 1. Determine which devices to sync
    std::vector<Device> devices;
    if (syncDevices.has_value()) {
        devices = syncDevices.value();
    } else {
        // Sync all GPU devices plus CPU
        devices = graph.GetClusterInfo().GetGpuDevices();
        devices.push_back(Device(DeviceKind::kCpu, 0));
    }

    // Remove duplicate devices via DeviceKeySet
    {
        DeviceKeySet deviceKeySet;
        for (const auto& device : devices) {
            deviceKeySet.insert(DeviceKey::FromDevice(device));
        }
        devices.clear();
        for (const auto& key : deviceKeySet) {
            devices.push_back(DeviceKey::ToDevice(key));
        }
    }
    DAlwaysAssert(!devices.empty());

    // 2. Determine Promise type — use the per-graph setting
    bool perDevicePerProcess = graph.GetGraphOption().perDevicePerProcess.value_or(false);
    auto promiseType = core::communication::GetVoidPromiseType(perDevicePerProcess);

    // 3. Create one VoidPromise per device, extract corresponding VoidFuture
    std::vector<std::unique_ptr<core::communication::VoidPromise>> promises;
    VoidFutureCollect futures;

    for (size_t i = 0; i < devices.size(); i++) {
        auto promise = core::communication::CreateVoidPromise(promiseType);
        auto future = promise->GetFuture();
        futures.AddFuture(std::move(future));
        promises.push_back(std::move(promise));
    }

    // 4. Create one SyncOp holding all devices and promises.
    //    SyncOp has 0 inputs, 0 outputs — pass graph explicitly.
    auto param = std::make_unique<core::SyncParam>(std::move(devices), std::move(promises));
    core::GraphConstructor::AddOperator(std::move(param), {}, true, graph);

    // 5. Return VoidFutureCollect
    return futures;
}

}  // namespace functional
}  // namespace cpp
}  // namespace api
}  // namespace dtorch

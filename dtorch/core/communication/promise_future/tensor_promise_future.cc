/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "tensor_promise_future.h"

#include "dtorch/common/debug.h"
#include "dtorch/core/operand.h"
#include "file_tensor_promise_future.h"
#include "memory_tensor_promise_future.h"

namespace dtorch {
namespace core {
namespace communication {

std::unique_ptr<TensorPromise> CreateTensorPromise(TensorPromiseType type) {
    switch (type) {
        case TensorPromiseType::kMemory:
            return std::make_unique<MemoryTensorPromise>();
        case TensorPromiseType::kFile:
            return std::make_unique<FileTensorPromise>();
        case TensorPromiseType::kNetwork:
            DUnimplemented();
            return nullptr;
    }
    DUnimplemented();
    return nullptr;
}

std::unique_ptr<TensorPromise> CreateTensorPromiseFromSerialized(TensorPromiseType type, const std::string& data) {
    switch (type) {
        case TensorPromiseType::kMemory: {
            // Memory mode should never be deserialized (same-process only)
            // DDebugAssert(false && "MemoryTensorPromise does not support deserialization");
            return nullptr;
        }
        case TensorPromiseType::kFile: {
            auto promise = std::make_unique<FileTensorPromise>();
            promise->Deserialize(data);
            return promise;
        }
        case TensorPromiseType::kNetwork:
            DUnimplemented();
            return nullptr;
    }
    DUnimplemented();
    return nullptr;
}

TensorPromiseType GetTensorPromiseTypeFromOperand(const core::Operand& operand, bool perDevicePerProcess) {
    // CPU tensors are always executed in-process via PerDeviceProcessNodeRunner::mCpuRunner (a NaiveRunner),
    // even when perDevicePerProcess is true. Only GPU tensors may run in separate processes.
    if (operand.GetDeviceKind() == DeviceKind::kCpu) {
        return TensorPromiseType::kMemory;
    }
    // GPU tensors: use File (Boost IPC) when perDevicePerProcess is set on the graph option.
    // Note: we use the per-graph setting, NOT GlobalOption, because GlobalOption only reflects the
    // DTORCH_PER_DEVICE_PER_PROCESS environment variable, not per-graph overrides.
    if (perDevicePerProcess) {
        return TensorPromiseType::kFile;
    }
    return TensorPromiseType::kMemory;
}

}  // namespace communication
}  // namespace core
}  // namespace dtorch

/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "memory_kernel.h"

#include <memory>

#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/torch.h>

#include "dtorch/common/debug.h"
#include "dtorch/core/operators/system/memory_op.h"

namespace dtorch {
namespace core {

void MemoryKernel::Compute(const TorchTensorOptArray& /*inputs*/, TorchTensorArray& outputs) {
    // Torch only supports memory operation on GPU
    DDebugAssert(mGlobalDevice.deviceKind == DeviceKind::kGpu);

    const auto& param = GetOpParam<MemoryParam>();
    if (param.memoryOperationType == MemoryOperationType::kEmptyCache) {
        DDebugAssert(torch::cuda::is_available());
        c10::cuda::CUDACachingAllocator::emptyCache();
    } else if (param.memoryOperationType == MemoryOperationType::kMemoryStats) {
        // torch\csrc\cuda\Module.cpp:THCPModule_memoryStats()
        // torch\cuda\memory.py:memory_stats()
        // "all", "small_pool", "large_pool" -> 0, 1, 2
        // memory_allocated = "allocated_bytes.all.current"
        // memory_reserved = "reserved_bytes.all.current"
        // max_memory_allocated = "allocated_bytes.all.peak"
        // max_memory_reserved = "reserved_bytes.all.peak"
        MemoryStat memoryStat;
        c10::CachingDeviceAllocator::DeviceStats stats;
        try {
            stats = c10::cuda::CUDACachingAllocator::getDeviceStats(mLocalDevice.deviceId);
            memoryStat.allocated = stats.allocated_bytes[0].current;
            memoryStat.reserved = stats.reserved_bytes[0].current;
            memoryStat.maxAllocated = stats.allocated_bytes[0].peak;
            memoryStat.maxReserved = stats.reserved_bytes[0].peak;
        } catch (const std::exception& e) {
            DLogWarning() << "Failed to get device stats for device " << mLocalDevice.deviceId << ": " << e.what()
                          << ", PyTorch not support get memory stats before use this cuda device.";
        }
        MemoryStats memoryStats;
        memoryStats.Insert({DeviceKey::FromDevice(mGlobalDevice), memoryStat});

        // Create result tensor on local device
        torch::Tensor tensor = core::MemoryOp::ToTorchTensor(memoryStats);
        auto options = torch::TensorOptions();
        options = options.device(external::torch::TorchUtil::ToDevice(mLocalDevice));
        tensor = tensor.to(options);
        outputs.push_back(tensor);
    } else {
        DUnimplemented();
    }
}

}  // namespace core
}  // namespace dtorch

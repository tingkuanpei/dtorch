/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "memory_op.h"

#include <torch/torch.h>

#include "dtorch/common/debug.h"
#include "dtorch/external/boost/boost_serialization.h"

namespace dtorch {
namespace core {

torch::Tensor MemoryOp::ToTorchTensor(const MemoryStats& memoryStats) {
    DDebugAssert(memoryStats.Size() == 1);
    std::stringstream ss(std::ios::out | std::ios::binary);
    external::boost::BinaryOArchive boa(ss);
    boa << memoryStats;
    std::string tmpString = ss.str();
    DDebugAssert(tmpString.size() < kMemoryStatSize);
    std::vector<char> dataBuffer;
    dataBuffer.insert(dataBuffer.end(), tmpString.begin(), tmpString.end());
    dataBuffer.resize(kMemoryStatSize);
    Shape shape(std::initializer_list<size_t>{static_cast<size_t>(1), kMemoryStatSize});
    return external::torch::TorchUtil::CreateTensor(shape, Device::GetDefaultCpuDevice(), DataKind::kUInt8, dataBuffer);
}

MemoryStats MemoryOp::ToMemoryStats(const torch::Tensor& tensor) {
    MemoryStats result;
    DDebugAssert(tensor.dim() == 2);
    DDebugAssert(tensor.size(1) == kMemoryStatSize);

    auto chunkTensors = tensor.chunk(tensor.size(0), 0);
    for (const auto& chunkTensor : chunkTensors) {
        std::vector<char> dataBuffer = external::torch::TorchUtil::ToCharVec(chunkTensor);
        DDebugAssert(dataBuffer.size() == kMemoryStatSize);
        std::string dataBufferStr(dataBuffer.begin(), dataBuffer.end());
        std::stringstream ss(dataBufferStr, std::ios::in | std::ios::binary);
        external::boost::BinaryIArchive bia(ss);
        MemoryStats memoryStats;
        bia >> memoryStats;
        result.Merge(memoryStats);
    }

    return result;
}

size_t MemoryOp::InferOutputSize() const {
    const auto& param = GetOpParam<MemoryParam>();
    switch (param.memoryOperationType) {
        case MemoryOperationType::kEmptyCache:
            return 0;
        case MemoryOperationType::kMemoryStats:
            return 1;
        default:
            DUnimplemented();
            return 0;
    }
}

void MemoryOp::InferOutputMetaInfo() const {
    DDebugAssert(GetInputSize() == 0);
    const auto& param = GetOpParam<MemoryParam>();
    size_t deviceCount = param.devices.size();

    // Torch only supports memory operation on GPU
    for (const auto& device : param.devices) {
        if (device.deviceKind != DeviceKind::kGpu) {
            throw std::invalid_argument("Memory operation is only supported on GPU");
        }
    }

    if (param.memoryOperationType == MemoryOperationType::kMemoryStats) {
        DDebugAssert(GetOutputSize() == 1);
        DDebugAssert(deviceCount > 0);
        OperandY()->SetShapeAndStride(Shape({deviceCount, kMemoryStatSize}));
        OperandY()->SetDataKind(DataKind::kUInt8);

        DeviceKind deviceKind = param.devices[0].deviceKind;
        std::vector<int64_t> meshVec;
        for (const auto& device : param.devices) {
            DDebugAssert(device.deviceKind == deviceKind);
            meshVec.push_back(device.deviceId);
        }
        DeviceMesh deviceMesh(deviceKind, meshVec);
        PlacementSeq placementSeq({Placement("S0")});
        OperandY()->SetDeviceMeshAndPlacementSeq(deviceMesh, placementSeq);
    } else {
        DDebugAssert(GetOutputSize() == 0);
    }
}

void MemoryOp::InferOperatorAssignInfo() {
    DDebugAssert(mOperatorAssignInfo.NumKernelForThisOp() == 0);

    for (const auto& device : GetOpParam<MemoryParam>().devices) {
        KernelStreamKey streamKey;
        streamKey.Init(device, KernelStreamType::kCompute);
        mOperatorAssignInfo.Insert(streamKey);
    }
}

}  // namespace core
}  // namespace dtorch

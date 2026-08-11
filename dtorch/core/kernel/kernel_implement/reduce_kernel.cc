/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "reduce_kernel.h"

#include <memory>

#include <torch/torch.h>

#include "dtorch/common/debug.h"
#include "dtorch/core/operators/standard/reduce_op.h"

namespace dtorch {
namespace core {

void ReduceKernel::ComputeLocalResult(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) {
    const auto& param = GetOpParam<ReduceParam>();
    DDebugAssert(inputs.size() == 1);
    torch::Tensor output;
    torch::Tensor input = inputs[0].value();

    switch (param.reduceKind) {
        case ReduceKind::kSum: {
            std::optional<at::ScalarType> dtype;
            if (param.dataKind.has_value()) {
                dtype = external::torch::TorchUtil::ToScalarType(param.dataKind.value());
            }
            output = torch::sum(input, param.dim, param.keepdim, dtype);
        } break;
        case ReduceKind::kMean: {
            std::optional<at::ScalarType> dtype;
            if (param.dataKind.has_value()) {
                dtype = external::torch::TorchUtil::ToScalarType(param.dataKind.value());
            }
            output = torch::mean(input, param.dim, param.keepdim, dtype);
        } break;
        case ReduceKind::kAny:
            if (param.dim.size() > 0) {
                output = torch::any(input, param.dim, param.keepdim);
            } else {
                output = torch::any(input);
            }
            break;
        case ReduceKind::kAll:
            if (param.dim.size() > 0) {
                output = torch::all(input, param.dim, param.keepdim);
            } else {
                output = torch::all(input);
            }
            break;
        default:
            DLogError() << "Unsupport param.reduceKind: " << EnumAsInteger(param.reduceKind);
            DUnimplemented();
            break;
    }

    outputs.push_back(output);
}

void ReduceKernel::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) {
    ComputeLocalResult(inputs, outputs);

    const Shape& inShape = mOp->OperandX()->GetShape();
    const auto& param = GetOpParam<ReduceParam>();
    std::unordered_set<size_t> dimSet = param.GetDimSet(inShape);

    std::vector<size_t> commDim;
    const PlacementSeq& placements = mOp->OperandX()->GetPlacementSeq();
    for (size_t i = 0; i < placements.Size(); i++) {
        if (!placements[i].IsShard()) {
            continue;
        }

        int64_t shardIdx = placements[i].GetShardIndex();
        if (dimSet.find(shardIdx) != dimSet.end()) {
            commDim.push_back(i);
        }
    }

    if (commDim.size() == 0) {
        return;
    }

    if (commDim.size() > 1) {
        DUnimplemented();
    }
    communication::ThreadGroupInfo info(mOp->OperandX()->GetDeviceMesh().GetMesh(), commDim[0], mGlobalDevice.deviceId);
    communication::ThreadGroup& threadGroup = mThreadGroupManager->GetThreadGroup(
        mOp->OperandX()->GetDeviceKind(), info.GetAllDeviceIds(), mGlobalDevice.deviceId);
    DeviceStream deviceStream = GetDeviceStream();
    threadGroup.SetStream(deviceStream);

    DDebugAssert(inputs.size() == 1 && outputs.size() == 1);
    std::vector<torch::Tensor> allGatherResult = threadGroup.AllGatherIntoVec(outputs[0], *(mOp->OperandY()));

    for (size_t i = 1; i < allGatherResult.size(); i++) {
        if (param.reduceKind == ReduceKind::kSum || param.reduceKind == ReduceKind::kMean) {
            allGatherResult[0] = allGatherResult[0] + allGatherResult[i];
        } else if (param.reduceKind == ReduceKind::kAll) {
            allGatherResult[0] = torch::logical_and(allGatherResult[0], allGatherResult[i]);
        } else if (param.reduceKind == ReduceKind::kAny) {
            allGatherResult[0] = torch::logical_or(allGatherResult[0], allGatherResult[i]);
        } else {
            DLogError() << "Unsupport param.reduceKind: " << EnumAsInteger(param.reduceKind);
            DUnimplemented();
        }
    }

    if (param.reduceKind == ReduceKind::kMean) {
        allGatherResult[0] = allGatherResult[0] / static_cast<int64_t>(allGatherResult.size());
    }
    outputs[0] = allGatherResult[0];
}

}  // namespace core
}  // namespace dtorch

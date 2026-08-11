/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "create_op.h"

#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>

#include "dtorch/external/torch/torch_util.h"

namespace dtorch {
namespace core {

Shape CreateOp::ComputeShapeForArange(double start, double end, double step) const {
    int64_t size = static_cast<int64_t>(std::ceil((end - start) / step));
    if (size <= 0) {
        std::stringstream ss;
        ss << "Arange operator invalid param: start(" << start << ") end(" << end << ") step (" << step << ")";
        throw std::invalid_argument(ss.str());
    }
    return Shape({size});
}

void CreateOp::InferOutputMetaInfo() const {
    DDebugAssert(GetInputSize() == 0);

    const auto& param = GetOpParam<CreateParam>();
    Shape outShape;
    switch (param.createKind) {
        case CreateKind::kEmpty:
        case CreateKind::kFull:
        case CreateKind::kOnes:
        case CreateKind::kZeros:
        case CreateKind::kRand:
        case CreateKind::kRandn:
        case CreateKind::kRandInt:
            outShape = param.shape;
            break;
        case CreateKind::kArange:
            outShape = ComputeShapeForArange(param.doubleArg0, param.doubleArg1, param.doubleArg2);
            break;
        case CreateKind::kFromTorch:
            outShape = param.shape;
            // Cpu tensor torchValue too large, we don't serialize it, will copy it to
            // PerDeviceProcessNodeRunner::mCpuRunner directly.
            if (param.torchValue.has_value()) {
                DDebugAssert(outShape == external::torch::TorchUtil::GetShape(param.torchValue.value()));
                DDebugAssert(param.dataKind == external::torch::TorchUtil::GetDataKind(param.torchValue.value()));
            }
            break;
        default:
            DUnimplemented();
            break;
    }

    // TODO: add param.generator's device check & add for distributed generator
    OperandY()->SetShapeAndStride(outShape);
    OperandY()->SetDataKind(param.dataKind);
    OperandY()->SetDeviceMeshAndPlacementSeq(param.deviceMesh, param.placementSeq);
}

}  // namespace core
}  // namespace dtorch

/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "view_kernel.h"

#include <torch/torch.h>

#include "dtorch/common/debug.h"
#include "dtorch/core/operators/standard/view_op.h"

namespace dtorch {
namespace core {

void ViewKernel::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) {
    DDebugAssert(inputs.size() == 1);
    if (mOp->GetOpType() == OperatorType::kView) {
        const auto& param = GetOpParam<ViewParam>();
        if (param.shape.has_value()) {
            outputs.push_back(inputs[0].value().view(mOp->OperandY()->GetLocalShape(GetGlobalDeviceId()).Vec()));
        } else {
            outputs.push_back(inputs[0].value());
        }
    } else {
        DDebugAssert(mOp->GetOpType() == OperatorType::kReshape);
        outputs.push_back(torch::reshape(inputs[0].value(), mOp->OperandY()->GetLocalShape(GetGlobalDeviceId()).Vec()));
    }
}

}  // namespace core
}  // namespace dtorch

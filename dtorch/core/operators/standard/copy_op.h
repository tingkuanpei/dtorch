/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <sstream>

#include "../operator.h"
#include "dtorch/common/debug.h"
#include "dtorch/core/operand.h"

namespace dtorch {
namespace core {

using CopyParam = NoElementOpParam<OperatorType::kCopy>;

class CopyOp : public Operator {
public:
public:
    CopyOp(std::shared_ptr<OpParam> opParamPtr) : Operator(opParamPtr) {}

    void CheckInput() const override {
        CheckInputAllDistributedOrNot();
        CheckInputDistributedSpec();
    }

    size_t InferOutputSize() const override { return 0; };

    void InferOutputMetaInfo() const override {
        DDebugAssert(GetInputSize() == 2);
        DDebugAssert(GetOutputSize() == 0);

        DDebugAssert(OperandA()->GetShape() == OperandB()->GetShape());
        DDebugAssert(OperandA()->IsDistributed() == OperandB()->IsDistributed());
        if (OperandA()->IsDistributed()) {
            DDebugAssert(OperandA()->GetDeviceMesh() == OperandB()->GetDeviceMesh());
            DDebugAssert(OperandA()->GetPlacementSeq() == OperandB()->GetPlacementSeq());
        }
    }

    DTORCH_FORCEINLINE void InferOutputDistributedSpecFromPlacementSignature() const override { return; }
};

}  // namespace core
}  // namespace dtorch

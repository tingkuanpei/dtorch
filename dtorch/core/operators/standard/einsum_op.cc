/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "einsum_op.h"

#include "dtorch/common/debug.h"

namespace dtorch {
namespace core {

void EinsumOp::InferOutputMetaInfo() const {
    DDebugAssert(GetInputSize() > 0);
    const auto& param = GetOpParam<EinsumParam>();
    std::stringstream ss;
    ss << "Unsupported equation: " << param.equation;

    if (GetInputSize() > 1) {
        DLogError() << ss.str();
        DUnimplemented();
    }

    size_t arrow_pos = param.equation.find("->");
    if (arrow_pos == std::string::npos) {
        throw std::runtime_error(ss.str());
    }

    const Shape& inputShape = OperandX()->GetShape();
    std::string inputExpr, outputExpr;
    DDebugAssert(arrow_pos != std::string::npos);
    inputExpr = param.equation.substr(0, arrow_pos);
    outputExpr = param.equation.substr(arrow_pos + 2);

    if (OperandX()->IsDistributed()) {
        mPlacementSignatureBuilder = std::make_unique<PlacementSignature::Builder>(GetInputSize(), GetOutputSize());
    }

    std::unordered_map<char, size_t> inputIdxMap;
    for (size_t i = 0; i < inputExpr.size(); i++) {
        if (!std::isalpha(inputExpr[i])) {
            DLogError() << ss.str();
            DUnimplemented();
        }
        if (inputIdxMap.find(inputExpr[i]) != inputIdxMap.end()) {
            DLogError() << ss.str();
            DUnimplemented();
        }
        if (i >= inputShape.NumAxis()) {
            DLogError() << ss.str();
            DUnimplemented();
        }

        inputIdxMap[inputExpr[i]] = i;
    }

    Shape outputShape;
    for (char c : outputExpr) {
        if (!std::isalpha(c)) {
            DLogError() << ss.str();
            DUnimplemented();
        }
        if (inputIdxMap.find(c) == inputIdxMap.end()) {
            DLogError() << ss.str();
            DUnimplemented();
        }
        DDebugAssert(inputIdxMap[c] < inputShape.NumAxis());
        if (mPlacementSignatureBuilder) {
            mPlacementSignatureBuilder->AddInput(Shard(inputIdxMap[c])).AddOutput(Shard(outputShape.NumAxis())).Build();
        }
        outputShape.PushBack(inputShape[inputIdxMap[c]]);
    }

    OperandY()->MetaDataSameAs(OperandX());
    OperandY()->SetShapeAndStride(outputShape);
}

PlacementSignature EinsumOp::GetPlacementSignature() const {
    DDebugAssert(GetInputSize() == 1);
    mPlacementSignatureBuilder->AddInput("P").AddOutput("P").Build();
    return mPlacementSignatureBuilder->Finish();
}

}  // namespace core
}  // namespace dtorch

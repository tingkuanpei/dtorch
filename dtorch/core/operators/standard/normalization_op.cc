/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "normalization_op.h"

#include <cstdint>
#include <stdexcept>

namespace dtorch {
namespace core {

void NormalizationOp::InferOutputMetaInfo() const {
    DDebugAssert(GetInputSize() == 3);
    const auto& param = GetOpParam<NormalizationParam>();
    const Shape& inShape = OperandX()->GetShape();

    if (param.normKind == NormalizationKind::kGroupNorm) {
        if (inShape.NumAxis() <= 2) {
            std::stringstream ss;
            ss << "Input tensor dim MUST larger than 2, but get: " << inShape;
            throw std::invalid_argument(ss.str());
        }

        int64_t numChannels = inShape[1];
        if (numChannels % param.numGroups != 0) {
            throw std::invalid_argument("num_channels must be divisible by num_groups");
        }

        if (!OperandWeight()->IsNullTensorShape() && OperandWeight()->GetShape() != Shape({numChannels})) {
            throw std::invalid_argument("Normalization op InferOutputShape error, invalid weight shape");
        }
        if (!OperandBias()->IsNullTensorShape() && OperandBias()->GetShape() != Shape({numChannels})) {
            throw std::invalid_argument("Normalization op InferOutputShape error, invalid bias shape");
        }

        OperandY()->MetaDataSameAs(OperandX());
    } else {
        const Shape& normShape = param.normalizedShape;
        int shift = static_cast<int>(inShape.NumAxis()) - static_cast<int>(normShape.NumAxis());
        std::stringstream ss;
        ss << "Input shape not compatiale with normalizedShape: " << inShape << " vs " << param.normalizedShape;
        std::string errorMsg = ss.str();

        if (shift <= 0) {
            throw std::invalid_argument(ss.str());
        }
        for (size_t i = 0; i < normShape.NumAxis(); i++) {
            if (inShape[shift + i] != normShape[i]) {
                throw std::invalid_argument(ss.str());
            }
        }

        if (!OperandWeight()->IsNullTensorShape() && param.normalizedShape != OperandWeight()->GetShape()) {
            throw std::invalid_argument("Normalization op InferOutputShape error, invalid weight shape");
        }

        if (param.normKind == NormalizationKind::kRmsNorm) {
            DDebugAssert(OperandBias()->IsNullTensorShape());
        }
        if (!OperandBias()->IsNullTensorShape() && param.normalizedShape != OperandBias()->GetShape()) {
            throw std::invalid_argument("Normalization op InferOutputShape error, invalid bias shape");
        }

        OperandY()->MetaDataSameAs(OperandX());
    }
}

PlacementSignature NormalizationOp::GetPlacementSignature() const {
    DDebugAssert(GetInputSize() == 3);
    const auto& param = GetOpParam<NormalizationParam>();
    PlacementSignature::Builder builder(GetInputSize(), GetOutputSize());
    if (param.normKind == NormalizationKind::kGroupNorm) {
        builder.AddInput("S0").AddOptionalInput("R").AddOptionalInput("R").AddOutput("S0").Build();
    } else {
        // -------------------------
        // |  X  |  W  |  B  |  Y  |
        // -------------------------
        // | S0  |  R  |  R  | S0  |
        size_t numAixs = OperandX()->GetShape().NumAxis() - GetOpParam<NormalizationParam>().normalizedShape.NumAxis();
        for (size_t i = 0; i < numAixs; i++) {
            builder.AddInput(Shard(i)).AddOptionalInput("R").AddOptionalInput("R").AddOutput(Shard(i)).Build();
        }
    }

    return builder.Finish();
}

}  // namespace core
}  // namespace dtorch

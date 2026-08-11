/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "batch_norm_op.h"

namespace dtorch {
namespace core {

void BatchNormOp::InferOutputMetaInfo() const {
    DDebugAssert(GetInputSize() == 5);
    // Check input shape
    const Shape& xShape = OperandX()->GetShape();
    if (xShape.NumAxis() != 4) {
        throw std::invalid_argument("BatchNorm op InferOutputShape error, invalid input shape: " + xShape.ToString());
    }
    OperandY()->MetaDataSameAs(OperandX());

    // Check mean, variance, scale, bias shape
    const auto& param = GetOpParam<BatchNormParam>();
    bool isChannelLast = param.format == OperatorFormat::kNHWC;
    Shape::DataType C;
    if (isChannelLast) {
        C = xShape[3];
    } else {
        C = xShape[1];
    }

    auto& meanShape = OperandB()->GetShape();
    auto& varShape = OperandC()->GetShape();
    if (!OperandB()->IsNullTensorShape() && (meanShape.NumAxis() != 1 || meanShape[0] != C)) {
        throw std::invalid_argument("BatchNorm op InferOutputShape error, invalid mean shape: " + meanShape.ToString());
    }
    if (!OperandC()->IsNullTensorShape() && (varShape.NumAxis() != 1 || varShape[0] != C)) {
        throw std::invalid_argument("BatchNorm op InferOutputShape error, invalid variance shape: " +
                                    varShape.ToString());
    }

    auto& scaleShape = OperandD()->GetShape();
    auto& biasShape = OperandE()->GetShape();
    if (!OperandD()->IsNullTensorShape() && (scaleShape.NumAxis() != 1 || scaleShape[0] != C)) {
        throw std::invalid_argument("BatchNorm op InferOutputShape error, invalid scale shape: " +
                                    scaleShape.ToString());
    }
    if (!OperandE()->IsNullTensorShape() && (biasShape.NumAxis() != 1 || biasShape[0] != C)) {
        throw std::invalid_argument("BatchNorm op InferOutputShape error, invalid bias shape: " + biasShape.ToString());
    }
}

std::string BatchNormOp::GetDescribeString() const {
    const auto& param = GetOpParam<BatchNormParam>();
    std::stringstream ss;
    ss << GetOpType() << ": epsilon: " << param.epsilon << " momentum: " << param.momentum;
    return ss.str();
}

}  // namespace core
}  // namespace dtorch

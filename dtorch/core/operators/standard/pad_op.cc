/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "pad_op.h"

#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "dtorch/common/debug.h"

namespace dtorch {
namespace core {

void PadOp::InferOutputMetaInfo() const {
    DDebugAssert(GetInputSize() == 1);
    const Shape& inShape = OperandX()->GetShape();
    const auto& param = GetOpParam<PadParam>();

    std::unordered_set<std::string> allMode = {"constant", "reflect", "replicate", "circular"};
    if (allMode.find(param.mode) == allMode.end()) {
        throw std::invalid_argument(" Unrecognised padding mode " + param.mode);
    }

    DDebugAssert(param.pad.size() > 0);
    if (param.pad.size() < 2 || param.pad.size() % 2 == 1) {
        throw std::invalid_argument("Padding length must be divisible by 2");
    }

    if (param.pad.size() > inShape.NumAxis() * 2) {
        std::stringstream ss;
        ss << "Padding length should be less than or equal to two times the input dimension but got padding length "
           << param.pad.size() << " and input of dimension " << inShape.NumAxis();
        throw std::invalid_argument(ss.str());
    }

    Shape outShape = inShape;
    for (size_t i = 0; i < inShape.NumAxis(); i++) {
        size_t dim = inShape.NumAxis() - 1 - i;
        int64_t size = inShape[dim];
        if (param.pad.size() >= 2 * (i + 1)) {
            size = size + param.pad[2 * i] + param.pad[2 * i + 1];
        }
        if (size < 0) {
            throw std::invalid_argument("Output tensor's length must be non-negative.");
        }

        outShape[dim] = size;
    }

    OperandY()->MetaDataSameAs(OperandX());
    OperandY()->SetShapeAndStride(outShape);
}

}  // namespace core
}  // namespace dtorch

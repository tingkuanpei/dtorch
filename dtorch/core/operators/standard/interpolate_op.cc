/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "interpolate_op.h"

#include <sstream>
#include <unordered_set>

#include "dtorch/common/debug.h"

namespace dtorch {
namespace core {

void InterpolateOp::InferOutputMetaInfo() const {
    DDebugAssert(GetInputSize() == 1);
    const Shape& inputShape = OperandX()->GetShape();
    Shape outputShape = inputShape;
    const auto& param = GetOpParam<InterpolateParam>();

    if (inputShape.NumAxis() < 3 || inputShape.NumAxis() > 5) {
        std::stringstream ss;
        ss << "Input Error: Only 3D, 4D and 5D input Tensors supported (got 1D) for the modes: "
           << "nearest | linear | bilinear | bicubic | trilinear | area | nearest-exact (got nearest)";
        throw std::invalid_argument(ss.str());
    }

    if (!param.shape.has_value() && !param.scaleFactor.has_value()) {
        throw std::invalid_argument("Either size or scale_factor should be defined");
    }

    if (param.shape.has_value() && param.scaleFactor.has_value()) {
        throw std::invalid_argument("Only one of size or scale_factor should be defined");
    }

    // Check param.scaleFactor
    if (param.scaleFactor.has_value()) {
        size_t scaleFactorSize = param.scaleFactor.value().size();
        if (scaleFactorSize != 1 && scaleFactorSize != inputShape.NumAxis() - 2) {
            std::stringstream ss;
            ss << "Input and scale_factor must have the same number of spatial dimensions, but got input with spatial "
               << "dimensions of " << inputShape.NumAxis() - 2 << " and scale_factor of " << scaleFactorSize
               << ". Please provide input tensor in (N, C, d1, d2, ...,dK) format and scale_factor in (s1, s2, ...,sK) "
               << "format.";
            throw std::invalid_argument(ss.str());
        }
    }

    // Check param.shape
    if (param.shape.has_value() && param.recomputeScaleFactor.has_value()) {
        throw std::invalid_argument("Parameter recompute_scale_factor is not meaningful with an explicit size.");
    }
    if (param.shape.has_value() && inputShape.NumAxis() - 2 != param.shape->NumAxis()) {
        std::stringstream ss;
        ss << "Input and output must have the same number of spatial dimensions, but got input with spatial dimensions"
           << "of " << inputShape.NumAxis() - 2 << " and output size of " << param.shape->NumAxis() << ". Please "
           << "provide input tensor in (N, C, d1, d2, ...,dK) format and output size in (o1, o2, ...,oK) format.";
        throw std::invalid_argument(ss.str());
    }

    // Check mode
    std::unordered_set<std::string> modeSets = {"nearest",   "linear", "bilinear",     "bicubic",
                                                "trilinear", "area",   "nearest-exact"};
    if (modeSets.count(param.mode) == 0) {
        std::stringstream ss;
        ss << "Input Error: Only 3D, 4D and 5D input Tensors supported (got 4D) for the modes: nearest | linear | "
           << "bilinear | bicubic | trilinear | area | nearest-exact (got " << param.mode << ")";
        throw std::invalid_argument(ss.str());
    }
    if (param.mode == "linear" && inputShape.NumAxis() != 3) {
        std::stringstream ss;
        ss << "Got " << inputShape.NumAxis() << "D input, but linear mode needs 3D input";
        throw std::invalid_argument(ss.str());
    } else if ((param.mode == "bilinear" || param.mode == "bicubic") && inputShape.NumAxis() != 4) {
        std::stringstream ss;
        ss << "Got " << inputShape.NumAxis() << "D input, but linear mode needs 4D input";
        throw std::invalid_argument(ss.str());
    } else if ((param.mode == "trilinear" || param.mode == "area" || param.mode == "nearest-exact") &&
               inputShape.NumAxis() != 5) {
        std::stringstream ss;
        ss << "Got " << inputShape.NumAxis() << "D input, but linear mode needs 5D input";
        throw std::invalid_argument(ss.str());
    }

    if (param.shape.has_value()) {
        DDebugAssert(outputShape.NumAxis() - 2 == param.shape->NumAxis());
        for (size_t i = 0; i < param.shape->NumAxis(); i++) {
            outputShape[i + 2] = param.shape->At(static_cast<int>(i));
        }
    } else {
        const auto& scaleFactorVec = param.scaleFactor.value();
        for (size_t i = 0; i < outputShape.NumAxis() - 2; i++) {
            double scaleFactor = 0;
            if (scaleFactorVec.size() == 1) {
                scaleFactor = scaleFactorVec[0];
            } else {
                scaleFactor = scaleFactorVec[i];
            }

            outputShape[i + 2] = outputShape[i + 2] * scaleFactor;
        }
    }

    for (size_t i = 0; i < outputShape.NumAxis(); i++) {
        if (outputShape[i] < 0) {
            std::stringstream ss;
            ss << "Input and output sizes should be greater than 0, but got input " << inputShape << ", output "
               << outputShape;
            throw std::invalid_argument(ss.str());
        }
    }

    OperandY()->MetaDataSameAs(OperandX());
    OperandY()->SetShapeAndStride(outputShape);
}

}  // namespace core
}  // namespace dtorch

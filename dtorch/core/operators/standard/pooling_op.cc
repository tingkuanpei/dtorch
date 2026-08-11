/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "pooling_op.h"

namespace dtorch {
namespace core {

PoolingParam::PoolingParam(PoolingKind poolingKind, bool isGlobalPooling, OperatorFormat format)
    : OpParam(OperatorType::kPooling),
      poolingKind(poolingKind),
      dilations(),
      ceilMode(false),
      kernelSize(),
      paddingType(PaddingType::kNotSet),
      pads(),
      strides(),
      countIncludePad(false),
      isGlobalPooling(true),
      format(format) {
    DDebugAssert(isGlobalPooling);
}

PoolingParam::PoolingParam(PoolingKind poolingKind, const IntOrIntArray& dilations, bool ceilMode,
                           const IntOrIntArray& kernelSize, PaddingType paddingType, const IntOrIntArray& pads,
                           const IntOrIntArray& strides, bool countIncludePad, OperatorFormat format)
    : OpParam(OperatorType::kPooling),
      poolingKind(poolingKind),
      dilations(OpParamUtil::IntOrIntArrayTo2DParam(dilations, "dilations")),
      ceilMode(ceilMode),
      kernelSize(OpParamUtil::IntOrIntArrayTo2DParam(kernelSize, "kernelSize")),
      paddingType(paddingType),
      pads(OpParamUtil::IntOrIntArrayTo2DPad(pads)),
      strides(OpParamUtil::IntOrIntArrayTo2DParam(strides, "strides")),
      countIncludePad(countIncludePad),
      isGlobalPooling(false),
      format(format) {
    if (paddingType == PaddingType::kNotSet && this->pads.empty()) {
        throw std::invalid_argument("You need to set explicit pad, when paddingType == PaddingType::kNotSet");
    } else if ((paddingType == PaddingType::kSame || paddingType == PaddingType::kValid) &&
               !OpParamUtil::IsPadEmptyOrZero(this->pads)) {
        throw std::invalid_argument("You can't set explicit pad, when paddingType == PaddingType::kSame or kValid");
    }

    if (!OpParamUtil::IsPadEmptyOrZero(this->pads) && this->pads.size() != 4) {
        throw std::invalid_argument("Size of pads MUST be 4");
    }

    if (paddingType == PaddingType::kSame || paddingType == PaddingType::kValid) {
        ceilMode = false;
    }
}

Shape::DataType PoolingOp::CalculatePoolingOutput(Shape::DataType input, int64_t kernel, int64_t dilation,
                                                  int64_t stride, int64_t padBefore, int64_t padAfter, bool ceilMode) {
    DDebugAssert(kernel > 0);
    DDebugAssert(dilation > 0);
    DDebugAssert(stride > 0);

    int64_t kernelExtent = (kernel - 1) * dilation;
    Shape::DataType output;

    if (ceilMode) {
        output = (input + padBefore + padAfter - kernelExtent - 1 + stride - 1) / stride + 1;
        // or:
        // std::ceil((input + padBefore + padAfter - kernelExtent - 1) / stride + 1);

        // Ensure that the last pooling starts inside the image needed to avoid problems in ceil mode
        if ((padBefore + padAfter) != 0) {
            if ((output - 1) * stride >= input + padBefore) --output;

            DDebugAssert((output - 1) * stride < input + padBefore);
        }
    } else {
        return (input + padBefore + padAfter - kernelExtent - 1) / stride + 1;
        // or:
        // std::floor((input + padBefore + padAfter - kernelExtent) / stride)
    }

    return output;
}

Shape::DataType PoolingOp::CalculateConvOutputForSamePad(Shape::DataType input, int64_t stride) {
    DDebugAssert(stride > 0);

    return (input + stride - 1) / stride;
    // or:
    // return std::ceil(input / stride)
}

void PoolingOp::CalculatePadForSamePad(Shape::DataType input, Shape::DataType output, int64_t kernel, int64_t dilation,
                                       int64_t stride, int64_t& padBefore, int64_t& padAfter) {
    const int64_t kernelExtent = (kernel - 1) * dilation;

    int64_t paddingNeeded = static_cast<int64_t>((output - 1) * stride + kernelExtent + 1 - input);
    if (paddingNeeded < 0) {
        paddingNeeded = 0;
    }

    padBefore = paddingNeeded / 2;
    padAfter = paddingNeeded - padBefore;
}

void PoolingOp::InferOutputMetaInfo() const {
    DDebugAssert(GetInputSize() == 1);
    auto& param = const_cast<PoolingParam&>(GetOpParam<PoolingParam>());
    bool isChannelLast = param.format == OperatorFormat::kNHWC;

    Operand* in = OperandX();
    Operand* out = OperandY();
    const Shape& inShape = OperandX()->GetShape();
    DDebugAssert(inShape.NumAxis() == 4);
    Shape outShape(inShape.NumAxis());
    Shape::DataType outN = inShape[0];
    Shape::DataType outC = isChannelLast ? inShape[3] : inShape[1];
    Shape::DataType inH = isChannelLast ? inShape[1] : inShape[2];
    Shape::DataType inW = isChannelLast ? inShape[2] : inShape[3];
    Shape::DataType outH = 0, outW = 0;

    if (param.isGlobalPooling) {
        outH = 1;
        outW = 1;

        param.kernelSize.resize(2);
        param.kernelSize[0] = inH;
        param.kernelSize[1] = inW;
        param.strides.resize(2);
        param.strides[0] = 1;
        param.strides[1] = 1;
        param.pads.resize(4);
        param.pads[0] = 0;
        param.pads[1] = 0;
        param.pads[2] = 0;
        param.pads[3] = 0;
        param.dilations.resize(2);
        param.dilations[0] = 1;
        param.dilations[1] = 1;
    } else {
        if (param.paddingType == PaddingType::kNotSet) {
            outH = CalculatePoolingOutput(inH, param.kernelSize[0], param.dilations[0], param.strides[0], param.pads[0],
                                          param.pads[2], param.ceilMode);
            outW = CalculatePoolingOutput(inW, param.kernelSize[1], param.dilations[1], param.strides[1], param.pads[1],
                                          param.pads[1], param.ceilMode);
        } else if (param.paddingType == PaddingType::kValid) {
            outH = CalculatePoolingOutput(inH, param.kernelSize[0], param.dilations[0], param.strides[0], 0, 0, false);
            outW = CalculatePoolingOutput(inW, param.kernelSize[1], param.dilations[1], param.strides[1], 0, 0, false);

            param.pads[0] = 0;
            param.pads[1] = 0;
            param.pads[2] = 0;
            param.pads[3] = 0;
        } else if (param.paddingType == PaddingType::kSame) {
            outH = CalculateConvOutputForSamePad(inH, param.strides[0]);
            outW = CalculateConvOutputForSamePad(inW, param.strides[1]);

            int64_t paddingT = 0, paddingB = 0, paddingL = 0, paddingR = 0;
            PoolingOp::CalculatePadForSamePad(inH, outH, param.kernelSize[0], param.dilations[0], param.strides[0],
                                              paddingT, paddingB);
            PoolingOp::CalculatePadForSamePad(inW, outW, param.kernelSize[1], param.dilations[1], param.strides[1],
                                              paddingL, paddingR);
            param.pads[0] = paddingT;
            param.pads[1] = paddingL;
            param.pads[2] = paddingB;
            param.pads[3] = paddingR;
        } else {
            DUnsupportedImpl();
        }
    }

    if (isChannelLast) {
        outShape[0] = outN;
        outShape[1] = outH;
        outShape[2] = outW;
        outShape[3] = outC;
    } else {
        outShape[0] = outN;
        outShape[1] = outC;
        outShape[2] = outH;
        outShape[3] = outW;
    }

    out->MetaDataSameAs(in);
    out->SetShapeAndStride(outShape);
}

std::string PoolingOp::GetDescribeString() const {
    const auto& param = GetOpParam<PoolingParam>();
    std::stringstream ss;
    ss << GetOpType() << ": kind: " << PoolingKindToString(param.poolingKind);
    return ss.str();
}

}  // namespace core
}  // namespace dtorch

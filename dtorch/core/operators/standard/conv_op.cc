/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "conv_op.h"

namespace dtorch {
namespace core {

ConvParam::ConvParam(const IntOrIntArray& dilations, int64_t group, const IntOrIntArray& kernelSize,
                     PaddingType paddingType, const IntOrIntArray& pads, const IntOrIntArray& strides,
                     OperatorFormat format)
    : OpParam(OperatorType::kConv),
      dilations(OpParamUtil::IntOrIntArrayTo2DParam(dilations, "dilations")),
      group(group),
      kernelSize(OpParamUtil::IntOrIntArrayTo2DParam(kernelSize, "kernelSize")),
      paddingType(paddingType),
      pads(OpParamUtil::IntOrIntArrayTo2DPad(pads)),
      strides(OpParamUtil::IntOrIntArrayTo2DParam(strides, "strides")),
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
}

std::vector<int64_t> ConvParam::GetKernelSize(const Shape& weightShape) {
    DDebugAssert(weightShape.NumAxis() == 4);
    std::vector<int64_t> kernelSize = {static_cast<int64_t>(weightShape[2]), static_cast<int64_t>(weightShape[3])};
    return kernelSize;
}

void ConvParam::Get2DParam(int64_t& dilationH, int64_t& dilationW, int64_t& kernelH, int64_t& kernelW, int64_t& strideH,
                           int64_t& strideW, int64_t& groupSize) const {
    dilationH = this->dilations[0];
    dilationW = this->dilations[1];
    kernelH = this->kernelSize[0];
    kernelW = this->kernelSize[1];
    strideH = this->strides[0];
    strideW = this->strides[1];
    groupSize = this->group;
}

Shape::DataType ConvOp::CalculateConvOutput(Shape::DataType input, Shape::DataType kernel, Shape::DataType dilation,
                                            Shape::DataType stride, Shape::DataType padBefore,
                                            Shape::DataType padAfter) {
    DDebugAssert(kernel > 0);
    DDebugAssert(dilation > 0);
    DDebugAssert(stride > 0);

    Shape::DataType kernelExtent = (kernel - 1) * dilation;
    Shape::DataType result = (input + padBefore + padAfter - kernelExtent - 1) / stride + 1;
    // or：
    // kernelExtent = (kernel - 1) * dilation;
    // result = std::ceil((input + padBefore + padAfter - kernelExtent) / stride)

    DDebugAssert(result > 0);
    return result;
}

Shape::DataType ConvOp::CalculateConvOutputForSamePad(Shape::DataType input, Shape::DataType stride) {
    DDebugAssert(stride > 0);

    return (input + stride - 1) / stride;
    // or:
    // return std::ceil(input / stride)
}

Shape::DataType ConvOp::CalculatePadBeforeForSamePad(Shape::DataType input, Shape::DataType output,
                                                     Shape::DataType kernel, Shape::DataType dilation,
                                                     Shape::DataType stride) {
    const Shape::DataType kernelExtent = (kernel - 1) * dilation;

    Shape::DataType paddingNeeded = (output - 1) * stride + kernelExtent + 1 - input;
    if (paddingNeeded < 0) {
        paddingNeeded = 0;
    }

    return paddingNeeded / 2;
}

void ConvOp::InferOutputMetaInfo() const {
    DDebugAssert(GetInputSize() == 3);
    const auto& param = GetOpParam<ConvParam>();
    bool isChannelLast = param.format == OperatorFormat::kNHWC;

    Operand* in = OperandX();
    Operand* out = OperandY();
    const Shape& inShape = OperandX()->GetShape();
    const Shape& weightShape = OperandWeight()->GetShape();
    DDebugAssert(inShape.NumAxis() == 4);
    DDebugAssert(weightShape.NumAxis() == 4);
    Shape outShape(inShape.NumAxis());
    Shape::DataType outN = inShape[0];
    Shape::DataType outC = weightShape[0];
    Shape::DataType inH = isChannelLast ? inShape[1] : inShape[2];
    Shape::DataType inW = isChannelLast ? inShape[2] : inShape[3];
    Shape::DataType outH = 0, outW = 0;

    if (param.paddingType == PaddingType::kNotSet) {
        outH = CalculateConvOutput(inH, param.kernelSize[0], param.dilations[0], param.strides[0], param.pads[0],
                                   param.pads[2]);
        outW = CalculateConvOutput(inW, param.kernelSize[1], param.dilations[1], param.strides[1], param.pads[1],
                                   param.pads[3]);
    } else if (param.paddingType == PaddingType::kValid) {
        outH = CalculateConvOutput(inH, param.kernelSize[0], param.dilations[0], param.strides[0], 0, 0);
        outW = CalculateConvOutput(inW, param.kernelSize[1], param.dilations[1], param.strides[1], 0, 0);
    } else if (param.paddingType == PaddingType::kSame) {
        outH = CalculateConvOutputForSamePad(inH, param.strides[0]);
        outW = CalculateConvOutputForSamePad(inW, param.strides[1]);
    } else {
        DUnsupportedImpl();
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

    // validate weight shape
    Shape::DataType inC = isChannelLast ? inShape[3] : inShape[1];
    Shape::DataType weightInC = inC;
    Shape::DataType weightH = param.kernelSize[0];
    Shape::DataType weightW = param.kernelSize[1];
    int64_t group = param.group;
    DDebugAssert(group > 0);
    DDebugAssert(group <= outC);
    DDebugAssert(outC % group == 0);
    DDebugAssert(inC % group == 0);

    if (group != 1) {
        weightInC = inC / group;
    }

    if (weightShape[1] != weightInC || weightShape[2] != weightH || weightShape[3] != weightW) {
        throw std::invalid_argument("Conv op InferOutputShape error, invalid param or weight shape");
    }

    // validate bias shape
    if (!OperandBias()->IsNullTensorShape()) {
        const Shape& biasShape = OperandBias()->GetShape();
        if (biasShape.NumAxis() != 1 || biasShape[0] != outC) {
            throw std::invalid_argument("Conv op InferOutputShape error, invalid bias shape");
        }
    }

    out->MetaDataSameAs(in);
    out->SetShapeAndStride(outShape);
}

std::string ConvOp::GetDescribeString() const {
    DDebugAssert(GetInputSize() >= 2);
    const Shape& weightShape = OperandWeight()->GetShape();
    DDebugAssert(weightShape.NumAxis() == 4);
    std::stringstream ss;

    ss << GetOpType() << ": inChannal: " << weightShape[1] << " outChannel: " << weightShape[0]
       << " kernelSize: " << weightShape[2] << ", " << weightShape[3];

    return ss.str();
}

PlacementSignature ConvOp::GetPlacementSignature() const {
    DDebugAssert(GetInputSize() == 3);

    const auto& param = GetOpParam<ConvParam>();
    const Shape& inShape = OperandX()->GetShape();
    DDebugAssert(inShape.NumAxis() == 4);

    PlacementSignature::Builder builder(GetInputSize(), GetOutputSize());
    builder.AddInput(Shard(0)).AddInput("R").AddOptionalInput("R").AddOutput(Shard(0)).Build();
    if (param.pads[0] == 0 && param.pads[1] == 0 && param.dilations[0] == 1 && param.dilations[1] == 1) {
        if (param.kernelSize[0] == param.strides[0]) {
            builder.AddInput(Shard(2)).AddInput("R").AddOptionalInput("R").AddOutput(Shard(2)).Build();
        }
        if (param.kernelSize[1] == param.strides[1]) {
            builder.AddInput(Shard(3)).AddInput("R").AddOptionalInput("R").AddOutput(Shard(3)).Build();
        }
    }
    return builder.Finish();
}

// FLOPs derivation for 2-D convolution.
//
// GetOperatorCost() runs after Infer(), so the output shape is already inferred — read it from OperandY()
// directly instead of redoing the padding/output-size derivation from InferOutputMetaInfo here.
//
// Each of the (outN * outC * outH * outW) output elements is produced by a stencil of
// (inCPerGroup * kH * kW) multiply-accumulates: every input channel within its group (inC/group)
// against the kH×kW kernel window. At 2 FLOPs per MAC that is
//   2 * outN * outC * outH * outW * inCPerGroup * kH * kW.
// outN/outC/outH/outW come from the output operand (channel position depends on NCHW vs NHWC);
// kH/kW come from param.kernelSize; inCPerGroup = inC / group. Bias add is memory-bound, not counted.
OperatorCost ConvOp::GetOperatorCost() const {
    DDebugAssert(GetInputSize() == 3);
    const auto& param = GetOpParam<ConvParam>();
    bool isChannelLast = param.format == OperatorFormat::kNHWC;

    const Shape& inShape = OperandX()->GetShape();
    const Shape& outShape = OperandY()->GetShape();
    DDebugAssert(inShape.NumAxis() == 4);
    DDebugAssert(outShape.NumAxis() == 4);

    int64_t outN = static_cast<int64_t>(outShape[0]);
    int64_t outC = static_cast<int64_t>(isChannelLast ? outShape[3] : outShape[1]);
    int64_t outH = static_cast<int64_t>(isChannelLast ? outShape[1] : outShape[2]);
    int64_t outW = static_cast<int64_t>(isChannelLast ? outShape[2] : outShape[3]);

    int64_t inC = static_cast<int64_t>(isChannelLast ? inShape[3] : inShape[1]);
    int64_t kH = param.kernelSize[0];
    int64_t kW = param.kernelSize[1];
    int64_t inCPerGroup = inC / param.group;  // == weightShape[1]

    return OperatorCost::Compute(2 * outN * outC * outH * outW * inCPerGroup * kH * kW);
}

}  // namespace core
}  // namespace dtorch

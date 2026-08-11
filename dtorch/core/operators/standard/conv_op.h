/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "../operator.h"
#include "dtorch/core/type.h"

namespace dtorch {
namespace core {

struct ConvParam : public OpParam {
    std::vector<int64_t> dilations;
    int64_t group;
    std::vector<int64_t> kernelSize;
    PaddingType paddingType;
    // [h_begin, w_begin, h_end, w_end]
    std::vector<int64_t> pads;
    std::vector<int64_t> strides;
    OperatorFormat format;

public:
    ConvParam()
        : OpParam(OperatorType::kConv),
          dilations(1),
          group(1),
          kernelSize(1),
          paddingType(PaddingType::kSame),
          pads(0),
          strides(1),
          format(OperatorFormat::kNCHW) {}

    ConvParam(const IntOrIntArray& dilations, int64_t group, const IntOrIntArray& kernelSize, PaddingType paddingType,
              const IntOrIntArray& pads, const IntOrIntArray& strides, OperatorFormat format);

    static std::vector<int64_t> GetKernelSize(const Shape& weightShape);

    void Get2DParam(int64_t& dilationH, int64_t& dilationW, int64_t& kernelH, int64_t& kernelW, int64_t& strideH,
                    int64_t& strideW, int64_t& groupSize) const;

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar& BaseObject<OpParam>(*this);
        ar & dilations;
        ar & group;
        ar & kernelSize;
        ar & paddingType;
        ar & pads;
        ar & strides;
        ar & format;
    }
};

class ConvOp : public Operator {
public:
    static Shape::DataType CalculateConvOutput(Shape::DataType input, Shape::DataType kernel, Shape::DataType dilation,
                                               Shape::DataType stride, Shape::DataType padBefore,
                                               Shape::DataType padAfter);

    static Shape::DataType CalculateConvOutputForSamePad(Shape::DataType input, Shape::DataType stride);

    static Shape::DataType CalculatePadBeforeForSamePad(Shape::DataType input, Shape::DataType output,
                                                        Shape::DataType kernel, Shape::DataType dilation,
                                                        Shape::DataType stride);

public:
    ConvOp(std::shared_ptr<OpParam> opParamPtr) : Operator(opParamPtr) {}

    void InferOutputMetaInfo() const override;

    std::string GetDescribeString() const override;

    void Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const override;

    PlacementSignature GetPlacementSignature() const override;

    OperatorCost GetOperatorCost() const override;
};

}  // namespace core
}  // namespace dtorch

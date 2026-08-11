/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "../operator.h"

namespace dtorch {
namespace core {

// TODO: support dilation or not. cudnn 不支持 dilation.

// According to definition of ONNX AveragePool, MaxPool, GlobalAveragePool and GlobalMaxPool
//
// Note:
//  1. dilation
//      PyTorch 中，MaxPool 支持 dilation，而 AvgPool 不支持。
//      https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html?highlight=maxpool#torch.nn.MaxPool2d
//      https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html?highlight=avgpool#torch.nn.AvgPool2d
//
//  2. output Shape
//      TensorFlow 和 PyTorch 中的计算公式如下，其中 TensorFlow 的 padding == "VALID" 等价于没有 padding 的情况。
//      TensorFlow: https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2
//      If padding == "SAME": ceil(in_height / stride_height)
//      If padding == "VALID": out_height = ceil((in_height - filter_height + 1) / stride_height)
//
//      PyTorch: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html?highlight=maxpool#torch.nn.MaxPool2d
//      If ceil_mode == false: output_shape = std::floor((input_shape + 2 * pad - dilation * (kernel_size - 1) - 1) /
//      stride + 1) If ceil_mode == true: output_shape = std::ceil((input_shape + 2 * pad - dilation * (kernel_size - 1)
//      - 1) / stride + 1)
//
//  3. Pad size for same pad
//      参照 ONNX 中的计算公式：
//      https://github.com/onnx/onnx/blob/master/docs/Operators.md#MaxPool
//      pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + ((kernel_spatial_shape[i] - 1) *
//      dilations[i] + 1) - input_spatial_shape[i]
//
// 4. PaddingType
//      当 PaddingType == kSame 或 kValid 时，ceilMode 没有意义。也可理解为 ceil_mode = false。
//
// Reference:
//      1. https://github.com/onnx/onnx/blob/master/docs/Operators.md#AveragePool
//      2. https://github.com/onnx/onnx/blob/master/docs/Operators.md#MaxPool
//      3. https://github.com/onnx/onnx/blob/main/docs/Operators.md#GlobalAveragePool
//      4. https://github.com/onnx/onnx/blob/main/docs/Operators.md#GlobalMaxPool

struct PoolingParam : public OpParam {
    PoolingKind poolingKind;
    std::vector<int64_t> dilations;
    bool ceilMode;
    std::vector<int64_t> kernelSize;
    PaddingType paddingType;
    // [h_begin, w_begin, h_end, w_end]
    std::vector<int64_t> pads;
    std::vector<int64_t> strides;
    bool countIncludePad;
    bool isGlobalPooling;
    OperatorFormat format;

public:
    PoolingParam(PoolingKind poolingKind = PoolingKind::kAvg, bool isGlobalPooling = true,
                 OperatorFormat format = OperatorFormat::kNCHW);

    PoolingParam(PoolingKind poolingKind, const IntOrIntArray& dilations, bool ceilMode,
                 const IntOrIntArray& kernelSize, PaddingType paddingType, const IntOrIntArray& pads,
                 const IntOrIntArray& strides, bool countIncludePad, OperatorFormat format);

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar& BaseObject<OpParam>(*this);
        ar & poolingKind;
        ar & dilations;
        ar & ceilMode;
        ar & kernelSize;
        ar & paddingType;
        ar & pads;
        ar & strides;
        ar & countIncludePad;
        ar & isGlobalPooling;
        ar & format;
    }
};

class PoolingOp : public Operator {
public:
    static Shape::DataType CalculatePoolingOutput(Shape::DataType input, int64_t kernel, int64_t dilation,
                                                  int64_t stride, int64_t padBefore, int64_t padAfter, bool ceilMode);

    static Shape::DataType CalculateConvOutputForSamePad(Shape::DataType input, int64_t stride);

    static void CalculatePadForSamePad(Shape::DataType input, Shape::DataType output, int64_t kernel, int64_t dilation,
                                       int64_t stride, int64_t& padBefore, int64_t& padAfter);

public:
    PoolingOp(std::shared_ptr<OpParam> opParamPtr) : Operator(opParamPtr) {}

    void InferOutputMetaInfo() const override;

    std::string GetDescribeString() const override;

    void Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const override;
};

}  // namespace core
}  // namespace dtorch

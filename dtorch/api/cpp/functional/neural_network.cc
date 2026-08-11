/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "neural_network.h"

#include "dtorch/common/debug.h"
#include "dtorch/core/graph/graph_constructor.h"
#include "dtorch/core/operators/standard/batch_norm_op.h"
#include "dtorch/core/operators/standard/conv_op.h"
#include "dtorch/core/operators/standard/dropout_op.h"
#include "dtorch/core/operators/standard/embedding_op.h"
#include "dtorch/core/operators/standard/flatten_op.h"
#include "dtorch/core/operators/standard/interpolate_op.h"
#include "dtorch/core/operators/standard/linear_op.h"
#include "dtorch/core/operators/standard/matmul_op.h"
#include "dtorch/core/operators/standard/normalization_op.h"
#include "dtorch/core/operators/standard/pooling_op.h"
#include "dtorch/core/operators/standard/reshape_op.h"
#include "dtorch/core/operators/standard/scaled_dot_product_attention_op.h"
#include "dtorch/core/operators/standard/softmax_op.h"

namespace dtorch {
namespace api {
namespace cpp {
namespace functional {

Tensor BatchNorm(const Tensor& input, const std::optional<Tensor>& runningMean, const std::optional<Tensor>& runningVar,
                 const std::optional<Tensor>& scale, const std::optional<Tensor>& bias, bool training, double momentum,
                 double eps, OperatorFormat format) {
    DAlwaysAssert(training == false);

    std::unique_ptr<core::OpParam> param(new core::BatchNormParam(eps, momentum, format));
    if (scale.has_value() != bias.has_value()) {
        throw std::invalid_argument("scale and bias MUST set or not set same");
    }

    if (!training && (!runningMean.has_value() || !runningVar.has_value())) {
        throw std::invalid_argument("runningMean and runningVar must be defined in evaluation mode");
    }

    TensorArray inputs = {input};
    TensorArrayPushOptional(inputs, runningMean);
    TensorArrayPushOptional(inputs, runningVar);
    TensorArrayPushOptional(inputs, scale);
    TensorArrayPushOptional(inputs, bias);
    return core::GraphConstructor::AddOperator(std::move(param), inputs);
}

Tensor GroupNorm(const Tensor& input, int64_t numGroups, const std::optional<Tensor>& scale,
                 const std::optional<Tensor>& bias, double eps) {
    std::unique_ptr<core::OpParam> param(
        new core::NormalizationParam(core::NormalizationKind::kGroupNorm, Shape(), eps, numGroups));

    TensorArray inputs = {input};
    TensorArrayPushOptional(inputs, scale);
    TensorArrayPushOptional(inputs, bias);
    return core::GraphConstructor::AddOperator(std::move(param), inputs);
}

Tensor LayerNorm(const Tensor& input, const IntOrIntArray& normalizedShape, const std::optional<Tensor>& scale,
                 const std::optional<Tensor>& bias, double eps) {
    std::unique_ptr<core::OpParam> param(
        new core::NormalizationParam(core::NormalizationKind::kLayerNorm, Shape(normalizedShape), eps));

    TensorArray inputs = {input};
    TensorArrayPushOptional(inputs, scale);
    TensorArrayPushOptional(inputs, bias);
    return core::GraphConstructor::AddOperator(std::move(param), inputs);
}

Tensor RmsNorm(const Tensor& input, const IntOrIntArray& normalizedShape, const std::optional<Tensor>& weight,
               double eps) {
    std::unique_ptr<core::OpParam> param(
        new core::NormalizationParam(core::NormalizationKind::kRmsNorm, Shape(normalizedShape), eps));

    TensorArray inputs = {input};
    TensorArrayPushOptional(inputs, weight);
    TensorArrayPushOptional(inputs, std::nullopt);
    return core::GraphConstructor::AddOperator(std::move(param), inputs);
}

Tensor Conv2d(const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias, const IntOrIntArray& stride,
              PaddingType paddingType, const IntOrIntArray& padding, const IntOrIntArray& dilation, int64_t group,
              OperatorFormat format) {
    std::unique_ptr<core::OpParam> param(new core::ConvParam(
        dilation, group, core::ConvParam::GetKernelSize(weight.GetShape()), paddingType, padding, stride, format));

    TensorArray inputs = {input, weight};
    TensorArrayPushOptional(inputs, bias);
    return core::GraphConstructor::AddOperator(std::move(param), inputs);
}

Tensor Dropout(const Tensor& input, double p, bool training, bool inplace) {
    DAlwaysAssert(training == false);
    IgnoreUnused(p, inplace);
    return input;

    // IgnoreUnused(inplace);
    // std::unique_ptr<core::OpParam> param(new core::DropoutParam(p));
    // return core::GraphConstructor::AddOperator(std::move(param), {input});
}

Tensor _Flatten(const Tensor& input, int64_t startDim, int64_t endDim) {
    std::unique_ptr<core::OpParam> param(new core::FlattenParam(startDim, endDim));
    return core::GraphConstructor::AddOperator(std::move(param), {input});
}

Tensor Linear(const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias) {
    std::unique_ptr<core::OpParam> param(new core::LinearParam());
    TensorArray inputs = {input, weight};
    TensorArrayPushOptional(inputs, bias);
    return core::GraphConstructor::AddOperator(std::move(param), inputs);
}

Tensor _Matmul(const Tensor& input, const Tensor& other) {
    std::unique_ptr<core::OpParam> param(new core::MatmulParam());
    TensorArray inputs = {input, other};
    return core::GraphConstructor::AddOperator(std::move(param), inputs);
}

Tensor _Pooling2d(const Tensor& input, PoolingKind poolingKind, const IntOrIntArray& dilation,
                  const IntOrIntArray& kernelSize, const IntOrIntArray& stride, PaddingType paddingType,
                  const IntOrIntArray& padding, bool ceilMode, bool countIncludePad, OperatorFormat format) {
    std::unique_ptr<core::OpParam> param(new core::PoolingParam(poolingKind, dilation, ceilMode, kernelSize,
                                                                paddingType, padding, stride, countIncludePad, format));
    return core::GraphConstructor::AddOperator(std::move(param), {input});
}

Tensor _GlobalPooling2d(const Tensor& input, PoolingKind poolingKind, OperatorFormat format) {
    std::unique_ptr<core::OpParam> param(new core::PoolingParam(poolingKind, true, format));
    return core::GraphConstructor::AddOperator(std::move(param), {input});
}

Tensor _Reshape(const Tensor& input, const Shape& shape) {
    std::unique_ptr<core::OpParam> param(new core::ReshapeParam(shape));
    return core::GraphConstructor::AddOperator(std::move(param), {input});
}

Tensor Embedding(const Tensor& input, const Tensor& weight) {
    std::unique_ptr<core::OpParam> param(new core::EmbeddingParam());
    return core::GraphConstructor::AddOperator(std::move(param), {input, weight});
}

Tensor _ScaledDotProductAttention(const Tensor& query, const Tensor& key, const Tensor& value,
                                  const std::optional<Tensor>& attnMask, double /*dropoutP*/, bool isCausal,
                                  std::optional<double> scale, bool enableGqa, std::optional<SdpaOption> sdpa_option) {
    std::unique_ptr<core::OpParam> param(new core::SdpaParam(isCausal, scale, enableGqa, sdpa_option));
    TensorArray inputs = {query, key, value};
    TensorArrayPushOptional(inputs, attnMask);
    return core::GraphConstructor::AddOperator(std::move(param), inputs);
}

Tensor Interpolate(const Tensor& input, const std::optional<Shape>& size,
                   const std::optional<std::vector<double>>& scaleFactor, const std::string& mode,
                   const std::optional<bool>& alignCorners, const std::optional<bool>& recomputeScaleFactor,
                   const std::optional<bool>& antialias) {
    std::unique_ptr<core::OpParam> param(
        new core::InterpolateParam(size, scaleFactor, mode, alignCorners, recomputeScaleFactor, antialias));
    return core::GraphConstructor::AddOperator(std::move(param), {input});
}

Tensor Interpolate(const Tensor& input, const std::optional<Shape>& size, const std::optional<double>& scaleFactor,
                   const std::string& mode, const std::optional<bool>& alignCorners,
                   const std::optional<bool>& recomputeScaleFactor, const std::optional<bool>& antialias) {
    std::optional<std::vector<double>> vecScaleFactor = std::nullopt;
    if (scaleFactor.has_value()) {
        vecScaleFactor = std::vector<double>{scaleFactor.value()};
    }
    return Interpolate(input, size, vecScaleFactor, mode, alignCorners, recomputeScaleFactor, antialias);
}

Tensor Softmax(const Tensor& input, std::optional<int64_t> dim) {
    if (!dim.has_value()) {
        throw std::invalid_argument("Softmax dim parameter can't be None");
    }
    std::unique_ptr<core::OpParam> param(new core::SoftmaxParam(dim.value()));
    return core::GraphConstructor::AddOperator(std::move(param), {input});
}

}  // namespace functional
}  // namespace cpp
}  // namespace api
}  // namespace dtorch

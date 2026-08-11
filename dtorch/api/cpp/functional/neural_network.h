/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <cstdint>
#include <optional>

#include "../int_or_int_array.h"
#include "../tensor.h"
#include "./functional_option.h"

namespace dtorch {
namespace api {
namespace cpp {
namespace functional {

Tensor BatchNorm(const Tensor& input, const std::optional<Tensor>& runningMean, const std::optional<Tensor>& runningVar,
                 const std::optional<Tensor>& scale = std::nullopt, const std::optional<Tensor>& bias = std::nullopt,
                 bool training = false, double momentum = 0.1f, double eps = 1e-5f,
                 OperatorFormat format = OperatorFormat::kNCHW);

Tensor GroupNorm(const Tensor& input, int64_t numGroups, const std::optional<Tensor>& scale = std::nullopt,
                 const std::optional<Tensor>& bias = std::nullopt, double eps = 1e-5f);

Tensor LayerNorm(const Tensor& input, const IntOrIntArray& normalizedShape,
                 const std::optional<Tensor>& scale = std::nullopt, const std::optional<Tensor>& bias = std::nullopt,
                 double eps = 1e-5f);

Tensor RmsNorm(const Tensor& input, const IntOrIntArray& normalizedShape, const std::optional<Tensor>& weight,
               double eps = 1e-5f);

Tensor Conv2d(const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias = std::nullopt,
              const IntOrIntArray& stride = 1, PaddingType paddingType = PaddingType::kNotSet,
              const IntOrIntArray& padding = 0, const IntOrIntArray& dilation = 1, int64_t group = 1,
              OperatorFormat format = OperatorFormat::kNCHW);

Tensor Dropout(const Tensor& input, double p = 0.5f, bool training = true, bool inplace = false);

Tensor _Flatten(const Tensor& input, int64_t startDim = 0, int64_t endDim = -1);

Tensor Linear(const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias = std::nullopt);

Tensor _Matmul(const Tensor& input, const Tensor& other);

Tensor _Pooling2d(const Tensor& input, PoolingKind poolingKind, const IntOrIntArray& dilation,
                  const IntOrIntArray& kernelSize, const IntOrIntArray& stride, PaddingType paddingType,
                  const IntOrIntArray& padding, bool ceilMode, bool countIncludePad,
                  OperatorFormat format = OperatorFormat::kNCHW);

Tensor _GlobalPooling2d(const Tensor& input, PoolingKind poolingKind, OperatorFormat format = OperatorFormat::kNCHW);

Tensor _Reshape(const Tensor& input, const Shape& shape);

Tensor Embedding(const Tensor& input, const Tensor& weight);

Tensor _ScaledDotProductAttention(const Tensor& query, const Tensor& key, const Tensor& value,
                                  const std::optional<Tensor>& attnMask = std::nullopt, double dropoutP = 0.0,
                                  bool isCausal = false, std::optional<double> scale = std::nullopt,
                                  bool enableGqa = false, std::optional<SdpaOption> sdpa_option = std::nullopt);

Tensor Interpolate(const Tensor& input, const std::optional<Shape>& size = std::nullopt,
                   const std::optional<std::vector<double>>& scaleFactor = std::nullopt,
                   const std::string& mode = "nearest", const std::optional<bool>& alignCorners = std::nullopt,
                   const std::optional<bool>& recomputeScaleFactor = std::nullopt,
                   const std::optional<bool>& antialias = false);

Tensor Interpolate(const Tensor& input, const std::optional<Shape>& size = std::nullopt,
                   const std::optional<double>& scaleFactor = std::nullopt, const std::string& mode = "nearest",
                   const std::optional<bool>& alignCorners = std::nullopt,
                   const std::optional<bool>& recomputeScaleFactor = std::nullopt,
                   const std::optional<bool>& antialias = false);

Tensor Softmax(const Tensor& input, std::optional<int64_t> dim = std::nullopt);

}  // namespace functional
}  // namespace cpp
}  // namespace api
}  // namespace dtorch

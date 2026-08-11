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

Tensor _ApplyRotaryEmb(const Tensor& x, const TensorArray& freqsCis, bool useReal = true,
                       int64_t useRealUnbindDim = -1);

TensorArray _SiluLinearChunk(const Tensor& emb, const Tensor& linearWeight, const Tensor& linearBias,
                             int64_t chunkSize);

Tensor _LayerNormMulAdd(const Tensor& x, const Tensor& scale, const Tensor& shift, const IntOrIntArray& normalizedShape,
                        double normEps = 1e-5f, const std::optional<Tensor>& normScale = std::nullopt,
                        const std::optional<Tensor>& normBias = std::nullopt);

}  // namespace functional
}  // namespace cpp
}  // namespace api
}  // namespace dtorch

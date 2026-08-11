/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "fused_compile_op.h"

#include "dtorch/common/debug.h"
#include "dtorch/core/graph/graph_constructor.h"
#include "dtorch/core/operators/fused_compile/apply_rotary_emb_op.h"
#include "dtorch/core/operators/fused_compile/layer_norm_mul_add.h"
#include "dtorch/core/operators/fused_compile/silu_linear_chunk.h"
#include "implement/util.h"

namespace dtorch {
namespace api {
namespace cpp {
namespace functional {

Tensor _ApplyRotaryEmb(const Tensor& x, const TensorArray& freqsCis, bool useReal, int64_t useRealUnbindDim) {
    std::unique_ptr<core::OpParam> param(new core::ApplyRotaryEmbParam(useReal, useRealUnbindDim));
    if (freqsCis.size() != 2) {
        throw std::invalid_argument("Size of freqsCis must be two");
    }
    Tensor freqsCis0Placed = PlacementR2S(freqsCis[0], x);
    Tensor freqsCis1Placed = PlacementR2S(freqsCis[1], x);
    return core::GraphConstructor::AddOperator(std::move(param), {x, freqsCis0Placed, freqsCis1Placed});
}

TensorArray _SiluLinearChunk(const Tensor& emb, const Tensor& linearWeight, const Tensor& linearBias,
                             int64_t chunkSize) {
    std::unique_ptr<core::OpParam> param(new core::SiluLinearChunkParam(chunkSize));
    return core::GraphConstructor::AddOperator(std::move(param), {emb, linearWeight, linearBias}, true);
}

Tensor _LayerNormMulAdd(const Tensor& x, const Tensor& scale, const Tensor& shift, const IntOrIntArray& normalizedShape,
                        double normEps, const std::optional<Tensor>& normScale, const std::optional<Tensor>& normBias) {
    std::unique_ptr<core::OpParam> param(new core::LayerNormMulAddParam(Shape(normalizedShape), normEps));

    TensorArray inputs = {x, scale, shift};
    TensorArrayPushOptional(inputs, normScale);
    TensorArrayPushOptional(inputs, normBias);
    return core::GraphConstructor::AddOperator(std::move(param), inputs);
}

}  // namespace functional
}  // namespace cpp
}  // namespace api
}  // namespace dtorch

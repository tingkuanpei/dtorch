/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "broadcast_op_imlp.h"

#include <optional>

#include "dtorch/common/debug.h"
#include "util.h"

namespace dtorch {
namespace api {
namespace cpp {
namespace functional {

Tensor BroadcastBinaryOpImpl(core::BroadcastBinaryKind binaryKind, const std::optional<Tensor>& input,
                             const std::optional<Tensor>& other, const std::optional<Scalar>& inputA,
                             const std::optional<Scalar>& inputB, const std::optional<double>& scaleB) {
    DAlwaysAssert(input.has_value() || other.has_value());
    std::unique_ptr<core::OpParam> param(new core::BroadcastBinaryParam(binaryKind, inputA, inputB, scaleB));
    TensorArray inputs;
    if (!input.has_value()) {
        inputs.push_back(Tensor::GetNullTensorLike(other.value()));
    } else {
        inputs.push_back(input.value());
    }
    TensorArrayPushOptional(inputs, other);

    // Convert placements from Replicate to Shard when one of inputs is Replicate and other is Shard.
    if (input.has_value() && other.has_value()) {
        inputs[0] = PlacementR2S(inputs[0], inputs[1]);
        inputs[1] = PlacementR2S(inputs[1], inputs[0]);
    }

    return core::GraphConstructor::AddOperator(std::move(param), inputs);
}

}  // namespace functional
}  // namespace cpp
}  // namespace api
}  // namespace dtorch

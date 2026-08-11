/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "dtorch/core/graph/graph_constructor.h"
#include "dtorch/core/operators/standard/broadcast_binary_op.h"

namespace dtorch {
namespace api {
namespace cpp {
namespace functional {

Tensor BroadcastBinaryOpImpl(core::BroadcastBinaryKind binaryKind, const std::optional<Tensor>& input,
                             const std::optional<Tensor>& other, const std::optional<Scalar>& inputA = std::nullopt,
                             const std::optional<Scalar>& inputB = std::nullopt,
                             const std::optional<double>& scaleB = std::nullopt);

}  // namespace functional
}  // namespace cpp
}  // namespace api
}  // namespace dtorch

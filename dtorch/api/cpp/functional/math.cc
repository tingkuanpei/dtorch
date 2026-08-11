/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "math.h"

#include <optional>

#include "dtorch/api/cpp/functional/implement/broadcast_op_imlp.h"
#include "dtorch/api/cpp/scalar.h"
#include "dtorch/common/debug.h"
#include "dtorch/core/graph/graph_constructor.h"
#include "dtorch/core/operators/standard/base_math_op.h"
#include "dtorch/core/operators/standard/broadcast_binary_op.h"
#include "dtorch/core/operators/standard/einsum_op.h"
#include "dtorch/core/operators/standard/outer_op.h"

namespace dtorch {
namespace api {
namespace cpp {
namespace functional {

using core::BaseMathType;
using core::BroadcastBinaryKind;

Tensor _Add(const Tensor& input, const Tensor& other, double alpha) {
    return BroadcastBinaryOpImpl(BroadcastBinaryKind::kAdd, input, other, std::nullopt, std::nullopt, alpha);
}

Tensor _Add(const Tensor& input, const Scalar& other, double alpha) {
    return BroadcastBinaryOpImpl(BroadcastBinaryKind::kAdd, input, std::nullopt, std::nullopt, other, alpha);
}

Tensor _Sub(const Tensor& input, const Tensor& other, double alpha) {
    return BroadcastBinaryOpImpl(BroadcastBinaryKind::kSub, input, other, std::nullopt, std::nullopt, alpha);
}

Tensor _Sub(const Tensor& input, const Scalar& other, double alpha) {
    return BroadcastBinaryOpImpl(BroadcastBinaryKind::kSub, input, std::nullopt, std::nullopt, other, alpha);
}

#define BROADCAST_BINARY_EXPAND(name)                                                                        \
    Tensor _##name(const Tensor& input, const Tensor& other) {                                               \
        return BroadcastBinaryOpImpl(BroadcastBinaryKind::k##name, input, other, std::nullopt, std::nullopt, \
                                     std::nullopt);                                                          \
    }                                                                                                        \
    Tensor _##name(const Tensor& input, const Scalar& other) {                                               \
        return BroadcastBinaryOpImpl(BroadcastBinaryKind::k##name, input, std::nullopt, std::nullopt, other, \
                                     std::nullopt);                                                          \
    }

BROADCAST_BINARY_EXPAND(Mul)
BROADCAST_BINARY_EXPAND(Div)
BROADCAST_BINARY_EXPAND(Equal)
BROADCAST_BINARY_EXPAND(Greater)
BROADCAST_BINARY_EXPAND(GreaterEqual)
BROADCAST_BINARY_EXPAND(Less)
BROADCAST_BINARY_EXPAND(LessEqual)
BROADCAST_BINARY_EXPAND(Pow)

Tensor _Minimum(const Tensor& input, const Tensor& other) {
    return BroadcastBinaryOpImpl(BroadcastBinaryKind::kMinimum, input, other, std::nullopt, std::nullopt, std::nullopt);
}

Tensor _Maximum(const Tensor& input, const Tensor& other) {
    return BroadcastBinaryOpImpl(BroadcastBinaryKind::kMaximum, input, other, std::nullopt, std::nullopt, std::nullopt);
}

Tensor _LogicalAnd(const Tensor& input, const Tensor& other) {
    return BroadcastBinaryOpImpl(BroadcastBinaryKind::kLogicalAnd, input, other);
}

Tensor _LogicalOr(const Tensor& input, const Tensor& other) {
    return BroadcastBinaryOpImpl(BroadcastBinaryKind::kLogicalOr, input, other);
}

Tensor _Pow(const Scalar& self, const Tensor& exponent) {
    return BroadcastBinaryOpImpl(BroadcastBinaryKind::kPow, std::nullopt, exponent, self, std::nullopt, std::nullopt);
}

Tensor _Einsum(const std::string& equation, const Tensor& operands) {
    TensorArray array = {operands};
    return _Einsum(equation, array);
}

Tensor _Einsum(const std::string& equation, const TensorArray& operands) {
    DAlwaysAssert(operands.size() > 0);
    std::unique_ptr<core::OpParam> param(new core::EinsumParam(equation));
    return core::GraphConstructor::AddOperator(std::move(param), operands);
}

Tensor BaseMathImp(const Tensor& input, BaseMathType baseMathType) {
    std::unique_ptr<core::OpParam> param(new core::BaseMathParam(baseMathType));
    TensorArray inputs = {input};
    return core::GraphConstructor::AddOperator(std::move(param), inputs);
}

Tensor _Exp(const Tensor& input) { return BaseMathImp(input, BaseMathType::kExp); }

Tensor _Square(const Tensor& input) { return BaseMathImp(input, BaseMathType::kSquare); }

Tensor _Rsqrt(const Tensor& input) { return BaseMathImp(input, BaseMathType::kRsqrt); }

Tensor _Abs(const Tensor& input) { return BaseMathImp(input, BaseMathType::kAbs); }

Tensor _Round(const Tensor& input) { return BaseMathImp(input, BaseMathType::kRound); }

Tensor _Floor(const Tensor& input) { return BaseMathImp(input, BaseMathType::kFloor); }

Tensor _Cos(const Tensor& input) { return BaseMathImp(input, BaseMathType::kCos); }

Tensor _Sin(const Tensor& input) { return BaseMathImp(input, BaseMathType::kSin); }

Tensor _Asin(const Tensor& input) { return BaseMathImp(input, BaseMathType::kAsin); }

Tensor Tanh(const Tensor& input) { return BaseMathImp(input, BaseMathType::kTanh); }

Tensor _Neg(const Tensor& input) { return BaseMathImp(input, BaseMathType::kNeg); }

Tensor _Reciprocal(const Tensor& input) { return BaseMathImp(input, BaseMathType::kReciprocal); }

Tensor _Log(const Tensor& input) { return BaseMathImp(input, BaseMathType::kLog); }

Tensor _Log2(const Tensor& input) { return BaseMathImp(input, BaseMathType::kLog2); }

Tensor _Log10(const Tensor& input) { return BaseMathImp(input, BaseMathType::kLog10); }

Tensor _Isinf(const Tensor& input) { return BaseMathImp(input, BaseMathType::kIsInf); }

Tensor _Isnan(const Tensor& input) { return BaseMathImp(input, BaseMathType::kIsNan); }

Tensor _Outer(const Tensor& input, const Tensor& vec2) {
    std::unique_ptr<core::OpParam> param(new core::OuterParam());
    TensorArray inputs = {input, vec2};
    return core::GraphConstructor::AddOperator(std::move(param), inputs);
}

}  // namespace functional
}  // namespace cpp
}  // namespace api
}  // namespace dtorch

/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "../scalar.h"
#include "../tensor.h"

namespace dtorch {
namespace api {
namespace cpp {
namespace functional {

Tensor _Add(const Tensor& input, const Tensor& other, double alpha = 1.0);

Tensor _Add(const Tensor& input, const Scalar& other, double alpha = 1.0);

Tensor _Sub(const Tensor& input, const Tensor& other, double alpha = 1.0);

Tensor _Sub(const Tensor& input, const Scalar& other, double alpha = 1.0);

Tensor _Mul(const Tensor& input, const Tensor& other);

Tensor _Mul(const Tensor& input, const Scalar& other);

Tensor _Div(const Tensor& input, const Tensor& other);

Tensor _Div(const Tensor& input, const Scalar& other);

Tensor _Pow(const Tensor& input, const Tensor& exponent);

Tensor _Pow(const Tensor& input, const Scalar& exponent);

Tensor _Pow(const Scalar& self, const Tensor& exponent);

Tensor _Equal(const Tensor& input, const Tensor& other);

Tensor _Equal(const Tensor& input, const Scalar& other);

Tensor _Greater(const Tensor& input, const Tensor& other);

Tensor _Greater(const Tensor& input, const Scalar& other);

Tensor _GreaterEqual(const Tensor& input, const Tensor& other);

Tensor _GreaterEqual(const Tensor& input, const Scalar& other);

Tensor _Less(const Tensor& input, const Tensor& other);

Tensor _Less(const Tensor& input, const Scalar& other);

Tensor _LessEqual(const Tensor& input, const Tensor& other);

Tensor _LessEqual(const Tensor& input, const Scalar& other);

Tensor _Minimum(const Tensor& input, const Tensor& other);

Tensor _Maximum(const Tensor& input, const Tensor& other);

Tensor _LogicalAnd(const Tensor& input, const Tensor& other);

Tensor _LogicalOr(const Tensor& input, const Tensor& other);

Tensor _Einsum(const std::string& equation, const Tensor& operands);

Tensor _Einsum(const std::string& equation, const TensorArray& operands);

Tensor _Exp(const Tensor& input);

Tensor _Square(const Tensor& input);

Tensor _Rsqrt(const Tensor& input);

Tensor _Abs(const Tensor& input);

Tensor _Round(const Tensor& input);

Tensor _Floor(const Tensor& input);

Tensor _Cos(const Tensor& input);

Tensor _Sin(const Tensor& input);

Tensor _Asin(const Tensor& input);

Tensor Tanh(const Tensor& input);

Tensor _Neg(const Tensor& input);

Tensor _Reciprocal(const Tensor& input);

Tensor _Log(const Tensor& input);

Tensor _Log2(const Tensor& input);

Tensor _Log10(const Tensor& input);

Tensor _Isinf(const Tensor& input);

Tensor _Isnan(const Tensor& input);

Tensor _Outer(const Tensor& input, const Tensor& vec2);

}  // namespace functional
}  // namespace cpp
}  // namespace api
}  // namespace dtorch

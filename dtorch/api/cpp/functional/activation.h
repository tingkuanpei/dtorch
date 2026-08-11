/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "../tensor.h"

namespace dtorch {
namespace api {
namespace cpp {
namespace functional {

Tensor Relu(const Tensor& input, bool inplace = false);

Tensor Sigmoid(const Tensor& input);

Tensor LeakyRelu(const Tensor& input, double negative_slope = 0.01, bool inplace = false);

Tensor Elu(const Tensor& input, double alpha = 1.0, bool inplace = false);

Tensor Gelu(const Tensor& input, const std::string& approximate = "none");

Tensor Silu(const Tensor& input, bool inplace = false);

}  // namespace functional
}  // namespace cpp
}  // namespace api
}  // namespace dtorch

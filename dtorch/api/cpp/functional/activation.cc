/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "activation.h"

#include "dtorch/core/graph/graph_constructor.h"
#include "dtorch/core/operators/standard/activation_op.h"

namespace dtorch {
namespace api {
namespace cpp {
namespace functional {

Tensor ActivationOpImpl(const Tensor& input, core::ActivationType activationType, bool inplace = false,
                        double alpha = 0.0f, double beta = 0.0f, const std::string& approximate = "") {
    std::unique_ptr<core::OpParam> param(new core::ActivationParam(activationType, inplace, alpha, beta, approximate));
    return core::GraphConstructor::AddOperator(std::move(param), {input});
}

Tensor Relu(const Tensor& input, bool inplace) { return ActivationOpImpl(input, core::ActivationType::kReLU, inplace); }

Tensor Sigmoid(const Tensor& input) { return ActivationOpImpl(input, core::ActivationType::kSigmoid); }

Tensor LeakyRelu(const Tensor& input, double negative_slope, bool inplace) {
    return ActivationOpImpl(input, core::ActivationType::kLeakyRelu, inplace, negative_slope);
}

Tensor Elu(const Tensor& input, double alpha, bool inplace) {
    return ActivationOpImpl(input, core::ActivationType::kELU, inplace, alpha);
}

Tensor Gelu(const Tensor& input, const std::string& approximate) {
    return ActivationOpImpl(input, core::ActivationType::kGELU, false, 0.0f, 0.0f,
                            approximate.empty() ? "none" : approximate);
}

Tensor Silu(const Tensor& input, bool inplace) { return ActivationOpImpl(input, core::ActivationType::kSiLU, inplace); }

}  // namespace functional
}  // namespace cpp
}  // namespace api
}  // namespace dtorch

/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <optional>

#include "dtorch/external/torch/torch_util.h"

namespace dtorch {
namespace external {
namespace sage {

::torch::Tensor SageAttnAdapter(const ::torch::Tensor& q, const ::torch::Tensor& k, const ::torch::Tensor& v,
                                bool isCausal = false, std::optional<float> smScale = std::nullopt,
                                const std::string& sageAttentionType = "auto");

}  // namespace sage
}  // namespace external
}  // namespace dtorch

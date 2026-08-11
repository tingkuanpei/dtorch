/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "../../tensor.h"

namespace dtorch {
namespace api {
namespace cpp {
namespace functional {

Tensor PlacementR2S(const Tensor& replicateTensor, const Tensor& shardTensor);

}  // namespace functional
}  // namespace cpp
}  // namespace api
}  // namespace dtorch

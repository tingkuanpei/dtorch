/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <string>

// forward declaration for torch::Tensor
namespace at {
class Tensor;
class Generator;
}  // namespace at

namespace c10 {
class Device;
}

namespace torch {
using at::Generator;
using at::Tensor;
using c10::Device;
}  // namespace torch

namespace dtorch {
namespace api {
namespace cpp {

enum class OperatorFormat {
    kNCHW = 0,
    kNHWC,
    kCount,
};

std::string OperatorFormatToString(OperatorFormat operatorFormat);

std::ostream& operator<<(std::ostream& os, OperatorFormat operatorFormat);

bool StringToOperatorFormat(const std::string& str, OperatorFormat& operatorFormat);

enum class PaddingType {
    kNotSet = 0,
    kSame,
    kValid,
    kCount,
};

std::string PaddingTypeToString(PaddingType paddingType);

std::ostream& operator<<(std::ostream& os, PaddingType paddingType);

bool StringToPaddingType(const std::string& str, PaddingType& paddingType);

enum class PoolingKind {
    kAvg = 0,
    kMax,
    kCount,
};

std::string PoolingKindToString(PoolingKind PoolingKind);

std::ostream& operator<<(std::ostream& os, PoolingKind PoolingKind);

bool StringToPoolingKind(const std::string& str, PoolingKind& poolingKind);

}  // namespace cpp
}  // namespace api
}  // namespace dtorch

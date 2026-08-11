/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "index.h"

#include <array>
#include <string>

#include <ATen/Tensor.h>
#include <torch/torch.h>

#include "dtorch/common/debug.h"
#include "dtorch/common/logging.h"

namespace dtorch {
namespace api {
namespace cpp {

std::string IndexTypeToString(IndexType indexType) {
    static const std::array<std::string, 5> kStringMap = {"None", "Ellipsis", "Integer", "Slice", "Tensor"};
    static_assert(static_cast<int>(kStringMap.size()) == EnumAsInteger(IndexType::kCount), "IndexType size not equal");

    if (indexType == IndexType::kCount) {
        DLogError() << "indexType invalid";
        return "";
    }

    return kStringMap[EnumAsInteger(indexType)];
}

std::ostream& operator<<(std::ostream& os, IndexType indexType) {
    os << IndexTypeToString(indexType);
    return os;
}

Index::Index(at::Tensor tensor)
    : mType(IndexType::kTensor), mInteger(), mSlice(), mTensor(std::make_shared<at::Tensor>(std::move(tensor))) {}

Index::Index(std::vector<int64_t> vec)
    : mType(IndexType::kTensor),
      mInteger(),
      mSlice(),
      mTensor(std::make_shared<at::Tensor>(torch::from_blob(const_cast<int64_t*>(vec.data()),
                                                            {static_cast<int64_t>(vec.size())},
                                                            at::TensorOptions().dtype(at::kLong))
                                               .clone())) {}

std::ostream& operator<<(std::ostream& os, const Slice& slice) {
    os << "slice(";
    if (slice.start) {
        os << slice.start.value();
    } else {
        os << "None";
    }
    os << ", ";
    if (slice.stop) {
        os << slice.stop.value();
    } else {
        os << "None";
    }
    os << ", ";
    if (slice.step) {
        os << slice.step.value();
    } else {
        os << "None";
    }
    os << ")";
    return os;
}

std::ostream& operator<<(std::ostream& os, const Index& index) {
    if (index.IsNone()) {
        os << "None";
    } else if (index.IsEllipsis()) {
        os << "Ellipsis";
    } else if (index.IsInteger()) {
        os << index.GetInteger();
    } else if (index.IsSlice()) {
        os << index.GetSlice();
    } else if (index.IsTensor()) {
        os << "Tensor(" << index.GetTensor().sizes() << ")";
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const std::vector<Index>& indexs) {
    os << "Index(";
    for (auto& it : indexs) {
        os << it << ", ";
    }
    os << ")";
    return os;
}

}  // namespace cpp
}  // namespace api
}  // namespace dtorch

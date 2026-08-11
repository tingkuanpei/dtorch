/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "api_utilities.h"
#include "serialization.h"
#include "tensor.h"

// Forward declaration: std::shared_ptr works with incomplete types
namespace at {
class Tensor;
}

namespace dtorch {
namespace api {
namespace cpp {

// https://www.runoob.com/numpy/numpy-indexing-and-slicing.html
// https://github.com/pytorch/pytorch/blob/f6801ba4b301e20fa84b7a350bc29cb40134934c/aten/src/ATen/TensorIndexing.h

enum class IndexType { kNone = 0, kEllipsis, kInteger, kSlice, kTensor, kCount };

std::string IndexTypeToString(IndexType indexType);

std::ostream& operator<<(std::ostream& os, IndexType indexType);

class EllipsisIndexType final {
public:
    EllipsisIndexType() = default;
};
constexpr EllipsisIndexType Ellipsis;

class Slice final {
public:
    std::optional<int64_t> start;
    std::optional<int64_t> stop;
    std::optional<int64_t> step;

public:
    Slice(std::optional<int64_t> start = std::nullopt, std::optional<int64_t> stop = std::nullopt,
          std::optional<int64_t> step = std::nullopt)
        : start(start), stop(stop), step(step) {}

    friend std::ostream& operator<<(std::ostream& os, const Slice& slice);

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar & start;
        ar & stop;
        ar & step;
    }
};

class Index final {
public:
    Index() : mType(IndexType::kNone), mInteger(), mSlice(), mTensor() {}

    // Case 1: None
    Index(std::nullopt_t) : mType(IndexType::kNone), mInteger(), mSlice(), mTensor() {}

    // Case 2: ... and Ellipsis
    Index(EllipsisIndexType) : mType(IndexType::kEllipsis), mInteger(), mSlice(), mTensor() {}
    Index(const std::string str) : Index(Ellipsis) { DApiAssert(str == "..."); }

    // Case 3: integer
    Index(int64_t integer) : mType(IndexType::kInteger), mInteger(integer), mSlice(), mTensor() {}
    Index(int integer) : mType(IndexType::kInteger), mInteger(integer), mSlice(), mTensor() {}

    // Case 4: Slice
    Index(Slice slice) : mType(IndexType::kSlice), mInteger(), mSlice(std::move(slice)), mTensor() {}

    // Case 5: Tensor (integer array indexing) — defined in index.cc
    Index(at::Tensor tensor);

    // Case 6: List of integers (Python list index) — defined in index.cc
    Index(std::vector<int64_t> vec);

    DTORCH_API_FORCEINLINE IndexType GetType() const { return mType; }

    DTORCH_API_FORCEINLINE bool IsNone() const { return mType == IndexType::kNone; }

    DTORCH_API_FORCEINLINE bool IsEllipsis() const { return mType == IndexType::kEllipsis; }

    DTORCH_API_FORCEINLINE bool IsInteger() const { return mType == IndexType::kInteger; }

    DTORCH_API_FORCEINLINE int64_t GetInteger() const { return mInteger; }

    DTORCH_API_FORCEINLINE bool IsSlice() const { return mType == IndexType::kSlice; }

    DTORCH_API_FORCEINLINE const Slice& GetSlice() const { return mSlice; }

    DTORCH_API_FORCEINLINE bool IsTensor() const { return mType == IndexType::kTensor; }

    DTORCH_API_FORCEINLINE const at::Tensor& GetTensor() const { return *mTensor; }

    friend std::ostream& operator<<(std::ostream& os, const Index& index);

    friend std::ostream& operator<<(std::ostream& os, const std::vector<Index>& indexs);

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar & mType;
        ar & mInteger;
        ar & mSlice;
        bool hasTensor = (mTensor != nullptr);
        ar & hasTensor;
        if (hasTensor) {
            if constexpr (Archive::is_saving::value) {
                // clang-format off
                ar & (*mTensor);
                // clang-format on
            } else {
                mTensor = std::make_shared<at::Tensor>();
                // clang-format off
                ar & (*mTensor);
                // clang-format on
            }
        }
    }

private:
    IndexType mType;
    int64_t mInteger;
    Slice mSlice;
    std::shared_ptr<at::Tensor> mTensor;
};

}  // namespace cpp
}  // namespace api
}  // namespace dtorch

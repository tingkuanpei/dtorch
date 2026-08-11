/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "stride.h"

#include <sstream>

#include "dtorch/common/debug.h"

namespace dtorch {
namespace api {
namespace cpp {

Stride::Stride(size_t dimention, DataType value) : mStride(dimention, value) {}

Stride::Stride(const Shape& shape) : mStride(shape.NumAxis()) {
    DataType count = 1;
    for (int i = static_cast<int>(shape.NumAxis() - 1); i >= 0; i--) {
        mStride[i] = count;
        count = count * shape[i];
    }
}

Stride::Stride(std::vector<int64_t> list) : mStride(list.size()) {
    for (size_t i = 0; i < mStride.size(); i++) {
        int64_t value = list.begin()[i];
        mStride[i] = value;
    }
}

Stride::Stride(std::initializer_list<int64_t> list) : mStride(list.size()) {
    for (size_t i = 0; i < mStride.size(); i++) {
        int64_t value = list.begin()[i];
        mStride[i] = value;
    }
}

std::string Stride::ToString() const {
    std::stringstream ss;
    ss << "(";

    if (mStride.size() > 0) {
        ss << mStride[0];
    }

    for (size_t i = 1; i < mStride.size(); i++) {
        ss << ", " << mStride[i];
    }
    ss << ")";
    return ss.str();
}

void Stride::Set(int index, DataType value) noexcept {
    DDebugAssert(index < static_cast<int>(mStride.size()));
    if (index < 0) index += static_cast<int>(mStride.size());
    DDebugAssert(index >= 0);
    DDebugAssert(value >= 0);

    mStride[index] = value;
}

int64_t Stride::At(int index) const noexcept {
    DDebugAssert(index < static_cast<int>(mStride.size()));
    if (index < 0) index += static_cast<int>(mStride.size());
    DDebugAssert(index >= 0);
    return mStride[index];
}

bool Stride::operator==(const Stride& other) const noexcept {
    if (this == &other) {
        return true;
    }

    if (mStride.size() != other.mStride.size()) return false;

    for (size_t i = 0; i < mStride.size(); i++) {
        if (mStride[i] != other.mStride[i]) return false;
    }

    return true;
}

const Stride::DataType& Stride::operator[](int index) const noexcept {
    DDebugAssert(index < static_cast<int>(mStride.size()));
    if (index < 0) index += static_cast<int>(mStride.size());
    DDebugAssert(index >= 0);
    return mStride[index];
}

const Stride::DataType& Stride::operator[](size_t index) const noexcept {
    DDebugAssert(index < mStride.size());
    return mStride[index];
}

Stride Stride::ExpandBefore(size_t dim, DataType scaleValue) const {
    DDebugAssert(dim > NumAxis());
    DataType expandValue = mStride[0] * scaleValue;

    Stride result(dim);
    size_t expandDims = dim - NumAxis();

    for (size_t i = 0; i < expandDims; i++) {
        result.mStride[i] = expandValue;
    }
    for (size_t i = 0; i < NumAxis(); i++) {
        result.mStride[i + expandDims] = mStride[i];
    }

    return result;
}

}  // namespace cpp
}  // namespace api
}  // namespace dtorch

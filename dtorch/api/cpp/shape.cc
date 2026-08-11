/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "shape.h"

#include <algorithm>
#include <sstream>

#include "dtorch/common/debug.h"

namespace dtorch {
namespace api {
namespace cpp {

Shape::Shape(size_t dimention, DataType value) : mShape(dimention, value) {}

Shape::Shape(const IntOrIntArray& intArray) : mShape(intArray.size()) {
    for (size_t i = 0; i < intArray.size(); i++) {
        int64_t value = intArray[i];
        mShape[i] = value;
    }
}

Shape::DataType Shape::Count() const noexcept {
    // Shape dimention == 0, meaning scalar
    if (NumAxis() == 0) {
        return 1;
    }

    return Prod(0);
}

std::string Shape::ToString() const {
    std::stringstream ss;
    ss << "(";

    if (mShape.size() > 0) {
        ss << mShape[0];
    }

    for (size_t i = 1; i < mShape.size(); i++) {
        ss << ", " << mShape[i];
    }
    ss << ")";
    return ss.str();
}

bool Shape::IsSameShape(const Shape& otherShape) const noexcept {
    if (mShape.size() != otherShape.mShape.size()) {
        return false;
    }

    for (size_t i = 0; i < mShape.size(); i++) {
        if (mShape[i] != otherShape.mShape[i]) {
            return false;
        }
    }

    return true;
}

int64_t Shape::Prod(int begin_dim, int end_dim) const noexcept {
    DDebugAssert(begin_dim <= end_dim);
    DDebugAssert(begin_dim >= 0);
    DDebugAssert(end_dim < static_cast<int>(mShape.size()));

    int64_t result = 1;
    for (int i = begin_dim; i <= end_dim; i++) {
        result *= mShape[i];
    }
    return result;
}

void Shape::Set(int index, DataType value) noexcept {
    DDebugAssert(index < static_cast<int>(mShape.size()));
    if (index < 0) index += static_cast<int>(mShape.size());
    DDebugAssert(index >= 0);
    DDebugAssert(value >= 0);

    mShape[index] = value;
}

int64_t Shape::At(int index) const noexcept {
    DDebugAssert(index < static_cast<int>(mShape.size()));
    if (index < 0) index += static_cast<int>(mShape.size());
    DDebugAssert(index >= 0);
    return mShape[index];
}

bool Shape::operator==(const Shape& other) const noexcept {
    if (this == &other) {
        return true;
    }

    if (mShape.size() != other.mShape.size()) return false;

    for (size_t i = 0; i < mShape.size(); i++) {
        if (mShape[i] != other.mShape[i]) return false;
    }

    return true;
}

const Shape::DataType& Shape::operator[](int64_t index) const noexcept {
    DDebugAssert(index < static_cast<int64_t>(mShape.size()));
    if (index < 0) index += static_cast<int64_t>(mShape.size());
    DDebugAssert(index >= 0);
    return mShape[index];
}

const Shape::DataType& Shape::operator[](size_t index) const noexcept {
    DDebugAssert(index < mShape.size());
    return mShape[index];
}

bool Shape::CanBroadcastWith(const Shape& otherShape) const {
    size_t thisSize = NumAxis();
    size_t otherSize = otherShape.NumAxis();

    if (thisSize == 0 || otherSize == 0) {
        return true;
    }

    size_t size = std::min(thisSize, otherSize);
    size_t thisShapeBegin = thisSize - size;
    size_t otherShapeBegin = otherSize - size;

    for (size_t i = 0; i < size; i++) {
        if (mShape[i + thisShapeBegin] == 1 || otherShape.mShape[i + otherShapeBegin] == 1) {
            continue;
        }

        if (mShape[i + thisShapeBegin] != otherShape.mShape[i + otherShapeBegin]) {
            return false;
        }
    }

    return true;
}

Shape Shape::ExpandBefore(size_t targetSize) const {
    DDebugAssert(targetSize >= NumAxis());

    if (targetSize == NumAxis()) {
        return *this;
    }

    Shape result(targetSize);
    size_t expandDims = targetSize - NumAxis();

    for (size_t i = 0; i < expandDims; i++) {
        result.mShape[i] = 1;
    }
    for (size_t i = 0; i < NumAxis(); i++) {
        result.mShape[i + expandDims] = mShape[i];
    }

    return result;
}

Shape Shape::KeepSelectedDim(const std::unordered_set<size_t>& selectDims) const {
    Shape result;
    for (size_t i = 0; i < NumAxis(); i++) {
        if (selectDims.find(i) != selectDims.end()) {
            result.PushBack(mShape[i]);
        }
    }
    return result;
}

Shape Shape::RemoveSelectedDim(const std::unordered_set<size_t>& selectDims) const {
    Shape result;
    for (size_t i = 0; i < NumAxis(); i++) {
        if (selectDims.find(i) == selectDims.end()) {
            result.PushBack(mShape[i]);
        }
    }
    return result;
}

Shape Shape::BroadcastOutputShape(const Shape& shape, const Shape& otherShape) {
    size_t oneSize = shape.NumAxis();
    size_t otherSize = otherShape.NumAxis();
    size_t minSize = std::min(oneSize, otherSize);
    size_t maxSize = std::max(oneSize, otherSize);

    if (oneSize == 0) return otherShape;
    if (otherSize == 0) return shape;

    Shape result(maxSize);

    if (oneSize > otherSize) {
        for (size_t i = 0; i < oneSize - otherSize; i++) {
            result[i] = shape[i];
        }
    } else {
        for (size_t i = 0; i < otherSize - oneSize; i++) {
            result[i] = otherShape[i];
        }
    }

    size_t oneBegin = maxSize - otherSize;
    size_t otherBegin = maxSize - oneSize;
    size_t destBegin = maxSize - minSize;

    for (size_t i = 0; i < minSize; i++) {
        DDebugAssert(shape[i + oneBegin] == 1 || otherShape[i + otherBegin] == 1 ||
                     shape[i + oneBegin] == otherShape[i + otherBegin]);
        result[destBegin + i] = std::max(shape[oneBegin + i], otherShape[otherBegin + i]);
    }

    return result;
}

}  // namespace cpp
}  // namespace api
}  // namespace dtorch

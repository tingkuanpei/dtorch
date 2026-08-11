/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <initializer_list>
#include <string>
#include <unordered_set>
#include <vector>

#include "api_utilities.h"
#include "int_or_int_array.h"
#include "serialization.h"

namespace dtorch {
namespace api {
namespace cpp {

// Support shape with zero
// 1. Support dimention == 0, ie: shape = (), meaning scalar
// 2. Support shape element == 0, ie: shape = (0, 4), meaning tensor with no element
//
// a = torch.Tensor([])
// print(a)             # tensor([])
// print(a.shape)       # torch.Size([0])
// print(a.dim())       # 1
//
// b = torch.tensor(3.1415)
// print(b)             # tensor(3.1415)
// print(b.shape)       # torch.Size([])
// print(b.dim())       # 0

// Null tensor shape = (-100,), representing an absent optional input (e.g., bias in matmul).
class Shape {
public:
    using DataType = int64_t;

    DTORCH_API_FORCEINLINE static Shape GetScalarShape() { return Shape(0); }

    DTORCH_API_FORCEINLINE static bool IsScalar(const Shape& shape) { return shape.NumAxis() == 0; }

    DTORCH_API_FORCEINLINE static Shape GetNullShape() { return Shape(1, -100); }

public:
    // TODO: add explicit
    Shape(size_t dimention = 0, DataType value = 0);

    template <typename T>
    Shape(const std::initializer_list<T>& list) : mShape(list.size()) {
        for (size_t i = 0; i < mShape.size(); i++) {
            int64_t value = static_cast<int64_t>(list.begin()[i]);
            mShape[i] = value;
        }
    }

    template <typename T>
    Shape(const std::vector<T>& vec) : mShape(vec.size()) {
        for (size_t i = 0; i < mShape.size(); i++) {
            int64_t value = static_cast<int64_t>(vec[i]);
            mShape[i] = value;
        }
    }

    Shape(const IntOrIntArray& intArray);

    ~Shape() = default;

    DTORCH_API_FORCEINLINE bool Empty() const noexcept { return mShape.empty(); }

    DTORCH_API_FORCEINLINE size_t NumAxis() const noexcept { return mShape.size(); }

    DataType Count() const noexcept;

    bool IsSameShape(const Shape& otherShape) const noexcept;

    // https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md
    // https://numpy.org/doc/stable/user/basics.broadcasting.html
    // 1. they are equal, or
    // 2. one of them is 1
    // TODO: support shape with zero
    // shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar ==> shape(result) = (2, 3, 4, 5)
    // shape(A) = (2, 3, 4, 5), shape(B) = (5,), ==> shape(result) = (2, 3, 4, 5)
    // shape(A) = (4, 5), shape(B) = (2, 3, 4, 5), ==> shape(result) = (2, 3, 4, 5)
    bool CanBroadcastWith(const Shape& otherShape) const;

    DTORCH_API_FORCEINLINE DataType Prod(int begin_dim = 0) const noexcept {
        return Prod(begin_dim, static_cast<int>(NumAxis() - 1));
    }

    DataType Prod(int begin_dim, int end_dim) const noexcept;

    void Set(int index, DataType value) noexcept;

    DataType At(int index) const noexcept;

    DTORCH_API_FORCEINLINE const DataType* Data() const noexcept { return mShape.data(); }
    DTORCH_API_FORCEINLINE DataType* Data() noexcept { return mShape.data(); }

    bool operator==(const Shape& other) const noexcept;
    DTORCH_API_FORCEINLINE bool operator!=(const Shape& other) const noexcept { return !(this->operator==(other)); }

    const DataType& operator[](int index) const noexcept {
        return const_cast<DataType&>(static_cast<const Shape&>(*this)[static_cast<int64_t>(index)]);
    }
    DTORCH_API_FORCEINLINE DataType& operator[](int index) noexcept {
        return const_cast<DataType&>(static_cast<const Shape&>(*this)[static_cast<int64_t>(index)]);
    }
    const DataType& operator[](int64_t index) const noexcept;
    DTORCH_API_FORCEINLINE DataType& operator[](int64_t index) noexcept {
        return const_cast<DataType&>(static_cast<const Shape&>(*this)[index]);
    }
    const DataType& operator[](size_t index) const noexcept;
    DTORCH_API_FORCEINLINE DataType& operator[](size_t index) noexcept {
        return const_cast<DataType&>(static_cast<const Shape&>(*this)[index]);
    }

    DTORCH_API_FORCEINLINE void PushBack(DataType value) { mShape.push_back(value); }

    Shape ExpandBefore(size_t targetSize) const;

    Shape KeepSelectedDim(const std::unordered_set<size_t>& selectDims) const;

    Shape RemoveSelectedDim(const std::unordered_set<size_t>& selectDims) const;

    DTORCH_API_FORCEINLINE std::vector<DataType> Vec() const { return mShape; }

    DTORCH_API_FORCEINLINE bool IsNullTensorShape() const noexcept { return mShape.size() == 1 && mShape[0] == -100; }

    std::string ToString() const;

    DTORCH_API_FORCEINLINE friend std::ostream& operator<<(std::ostream& os, const Shape& shape) {
        os << shape.ToString();
        return os;
    }

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar & mShape;
    }

public:
    static Shape BroadcastOutputShape(const Shape& shape, const Shape& otherShape);

private:
    std::vector<DataType> mShape;
};

}  // namespace cpp
}  // namespace api
}  // namespace dtorch

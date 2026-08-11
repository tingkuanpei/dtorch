/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "dtorch/api/cpp/api_utilities.h"
#include "serialization.h"
#include "shape.h"

namespace dtorch {
namespace api {
namespace cpp {

class Stride {
public:
    using DataType = int64_t;

public:
    Stride(size_t dimention = 0, DataType value = 0);

    Stride(const Shape& shape);

    Stride(std::vector<DataType> list);

    Stride(std::initializer_list<DataType> list);

    ~Stride() = default;

    DTORCH_API_FORCEINLINE bool Empty() const noexcept { return mStride.empty(); }

    DTORCH_API_FORCEINLINE size_t NumAxis() const noexcept { return mStride.size(); }

    void Set(int index, DataType value) noexcept;

    DataType At(int index) const noexcept;

    DTORCH_API_FORCEINLINE const DataType* Data() const noexcept { return mStride.data(); }
    DTORCH_API_FORCEINLINE DataType* Data() noexcept { return mStride.data(); }

    bool operator==(const Stride& other) const noexcept;
    DTORCH_API_FORCEINLINE bool operator!=(const Stride& other) const noexcept { return !(this->operator==(other)); }

    const DataType& operator[](int index) const noexcept;
    DTORCH_API_FORCEINLINE DataType& operator[](int index) noexcept {
        return const_cast<DataType&>(static_cast<const Stride&>(*this)[index]);
    }
    const DataType& operator[](size_t index) const noexcept;
    DTORCH_API_FORCEINLINE DataType& operator[](size_t index) noexcept {
        return const_cast<DataType&>(static_cast<const Stride&>(*this)[index]);
    }

    DTORCH_API_FORCEINLINE void PushBack(DataType value) { mStride.push_back(value); }

    // Shape expand before with 1
    // Stride expand with shape[0] * stride[0], scaleValue is shape[0]
    Stride ExpandBefore(size_t dim, DataType scaleValue) const;

    template <typename T>
    T ComputeIndex(const std::vector<T>& coordinate) const;

    template <typename T>
    std::vector<T> ComputeCoordinate(T index) const;

    DTORCH_API_FORCEINLINE std::vector<DataType> Vec() const { return mStride; }

    std::string ToString() const;

    DTORCH_API_FORCEINLINE friend std::ostream& operator<<(std::ostream& os, const Stride& stride) {
        os << stride.ToString();
        return os;
    }

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar & mStride;
    }

private:
    std::vector<DataType> mStride;
};

template <typename T>
T Stride::ComputeIndex(const std::vector<T>& coordinate) const {
    DApiAssert(coordinate.size() == this->NumAxis());
    T result = 0;
    for (size_t i = 0; i < coordinate.size(); i++) {
        T coord = coordinate[i];
        DApiAssert(coord >= 0);
        result += coord * mStride[i];
    }
    return result;
}

template <typename T>
std::vector<T> Stride::ComputeCoordinate(T index) const {
    std::vector<T> result;
    for (size_t i = 0; i < NumAxis(); i++) {
        result.push_back(index / mStride[i]);
        index = index % mStride[i];
    }
    return result;
}

}  // namespace cpp
}  // namespace api
}  // namespace dtorch

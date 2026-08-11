/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "api_type.h"
#include "serialization.h"
#include "shape.h"

namespace dtorch {
namespace api {
namespace cpp {

class SimpleArray {
public:
    SimpleArray();

    SimpleArray(const Shape& shape, const std::vector<int64_t>& data,
                const std::vector<std::string>& dimensionNames = {});

    SimpleArray(const std::vector<int64_t>& data, const std::vector<std::string>& dimensionNames = {});

    SimpleArray(const torch::Tensor& torchTensor, const std::vector<std::string>& dimensionNames = {});

    DTORCH_API_FORCEINLINE const std::vector<int64_t>& GetData() const noexcept { return mData; }

    DTORCH_API_FORCEINLINE const std::unordered_set<int64_t>& GetDataSet() const noexcept { return mDataSet; }

    torch::Tensor ToTrochTensor() const;

    DTORCH_API_FORCEINLINE size_t NumAxis() const noexcept { return mShape.NumAxis(); }

    DTORCH_API_FORCEINLINE size_t Count() const noexcept { return mData.size(); }

    DTORCH_API_FORCEINLINE const Shape& GetShape() const noexcept { return mShape; }

    DTORCH_API_FORCEINLINE int64_t Size(int64_t dims) const noexcept { return mShape[dims]; }

    DTORCH_API_FORCEINLINE int64_t Size(const std::string& dimensionName) const noexcept {
        return mShape[GetDimensionNameIndex(dimensionName)];
    }

    DTORCH_API_FORCEINLINE bool HasDimensionName(const std::string& dimensionName) const noexcept {
        return mDimensionNamesMap.count(dimensionName) > 0;
    }

    DTORCH_API_FORCEINLINE size_t GetDimensionNameIndex(const std::string& dimensionName) const noexcept {
        DApiAssertMsg(mDimensionNamesMap.count(dimensionName) > 0, "Can't find dim name: " + dimensionName);
        return mDimensionNamesMap.at(dimensionName);
    }

    std::vector<std::string> GetDimensionNames() const noexcept;

    std::string ToString() const noexcept;

    DTORCH_API_FORCEINLINE friend std::ostream& operator<<(std::ostream& os, const SimpleArray& array) {
        os << array.ToString();
        return os;
    }

    bool operator==(const SimpleArray& other) const noexcept;

    DTORCH_API_FORCEINLINE bool operator!=(const SimpleArray& other) const noexcept {
        return !(this->operator==(other));
    }

    std::vector<SimpleArray> Unbind(const std::vector<int64_t>& dims) const;

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar & mShape;
        ar & mDimensionNamesMap;
        ar & mData;
        ar & mDataSet;
    }

private:
    Shape mShape;
    std::unordered_map<std::string, size_t> mDimensionNamesMap;
    std::vector<int64_t> mData;
    std::unordered_set<int64_t> mDataSet;
};

}  // namespace cpp
}  // namespace api
}  // namespace dtorch

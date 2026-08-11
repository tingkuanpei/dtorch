/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "simple_array.h"

#include <algorithm>
#include <cstdint>
#include <sstream>

#include <torch/torch.h>

#include "dtorch/common/debug.h"
#include "dtorch/common/logging.h"
#include "dtorch/common/string.h"
#include "dtorch/external/torch/torch_util.h"

namespace dtorch {
namespace api {
namespace cpp {

SimpleArray::SimpleArray() : mShape(), mDimensionNamesMap(), mData(), mDataSet() {}

SimpleArray::SimpleArray(const Shape& shape, const std::vector<int64_t>& data,
                         const std::vector<std::string>& dimensionNames)
    : mShape(shape), mDimensionNamesMap(), mData(data), mDataSet(data.begin(), data.end()) {
    DAlwaysAssert(mData.size() > 0);
    DAlwaysAssert(mShape.Prod() == static_cast<int64_t>(mData.size()));
    DAlwaysAssert(mData.size() == mDataSet.size());
    if (dimensionNames.size() > 0) {
        for (size_t i = 0; i < dimensionNames.size(); i++) {
            mDimensionNamesMap[dimensionNames[i]] = i;
        }

        if (mShape.NumAxis() != dimensionNames.size() || mDimensionNamesMap.size() != dimensionNames.size()) {
            throw std::invalid_argument("Invalid dim_names: " + String::ToString(dimensionNames));
        }
    }
}

SimpleArray::SimpleArray(const std::vector<int64_t>& data, const std::vector<std::string>& dimensionNames)
    : SimpleArray(Shape({data.size()}), data, dimensionNames) {}

SimpleArray::SimpleArray(const torch::Tensor& torchTensor, const std::vector<std::string>& dimensionNames)
    : SimpleArray(external::torch::TorchUtil::GetShape(torchTensor),
                  external::torch::TorchUtil::ToInt64Vec(torchTensor), dimensionNames) {}

torch::Tensor SimpleArray::ToTrochTensor() const {
    const char* dataPtr = reinterpret_cast<const char*>(mData.data());
    const size_t totalBytes = mData.size() * sizeof(int64_t);
    std::vector<char> dataBuffer(dataPtr, dataPtr + totalBytes);

    torch::Tensor torchTensor =
        external::torch::TorchUtil::CreateTensor(mShape, Device::GetDefaultCpuDevice(), DataKind::kInt64, dataBuffer);
    DDebugAssert(torchTensor.device().is_cpu());
    DDebugAssert(torchTensor.dtype() == torch::kInt64);
    DDebugAssert(torchTensor.numel() == static_cast<int64_t>(mData.size()));
    return torchTensor;
}

std::vector<std::string> SimpleArray::GetDimensionNames() const noexcept {
    std::vector<std::string> result;
    result.resize(mDimensionNamesMap.size());

    for (const auto& it : mDimensionNamesMap) {
        result[it.second] = it.first;
    }

    return result;
}

std::string SimpleArray::ToString() const noexcept {
    DDebugAssert(mShape.NumAxis() >= 1);
    int64_t lastDim = mShape[-1];
    DDebugAssert(lastDim > 0);
    std::stringstream ss;
    ss << "{Shape: " << mShape << " ";
    ss << "DimensionName: " << String::ToString(GetDimensionNames()) << " ";
    ss << "Data: " << String::ToString(mData) << "}";
    return ss.str();
}

bool SimpleArray::operator==(const SimpleArray& other) const noexcept {
    if (this == &other) {
        return true;
    }

    if (mShape != other.mShape) {
        return false;
    }

    if (mDimensionNamesMap != other.mDimensionNamesMap) {
        return false;
    }

    DDebugAssert(mData.size() == other.mData.size());
    for (size_t i = 0; i < mData.size(); i++) {
        if (mData[i] != other.mData[i]) {
            return false;
        }
    }

    return true;
}

std::vector<SimpleArray> SimpleArray::Unbind(const std::vector<int64_t>& dims) const {
    // validDims
    std::vector<int64_t> validDims;
    size_t numAixs = NumAxis();
    for (auto dim : dims) {
        int64_t validDim = dim < 0 ? dim + numAixs : dim;
        if (validDim < 0 || validDim > static_cast<int64_t>(numAixs)) {
            std::stringstream ss;
            ss << "Dimension out of range (expected to be in range of [-" << numAixs << ", " << numAixs - 1
               << "], but got " << dim << ")";
            throw std::invalid_argument(ss.str());
        }
        validDims.push_back(validDim);
    }
    std::sort(validDims.begin(), validDims.end());

    auto Imp = [](const SimpleArray& array, int64_t dim) {
        DDebugAssert(dim >= 0 && dim < static_cast<int64_t>(array.NumAxis()));

        std::vector<std::string> dimensionNames = array.GetDimensionNames();
        if (dimensionNames.size() > 0) {
            DDebugAssert(dim < static_cast<int64_t>(dimensionNames.size()));
            dimensionNames.erase(dimensionNames.begin() + dim);
        }

        std::vector<SimpleArray> result;
        torch::Tensor torchTensor = array.ToTrochTensor();

        for (auto unbindTensor : torchTensor.unbind(dim)) {
            result.push_back(SimpleArray(unbindTensor, dimensionNames));
        }
        return result;
    };

    // recursive
    std::vector<SimpleArray> tmpInput;
    std::vector<SimpleArray> tmpOutput;
    tmpInput.push_back(*this);

    for (int64_t i = static_cast<int64_t>(validDims.size()) - 1; i >= 0; i--) {
        DDebugAssert(tmpOutput.size() == 0);

        for (const auto& array : tmpInput) {
            auto tmp = Imp(array, validDims[i]);
            tmpOutput.insert(tmpOutput.end(), tmp.begin(), tmp.end());
        }

        tmpInput = std::move(tmpOutput);
        tmpOutput.clear();
    }

    return tmpInput;
}

}  // namespace cpp
}  // namespace api
}  // namespace dtorch

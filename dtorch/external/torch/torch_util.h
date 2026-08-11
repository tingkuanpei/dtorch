/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <sstream>
#include <unordered_set>
#include <vector>

#include "dtorch/core/type.h"

namespace c10d {
class Backend;
}  // namespace c10d

namespace c10 {
enum class ScalarType : int8_t;
class Scalar;
namespace cuda {
class CUDAStream;
}  // namespace cuda
}  // namespace c10

// forward declaration for ::torch::Tensor
namespace at {
class Tensor;
class Generator;
namespace cuda {
using c10::cuda::CUDAStream;
}
namespace indexing {
class TensorIndex;
class Slice;
}  // namespace indexing
}  // namespace at

namespace torch {
using at::Generator;
using at::Tensor;
using c10::ScalarType;
namespace indexing {
using at::indexing::Slice;
using at::indexing::TensorIndex;
}  // namespace indexing
}  // namespace torch

namespace dtorch {
namespace external {
namespace torch {

using core::DataKind;
using core::Device;
using core::DeviceKind;
using core::Index;
using core::Scalar;
using core::Shape;
using core::Slice;
using core::Stride;

class TorchUtil {
public:
    static size_t CudaDeviceCount();

    static Device ToDevice(const ::torch::Device& torchDevice);

    static ::torch::Device ToDevice(const Device& device);

    static DataKind ToDataKind(const ::torch::ScalarType& torchScalarType);

    static ::torch::ScalarType ToScalarType(DataKind dataKind);

    static c10::Scalar ToScalar(const Scalar& scalar);

    static Shape GetShape(const ::torch::Tensor& torchTensor);

    static Stride GetStride(const ::torch::Tensor& torchTensor);

    static DataKind GetDataKind(const ::torch::Tensor& torchTensor);

    static Device GetDevice(const ::torch::Tensor& torchTensor);

    static std::vector<int64_t> ToInt64Vec(const ::torch::Tensor& torchTensor);

    static std::unordered_set<int64_t> ToInt64Set(const ::torch::Tensor& torchTensor);

    static ::torch::indexing::Slice ToSlice(const Slice& slice);

    static ::torch::indexing::TensorIndex ToIndex(const Index& index);

    static std::vector<::torch::indexing::TensorIndex> ToIndex(const std::vector<Index>& indexs);

    static ::torch::Tensor CreateTensor(const Shape& shape, const Device& device, DataKind dataKind,
                                        const std::vector<char>& dataBuffer);

    static std::vector<char> ToCharVec(const ::torch::Tensor& torchTensor);

    static std::string ToIpcMemHandle(const ::torch::Tensor& torchTensor);

    static ::torch::Tensor FromIpcMemHandle(const std::string& str);

    static std::shared_ptr<::torch::Generator> GetGenerator(const Device& device);
};

using TorchTensorOptArray = std::vector<std::optional<::torch::Tensor>>;
using TorchTensorArray = std::vector<::torch::Tensor>;

std::string CUDAStreamToString(const c10::cuda::CUDAStream& stream);

std::ostream& operator<<(std::ostream& os, const c10::cuda::CUDAStream& stream);

}  // namespace torch
}  // namespace external
}  // namespace dtorch

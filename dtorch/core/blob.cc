/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "dtorch/core/blob.h"

#include <memory>
#include <mutex>

#include <ATen/core/TensorBody.h>
#include <torch/torch.h>

#include "dtorch/common/debug.h"
#include "dtorch/core/operand.h"
#include "dtorch/core/type.h"

namespace dtorch {
namespace core {

struct Blob::Impl {
    Impl() : mutex(), tensor() {}

    Impl(const std::shared_ptr<torch::Tensor>& tensor) : mutex(), tensor(tensor) {}

    ~Impl() {}

    std::mutex mutex;
    std::shared_ptr<torch::Tensor> tensor;
};

// Blob 必须唯一且支持拷贝，因此在初始化时就需要创建 mImplPtr。
Blob::Blob() : mImplPtr(std::make_shared<Blob::Impl>()) {}

Blob::~Blob() {}

bool Blob::IsEmpty() const noexcept {
    std::unique_lock<std::mutex> lock(mImplPtr->mutex);
    return mImplPtr->tensor == nullptr;
}

void Blob::SetTensor(const std::shared_ptr<torch::Tensor>& tensor) {
    std::unique_lock<std::mutex> lock(mImplPtr->mutex);
    DAlwaysAssert(mImplPtr->tensor == nullptr);
    mImplPtr->tensor = tensor;
}

std::shared_ptr<torch::Tensor> Blob::GetTensor() const {
    std::unique_lock<std::mutex> lock(mImplPtr->mutex);
    DAlwaysAssert(mImplPtr->tensor != nullptr);
    return mImplPtr->tensor;
}

void Blob::CreateTensor(const Shape& shape, const Device& device, DataKind dataKind) {
    torch::Device torchDevice = external::torch::TorchUtil::ToDevice(device);
    torch::ScalarType scalarType = external::torch::TorchUtil::ToScalarType(dataKind);

    auto options = torch::TensorOptions().dtype(scalarType).device(torchDevice);
    torch::Tensor tensor = torch::empty(shape.Vec(), options);

    SetTensor(std::make_shared<torch::Tensor>(tensor));
}

}  // namespace core
}  // namespace dtorch

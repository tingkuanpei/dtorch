/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <memory>
#include <unordered_map>

#include "dtorch/common/utilities.h"
#include "dtorch/external/torch/torch_util.h"

namespace dtorch {
namespace core {

class Operand;

class Blob {
public:
    Blob();

    ~Blob();

    DTORCH_DEFAULT_COPY_AND_MOVE(Blob);

    bool IsEmpty() const noexcept;

    void SetTensor(const std::shared_ptr<torch::Tensor>& tensor);

    std::shared_ptr<torch::Tensor> GetTensor() const;

    void CreateTensor(const Shape& shape, const Device& device, DataKind dataKind);

private:
    struct Impl;
    std::shared_ptr<Impl> mImplPtr;
};

using AllDeviceBlobs = std::unordered_map<int64_t, Blob>;

}  // namespace core
}  // namespace dtorch

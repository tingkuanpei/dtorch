/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <memory>
#include <optional>

#include "api_type.h"
#include "device.h"
#include "dtorch/common/utilities.h"

namespace dtorch {
namespace api {
namespace cpp {

class Generator {
public:
    Generator();

    Generator(const Device& device);

    Generator(const std::shared_ptr<torch::Generator>& torchGenerator);

    DTORCH_FORCEINLINE bool IsEmpty() const noexcept { return mImplPtr == nullptr; }

    Device GetDevice() const noexcept;

    torch::Generator GetTorchGenerator() const;

    std::shared_ptr<torch::Generator> GetSharedTorchGenerator() const;

    std::optional<torch::Generator> GetOptTorchGenerator() const;

private:
    struct Impl;
    mutable std::shared_ptr<Impl> mImplPtr;
};

}  // namespace cpp
}  // namespace api
}  // namespace dtorch

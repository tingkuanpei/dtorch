/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "generator.h"

#include <c10/util/intrusive_ptr.h>
#include <torch/extension.h>

#include "dtorch/common/config.h"
#include "dtorch/external/torch/torch_util.h"
#if DTORCH_WITH_CUDA
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif

#include "dtorch/common/debug.h"
#include "dtorch/common/logging.h"

namespace dtorch {
namespace api {
namespace cpp {

struct Generator::Impl {
    Impl(const Device& device) : torchGenerator(external::torch::TorchUtil::GetGenerator(device)) {}

    Impl(const std::shared_ptr<torch::Generator>& torchGenerator) : torchGenerator(torchGenerator) {}

    DTORCH_API_DISABLE_COPY_AND_MOVE(Impl);

    std::shared_ptr<torch::Generator> torchGenerator;
};

Generator::Generator() : mImplPtr() {}

Generator::Generator(const Device& device) : mImplPtr(std::make_shared<Impl>(device)) {}

Generator::Generator(const std::shared_ptr<torch::Generator>& torchGenerator)
    : mImplPtr(std::make_shared<Impl>(torchGenerator)) {}

Device Generator::GetDevice() const noexcept {
    DDebugAssert(mImplPtr != nullptr);
    return external::torch::TorchUtil::ToDevice(mImplPtr->torchGenerator->device());
}

torch::Generator Generator::GetTorchGenerator() const {
    DDebugAssert(mImplPtr != nullptr && mImplPtr->torchGenerator != nullptr);
    return *mImplPtr->torchGenerator.get();
}

std::shared_ptr<torch::Generator> Generator::GetSharedTorchGenerator() const {
    DDebugAssert(mImplPtr != nullptr);
    return mImplPtr->torchGenerator;
}

std::optional<torch::Generator> Generator::GetOptTorchGenerator() const {
    if (mImplPtr == nullptr) {
        return std::nullopt;
    } else {
        return GetTorchGenerator();
    }
}

}  // namespace cpp
}  // namespace api
}  // namespace dtorch

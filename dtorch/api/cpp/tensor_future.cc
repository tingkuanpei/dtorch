/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "tensor_future.h"

#include "dtorch/api/cpp/api_utilities.h"
#include "dtorch/api/cpp/tensor.h"
#include "dtorch/common/debug.h"
#include "dtorch/core/communication/promise_future/tensor_promise_future.h"
#include "dtorch/external/python/python_gil.h"

namespace dtorch {
namespace api {
namespace cpp {

struct TensorFuture::Impl {
    // Avoid tensor release before get future
    Tensor tensor;
    std::unique_ptr<core::communication::TensorFuture> futureImpl;

    explicit Impl(const Tensor& tensor, std::unique_ptr<core::communication::TensorFuture> f)
        : tensor(tensor), futureImpl(std::move(f)) {}

    DTORCH_API_DISABLE_COPY_AND_MOVE(Impl);
};

TensorFuture::TensorFuture(const Tensor& tensor, std::unique_ptr<core::communication::TensorFuture> futureImpl)
    : mImplPtr(std::make_shared<Impl>(tensor, std::move(futureImpl))) {}

TensorFuture::~TensorFuture() = default;

torch::Tensor TensorFuture::Get() {
    DAlwaysAssert(mImplPtr != nullptr);
    DAlwaysAssert(mImplPtr->futureImpl != nullptr);
    // Release Python GIL before blocking, same as PushMessageAndGetResult in the old sync path
    auto scopedRelease = external::python::GetPythonGilScopedRelease();
    auto result = mImplPtr->futureImpl->Get();
    DAlwaysAssert(result != nullptr);
    return *result;
}

void TensorFuture::Wait() {
    DAlwaysAssert(mImplPtr != nullptr);
    DAlwaysAssert(mImplPtr->futureImpl != nullptr);
    auto scopedRelease = external::python::GetPythonGilScopedRelease();
    mImplPtr->futureImpl->Wait();
}

bool TensorFuture::WaitFor(int64_t timeoutMs) {
    DAlwaysAssert(mImplPtr != nullptr);
    DAlwaysAssert(mImplPtr->futureImpl != nullptr);
    auto scopedRelease = external::python::GetPythonGilScopedRelease();
    return mImplPtr->futureImpl->WaitFor(timeoutMs);
}

bool TensorFuture::IsReady() const {
    DAlwaysAssert(mImplPtr != nullptr);
    DAlwaysAssert(mImplPtr->futureImpl != nullptr);
    return mImplPtr->futureImpl->IsReady();
}

}  // namespace cpp
}  // namespace api
}  // namespace dtorch

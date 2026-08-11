/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "void_future_collect.h"

#include <vector>

#include "dtorch/core/communication/promise_future/void_promise_future.h"
#include "dtorch/external/python/python_gil.h"

namespace dtorch {
namespace api {
namespace cpp {

struct VoidFutureCollect::Impl {
    std::vector<std::unique_ptr<core::communication::VoidFuture>> futures;

    Impl() : futures() {}
};

VoidFutureCollect::VoidFutureCollect() : mImplPtr(std::make_shared<Impl>()) {}

VoidFutureCollect::~VoidFutureCollect() = default;

void VoidFutureCollect::AddFuture(std::unique_ptr<core::communication::VoidFuture> future) {
    mImplPtr->futures.push_back(std::move(future));
}

void VoidFutureCollect::Get() {
    auto scopedRelease = external::python::GetPythonGilScopedRelease();
    for (auto& future : mImplPtr->futures) {
        future->Get();
    }
}

void VoidFutureCollect::Wait() {
    auto scopedRelease = external::python::GetPythonGilScopedRelease();
    for (auto& future : mImplPtr->futures) {
        future->Wait();
    }
}

bool VoidFutureCollect::IsReady() const {
    for (auto& future : mImplPtr->futures) {
        if (!future->IsReady()) {
            return false;
        }
    }
    return true;
}

}  // namespace cpp
}  // namespace api
}  // namespace dtorch

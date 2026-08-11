/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <memory>
#include <type_traits>

#include "cuda_error.h"
#include "dtorch/common/debug.h"
#include "dtorch/common/utilities.h"

namespace dtorch {
namespace external {
namespace cuda {

template <typename T>
class CudaObject {
    using BaseType = typename std::remove_pointer<T>::type;

    static_assert(std::is_same<T, BaseType*>(), "T MUST be pointer");

public:
    CudaObject() noexcept : mObjectPtr(nullptr) {}

    virtual ~CudaObject() = default;

    DTORCH_FORCEINLINE virtual bool IsCreate() const noexcept { return mObjectPtr != nullptr; }

    DTORCH_FORCEINLINE T Get() noexcept {
        DDebugAssert(IsCreate());
        return mObjectPtr.get();
    }

    DTORCH_FORCEINLINE const T Get() const noexcept {
        DDebugAssert(IsCreate());
        return mObjectPtr.get();
    }

protected:
    template <typename Type, typename Deleter>
    DTORCH_FORCEINLINE void Reset(Type* ptr, Deleter d) {
        mObjectPtr.reset(ptr, d);
    }

private:
    std::shared_ptr<BaseType> mObjectPtr;
};

}  // namespace cuda
}  // namespace external
}  // namespace dtorch

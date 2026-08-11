/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <cstdint>

#include "cuda_error.h"
#include "dtorch/common/debug.h"

namespace dtorch {
namespace external {
namespace cuda {

class CudaUtil {
public:
    static size_t GetMemOffeset(void* dataPtr) {
        CUdeviceptr pbase;
        size_t psize;
        CUdeviceptr dptr = static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(dataPtr));
        CudaCheckError(cuMemGetAddressRange(&pbase, &psize, dptr));
        DDebugAssert(dptr >= pbase);
        size_t storageOffset = static_cast<size_t>(dptr - pbase);
        DDebugAssert(storageOffset < psize);
        return storageOffset;
    }
};

}  // namespace cuda
}  // namespace external
}  // namespace dtorch

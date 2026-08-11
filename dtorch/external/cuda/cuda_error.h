/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <sstream>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

#include "dtorch/common/debug.h"
#include "dtorch/common/logging.h"
#include "dtorch/common/utilities.h"

#define DCudaCheckError(expr) dtorch::external::cuda::CudaCheckError(expr)

namespace dtorch {
namespace external {
namespace cuda {

DTORCH_FORCEINLINE void CudaCheckError(cudaError_t error) noexcept {
    if (error != cudaError_t::cudaSuccess) {
        std::stringstream ss;
        ss << "CUDA error name: " << cudaGetErrorName(error) << ". "
           << "CUDA error string: " << cudaGetErrorString(error);

        DLogFatal() << ss.str();
    }
}

DTORCH_FORCEINLINE void CudaCheckLastError() noexcept { CudaCheckError(cudaGetLastError()); }

DTORCH_FORCEINLINE void CudaCheckError(CUresult error) noexcept {
    if (error != CUresult::CUDA_SUCCESS) {
        const char *errorName = nullptr;
        cuGetErrorName(error, &errorName);
        std::stringstream ss;
        ss << "CUDA error name: " << (errorName ? errorName : "Unknown");
        DLogFatal() << ss.str();
    }
}

}  // namespace cuda
}  // namespace external
}  // namespace dtorch

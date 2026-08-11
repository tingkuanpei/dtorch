/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <string>

#include "dtorch/common/config.h"
#include "dtorch/common/debug.h"
#include "dtorch/external/cuda/cuda_error.h"

#if DTORCH_WITH_CUDA
#define UNvtxNameOsThread(name) dtorch::external::cuda::NvtxProfile::NameOsThread(name)
#else
#define UNvtxNameOsThread(name) dtorch::IgnoreUnused(name)
#endif

namespace dtorch {
namespace external {
namespace cuda {

#if DTORCH_WITH_CUDA

class NvtxProfile {
public:
    static void NameOsThread(const std::string& threadName);

    static void RangePush(const std::string& message);

    static void RangePop();

    static void Mark(const std::string& message);

    static void* StreamRangePush(cudaStream_t stream, const std::string& message);

    static void StreamRangePop(void* handle, cudaStream_t stream);

    static void NameStream(cudaStream_t stream, const std::string& name);

    static void NameEvent(cudaEvent_t event, const std::string& name);
};

#endif

}  // namespace cuda
}  // namespace external
}  // namespace dtorch

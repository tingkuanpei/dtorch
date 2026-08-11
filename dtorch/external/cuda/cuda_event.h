/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "cuda_object.h"

namespace dtorch {
namespace external {
namespace cuda {

class CudaEvent : public CudaObject<cudaEvent_t> {
public:
    CudaEvent() = default;

    void Create(unsigned int flags = cudaEventDefault);

    float ElapsedTime(CudaEvent endEvent) const noexcept;

    bool Query() const noexcept;

    DTORCH_FORCEINLINE void Record(cudaStream_t stream) noexcept { CudaCheckError(cudaEventRecord(Get(), stream)); }

    DTORCH_FORCEINLINE void Synchronize() const noexcept { CudaCheckError(cudaEventSynchronize(Get())); }

    DTORCH_FORCEINLINE void StreamWaitEvent(cudaStream_t stream) noexcept {
        CudaCheckError(cudaStreamWaitEvent(stream, Get()));
    }

    void OpenEventHandle(const std::string& handleStr);

    std::string GetIpcEventHandle() const noexcept;
};

}  // namespace cuda
}  // namespace external
}  // namespace dtorch

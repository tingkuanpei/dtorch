/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "cuda_event.h"

#include <cstring>

#include "dtorch/common/debug.h"

namespace dtorch {
namespace external {
namespace cuda {

void CudaEvent::Create(unsigned int flags) {
    cudaEvent_t event = nullptr;
    CudaCheckError(cudaEventCreate(&event, flags));

    Reset(event, [](cudaEvent_t p) { CudaCheckError(cudaEventDestroy(p)); });
}

float CudaEvent::ElapsedTime(CudaEvent endEvent) const noexcept {
    float time = 0;
    CudaCheckError(cudaEventElapsedTime(&time, Get(), endEvent.Get()));
    return time;
}

bool CudaEvent::Query() const noexcept {
    cudaError_t error = cudaEventQuery(Get());
    if (error == cudaError_t::cudaErrorNotReady) return false;
    CudaCheckError(error);
    return true;
}

void CudaEvent::OpenEventHandle(const std::string& handleStr) {
    cudaIpcEventHandle_t handle;
    DAlwaysAssert(handleStr.size() == CUDA_IPC_HANDLE_SIZE);
    std::memcpy(handle.reserved, handleStr.data(), CUDA_IPC_HANDLE_SIZE);

    cudaEvent_t event = nullptr;

    CudaCheckError(cudaIpcOpenEventHandle(&event, handle));
    Reset(event, [](cudaEvent_t p) { CudaCheckError(cudaEventDestroy(p)); });
}

std::string CudaEvent::GetIpcEventHandle() const noexcept {
    cudaIpcEventHandle_t handle;
    CudaCheckError(cudaIpcGetEventHandle(&handle, Get()));

    std::string handleStr;
    handleStr.resize(CUDA_IPC_HANDLE_SIZE);
    std::memcpy(handleStr.data(), handle.reserved, CUDA_IPC_HANDLE_SIZE);
    return handleStr;
}

}  // namespace cuda
}  // namespace external
}  // namespace dtorch

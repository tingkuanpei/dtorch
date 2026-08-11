/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "nvtx_profiler.h"

#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtCudaRt.h>

#include "dtorch/common/utilities.h"

#if DTORCH_PLATFORM_WINDOWS
#include <windows.h>
#elif DTORCH_PLATFORM_ANDROID
#include <unistd.h>
#elif DTORCH_PLATFORM_LINUX || DTORCH_PLATFORM_APPLE
#include <sys/syscall.h>
#endif

namespace dtorch {
namespace external {
namespace cuda {

// NvtxRangeHandle, StreamRangePush and StreamRangePop modified from torch\csrc\cuda\shared\nvtx.cpp
struct NvtxRangeHandle {
    nvtxRangeId_t id;
    const char* msg;
};

static void NvtxRangeEndCallback(void* userData) {
    NvtxRangeHandle* handle = ((NvtxRangeHandle*)userData);
    nvtxRangeEnd(handle->id);
    free((void*)handle->msg);
    free((void*)handle);
}

static void NvtxRangeStartCallback(void* userData) {
    NvtxRangeHandle* handle = ((NvtxRangeHandle*)userData);
    handle->id = nvtxRangeStartA(handle->msg);
}

void NvtxProfile::NameOsThread(const std::string& threadName) {
    // https://nvidia.github.io/NVTX/doxygen/group___r_e_s_o_u_r_c_e___n_a_m_i_n_g.html#gaeb7d5b25e1147fc4aaf1f3acb8e719d0
#if DTORCH_PLATFORM_WINDOWS
    nvtxNameOsThreadA(GetCurrentThreadId(), threadName.c_str());
#elif DTORCH_PLATFORM_ANDROID
    nvtxNameOsThreadA(gettid(), threadName.c_str());
#elif DTORCH_PLATFORM_LINUX
    nvtxNameOsThreadA(syscall(SYS_gettid), threadName.c_str());
#elif DTORCH_PLATFORM_APPLE
    nvtxNameOsThreadA(syscall(SYS_thread_selfid), threadName.c_str());
#else
#error Unsupport platform for nvtxNameOsThread.
#endif
}

void NvtxProfile::RangePush(const std::string& message) { nvtxRangePushA(message.c_str()); }

void NvtxProfile::RangePop() { nvtxRangePop(); }

void NvtxProfile::Mark(const std::string& message) { nvtxMarkA(message.c_str()); }

void* NvtxProfile::StreamRangePush(cudaStream_t stream, const std::string& message) {
    auto handle = static_cast<NvtxRangeHandle*>(calloc(1, sizeof(NvtxRangeHandle)));
    handle->msg = strdup(message.c_str());
    handle->id = 0;
    CudaCheckError(cudaLaunchHostFunc(stream, NvtxRangeStartCallback, (void*)handle));
    return handle;
}

void NvtxProfile::StreamRangePop(void* handle, cudaStream_t stream) {
    CudaCheckError(cudaLaunchHostFunc(stream, NvtxRangeEndCallback, handle));
}

void NvtxProfile::NameStream(cudaStream_t stream, const std::string& name) {
    nvtxNameCudaStreamA(stream, name.c_str());
}

void NvtxProfile::NameEvent(cudaEvent_t event, const std::string& name) { nvtxNameCudaEventA(event, name.c_str()); }

}  // namespace cuda
}  // namespace external
}  // namespace dtorch

/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "cuda_kernel_stream.h"

#include <sstream>
#include <string>

#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include "dtorch/common/debug.h"
#include "dtorch/external/cuda/nvtx_profiler.h"
#include "dtorch/external/torch/torch_util.h"

namespace dtorch {
namespace core {

//
void CudaKernelStream::InitInAsyncThread() {
    // Set stream and device id for this thread
    c10::cuda::set_device(GetDevice().deviceId);

    // Not use default stream, because default stream is a special stream, it will synchronize with other streams, which
    // will cause performance loss.
    //
    // Kernels in PyTorch run on the default stream by default. If you want to align with PyTorch's behavior,
    // you can use the code below:
    // mStream = std::make_shared<at::cuda::CUDAStream>(at::cuda::getDefaultCUDAStream(GetDevice().deviceId));
    //
    // Reference:
    // https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#summary-of-asynchronous-execution
    //      The default stream is a blocking stream which synchronizes with all other blocking streams, but does not
    //      synchronize with non-blocking streams

    mStream = std::make_shared<at::cuda::CUDAStream>(at::cuda::getStreamFromPool(true, GetDevice().deviceId));
    at::cuda::setCurrentCUDAStream(*mStream);
    TorchCudaStreamGuarantee::SetStream(*mStream);

    // Nvtx
    UNvtxNameOsThread("DTorchCudaStreamThreadForDevice" + std::to_string(GetDevice().deviceId));
    size_t cudaDeviceCount = external::torch::TorchUtil::CudaDeviceCount();
    for (size_t i = 0; i < cudaDeviceCount; i++) {
        auto stream = at::cuda::getCurrentCUDAStream(i);
        std::stringstream ss;
        if (i == static_cast<size_t>(GetDevice().deviceId)) {
            ss << "DTorchCudaStreamForMainDevice" << i;
        } else {
            ss << "DTorchCudaStreamForDevice" << i;
        }
        external::cuda::NvtxProfile::NameStream(stream, ss.str());
    }
}

void CudaKernelStream::SyncImp() {
    DDebugAssert(mStream != nullptr);
    mStream->synchronize();
    KernelStream::SyncImp();
}

thread_local std::shared_ptr<at::cuda::CUDAStream> TorchCudaStreamGuarantee::mStream = nullptr;

void TorchCudaStreamGuarantee::SetStream(at::cuda::CUDAStream& stream) {
    DAlwaysAssert(mStream == nullptr);
    mStream = std::make_shared<at::cuda::CUDAStream>(stream);
}

void TorchCudaStreamGuarantee::CheckStream(const at::cuda::CUDAStream& stream) {
    DAlwaysAssert(mStream != nullptr);
    if (*mStream != stream) {
        DLogFatal() << "Stream mismatch, you can't switch cuda stream in kernel, expected: " << *mStream
                    << ", actual: " << stream;
    }
}

}  // namespace core
}  // namespace dtorch

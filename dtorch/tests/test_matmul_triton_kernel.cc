/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include "dtorch/api/cpp/device.h"
#include "test.h"

// From https://github.com/triton-lang/triton/blob/v3.1.0/python/test/unit/tools/test_aot.py

const std::string kKernelUtilTritonKernelBundle = R"(
{
    "file_name": "kernel_utils.py",
    "src_code": "$kKernelUtilTritonKernelSrc"
}
)";

const std::string kKernelUtilTritonKernelSrc = R"(
import triton

@triton.jit
def mul(x, y):
    return x * y
)";

const std::string kKernelTritonKernelBundle = R"(
{
    "file_name": "matmul_fp16_kernel.py",
    "src_code": "$kKernelTritonKernelSrc",
    "name": "matmul_fp16",
    "sig": "*fp32, *fp16, *fp16, i32, i32, i32, i32, i32, i32, i32, i32, i32, 16, 16, 16",
    "grid": "M/16, N/16, 1",
    "num_warps": 1
}
)";

const std::string kKernelTritonKernelSrc = R"(
import triton
import triton.language as tl
import kernel_utils

@triton.jit
def matmul_fp16(C, A, B, M, N, K,
           stride_cm, stride_cn,
           stride_am, stride_ak,
           stride_bk, stride_bn,
           BLOCK_M: tl.constexpr,
           BLOCK_N: tl.constexpr,
           BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = kernel_utils.mul(accumulator, accumulator)
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c)
)";

#include "triton_aot_kernel.h"

torch::Tensor TritonMatMul(at::cuda::CUDAStream stream, torch::Tensor tensorA, torch::Tensor tensorB) {
    const int32_t M = 16, N = 16, K = 16;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor result = torch::empty({M, N}, options);

    load_matmul_fp16();
    matmul_fp16_default(stream, reinterpret_cast<CUdeviceptr>(result.data_ptr()),
                        reinterpret_cast<CUdeviceptr>(tensorA.data_ptr()),
                        reinterpret_cast<CUdeviceptr>(tensorB.data_ptr()), M, N, K, N, 1, K, 1, N, 1);

    return result;
}

using namespace dtorch;
using namespace dtorch::api::cpp;

TEST(TritonKernelTest, SimpleTest) {
    if (!Device::IsAvailable(DeviceKind::kGpu)) {
        return;
    }

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    const int32_t M = 16, N = 16, K = 16;
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
    torch::Tensor tensorA = torch::randn({M, K}, options);
    torch::Tensor tensorB = torch::randn({K, N}, options);

    torch::Tensor tensorAFp32 = tensorA.to(torch::TensorOptions().dtype(torch::kFloat32));
    torch::Tensor tensorBFp32 = tensorB.to(torch::TensorOptions().dtype(torch::kFloat32));
    torch::Tensor refTensorC = torch::matmul(tensorAFp32, tensorBFp32);
    refTensorC = refTensorC.mul(refTensorC);

    torch::Tensor tritonTensorC = TritonMatMul(stream, tensorA, tensorB);
    stream.synchronize();

    EXPECT_TRUE(torch::allclose(refTensorC, tritonTensorC, 0.0, 1e-4));
}

/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "create_kernel.h"

#include <optional>

#include <ATen/core/Generator.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/randint.h>
#include <torch/torch.h>

#include "dtorch/common/debug.h"
#include "dtorch/core/kernel_stream/kernel_stream.h"
#include "dtorch/core/operators/standard/create_op.h"
#include "dtorch/external/torch/torch_util.h"

namespace dtorch {
namespace core {

void CreateKernel::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) {
    DAlwaysAssert(mNumKernelForThisOp == mOp->OperandY()->GetDeviceMesh().Count());
    const auto& param = GetOpParam<CreateParam>();
    DDebugAssert(inputs.size() == 0);

    torch::Device torchDevice = external::torch::TorchUtil::ToDevice(mStream->GetDevice());
    torch::ScalarType scalarType = external::torch::TorchUtil::ToScalarType(param.dataKind);
    Shape outputShape = mOp->OperandY()->GetLocalShape(GetGlobalDeviceId());

    auto options = torch::TensorOptions().dtype(scalarType).device(torchDevice);
    std::optional<torch::Generator> generator;
    if (param.generator.has_value()) {
        generator = param.generator.value().GetOptTorchGenerator();
    }

    switch (param.createKind) {
        case CreateKind::kZeros:
            outputs.push_back(torch::zeros(outputShape.Vec(), options));
            break;
        case CreateKind::kEmpty:
            // For Tensor::GetNullTensorLike
            // Null tensor shape = (-100,), representing an absent optional input (e.g., bias in matmul).
            if (outputShape.IsNullTensorShape()) {
                outputShape = Shape({0});
            }
            outputs.push_back(torch::empty(outputShape.Vec(), options));
            break;
        case CreateKind::kOnes:
            outputs.push_back(torch::ones(outputShape.Vec(), options));
            break;
        case CreateKind::kArange:
            outputs.push_back(torch::arange(param.doubleArg0, param.doubleArg1, param.doubleArg2, options));
            break;
        case CreateKind::kRandInt: {
            // TODO: add param.generator's device check & add for distributed generator
            outputs.push_back(
                torch::randint(param.doubleArg0, param.doubleArg1, outputShape.Vec(), generator, options));
        } break;
        case CreateKind::kRand: {
            outputs.push_back(torch::rand(outputShape.Vec(), generator, options));
        } break;
        case CreateKind::kRandn: {
            outputs.push_back(torch::randn(outputShape.Vec(), generator, options));
        } break;
        case CreateKind::kFull: {
            outputs.push_back(torch::full(outputShape.Vec(), param.doubleArg0, options));
        } break;
        case CreateKind::kFromTorch: {
            DAlwaysAssert(!param.deviceMesh.IsDistributed());
            DAlwaysAssert(param.torchValue.has_value());
            DAlwaysAssert(param.deviceMesh.ToDevice() ==
                          external::torch::TorchUtil::GetDevice(param.torchValue.value()));
            // param.torchValue need to clone here, because
            // 1. the tensor may be created in another process, we need to clone it to avoid race condition.
            // 2. torch may modify this tensor before dtorch use it in other kernels.
            outputs.push_back(param.torchValue.value().clone());
        } break;
        default:
            DUnimplemented();
            break;
    }
}

}  // namespace core
}  // namespace dtorch

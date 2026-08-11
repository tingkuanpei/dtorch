/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include <ATen/ops/all.h>
#include <ATen/ops/max.h>
#include <ATen/ops/rsqrt.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/where.h>
#include <nanobind/nanobind.h>
#include <torch/torch.h>

#include "dtorch/common/config.h"
#include "dtorch/common/debug.h"
#include "dtorch/common/logging.h"
#include "dtorch/core/operators/standard/activation_op.h"
#include "dtorch/core/operators/standard/base_math_op.h"
#include "dtorch/core/operators/standard/batch_norm_op.h"
#include "dtorch/core/operators/standard/chunk_op.h"
#include "dtorch/core/operators/standard/clamp_op.h"
#include "dtorch/core/operators/standard/clone_op.h"
#include "dtorch/core/operators/standard/concat_op.h"
#include "dtorch/core/operators/standard/contiguous_op.h"
#include "dtorch/core/operators/standard/conv_op.h"
#include "dtorch/core/operators/standard/dropout_op.h"
#include "dtorch/core/operators/standard/einsum_op.h"
#include "dtorch/core/operators/standard/embedding_op.h"
#include "dtorch/core/operators/standard/expand_op.h"
#include "dtorch/core/operators/standard/flatten_op.h"
#include "dtorch/core/operators/standard/get_item_op.h"
#include "dtorch/core/operators/standard/interpolate_op.h"
#include "dtorch/core/operators/standard/linear_op.h"
#include "dtorch/core/operators/standard/masked_op.h"
#include "dtorch/core/operators/standard/matmul_op.h"
#include "dtorch/core/operators/standard/max_min_op.h"
#include "dtorch/core/operators/standard/normalization_op.h"
#include "dtorch/core/operators/standard/outer_op.h"
#include "dtorch/core/operators/standard/pad_op.h"
#include "dtorch/core/operators/standard/permute_op.h"
#include "dtorch/core/operators/standard/pooling_op.h"
#include "dtorch/core/operators/standard/repeat_interleave_op.h"
#include "dtorch/core/operators/standard/repeat_op.h"
#include "dtorch/core/operators/standard/set_item_op.h"
#include "dtorch/core/operators/standard/softmax_op.h"
#include "dtorch/core/operators/standard/squeeze_op.h"
#include "dtorch/core/operators/standard/transpose_op.h"
#include "dtorch/core/operators/standard/unsqueeze_op.h"
#include "dtorch/core/operators/standard/where_op.h"
#include "dtorch/external/python/nanobind_util.h"
#include "dtorch/external/python/python_gil.h"
#include "dtorch/external/torch/torch_stream_guard.h"
#include "dtorch/external/torch/torch_util.h"

using dtorch::external::python::NanobindUtil;

// Gather all torch-related functions to speed up compilation speed.

namespace dtorch {
namespace core {

void ActivationOp::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    const auto& param = GetOpParam<ActivationParam>();
    torch::Tensor input = inputs[0].value();
    switch (param.activationType) {
        case ActivationType::kReLU:
            outputs.push_back(torch::nn::functional::relu(input, param.inplace));
            break;
        case ActivationType::kSigmoid:
            outputs.push_back(torch::sigmoid(input));
            break;
        case ActivationType::kLeakyRelu:
            if (param.inplace) {
                outputs.push_back(torch::leaky_relu_(input, param.alpha));
            } else {
                outputs.push_back(torch::leaky_relu(input, param.alpha));
            }
            break;
        case ActivationType::kELU:
            if (param.inplace) {
                outputs.push_back(torch::elu_(input, param.alpha));
            } else {
                outputs.push_back(torch::elu(input, param.alpha));
            }
            break;
        case ActivationType::kGELU: {
            std::string approximate = param.approximate;
            if (approximate == "") {
                approximate = "none";
            }
            outputs.push_back(torch::gelu(input, approximate));
        } break;
        case ActivationType::kSiLU:
            if (param.inplace) {
                outputs.push_back(torch::silu_(input));
            } else {
                outputs.push_back(torch::silu(input));
            }
            break;
        default:
            DLogError() << "Unsupport param.activationType: " << EnumAsInteger(param.activationType);
            DUnimplemented();
            break;
    }
}

void BaseMathOp::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    const auto& param = GetOpParam<BaseMathParam>();
    const auto& input = inputs[0].value();
    switch (param.baseMathType) {
        case BaseMathType::kExp:
            outputs.push_back(torch::exp(input));
            break;
        case BaseMathType::kSquare:
            outputs.push_back(torch::square(input));
            break;
        case BaseMathType::kRsqrt:
            outputs.push_back(torch::rsqrt(input));
            break;
        case BaseMathType::kAbs:
            outputs.push_back(torch::abs(input));
            break;
        case BaseMathType::kRound:
            outputs.push_back(torch::round(input));
            break;
        case BaseMathType::kFloor:
            outputs.push_back(torch::floor(input));
            break;
        case BaseMathType::kCos:
            outputs.push_back(torch::cos(input));
            break;
        case BaseMathType::kSin:
            outputs.push_back(torch::sin(input));
            break;
        case BaseMathType::kAsin:
            outputs.push_back(torch::asin(input));
            break;
        case BaseMathType::kTanh:
            outputs.push_back(torch::tanh(input));
            break;
        case BaseMathType::kNeg:
            outputs.push_back(torch::neg(input));
            break;
        case BaseMathType::kReciprocal:
            outputs.push_back(torch::reciprocal(input));
            break;
        case BaseMathType::kLog:
            outputs.push_back(torch::log(input));
            break;
        case BaseMathType::kLog2:
            outputs.push_back(torch::log2(input));
            break;
        case BaseMathType::kLog10:
            outputs.push_back(torch::log10(input));
            break;
        case BaseMathType::kIsInf:
            outputs.push_back(torch::isinf(input));
            break;
        case BaseMathType::kIsNan:
            outputs.push_back(torch::isnan(input));
            break;
        default:
            DLogError() << "Unsupport param.baseMathType: " << EnumAsInteger(param.baseMathType);
            DUnimplemented();
            break;
    }
}

void BatchNormOp::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    const auto& param = GetOpParam<BatchNormParam>();
    DDebugAssert(inputs.size() == 5);
    outputs.push_back(torch::batch_norm(inputs[0].value(), inputs[3], inputs[4], inputs[1], inputs[2], false,
                                        param.momentum, param.epsilon, true));
}

void ChunkOp::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    DDebugAssert(inputs.size() == 1);
    const auto& param = GetOpParam<ChunkParam>();
    outputs = inputs[0].value().chunk(param.chunks, param.dim);
    DDebugAssert(outputs.size() == GetOutputSize());
}

void ClampOp::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    DDebugAssert(inputs.size() == 3);
    const auto& param = GetOpParam<ClampParam>();
    const auto& input = inputs[0].value();
    if (param.min.has_value() || param.max.has_value()) {
        std::optional<at::Scalar> min;
        if (param.min.has_value()) {
            min = param.min.value();
        }
        std::optional<at::Scalar> max;
        if (param.max.has_value()) {
            max = param.max.value();
        }
        outputs.push_back(torch::clamp(input, min, max));
    } else {
        outputs.push_back(torch::clamp(input, inputs[1], inputs[2]));
    }
}

void ConcatOp::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    const auto& param = GetOpParam<ConcatParam>();
    std::vector<torch::Tensor> tensorList;
    for (size_t i = 0; i < inputs.size(); i++) {
        tensorList.push_back(inputs[i].value());
    }
    outputs.push_back(torch::concat(tensorList, param.dim));
}

void ContiguousOp::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    DDebugAssert(inputs.size() == 1);
    outputs.push_back(inputs[0].value().contiguous());
}

void CloneOp::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    DDebugAssert(inputs.size() == 1);
    outputs.push_back(inputs[0].value().clone());
}

void ConvOp::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    const auto& param = GetOpParam<ConvParam>();
    DDebugAssert(inputs.size() == 3);
    std::vector<int64_t> pads = {param.pads[0], param.pads[1]};
    outputs.push_back(torch::conv2d(inputs[0].value(), inputs[1].value(), inputs[2], param.strides, pads,
                                    param.dilations, param.group));
}

void DropoutOp::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    const auto& param = GetOpParam<DropoutParam>();
    DDebugAssert(inputs.size() == 1);
    outputs.push_back(torch::dropout(inputs[0].value(), param.probability, false));
}

void EinsumOp::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    DDebugAssert(inputs.size() > 0);
    const auto& param = GetOpParam<EinsumParam>();
    std::vector<torch::Tensor> tensorList;
    for (size_t i = 0; i < inputs.size(); i++) {
        tensorList.push_back(inputs[i].value());
    }
    outputs.push_back(torch::einsum(param.equation, tensorList));
}

void InterpolateOp::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    DDebugAssert(inputs.size() == 1);
    const auto& param = GetOpParam<InterpolateParam>();

    auto options = torch::nn::functional::InterpolateFuncOptions();
    if (param.shape.has_value()) {
        options = options.size(param.shape.value().Vec());
    }
    if (param.scaleFactor.has_value()) {
        std::vector<double> scaleFactorVec = param.scaleFactor.value();
        const Shape& inputShape = OperandX()->GetShape();
        DDebugAssert(inputShape.NumAxis() > 2);
        if (scaleFactorVec.size() == 1 && scaleFactorVec.size() != inputShape.NumAxis() - 2) {
            scaleFactorVec = std::vector<double>(scaleFactorVec[0], inputShape.NumAxis() - 2);
        }
        options = options.scale_factor(scaleFactorVec);
    }
    if (param.mode == "nearest") {
        options = options.mode(torch::kNearest);
    } else if (param.mode == "linear") {
        options = options.mode(torch::kLinear);
    } else if (param.mode == "bilinear") {
        options = options.mode(torch::kBilinear);
    } else if (param.mode == "bicubic") {
        options = options.mode(torch::kBicubic);
    } else if (param.mode == "trilinear") {
        options = options.mode(torch::kTrilinear);
    } else if (param.mode == "area") {
        options = options.mode(torch::kArea);
    } else if (param.mode == "nearest-exact") {
        options = options.mode(torch::kNearestExact);
    } else {
        DLogError() << "Unsupport mode: " << param.mode;
        DUnsupportedImpl();
    }
    if (param.alignCorners.has_value()) {
        options = options.align_corners(param.alignCorners.value());
    }
    if (param.recomputeScaleFactor) {
        options = options.recompute_scale_factor(param.recomputeScaleFactor.value());
    }
    if (param.antialias) {
        options = options.antialias(param.antialias.value());
    }

    outputs.push_back(torch::nn::functional::interpolate(inputs[0].value(), options));
}

void EmbeddingOp::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    DDebugAssert(inputs.size() == 2);
    outputs.push_back(torch::embedding(inputs[1].value(), inputs[0].value()));
}

void ExpandOp::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    const auto& param = GetOpParam<ExpandParam>();
    DDebugAssert(inputs.size() == 1);
    outputs.push_back(inputs[0].value().expand(param.shape.Vec()));
}

void FlattenOp::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    const auto& param = GetOpParam<FlattenParam>();
    DDebugAssert(inputs.size() == 1);
    outputs.push_back(torch::flatten(inputs[0].value(), param.startDim, param.endDim));
}

void LinearOp::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    DDebugAssert(inputs.size() == 3);
    outputs.push_back(torch::linear(inputs[0].value(), inputs[1].value(), inputs[2]));
}

void NormalizationOp::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    const auto& param = GetOpParam<NormalizationParam>();
    DDebugAssert(inputs.size() == 3);
    if (param.normKind == NormalizationKind::kLayerNorm) {
        outputs.push_back(
            torch::layer_norm(inputs[0].value(), param.normalizedShape.Vec(), inputs[1], inputs[2], param.epsilon));
    } else if (param.normKind == NormalizationKind::kGroupNorm) {
        outputs.push_back(torch::group_norm(inputs[0].value(), param.numGroups, inputs[1], inputs[2], param.epsilon));
    }
#if !DTORCH_INTEL_MAXOS_TORCH_2_2_2
    else if (param.normKind == NormalizationKind::kRmsNorm) {
        outputs.push_back(torch::rms_norm(inputs[0].value(), param.normalizedShape.Vec(), inputs[1], param.epsilon));
        // auto scopedAcquire = external::python::GilScopedAcquire();
        // auto streamGuard = external::torch::PythonCodeCudaStreamGuard();

        // const auto& weight = inputs[1].has_value() ? NanobindUtil::ToObject(inputs[1].value()) : nb::none();

        // nb::module_ compileKernelModule = nb::module_::import_("dtorch.compiled_op");
        // nb::object funcOut = compileKernelModule.attr("rms_norm")(
        //     NanobindUtil::ToObject(inputs[0].value()), param.normalizedShape.Vec(), weight, param.epsilon);

        // outputs = NanobindUtil::ToTensorArray(funcOut);
    }
#endif
    else {
        DUnimplemented();
    }
}

void MaxMinOp::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    const auto& param = GetOpParam<MaxMinParam>();
    DDebugAssert(inputs.size() == 1);
    if (param.maxMinKind == MaxMinKind::kMax) {
        if (!param.dim.has_value()) {
            outputs.push_back(torch::max(inputs[0].value()));
        } else {
            const auto [value, indice] = torch::max(inputs[0].value(), param.dim.value(), param.keepdim);
            outputs.push_back(value);
            outputs.push_back(indice);
        }
    } else {
        DDebugAssert(param.maxMinKind == MaxMinKind::kMin);
        if (!param.dim.has_value()) {
            outputs.push_back(torch::min(inputs[0].value()));
        } else {
            const auto [value, indice] = torch::min(inputs[0].value(), param.dim.value(), param.keepdim);
            outputs.push_back(value);
            outputs.push_back(indice);
        }
    }
}

void MatmulOp::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    DDebugAssert(inputs.size() == 2);
    outputs.push_back(torch::matmul(inputs[0].value(), inputs[1].value()));
}

void OuterOp::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    DDebugAssert(inputs.size() == 2);
    outputs.push_back(torch::outer(inputs[0].value(), inputs[1].value()));
}

void PadOp::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    const auto& param = GetOpParam<PadParam>();
    DDebugAssert(inputs.size() == 1);
    outputs.push_back(torch::pad(inputs[0].value(), param.pad, param.mode, param.value));
}

void PoolingOp::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    const auto& param = GetOpParam<PoolingParam>();
    DDebugAssert(inputs.size() == 1);
    std::vector<int64_t> pads = {param.pads[0], param.pads[1]};
    if (param.poolingKind == PoolingKind::kMax) {
        outputs.push_back(torch::max_pool2d(inputs[0].value(), param.kernelSize, param.strides, pads, param.dilations,
                                            param.ceilMode));
    } else {
        outputs.push_back(torch::avg_pool2d(inputs[0].value(), param.kernelSize, param.strides, pads, param.ceilMode,
                                            param.countIncludePad));
    }
}

void PermuteOp::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    const auto& param = GetOpParam<PermuteParam>();
    DDebugAssert(inputs.size() == 1);
    outputs.push_back(torch::permute(inputs[0].value(), param.dims));
}

void RepeatInterleaveOp::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    const auto& param = GetOpParam<RepeatInterleaveParam>();
    DDebugAssert(inputs.size() == 1);
    outputs.push_back(torch::repeat_interleave(inputs[0].value(), param.repeats, param.dim));
}

void SoftmaxOp::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    const auto& param = GetOpParam<SoftmaxParam>();
    DDebugAssert(inputs.size() == 1);
    outputs.push_back(torch::softmax(inputs[0].value(), param.dim));
}

void SqueezeOp::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    const auto& param = GetOpParam<SqueezeParam>();
    DDebugAssert(inputs.size() == 1);
    if (param.dims.size() == 0) {
        outputs.push_back(torch::squeeze(inputs[0].value()));
    } else {
        outputs.push_back(torch::squeeze(inputs[0].value(), param.dims));
    }
}

void TransposeOp::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    const auto& param = GetOpParam<TransposeParam>();
    DDebugAssert(inputs.size() == 1);
    outputs.push_back(torch::transpose(inputs[0].value(), param.dim0, param.dim1));
}

void UnsqueezeOp::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    const auto& param = GetOpParam<UnsqueezeParam>();
    DDebugAssert(inputs.size() == 1);
    outputs.push_back(torch::unsqueeze(inputs[0].value(), param.dim));
}

void RepeatOp::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    const auto& param = GetOpParam<RepeatParam>();
    DDebugAssert(inputs.size() == 1);
    outputs.push_back(inputs[0].value().repeat(param.repeat));
}

void WhereOp::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    const auto& param = GetOpParam<WhereParam>();
    DDebugAssert(inputs.size() == 3);

    torch::Tensor output;
    if (param.input.has_value() && param.other.has_value()) {
        output = torch::where(inputs[0].value(), param.input.value(), param.other.value());
    } else if (param.input.has_value() && !param.other.has_value()) {
        output = torch::where(inputs[0].value(), param.input.value(), inputs[2].value());
    } else if (!param.input.has_value() && param.other.has_value()) {
        output = torch::where(inputs[0].value(), inputs[1].value(), param.other.value());
    } else if (!param.input.has_value() && !param.other.has_value()) {
        output = torch::where(inputs[0].value(), inputs[1].value(), inputs[2].value());
    } else {
        DUnimplemented();
    }

    outputs.push_back(output);
}

void MaskedOp::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    const auto& param = GetOpParam<MaskedParam>();
    torch::Tensor output;
    switch (param.maskedType) {
        case MaskedType::kMaskedFill: {
            DDebugAssert(inputs.size() == 2);
            output = inputs[0].value().masked_fill(inputs[1].value(), param.value);
            break;
        }
        case MaskedType::kMaskedScatter: {
            DDebugAssert(inputs.size() == 3);
            output = inputs[0].value().masked_scatter(inputs[1].value(), inputs[2].value());
            break;
        }
    }
    outputs.push_back(output);
}

void SetItemOp::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    DDebugAssert(inputs.size() == 2);
    auto result = inputs[0].value();
    result.index_put_(external::torch::TorchUtil::ToIndex(mIndexVec), inputs[1].value());
    outputs.push_back(std::move(result));
}

}  // namespace core
}  // namespace dtorch

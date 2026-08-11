/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "layer_norm_mul_add.h"

#include <stdexcept>

#include <nanobind/nanobind.h>

#include "dtorch/external/python/nanobind_util.h"
#include "dtorch/external/python/python_gil.h"
#include "dtorch/external/torch/torch_stream_guard.h"

using dtorch::external::python::NanobindUtil;

namespace dtorch {
namespace core {

void LayerNormMulAddOp::InferOutputMetaInfo() const {
    DDebugAssert(GetInputSize() == 5 && GetOutputSize() == 1);
    OperandY()->MetaDataSameAs(OperandX());
}

void LayerNormMulAddOp::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    const auto& param = GetOpParam<LayerNormMulAddParam>();
    auto scopedAcquire = external::python::GilScopedAcquire();
    auto streamGuard = external::torch::PythonCodeCudaStreamGuard();

    const auto& x = NanobindUtil::ToObject(inputs[0].value());
    const auto& scale = NanobindUtil::ToObject(inputs[1].value());
    const auto& shift = NanobindUtil::ToObject(inputs[2].value());
    const auto& normScale = inputs[3].has_value() ? NanobindUtil::ToObject(inputs[3].value()) : nb::none();
    const auto& normBias = inputs[4].has_value() ? NanobindUtil::ToObject(inputs[4].value()) : nb::none();

    nb::module_ compileKernelModule = nb::module_::import_("dtorch.compiled_op");
    nb::object funcOut = compileKernelModule.attr("layer_norm_mul_add")(x, scale, shift, param.normalizedShape.Vec(),
                                                                        param.epsilon, normScale, normBias);

    outputs = NanobindUtil::ToTensorArray(funcOut);
}

}  // namespace core
}  // namespace dtorch

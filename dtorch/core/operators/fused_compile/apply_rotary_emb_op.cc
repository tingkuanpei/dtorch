/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "apply_rotary_emb_op.h"

#include <stdexcept>

#include <nanobind/nanobind.h>

#include "dtorch/external/python/nanobind_util.h"
#include "dtorch/external/python/python_gil.h"
#include "dtorch/external/torch/torch_stream_guard.h"

using dtorch::external::python::NanobindUtil;

namespace dtorch {
namespace core {

void ApplyRotaryEmbOp::InferOutputMetaInfo() const {
    DDebugAssert(GetInputSize() == 3 && GetOutputSize() == 1);
    OperandY()->MetaDataSameAs(OperandX());
}

void ApplyRotaryEmbOp::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    const auto& param = GetOpParam<ApplyRotaryEmbParam>();
    const auto& x = inputs[0].value();
    const auto& cos = inputs[1].value();
    const auto& sin = inputs[2].value();

    auto scopedAcquire = external::python::GilScopedAcquire();
    auto streamGuard = external::torch::PythonCodeCudaStreamGuard();

    nb::module_ compileKernelModule = nb::module_::import_("dtorch.compiled_op");
    nb::object funcOut = compileKernelModule.attr("apply_rotary_emb")(
        NanobindUtil::ToObject(x), NanobindUtil::ToObject(cos), NanobindUtil::ToObject(sin), param.useReal,
        param.useRealUnbindDim);

    outputs = NanobindUtil::ToTensorArray(funcOut);
}

}  // namespace core
}  // namespace dtorch

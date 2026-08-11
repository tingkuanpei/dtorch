/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "silu_linear_chunk.h"

#include <stdexcept>

#include <nanobind/nanobind.h>

#include "dtorch/external/python/nanobind_util.h"
#include "dtorch/external/python/python_gil.h"
#include "dtorch/external/torch/torch_stream_guard.h"

using dtorch::external::python::NanobindUtil;

namespace dtorch {
namespace core {

size_t SiluLinearChunkOp::InferOutputSize() const {
    const auto& param = GetOpParam<SiluLinearChunkParam>();
    return static_cast<size_t>(param.chunkSize);
}

void SiluLinearChunkOp::InferOutputMetaInfo() const {
    DDebugAssert(GetInputSize() == 3);
    const auto& param = GetOpParam<SiluLinearChunkParam>();
    for (size_t i = 0; i < static_cast<size_t>(param.chunkSize); i++) {
        GetOutputOperand(i)->MetaDataSameAs(OperandX());
    }
}

void SiluLinearChunkOp::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    const auto& param = GetOpParam<SiluLinearChunkParam>();
    const auto& emb = inputs[0].value();
    const auto& linearWeight = inputs[1].value();
    const auto& linearBias = inputs[2].value();

    auto scopedAcquire = external::python::GilScopedAcquire();
    auto streamGuard = external::torch::PythonCodeCudaStreamGuard();

    nb::module_ compileKernelModule = nb::module_::import_("dtorch.compiled_op");
    nb::object funcOut =
        compileKernelModule.attr("silu_linear_chunk")(NanobindUtil::ToObject(emb), NanobindUtil::ToObject(linearWeight),
                                                      NanobindUtil::ToObject(linearBias), param.chunkSize);

    outputs = NanobindUtil::ToTensorArray(funcOut);
    DDebugAssert(outputs.size() == static_cast<size_t>(param.chunkSize));
}

}  // namespace core
}  // namespace dtorch

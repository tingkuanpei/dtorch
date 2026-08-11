/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "sage_attn_adapter.h"

#include <sstream>

#include <nanobind/nanobind.h>
#include <torch/torch.h>

#include "dtorch/common/debug.h"
#include "dtorch/core/type.h"
#include "dtorch/external/cuda/cuda_device.h"
#include "dtorch/external/python/nanobind_util.h"
#include "dtorch/external/python/python_gil.h"
#include "dtorch/external/torch/torch_stream_guard.h"

using dtorch::external::python::GilScopedAcquire;
using dtorch::external::python::NanobindUtil;

namespace dtorch {
namespace external {
namespace sage {

std::vector<::torch::Tensor> SageAttn(const ::torch::Tensor& q, const ::torch::Tensor& k, const ::torch::Tensor& v,
                                      const std::string& tensorLayout = "HND", bool isCausal = false,
                                      std::optional<float> smScale = std::nullopt, bool returnLse = false) {
    auto scopedAcquire = GilScopedAcquire();
    auto streamGuard = external::torch::PythonCodeCudaStreamGuard();

    nb::module_ sageAttnModule = nb::module_::import_("sageattention");
    nb::object funcOut =
        sageAttnModule.attr("sageattn")(NanobindUtil::ToObject(q), NanobindUtil::ToObject(k), NanobindUtil::ToObject(v),
                                        tensorLayout, isCausal, smScale, returnLse);

    std::vector<::torch::Tensor> result = NanobindUtil::ToTensorArray(funcOut);
    DDebugAssert(result.size() == 1 || result.size() == 2);
    return result;
}

std::vector<::torch::Tensor> SageAttnQKInt8PVFp16Cuda(const ::torch::Tensor& q, const ::torch::Tensor& k,
                                                      const ::torch::Tensor& v, const std::string& tensorLayout = "HND",
                                                      bool isCausal = false,
                                                      const std::string& qkQuantGran = "per_thread",
                                                      std::optional<float> smScale = std::nullopt,
                                                      const std::string& pvAccumDType = "fp32", bool smoothK = true,
                                                      bool smoothV = false, bool returnLse = false) {
    auto scopedAcquire = GilScopedAcquire();
    auto streamGuard = external::torch::PythonCodeCudaStreamGuard();

    nb::module_ sageAttnModule = nb::module_::import_("sageattention");
    nb::object funcOut = sageAttnModule.attr("sageattn_qk_int8_pv_fp16_cuda")(
        NanobindUtil::ToObject(q), NanobindUtil::ToObject(k), NanobindUtil::ToObject(v), tensorLayout, isCausal,
        qkQuantGran, smScale, pvAccumDType, smoothK, smoothV, returnLse);

    std::vector<::torch::Tensor> result = NanobindUtil::ToTensorArray(funcOut);
    DDebugAssert(result.size() == 1 || result.size() == 2);
    return result;
}

std::vector<::torch::Tensor> SageAttnQKInt8PVFp8Cuda(const ::torch::Tensor& q, const ::torch::Tensor& k,
                                                     const ::torch::Tensor& v, const std::string& tensorLayout = "HND",
                                                     bool isCausal = false,
                                                     const std::string& qkQuantGran = "per_thread",
                                                     std::optional<float> smScale = std::nullopt,
                                                     const std::string& pvAccumDType = "fp32+fp16", bool smoothK = true,
                                                     bool smoothV = false, bool returnLse = false) {
    auto scopedAcquire = GilScopedAcquire();
    auto streamGuard = external::torch::PythonCodeCudaStreamGuard();

    nb::module_ sageAttnModule = nb::module_::import_("sageattention");
    nb::object funcOut = sageAttnModule.attr("sageattn_qk_int8_pv_fp8_cuda")(
        NanobindUtil::ToObject(q), NanobindUtil::ToObject(k), NanobindUtil::ToObject(v), tensorLayout, isCausal,
        qkQuantGran, smScale, pvAccumDType, smoothK, smoothV, returnLse);

    std::vector<::torch::Tensor> result = NanobindUtil::ToTensorArray(funcOut);
    DDebugAssert(result.size() == 1 || result.size() == 2);
    return result;
}

std::vector<::torch::Tensor> SageAttnQKInt8PVFp16Triton(
    const ::torch::Tensor& q, const ::torch::Tensor& k, const ::torch::Tensor& v,
    const std::string& tensorLayout = "HND", const std::string& quantizationBackend = "triton", bool isCausal = false,
    const std::optional<::torch::Tensor>& attn_mask = std::nullopt, std::optional<float> smScale = std::nullopt,
    bool smoothK = true, bool returnLse = false) {
    auto scopedAcquire = GilScopedAcquire();
    auto streamGuard = external::torch::PythonCodeCudaStreamGuard();

    nb::module_ sageAttnModule = nb::module_::import_("sageattention");
    nb::object funcOut = sageAttnModule.attr("sageattn_qk_int8_pv_fp16_triton")(
        NanobindUtil::ToObject(q), NanobindUtil::ToObject(k), NanobindUtil::ToObject(v), tensorLayout,
        quantizationBackend, isCausal, attn_mask.has_value() ? NanobindUtil::ToObject(attn_mask.value()) : nb::none(),
        smScale, smoothK, returnLse);

    std::vector<::torch::Tensor> result = NanobindUtil::ToTensorArray(funcOut);
    DDebugAssert(result.size() == 1 || result.size() == 2);
    return result;
}

std::vector<::torch::Tensor> SageAttnQKInt8PVFp8CudaSm90(const ::torch::Tensor& q, const ::torch::Tensor& k,
                                                         const ::torch::Tensor& v,
                                                         const std::string& tensorLayout = "HND", bool isCausal = false,
                                                         const std::string& qkQuantGran = "per_thread",
                                                         std::optional<float> smScale = std::nullopt,
                                                         const std::string& pvAccumDType = "fp32+fp32",
                                                         bool smoothK = true, bool returnLse = false) {
    auto scopedAcquire = GilScopedAcquire();
    auto streamGuard = external::torch::PythonCodeCudaStreamGuard();

    nb::module_ sageAttnModule = nb::module_::import_("sageattention");
    nb::object funcOut = sageAttnModule.attr("sageattn_qk_int8_pv_fp8_cuda_sm90")(
        NanobindUtil::ToObject(q), NanobindUtil::ToObject(k), NanobindUtil::ToObject(v), tensorLayout, isCausal,
        qkQuantGran, smScale, pvAccumDType, smoothK, returnLse);

    std::vector<::torch::Tensor> result = NanobindUtil::ToTensorArray(funcOut);
    DDebugAssert(result.size() == 1 || result.size() == 2);
    return result;
}

::torch::Tensor SageAttnAdapter(const ::torch::Tensor& q, const ::torch::Tensor& k, const ::torch::Tensor& v,
                                bool isCausal, std::optional<float> smScale, const std::string& sageAttentionType) {
    std::vector<::torch::Tensor> result;
    core::Device device = torch::TorchUtil::GetDevice(q);
    DDebugAssert(device.deviceKind == core::DeviceKind::kGpu);
    int major, minor;
    cuda::CudaDevice::GetDeviceCapability(device.deviceId, major, minor);
    int capability = major * 10 + minor;

    std::stringstream ss;
    ss << "Uncompatiable device capability(" << std::to_string(capability) << ") with " << sageAttentionType;

    if (sageAttentionType == "auto") {
        result = SageAttn(q, k, v, "HND", isCausal, smScale, false);
    } else if (sageAttentionType == "qk_int8_pv_fp16") {
        if (capability == 80) {
            result =
                SageAttnQKInt8PVFp16Cuda(q, k, v, "HND", isCausal, "per_thread", smScale, "fp32", true, false, false);
        } else if (capability > 80) {
            result = SageAttnQKInt8PVFp16Triton(q, k, v, "HND", "triton", isCausal, std::nullopt, smScale, true, false);
        } else {
            throw std::invalid_argument(ss.str());
        }
    } else if (sageAttentionType == "qk_int8_pv_fp8") {
        if (capability < 89) {
            throw std::invalid_argument(ss.str());
        }

        result = SageAttn(q, k, v, "HND", isCausal, smScale, false);
    } else {
        throw std::invalid_argument(ss.str());
    }

    DDebugAssert(result.size() == 1);
    return result[0];
}

}  // namespace sage
}  // namespace external
}  // namespace dtorch

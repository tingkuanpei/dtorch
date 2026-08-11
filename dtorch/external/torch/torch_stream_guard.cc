/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "torch_stream_guard.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <nanobind/nanobind.h>
#include <torch/torch.h>

#include "dtorch/common/utilities.h"

namespace nb = nanobind;

namespace dtorch {
namespace external {
namespace torch {

struct PythonCodeCudaStreamGuard::Impl {
public:
    Impl() : torchCudaModule(nb::module_::import_("torch.cuda")), originalPythonCudaStream(PythonGetCudaStream()) {
        PythonSetCudaStream(ToPythonCudaStream(at::cuda::getCurrentCUDAStream()));
    }

    ~Impl() { PythonSetCudaStream(originalPythonCudaStream); }

    DTORCH_DISABLE_COPY_AND_MOVE(Impl);

private:
    nb::object ToPythonCudaStream(const at::cuda::CUDAStream& cudaStream) {
        cudaStream_t handle = cudaStream.stream();
        int device = cudaStream.device_index();
        return torchCudaModule.attr("ExternalStream")(reinterpret_cast<uint64_t>(handle), nb::arg("device") = device);
    }

    nb::object PythonGetCudaStream() { return torchCudaModule.attr("current_stream")(); }

    void PythonSetCudaStream(const nb::object& object) { torchCudaModule.attr("set_stream")(object); }

public:
    nb::module_ torchCudaModule;
    nb::object originalPythonCudaStream;
};

PythonCodeCudaStreamGuard::PythonCodeCudaStreamGuard()
    : mImplPtr(std::make_shared<PythonCodeCudaStreamGuard::Impl>()) {}

PythonCodeCudaStreamGuard::~PythonCodeCudaStreamGuard() {}

}  // namespace torch
}  // namespace external
}  // namespace dtorch

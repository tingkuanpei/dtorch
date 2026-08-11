/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <torch/torch.h>

#include "dtorch/api/cpp/global_option.h"
#include "nanobind_register.h"

namespace dtorch {
namespace api {
namespace py {

class PyBindGlobalOption {
public:
    static std::string ModuleName() { return ""; }

    static void RegisterFunc(nb::module_& m) {
        using ::dtorch::core::GlobalOption;

        m.def("_global_option_get_comm_timeout_second",
              []() { return GlobalOption::GetSingleton().GetCommTimeoutSecond(); });

        m.def("_global_option_get_grpc_timeout_second",
              []() { return GlobalOption::GetSingleton().GetGrpcTimeoutSecond(); });

        m.def("_global_option_get_zmq_timeout_second",
              []() { return GlobalOption::GetSingleton().GetZmqTimeoutSecond(); });

        m.def("_global_option_get_dtensor_in_same_device",
              []() { return GlobalOption::GetSingleton().GetDTensorInSameDevice(); });

        m.def("_global_option_get_per_device_per_process",
              []() { return GlobalOption::GetSingleton().GetPerDevicePerProcess(); });

        m.def("_global_option_get_num_gpu_when_enable_dtensor_in_same_device",
              []() { return GlobalOption::GetSingleton().GetNumGpuWhenEnableDtensorInSameDevice(); });

        m.def("_global_option_get_validate_kernel_input_output",
              []() { return GlobalOption::GetSingleton().GetValidateKernelInputOutput(); });
    }
};

}  // namespace py
}  // namespace api
}  // namespace dtorch

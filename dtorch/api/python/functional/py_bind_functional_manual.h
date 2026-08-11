/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <nanobind/stl/vector.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include "../nanobind_register.h"
#include "dtorch/api/cpp/distributed_spec.h"
#include "dtorch/api/cpp/functional/tensor_functional.h"

namespace dtorch {
namespace api {
namespace py {

class PyBindFunctionalManual {
public:
    static std::string ModuleName() { return "nn.functional"; }

    static void RegisterFunc(nb::module_& m) {
        using api::cpp::DeviceMesh;
        using api::cpp::Graph;
        using api::cpp::Placement;
        using api::cpp::Tensor;

        // Add nanobind function binding, and add function name to skip_func_names in generate_py_bind_functional.py

        // m.def(
        //     "redistribute",
        //     [](const Tensor& input, const DeviceMesh& deviceMesh, const std::vector<Placement>& placementSeq,
        //        std::string opName, std::optional<Graph> graphOpt) {
        //         return api::cpp::functional::Redistribute(input, deviceMesh, placementSeq, opName, graphOpt);
        //     },
        //     nb::arg("input"), nb::arg("device_mesh"), nb::arg("placement"),
        //     nb::arg("op_name") = "", nb::arg("graph") = std::nullopt);
    }
};

}  // namespace py
}  // namespace api
}  // namespace dtorch

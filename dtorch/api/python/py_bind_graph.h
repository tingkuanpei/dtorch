/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <torch/extension.h>

#include "dtorch/api/cpp/graph.h"
#include "dtorch/api/cpp/tensor.h"
#include "dtorch/api/cpp/void_future_collect.h"
#include "dtorch/common/debug.h"
#include "dtorch/common/logging.h"
#include "dtorch/external/torch/torch_util.h"
#include "nanobind_register.h"

namespace dtorch {
namespace api {
namespace py {

class PyBindGraph {
public:
    static std::string ModuleName() { return ""; }

    static void RegisterFunc(nb::module_& m) {
        using api::cpp::Graph;
        using api::cpp::GraphOption;
        using api::cpp::Tensor;
        using api::cpp::VoidFutureCollect;

        nb::class_<GraphOption>(m, "GraphOption")
            .def(nb::init<>())
            .def(nb::init<const GraphOption&>())
            .def("_get_per_device_per_process",
                 [](const GraphOption& graphOption) { return graphOption.perDevicePerProcess; })
            .def("_set_per_device_per_process",
                 [](GraphOption& graphOption, std::optional<bool> perDevicePerProcess) {
                     graphOption.perDevicePerProcess = perDevicePerProcess;
                 })
            .def("_to_string", &GraphOption::ToString)
            .def("_is_equal", &GraphOption::operator==);

        nb::class_<Graph>(m, "Graph")
            .def(nb::init<>())
            .def(nb::init<const GraphOption&>(), nb::arg("graph_option"))
            .def(nb::init<const Graph&>())
            .def("_get_id", &Graph::GetId)
            .def("set_name", &Graph::SetName)
            .def("get_name", &Graph::GetName)
            .def("_set_default_device_mesh", &Graph::SetDefaultDeviceMesh)
            .def("_get_default_device_mesh", &Graph::GetDefaultDeviceMesh)
            .def("_satisfy", &Graph::Satisfy)
            .def("_set_default_dtype",
                 [](Graph& graph, const torch::ScalarType& scalarType) {
                     graph.SetDefaultDataKind(external::torch::TorchUtil::ToDataKind(scalarType));
                 })
            .def(
                "_get_default_dtype",
                [](const Graph& graph) { return external::torch::TorchUtil::ToScalarType(graph.GetDefaultDataKind()); })
            .def("_sync", [](Graph& graph) { graph.Sync(); })

            .def("_sync_future", [](Graph& graph) { return graph.SyncFuture(); });

        nb::class_<VoidFutureCollect>(m, "VoidFutureCollect")
            .def(nb::init<>())
            .def(nb::init<const VoidFutureCollect&>())
            .def("Get", &VoidFutureCollect::Get)
            .def("Wait", &VoidFutureCollect::Wait)
            .def("IsReady", &VoidFutureCollect::IsReady);

        m.def("_get_thread_local_default_graph", []() { return Graph::GetDefaultThreadLocalGraph(); });

        // Static global object may release after cuda driver shutting down. We have to manual release cuda related
        // object before main() return.
        // One possible method is release object in std::atexit(). But I can't find any NVIDIA document which descirbing
        // when cuda drive shutting down. I had try malloc cuda buffer first(init cuda drive), then call std::atexit(),
        // this is also failed in windows vs2017.
        //
        // Reference: https://segmentfault.com/a/1190000020656740
        // m.def("_release_thread_local_default_graph", []() { Graph::GetDefaultThreadLocalGraph().Destroy(); });
    }
};

}  // namespace py
}  // namespace api
}  // namespace dtorch

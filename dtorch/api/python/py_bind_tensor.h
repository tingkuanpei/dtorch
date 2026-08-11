/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <stdexcept>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include "dtorch/api/cpp/index.h"
#include "dtorch/api/cpp/tensor_future.h"
#include "dtorch/external/torch/torch_util.h"
#include "py_bind_graph.h"

namespace dtorch {
namespace api {
namespace py {

class PyBindTensor {
public:
    static std::string ModuleName() { return ""; }

    // TODO: PyTorch 中的 tensor 具有各种不同的行为：
    // https://pytorch.org/docs/stable/generated/torch.tensor.html#torch.tensor
    static void RegisterFunc(nb::module_& m) {
        using api::cpp::DeviceMesh;
        using api::cpp::Graph;
        using api::cpp::Index;
        using api::cpp::Placement;
        using api::cpp::PlacementSeq;
        using api::cpp::Slice;
        using api::cpp::Tensor;
        using api::cpp::TensorFuture;

        nb::class_<Index>(m, "Index")
            .def("__init__", [](Index* self, nb::ellipsis) { new (self) Index(api::cpp::Ellipsis); })
            .def("__init__",
                 [](Index* self, std::optional<int64_t> integer) {
                     if (integer) {
                         new (self) Index(integer.value());
                     } else {
                         new (self) Index(std::nullopt);
                     }
                 })
            .def("__init__",
                 [](Index* self, nb::slice slice) {
                     auto parseAttr = [](const nb::object& obj) {
                         if (obj.is_none()) {
                             return std::optional<int64_t>(std::nullopt);
                         } else if (PyLong_Check(obj.ptr())) {
                             return std::optional<int64_t>(nb::cast<int>(obj));
                         } else {
                             throw std::invalid_argument("Invalid slice indices");
                         }
                     };

                     std::optional<int64_t> start = parseAttr(getattr(slice, "start"));
                     std::optional<int64_t> stop = parseAttr(getattr(slice, "stop"));
                     std::optional<int64_t> step = parseAttr(getattr(slice, "step"));
                     new (self) Index(Slice(start, stop, step));
                 })
            .def("__init__", [](Index* self, Tensor& tensor) { new (self) Index(tensor.GetTorchTensor()); })
            .def("__init__", [](Index* self, std::vector<int64_t> vec) { new (self) Index(std::move(vec)); });

        nb::class_<TensorFuture>(m, "TensorFuture")
            .def(nb::init<const TensorFuture&>())
            .def("Get", &TensorFuture::Get)
            .def("Wait", &TensorFuture::Wait)
            .def("WaitFor", &TensorFuture::WaitFor)
            .def("IsReady", &TensorFuture::IsReady);

        nb::class_<Tensor>(m, "Tensor")
            .def(
                "__init__",
                [](Tensor* self, Graph graph, torch::Tensor& torchTensor, const std::optional<DeviceMesh>& deviceMesh,
                   const std::optional<std::vector<Placement>>& placements) {
                    std::optional<PlacementSeq> placementSeq;
                    if (placements.has_value()) {
                        placementSeq = PlacementSeq(placements.value());
                    }
                    new (self) Tensor(graph, torchTensor, deviceMesh, placementSeq);
                },
                nb::arg("graph"), nb::arg("tensor"), nb::arg("device_mesh") = std::nullopt,
                nb::arg("placements") = std::nullopt)
            .def(nb::init<const Tensor&>())
            .def("set_name", &Tensor::SetName)
            .def("get_name", &Tensor::GetName)
            .def("_get_shape", [](const Tensor& tensor) { return tensor.GetShape().Vec(); })
            .def("_get_stride", &Tensor::GetStride)
            .def("_get_dtype",
                 [](const Tensor& tensor) { return external::torch::TorchUtil::ToScalarType(tensor.GetDataKind()); })
            .def("_is_distributed", &Tensor::IsDistributed)
            .def("_get_default_device_mesh", &Tensor::GetDeviceMesh)
            .def("_get_placement_seq", [](const Tensor& tensor) { return tensor.GetPlacementSeq().Vec(); })
            .def("_get_torch_tensor", &Tensor::GetTorchTensor)
            .def("_get_torch_tensor_async", &Tensor::GetTorchTensorAsync)
            .def("_get_graph", &Tensor::GetGraph)
            .def("_inplace_assignment", &Tensor::InplaceAssignment);
    }
};

}  // namespace py
}  // namespace api
}  // namespace dtorch

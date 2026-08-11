/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <torch/torch.h>

#include "dtorch/api/cpp/api_type.h"
#include "dtorch/api/cpp/device.h"
#include "dtorch/api/cpp/distributed_spec.h"
#include "dtorch/api/cpp/memory_stats.h"
#include "dtorch/api/cpp/version.h"
#include "nanobind_register.h"

namespace dtorch {
namespace api {
namespace py {

class PyBindType {
public:
    static std::string ModuleName() { return ""; }

    static void RegisterFunc(nb::module_& m) {
        m.attr("_version") = api::cpp::GetVersionString();
        m.attr("_compile_arguments") = api::cpp::GetCompileArguments();

        using api::cpp::OperatorFormat;
        nb::enum_<OperatorFormat>(m, "OperatorFormat")
            .value("nchw", OperatorFormat::kNCHW)
            .value("nhwc", OperatorFormat::kNHWC)
            .def("_to_string", [](OperatorFormat operatorFormat) { return OperatorFormatToString(operatorFormat); });

        using api::cpp::PaddingType;
        nb::enum_<PaddingType>(m, "PaddingType")
            .value("not_set", PaddingType::kNotSet)
            .value("same", PaddingType::kSame)
            .value("valid", PaddingType::kValid)
            .def("_to_string", [](PaddingType paddingType) { return PaddingTypeToString(paddingType); });

        using api::cpp::PoolingKind;
        nb::enum_<PoolingKind>(m, "PoolingKind")
            .value("avg", PoolingKind::kAvg)
            .value("max", PoolingKind::kMax)
            .def("_to_string", [](PoolingKind poolingKind) { return PoolingKindToString(poolingKind); });

        using api::cpp::DeviceMesh;
        nb::class_<DeviceMesh>(m, "DeviceMesh")
            .def(nb::init<const DeviceMesh&>())
            .def(
                "__init__",
                [](DeviceMesh* self, const std::string& deviceStr, const torch::Tensor& mesh,
                   const std::optional<std::vector<std::string>>& dimensionNames) {
                    new (self) DeviceMesh(api::cpp::DeviceKindFromString(deviceStr), mesh,
                                          dimensionNames.value_or(std::vector<std::string>()));
                },
                nb::arg("device_str"), nb::arg("mesh"), nb::arg("dim_names") = std::nullopt)
            .def("__init__",
                 [](DeviceMesh* self, nb::handle deviceHandle) {
                     PyObject* strObj = PyObject_Str(deviceHandle.ptr());
                     if (!strObj) {
                         throw std::invalid_argument("Invalid device argument");
                     }
                     const char* str = PyUnicode_AsUTF8(strObj);
                     std::string deviceStr(str);
                     Py_DECREF(strObj);
                     new (self) DeviceMesh(torch::Device(deviceStr));
                 })
            .def("_get_device_str",
                 [](const DeviceMesh& deviceMesh) { return api::cpp::DeviceKindToString(deviceMesh.GetDeviceKind()); })
            .def("_get_mesh", [](const DeviceMesh& deviceMesh) { return deviceMesh.GetMesh().ToTrochTensor(); })
            .def("_get_mesh_shape", [](const DeviceMesh& deviceMesh) { return deviceMesh.GetShape().Vec(); })
            .def("_get_mesh_dimension_names",
                 [](const DeviceMesh& deviceMesh) { return deviceMesh.GetMesh().GetDimensionNames(); })
            .def("_has_dimension_name",
                 [](const DeviceMesh& deviceMesh, const std::string& dimensionName) {
                     return deviceMesh.GetMesh().HasDimensionName(dimensionName);
                 })
            .def("_get_dimension_name_index",
                 [](const DeviceMesh& deviceMesh, const std::string& dimensionName) {
                     return deviceMesh.GetMesh().GetDimensionNameIndex(dimensionName);
                 })
            .def("_get_mesh_data", [](const DeviceMesh& deviceMesh) { return deviceMesh.GetMesh().GetData(); })
            .def("_is_distributed", &DeviceMesh::IsDistributed)
            .def("_is_same", &DeviceMesh::operator==)
            .def("_unbind", &DeviceMesh::Unbind);

        using api::cpp::Placement;
        nb::class_<Placement>(m, "Placement")
            .def(nb::init<const std::string&>())
            .def(nb::init<const Placement&>())
            .def("_to_string", &Placement::ToString)
            .def("_is_replicate", &Placement::IsReplicate)
            .def("_is_partial", &Placement::IsPartial)
            .def("_is_shard", [](const Placement& placement) { return placement.IsShard(); })
            .def("_get_shard_index", &Placement::GetShardIndex)
            .def("_has_sub_split_coordinates", &Placement::HasSubSplitCoordinates)
            .def("_get_sub_split_coordinates", &Placement::GetSubSplitCoordinates)
            .def("_equal", [](const Placement& a, const Placement& b) { return a == b; });

        using api::cpp::MemoryStat;
        nb::class_<MemoryStat>(m, "MemoryStat")
            .def(nb::init<>())
            .def(nb::init<const MemoryStat&>())
            .def("_get_allocated", [](const MemoryStat& memoryStat) { return memoryStat.allocated; })
            .def("_set_allocated", [](MemoryStat& memoryStat, int64_t allocated) { memoryStat.allocated = allocated; })
            .def("_get_reserved", [](const MemoryStat& memoryStat) { return memoryStat.reserved; })
            .def("_set_reserved", [](MemoryStat& memoryStat, int64_t reserved) { memoryStat.reserved = reserved; })
            .def("_get_max_allocated", [](const MemoryStat& memoryStat) { return memoryStat.maxAllocated; })
            .def("_set_max_allocated",
                 [](MemoryStat& memoryStat, int64_t maxAllocated) { memoryStat.maxAllocated = maxAllocated; })
            .def("_get_max_reserved", [](const MemoryStat& memoryStat) { return memoryStat.maxReserved; })
            .def("_set_max_reserved",
                 [](MemoryStat& memoryStat, int64_t maxReserved) { memoryStat.maxReserved = maxReserved; })
            .def("_to_string", &MemoryStat::ToString)
            .def("_is_equal", &MemoryStat::operator==);

        using api::cpp::MemoryStats;
        nb::class_<MemoryStats>(m, "MemoryStats")
            .def(nb::init<>())
            .def(nb::init<const MemoryStats&>())
            .def("_get_size", &MemoryStats::Size)
            .def("_to_string", &MemoryStats::ToString)
            .def("_find", [](const MemoryStats& memoryStats, int64_t deviceId) {
                auto it = memoryStats.Find(deviceId);
                if (it != memoryStats.End()) {
                    return std::make_optional<MemoryStat>(it->second);
                } else {
                    return std::optional<MemoryStat>(std::nullopt);
                }
            });
    }
};

}  // namespace py
}  // namespace api
}  // namespace dtorch

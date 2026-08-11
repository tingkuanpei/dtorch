/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "dtorch/api/cpp/tensor.h"

namespace dtorch {
namespace api {
namespace py {

/// Wrap a C++ Tensor as a Python dtorch.Tensor (DTorchTensor).
///
/// We allocate a DTorchTensor subclass instance directly via nb::inst_alloc and copy-construct
/// the C++ Tensor into it with placement new. This avoids the double-wrapping that would occur
/// with `DTorchTensor(nb::cast(result))`, where nb::cast first creates a base _dtorch_py_api.Tensor
/// wrapper before the subclass constructor copies again.
inline nb::object WrapTensor(const api::cpp::Tensor& result) {
    static nb::object tensor_cls = nb::module_::import_("dtorch").attr("Tensor");
    nb::object obj = nb::inst_alloc(tensor_cls);
    api::cpp::Tensor* ptr = nb::inst_ptr<api::cpp::Tensor>(obj);
    new (ptr) api::cpp::Tensor(result);
    nb::inst_mark_ready(obj);
    return obj;
}

/// Wrap a vector of C++ Tensors as a Python list of dtorch.Tensor.
inline nb::object WrapTensorArray(const std::vector<api::cpp::Tensor>& results) {
    nb::list out;
    for (const auto& t : results) {
        out.append(WrapTensor(t));
    }
    return out;
}

}  // namespace py
}  // namespace api
}  // namespace dtorch

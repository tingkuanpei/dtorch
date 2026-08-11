/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <vector>

#include <nanobind/nanobind.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/script.h>
#include <torch/torch.h>

#include "dtorch/api/python/nanobind_type_casters.h"
#include "dtorch/common/debug.h"
#include "dtorch/common/utilities.h"
#include "py_unpack_common.h"

namespace nb = nanobind;

namespace dtorch {
namespace external {
namespace python {

class NanobindUtil {
public:
    static DTORCH_FORCEINLINE nb::object ToObject(const ::torch::Tensor& input) {
        return nb::steal<nb::object>(THPVariable_Wrap(input));
    }

    static DTORCH_FORCEINLINE ::torch::Tensor ToTensor(PyObject* obj) {
        nb::handle handle(obj);
        return nb::cast<::torch::Tensor>(handle);
    }

    static DTORCH_FORCEINLINE ::torch::Tensor ToTensor(const nb::object& object) {
        nb::handle handle(object);
        return nb::cast<::torch::Tensor>(handle);
    }

    static DTORCH_FORCEINLINE std::vector<::torch::Tensor> ToTensorArray(const nb::object& object) {
        return ToTensorArray(object.ptr());
    }

    static DTORCH_FORCEINLINE std::vector<::torch::Tensor> ToTensorArray(PyObject* obj) {
        std::vector<::torch::Tensor> result;

        if (!PyCheckIsTupleOrList(obj)) {
            result.push_back(ToTensor(obj));
        } else {
            DDebugAssert(PyCheckIsTupleOrList(obj));
            for (auto it : PyUnpackTupleOrList(obj)) {
                result.push_back(ToTensor(it));
            }
        }

        return result;
    }
};

}  // namespace python
}  // namespace external
}  // namespace dtorch

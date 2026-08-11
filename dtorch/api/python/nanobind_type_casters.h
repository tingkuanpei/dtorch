/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Generator.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include "dtorch/api/cpp/api_type.h"
#include "dtorch/api/cpp/data_kind.h"
#include "dtorch/api/cpp/distributed_spec.h"
#include "dtorch/api/cpp/functional/functional_option.h"
#include "dtorch/api/cpp/generator.h"
#include "dtorch/api/cpp/int_or_int_array.h"
#include "dtorch/api/cpp/scalar.h"
#include "dtorch/api/cpp/shape.h"
#include "dtorch/external/torch/torch_util.h"

namespace nb = nanobind;

// error: extra ‘;’ [-Werror=pedantic]
//    261 |     NB_TYPE_CASTER(dtorch::api::cpp::OperatorFormat, const_name("OperatorFormat"));
//        |                                                          ^
//        |                                                          -
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wpedantic"
#endif

namespace nanobind {
namespace detail {

// ===================================================================
// Scalar: from Python int or float
// ===================================================================
template <>
struct type_caster<dtorch::api::cpp::Scalar> {
public:
    NB_TYPE_CASTER(dtorch::api::cpp::Scalar, const_name("Scalar"));

    bool from_python(handle src, uint8_t, cleanup_list *) noexcept {
        PyObject *obj = src.ptr();
        if (PyLong_Check(obj)) {
            int64_t val = PyLong_AsLongLong(obj);
            if (val == -1 && PyErr_Occurred()) {
                PyErr_Clear();
                return false;
            }
            value = dtorch::api::cpp::Scalar(val);
            return true;
        }
        if (PyFloat_Check(obj)) {
            value = dtorch::api::cpp::Scalar(PyFloat_AsDouble(obj));
            return true;
        }
        return false;
    }

    static handle from_cpp(dtorch::api::cpp::Scalar src, rv_policy, cleanup_list *) {
        if (src.IsFloatingPoint()) {
            return PyFloat_FromDouble(src.Value<double>());
        }
        if (src.IsSigned()) {
            return PyLong_FromLongLong(src.Value<int64_t>());
        }
        return PyLong_FromUnsignedLongLong(src.Value<uint64_t>());
    }
};

// ===================================================================
// IntOrIntArray: from Python int or list/tuple of ints
// ===================================================================
template <>
struct type_caster<dtorch::api::cpp::IntOrIntArray> {
public:
    NB_TYPE_CASTER(dtorch::api::cpp::IntOrIntArray, const_name("IntOrIntArray"));

    bool from_python(handle src, uint8_t, cleanup_list *) noexcept {
        PyObject *obj = src.ptr();
        if (PyLong_Check(obj)) {
            int64_t val = PyLong_AsLongLong(obj);
            if (val == -1 && PyErr_Occurred()) {
                PyErr_Clear();
                return false;
            }
            value = dtorch::api::cpp::IntOrIntArray(val);
            return true;
        }
        if (PyTuple_Check(obj) || PyList_Check(obj)) {
            size_t n =
                PyTuple_Check(obj) ? static_cast<size_t>(PyTuple_Size(obj)) : static_cast<size_t>(PyList_Size(obj));
            std::vector<int64_t> vec;
            vec.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                PyObject *item = PyTuple_Check(obj) ? PyTuple_GetItem(obj, i) : PyList_GetItem(obj, i);
                if (!PyLong_Check(item)) return false;
                int64_t val = PyLong_AsLongLong(item);
                if (val == -1 && PyErr_Occurred()) {
                    PyErr_Clear();
                    return false;
                }
                vec.push_back(val);
            }
            value = dtorch::api::cpp::IntOrIntArray(vec);
            return true;
        }
        return false;
    }

    static handle from_cpp(const dtorch::api::cpp::IntOrIntArray &src, rv_policy, cleanup_list *) {
        return nb::cast(src.Vec()).release();
    }
};

// ===================================================================
// Shape: from Python int or list/tuple of ints
// ===================================================================
template <>
struct type_caster<dtorch::api::cpp::Shape> {
public:
    NB_TYPE_CASTER(dtorch::api::cpp::Shape, const_name("Shape"));

    bool from_python(handle src, uint8_t, cleanup_list *) noexcept {
        PyObject *obj = src.ptr();
        if (PyLong_Check(obj)) {
            int64_t val = PyLong_AsLongLong(obj);
            if (val == -1 && PyErr_Occurred()) {
                PyErr_Clear();
                return false;
            }
            value = dtorch::api::cpp::Shape(std::vector<int64_t>{val});
            return true;
        }
        if (PyTuple_Check(obj) || PyList_Check(obj)) {
            size_t n =
                PyTuple_Check(obj) ? static_cast<size_t>(PyTuple_Size(obj)) : static_cast<size_t>(PyList_Size(obj));
            std::vector<int64_t> vec;
            vec.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                PyObject *item = PyTuple_Check(obj) ? PyTuple_GetItem(obj, i) : PyList_GetItem(obj, i);
                if (!PyLong_Check(item)) return false;
                int64_t val = PyLong_AsLongLong(item);
                if (val == -1 && PyErr_Occurred()) {
                    PyErr_Clear();
                    return false;
                }
                vec.push_back(val);
            }
            value = dtorch::api::cpp::Shape(vec);
            return true;
        }
        return false;
    }

    static handle from_cpp(const dtorch::api::cpp::Shape &src, rv_policy, cleanup_list *) {
        return nb::cast(src.Vec()).release();
    }
};

// ===================================================================
// PlacementSeq: from a single Placement or list/tuple of Placements
// ===================================================================
template <>
struct type_caster<dtorch::api::cpp::PlacementSeq> {
public:
    NB_TYPE_CASTER(dtorch::api::cpp::PlacementSeq, const_name("PlacementSeq"));

    bool from_python(handle src, uint8_t, cleanup_list *) noexcept {
        try {
            // Single Placement object
            if (isinstance<dtorch::api::cpp::Placement>(src)) {
                value = dtorch::api::cpp::PlacementSeq(
                    std::vector<dtorch::api::cpp::Placement>{nb::cast<const dtorch::api::cpp::Placement &>(src)});
                return true;
            }
            // List/tuple of Placements
            if (PyList_Check(src.ptr()) || PyTuple_Check(src.ptr())) {
                nb::list lst = nb::borrow<nb::list>(src);
                std::vector<dtorch::api::cpp::Placement> vec;
                vec.reserve(lst.size());
                for (auto item : lst) {
                    if (!isinstance<dtorch::api::cpp::Placement>(item)) return false;
                    vec.push_back(nb::cast<const dtorch::api::cpp::Placement &>(item));
                }
                value = dtorch::api::cpp::PlacementSeq(vec);
                return true;
            }
        } catch (...) {
            return false;
        }
        return false;
    }

    static handle from_cpp(const dtorch::api::cpp::PlacementSeq &src, rv_policy, cleanup_list *) {
        nb::list result;
        for (const auto &p : src.Vec()) {
            result.append(nb::cast(p));
        }
        return result.release();
    }
};

// ===================================================================
// DataKind: from Python torch.dtype (via torch::ScalarType)
// ===================================================================
template <>
struct type_caster<dtorch::api::cpp::DataKind> {
public:
    NB_TYPE_CASTER(dtorch::api::cpp::DataKind, const_name("DataKind"));

    bool from_python(handle src, uint8_t, cleanup_list *) noexcept {
        try {
            torch::ScalarType scalarType = nb::cast<torch::ScalarType>(src);
            value = dtorch::external::torch::TorchUtil::ToDataKind(scalarType);
            return true;
        } catch (const nb::cast_error &) {
            return false;
        }
    }

    static handle from_cpp(dtorch::api::cpp::DataKind src, rv_policy, cleanup_list *) {
        return nb::cast(dtorch::external::torch::TorchUtil::ToScalarType(src)).release();
    }
};

// ===================================================================
// Generator: from Python torch.Generator
// ===================================================================
template <>
struct type_caster<dtorch::api::cpp::Generator> {
public:
    NB_TYPE_CASTER(dtorch::api::cpp::Generator, const_name("Generator"));

    bool from_python(handle src, uint8_t, cleanup_list *) noexcept {
        try {
            torch::Generator gen = nb::cast<torch::Generator>(src);
            value = dtorch::api::cpp::Generator(std::make_shared<torch::Generator>(gen));
            return true;
        } catch (const nb::cast_error &) {
            return false;
        }
    }

    static handle from_cpp(const dtorch::api::cpp::Generator &src, rv_policy, cleanup_list *) {
        auto optTorchGen = src.GetOptTorchGenerator();
        if (optTorchGen.has_value()) {
            return nb::cast(optTorchGen.value()).release();
        }
        return nb::none().release();
    }
};

// ===================================================================
// OperatorFormat: from Python int, nanobind enum value, or Python wrapper with _value
// ===================================================================
template <>
struct type_caster<dtorch::api::cpp::OperatorFormat> {
public:
    NB_TYPE_CASTER(dtorch::api::cpp::OperatorFormat, const_name("OperatorFormat"));

    bool from_python(handle src, uint8_t, cleanup_list *) noexcept {
        // Use nanobind's built-in enum conversion (handles Python ints and nanobind enum objects)
        int64_t result;
        if (detail::enum_from_python(&typeid(dtorch::api::cpp::OperatorFormat), src.ptr(), &result, 0)) {
            value = static_cast<dtorch::api::cpp::OperatorFormat>(result);
            return true;
        }
        // Python wrapper (e.g. dtorch.type.OperatorFormat): extract _value and retry step 1
        try {
            if (hasattr(src, "_value")) {
                nb::handle inner = src.attr("_value");
                if (detail::enum_from_python(&typeid(dtorch::api::cpp::OperatorFormat), inner.ptr(), &result, 0)) {
                    value = static_cast<dtorch::api::cpp::OperatorFormat>(result);
                    return true;
                }
            }
        } catch (...) {
        }
        return false;
    }

    static handle from_cpp(dtorch::api::cpp::OperatorFormat src, rv_policy, cleanup_list *) {
        return detail::enum_from_cpp(&typeid(dtorch::api::cpp::OperatorFormat), static_cast<int64_t>(src));
    }
};

// ===================================================================
// PaddingType: from Python int, nanobind enum value, or Python wrapper with _value
// ===================================================================
template <>
struct type_caster<dtorch::api::cpp::PaddingType> {
public:
    NB_TYPE_CASTER(dtorch::api::cpp::PaddingType, const_name("PaddingType"));

    bool from_python(handle src, uint8_t, cleanup_list *) noexcept {
        // Use nanobind's built-in enum conversion (handles Python ints and nanobind enum objects)
        int64_t result;
        if (detail::enum_from_python(&typeid(dtorch::api::cpp::PaddingType), src.ptr(), &result, 0)) {
            value = static_cast<dtorch::api::cpp::PaddingType>(result);
            return true;
        }
        // Python wrapper (e.g. dtorch.type.PaddingType): extract _value and retry step 1
        try {
            if (hasattr(src, "_value")) {
                nb::handle inner = src.attr("_value");
                if (detail::enum_from_python(&typeid(dtorch::api::cpp::PaddingType), inner.ptr(), &result, 0)) {
                    value = static_cast<dtorch::api::cpp::PaddingType>(result);
                    return true;
                }
            }
        } catch (...) {
        }
        return false;
    }

    static handle from_cpp(dtorch::api::cpp::PaddingType src, rv_policy, cleanup_list *) {
        return detail::enum_from_cpp(&typeid(dtorch::api::cpp::PaddingType), static_cast<int64_t>(src));
    }
};

// ===================================================================
// PoolingKind: from Python int, nanobind enum value, or Python wrapper with _value
// ===================================================================
template <>
struct type_caster<dtorch::api::cpp::PoolingKind> {
public:
    NB_TYPE_CASTER(dtorch::api::cpp::PoolingKind, const_name("PoolingKind"));

    bool from_python(handle src, uint8_t, cleanup_list *) noexcept {
        // Use nanobind's built-in enum conversion (handles Python ints and nanobind enum objects)
        int64_t result;
        if (detail::enum_from_python(&typeid(dtorch::api::cpp::PoolingKind), src.ptr(), &result, 0)) {
            value = static_cast<dtorch::api::cpp::PoolingKind>(result);
            return true;
        }
        // Python wrapper (e.g. dtorch.type.PoolingKind): extract _value and retry step 1
        try {
            if (hasattr(src, "_value")) {
                nb::handle inner = src.attr("_value");
                if (detail::enum_from_python(&typeid(dtorch::api::cpp::PoolingKind), inner.ptr(), &result, 0)) {
                    value = static_cast<dtorch::api::cpp::PoolingKind>(result);
                    return true;
                }
            }
        } catch (...) {
        }
        return false;
    }

    static handle from_cpp(dtorch::api::cpp::PoolingKind src, rv_policy, cleanup_list *) {
        return detail::enum_from_cpp(&typeid(dtorch::api::cpp::PoolingKind), static_cast<int64_t>(src));
    }
};

// ===================================================================
// SdpaOption: from a Python object with sage_attn_type attribute
// ===================================================================
template <>
struct type_caster<dtorch::api::cpp::functional::SdpaOption> {
public:
    NB_TYPE_CASTER(dtorch::api::cpp::functional::SdpaOption, const_name("SdpaOption"));

    bool from_python(handle src, uint8_t, cleanup_list *) noexcept {
        try {
            dtorch::api::cpp::functional::SdpaOption result;
            if (hasattr(src, "sage_attn_type")) {
                auto attr = src.attr("sage_attn_type");
                if (!attr.is_none()) {
                    result.sageAttentionType = nb::cast<std::string>(attr);
                }
            }
            value = result;
            return true;
        } catch (const std::exception &) {
            return false;
        }
    }

    static handle from_cpp(const dtorch::api::cpp::functional::SdpaOption &src, rv_policy, cleanup_list *) {
        nb::dict out;
        out["sage_attn_type"] = src.sageAttentionType;
        return out.release();
    }
};

// ===================================================================
// torch::Tensor: from Python torch.Tensor
// ===================================================================
template <>
struct type_caster<::torch::Tensor> {
    NB_TYPE_CASTER(::torch::Tensor, const_name("Tensor"));

    bool from_python(handle src, uint8_t, cleanup_list *) noexcept {
        try {
            value = THPVariable_Unpack(src.ptr());
            return true;
        } catch (...) {
            return false;
        }
    }

    static handle from_cpp(::torch::Tensor src, rv_policy, cleanup_list *) { return THPVariable_Wrap(src); }
};

// ===================================================================
// torch::ScalarType (c10::ScalarType / torch.dtype)
// ===================================================================
template <>
struct type_caster<::torch::ScalarType> {
    NB_TYPE_CASTER(::torch::ScalarType, const_name("ScalarType"));

    bool from_python(handle src, uint8_t, cleanup_list *) noexcept {
        PyObject *obj = src.ptr();
        if (THPDtype_Check(obj)) {
            value = reinterpret_cast<THPDtype *>(obj)->scalar_type;
            return true;
        }
        if (PyLong_Check(obj)) {
            value = static_cast<::torch::ScalarType>(PyLong_AsLong(obj));
            return true;
        }
        return false;
    }

    static handle from_cpp(::torch::ScalarType src, rv_policy, cleanup_list *) {
        return Py_NewRef(::torch::getTHPDtype(src));
    }
};

// ===================================================================
// torch::Generator (at::Generator)
// ===================================================================
template <>
struct type_caster<::torch::Generator> {
    NB_TYPE_CASTER(::torch::Generator, const_name("Generator"));

    bool from_python(handle src, uint8_t, cleanup_list *) noexcept {
        PyObject *obj = src.ptr();
        if (THPGenerator_Check(obj)) {
            value = reinterpret_cast<THPGenerator *>(obj)->cdata;
            return true;
        }
        return false;
    }

    static handle from_cpp(::torch::Generator src, rv_policy, cleanup_list *) { return THPGenerator_Wrap(src); }
};

}  // namespace detail
}  // namespace nanobind

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "py_unpack_common.h"

#include <limits>
#include <sstream>

#include "dtorch/api/cpp/distributed_spec.h"
#include "dtorch/common/debug.h"
#include "dtorch/common/logging.h"
#include "dtorch/common/string.h"

namespace nb = nanobind;

namespace dtorch {
namespace external {
namespace python {

bool StringToInt64Array(const std::string& str, std::vector<int64_t>& vec) {
    vec.clear();

    std::string betweenBraceString;
    if (String::FindStringBetweenLeftAndRight(str, "{", "}", betweenBraceString)) {
        if (betweenBraceString.empty()) {
            return true;
        }
        for (auto it : String::Split(betweenBraceString, ",")) {
            try {
                vec.push_back(std::stoi(it));
            } catch (std::exception& e) {
                DLogError() << e.what();
                return false;
            }
        }
    } else {
        try {
            vec.push_back(std::stoi(str));
        } catch (std::exception& e) {
            DLogError() << e.what();
            return false;
        }
    }

    return true;
}

int32_t PyUnpackInt32(PyObject* obj) {
    int64_t value = PyUnpackInt64(obj);

    if (value > std::numeric_limits<int32_t>::max() || value < std::numeric_limits<int32_t>::min()) {
        throw std::runtime_error("Python overflow when unpacking int32");
    }
    return (int32_t)value;
}

int64_t PyUnpackInt64(PyObject* obj) {
    DDebugAssert(PyLong_Check(obj));

    int overflow;
    long long value = PyLong_AsLongLongAndOverflow(obj, &overflow);
    if (value == -1 && PyErr_Occurred()) {
        throw std::runtime_error("Python unpack int64 failed");
    }
    if (overflow != 0) {
        throw std::runtime_error("Python overflow when unpacking int64");
    }
    return (int64_t)value;
}

double PyUnpackDouble(PyObject* obj) {
    if (PyFloat_Check(obj)) {
        return PyFloat_AS_DOUBLE(obj);
    }

    return static_cast<double>(PyUnpackInt64(obj));
}

std::vector<int64_t> PyUnpackInt64Array(PyObject* obj) {
    DDebugAssert(PyLong_Check(obj) || PyCheckIsTupleOrList(obj));

    if (PyLong_Check(obj)) {
        return {PyUnpackInt64(obj)};
    }

    std::vector<int64_t> result;
    for (auto it : PyUnpackTupleOrList(obj)) {
        result.push_back(PyUnpackInt64(it));
    }
    return result;
}

std::string PyUnpackString(PyObject* obj) {
    if (PyBytes_Check(obj)) {
        size_t size = PyBytes_GET_SIZE(obj);
        return std::string(PyBytes_AS_STRING(obj), size);
    } else {
        DDebugAssert(PyUnicode_Check(obj));
        Py_ssize_t size;
        const char* data = PyUnicode_AsUTF8AndSize(obj, &size);
        DDebugAssert(data != nullptr);
        return std::string(data, (size_t)size);
    }
}

std::vector<PyObject*> PyUnpackTupleOrList(PyObject* obj) {
    DDebugAssert(obj != nullptr);
    DDebugAssert(PyTuple_Check(obj) || PyList_Check(obj));
    std::vector<PyObject*> result;

    bool isTuple = PyTuple_Check(obj);
    const Py_ssize_t size = isTuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject* elementObj = isTuple ? PyTuple_GET_ITEM(obj, i) : PyList_GET_ITEM(obj, i);
        result.push_back(elementObj);
    }
    return result;
}

std::string GetObjectModuleName(PyObject* obj) {
    PyObject* moduleName = PyObject_GetAttrString(obj, "__module__");
    if (moduleName == nullptr || moduleName == Py_None) {
        return "";
    }
    return PyUnpackString(moduleName);
}

std::string GetObjectTypeString(PyObject* obj) {
    std::stringstream ss;
    if (PyCheckIsTupleOrList(obj)) {
        ss << "Sequence[";
        auto objVec = PyUnpackTupleOrList(obj);
        if (objVec.size() > 0) {
            std::string typeSting = GetObjectTypeString(objVec[0]);
            bool sameType = true;

            for (size_t i = 0; i < objVec.size(); i++) {
                if (GetObjectTypeString(objVec[i]) != typeSting) {
                    sameType = false;
                    break;
                }
            }
            if (sameType) {
                ss << typeSting;
            } else {
                ss << "Any";
            }
        }
        ss << "]";
    } else {
        std::string typeClassName = Py_TYPE(obj)->tp_name;
        if (typeClassName == "Tensor") {
            std::string moduleName = GetObjectModuleName(obj);
            if (!moduleName.empty()) {
                ss << moduleName << ".";
            }
        } else if (typeClassName == "DTorchTensor") {
            typeClassName = "dtorch.Tensor";
        }
        ss << typeClassName;
    }
    return ss.str();
}

bool PyCheckIsPlacementSeq(PyObject* obj) {
    if (!PyCheckIsTupleOrList(obj)) {
        return false;
    }

    auto objVec = PyUnpackTupleOrList(obj);
    for (auto it : objVec) {
        if (!nb::isinstance<api::cpp::Placement>(nb::borrow<nb::object>(it))) {
            return false;
        }
    }
    return true;
}

bool PyCheckIsIntArray(PyObject* obj) {
    if (!PyCheckIsTupleOrList(obj)) {
        return false;
    }

    auto objVec = PyUnpackTupleOrList(obj);
    for (auto it : objVec) {
        if (!PyLong_Check(it)) {
            return false;
        }
    }
    return true;
}

}  // namespace python
}  // namespace external
}  // namespace dtorch

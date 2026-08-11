/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <vector>

// TODO: #include <Python.h>
// Python.h will define HAVE_SNPRINTF, nanobind/nanobind.h define too
#include <nanobind/nanobind.h>

#include "dtorch/common/utilities.h"

namespace dtorch {
namespace external {
namespace python {

bool StringToInt64Array(const std::string& str, std::vector<int64_t>& vec);

int32_t PyUnpackInt32(PyObject* obj);

int64_t PyUnpackInt64(PyObject* obj);

double PyUnpackDouble(PyObject* obj);

std::vector<int64_t> PyUnpackInt64Array(PyObject* obj);

DTORCH_FORCEINLINE bool PyCheckIsString(PyObject* obj) { return PyBytes_Check(obj) || PyUnicode_Check(obj); }

std::string PyUnpackString(PyObject* obj);

DTORCH_FORCEINLINE bool PyCheckIsTupleOrList(PyObject* obj) { return PyTuple_Check(obj) || PyList_Check(obj); }

std::vector<PyObject*> PyUnpackTupleOrList(PyObject* obj);

std::string GetObjectModuleName(PyObject* obj);

std::string GetObjectTypeString(PyObject* obj);

bool PyCheckIsPlacementSeq(PyObject* obj);

bool PyCheckIsIntArray(PyObject* obj);

}  // namespace python
}  // namespace external
}  // namespace dtorch

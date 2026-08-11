/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "python_gil.h"

namespace dtorch {
namespace external {
namespace python {

bool IsPythonInitialized() { return Py_IsInitialized(); }

std::unique_ptr<GilScopedRelease> GetPythonGilScopedRelease() {
    std::unique_ptr<GilScopedRelease> result;
    if (Py_IsInitialized() && PyGILState_Check()) {
        result = std::make_unique<GilScopedRelease>();
    }
    return result;
}

std::unique_ptr<GilScopedAcquire> GetPythonGilScopedAcquire() {
    std::unique_ptr<GilScopedAcquire> result;
    if (Py_IsInitialized()) {
        result = std::make_unique<GilScopedAcquire>();
    }
    return result;
}

}  // namespace python
}  // namespace external
}  // namespace dtorch

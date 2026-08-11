/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <memory>

#include <Python.h>

#include "dtorch/common/debug.h"
#include "dtorch/common/utilities.h"

namespace dtorch {
namespace external {
namespace python {

bool IsPythonInitialized();

// The PyTorch C++ Tensor holds an impl::PyObjectSlot object. When this object is released, the Python Global
// Interpreter Lock (GIL) needs to be acquired. Therefore, in a multi-threaded environment, to avoid deadlocks,
// the GIL should be released at appropriate times.
//
// PyTorch code:
//      https://github.com/pytorch/pytorch/blob/v2.6.0/c10/core/TensorImpl.h#L2874

class GilScopedAcquire {
public:
    GilScopedAcquire() : state(PyGILState_Ensure()) {}

    ~GilScopedAcquire() { PyGILState_Release(state); }

    DTORCH_DISABLE_COPY_AND_DEFAULT_MOVE(GilScopedAcquire);

private:
    PyGILState_STATE state;
};

class GilScopedRelease {
public:
    GilScopedRelease() : state(nullptr) {
        DAlwaysAssert(PyGILState_Check());
        state = PyEval_SaveThread();
    }

    ~GilScopedRelease() { PyEval_RestoreThread(state); }

    DTORCH_DISABLE_COPY_AND_DEFAULT_MOVE(GilScopedRelease);

private:
    PyThreadState *state;
};

std::unique_ptr<GilScopedRelease> GetPythonGilScopedRelease();

std::unique_ptr<GilScopedAcquire> GetPythonGilScopedAcquire();

}  // namespace python
}  // namespace external
}  // namespace dtorch

/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "nanobind_register.h"

namespace dtorch {
namespace api {
namespace py {

NB_MODULE(_dtorch_py_api, m) {
    NanobindRegister nanobindRegister;
    nanobindRegister.RegisterAll(m);
}

}  // namespace py
}  // namespace api
}  // namespace dtorch

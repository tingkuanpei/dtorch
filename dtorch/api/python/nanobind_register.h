/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include <nanobind/nanobind.h>

#include "dtorch/external/python/nanobind_util.h"

namespace nb = nanobind;

namespace dtorch {
namespace api {
namespace py {

class NanobindRegister {
public:
    NanobindRegister();

    ~NanobindRegister() = default;

    void RegisterAll(nb::module_& m);

private:
    template <typename RegisterType>
    void Register();

    using RegisterFunc = std::function<void(nb::module_&)>;

    void RegisterModule(nb::module_& m, const std::string& moduleName, RegisterFunc func);

private:
    std::unordered_map<std::string, std::vector<RegisterFunc>> mRegisterMap;
};

}  // namespace py
}  // namespace api
}  // namespace dtorch

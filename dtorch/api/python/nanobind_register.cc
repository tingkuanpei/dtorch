/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "nanobind_register.h"

#include "dtorch/api/python/functional/py_bind_functional_generated.h"
#include "dtorch/common/string.h"
#include "functional/py_bind_functional_manual.h"
#include "py_bind_cluster.h"
#include "py_bind_global_option.h"
#include "py_bind_graph.h"
#include "py_bind_tensor.h"
#include "py_bind_type.h"

namespace dtorch {
namespace api {
namespace py {

template <typename RegisterType>
void NanobindRegister::Register() {
    std::string moduleName = RegisterType::ModuleName();
    RegisterFunc registerFunc = RegisterType::RegisterFunc;

    mRegisterMap[moduleName].emplace_back(registerFunc);
}

NanobindRegister::NanobindRegister() : mRegisterMap() {
    Register<PyBindCluster>();
    Register<PyBindGlobalOption>();
    Register<PyBindType>();
    Register<PyBindGraph>();
    Register<PyBindTensor>();
    Register<PyBindFunctionalGenerated>();
    Register<PyBindFunctionalManual>();
}

void NanobindRegister::RegisterAll(nb::module_& m) {
    for (const auto& it : mRegisterMap) {
        const std::string& moduleName = it.first;
        for (const auto& registerFunc : it.second) {
            RegisterModule(m, moduleName, registerFunc);
        }
    }
}

void NanobindRegister::RegisterModule(nb::module_& m, const std::string& moduleName, RegisterFunc registerFunc) {
    if (moduleName.empty()) {
        registerFunc(m);
        return;
    }

    std::vector<nb::module_> moduleVec;
    moduleVec.push_back(m);

    std::vector<std::string> nameVec = String::Split(moduleName, ".");
    for (auto it : nameVec) {
        DAlwaysAssert(!it.empty());
        moduleVec.push_back(moduleVec.back().def_submodule(it.data()));
    }
    registerFunc(moduleVec.back());
}

}  // namespace py
}  // namespace api
}  // namespace dtorch

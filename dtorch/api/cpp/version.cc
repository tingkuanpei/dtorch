/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "dtorch/api/cpp/version.h"

#include "dtorch/common/config.h"

namespace dtorch {
namespace api {
namespace cpp {

std::string GetVersionString() {
    static const std::string versionString = "0.0.1+" + kDTorchComileCommitId;
    return versionString;
}

std::string GetCompileArguments() { return kDTorchCompileArguments; }

}  // namespace cpp
}  // namespace api
}  // namespace dtorch

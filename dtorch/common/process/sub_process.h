/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "dtorch/common/utilities.h"

namespace dtorch {

class SubProcess {
public:
    SubProcess();

    SubProcess(const std::string& exeCmd, const std::unordered_map<std::string, std::string>& envs = {});

    ~SubProcess();

    DTORCH_DISABLE_COPY_AND_DEFAULT_MOVE(SubProcess);

private:
    struct Impl;
    std::shared_ptr<Impl> mImplPtr;
};

}  // namespace dtorch

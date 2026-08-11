/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <mutex>
#include <string>

#include "kernel_hook.h"

namespace dtorch {
namespace core {

class Operator;

class DumpOperatorIOHook : public KernelHook {
public:
    DumpOperatorIOHook(const std::string& path);

    void BeforeCompute(const Kernel& kernel) override;

    void AfterCompute(const Kernel& kernel) override;

private:
    std::string GenerateTensorNpyFileName(const Operator* op, bool isInput, size_t index, bool isModifyedInput = false);

private:
    size_t mOpIndex;
    std::string mPath;
    std::mutex mMutex;
};

}  // namespace core
}  // namespace dtorch

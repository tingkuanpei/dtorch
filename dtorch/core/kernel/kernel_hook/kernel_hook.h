/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

namespace dtorch {
namespace core {

class Kernel;

class KernelHook {
public:
    KernelHook() = default;

    virtual ~KernelHook() = default;

    virtual void BeforeCompute(const Kernel& kernel) = 0;

    virtual void AfterCompute(const Kernel& kernel) = 0;
};

}  // namespace core
}  // namespace dtorch

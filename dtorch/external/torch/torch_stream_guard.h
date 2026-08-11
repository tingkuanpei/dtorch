/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <memory>

namespace dtorch {
namespace external {
namespace torch {

class PythonCodeCudaStreamGuard {
public:
    PythonCodeCudaStreamGuard();

    ~PythonCodeCudaStreamGuard();

private:
    struct Impl;
    mutable std::shared_ptr<Impl> mImplPtr;
};

}  // namespace torch
}  // namespace external
}  // namespace dtorch

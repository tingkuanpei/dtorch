/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <string>

#include "dtorch/common/debug.h"
#include "dtorch/common/logging.h"
#include "dtorch/common/utilities.h"
#include "nccl.h"

#define UNcclCheckError(expr) dtorch::nccl::NcclCheckError(expr)

namespace dtorch {
namespace external {
namespace nccl {

DTORCH_FORCEINLINE void NcclCheckError(ncclResult_t error) noexcept {
    if (error != ncclResult_t::ncclSuccess) {
        DLogFatal() << "NCCL error string: " << ncclGetErrorString(error);
    }
}

}  // namespace nccl
}  // namespace external
}  // namespace dtorch

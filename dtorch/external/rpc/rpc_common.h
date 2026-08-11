/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <string>

#include "dtorch/common/filesystem.h"

namespace dtorch {
namespace external {
namespace rpc {

std::string GetRandomUdsAddress(size_t length = 24);

bool IsUdsAddress(const std::string& address);

std::string GetUdsActualFilePath(const std::string& address);

std::unique_ptr<FileRemoveGuard> GetUdsFileRemoveGuard(const std::string& address);

}  // namespace rpc
}  // namespace external
}  // namespace dtorch

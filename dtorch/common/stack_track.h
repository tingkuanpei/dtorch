/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <string>

namespace dtorch {

std::string GetStackTrack(size_t skip = 0, size_t max_depth = 64);

}  // namespace dtorch

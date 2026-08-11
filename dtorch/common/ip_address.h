/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <string>

namespace dtorch {

std::string GetValidNodeAddress(int64_t startPort = 13000, int64_t endPort = 14000);

bool CheckStringIsValidAddress(const std::string& str);

}  // namespace dtorch

/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <memory>
#include <string>

#include "utilities.h"

namespace dtorch {

std::unique_ptr<char[]> GetEnv(const std::string& envVar);

DTORCH_FORCEINLINE bool IsEnvVarDefined(const std::string& envVar) { return GetEnv(envVar) != nullptr; }

bool GetEnvVarAsBool(const std::string& envVar, bool defaultValue);

int GetEnvVarAsInt(const std::string& envVar, int defaultValue);

size_t GetEnvVarAsUInt(const std::string& envVar, size_t defaultValue);

std::string GetEnvVarAsString(const std::string& envVar, const std::string& defaultValue);

}  // namespace dtorch

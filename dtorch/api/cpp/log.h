/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <functional>
#include <string>

namespace dtorch {
namespace api {
namespace cpp {

enum class LogSeverity { kInfo = 0, kWarning, kError, kFatal };

// void(LogSeverity severity, const char* file, int line, const std::string& message)
using LogFuncType = std::function<void(LogSeverity, const char*, int, const std::string&)>;

// Not thread safe, call this function before dtorch run.
void SetLogFunc(LogFuncType func);

}  // namespace cpp
}  // namespace api
}  // namespace dtorch

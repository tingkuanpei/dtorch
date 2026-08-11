/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "log.h"

#include "dtorch/common/logging.h"

namespace dtorch {
namespace api {
namespace cpp {

void SetLogFunc(LogFuncType func) { dtorch::LogMessage::SetLogFunc(func); }

}  // namespace cpp
}  // namespace api
}  // namespace dtorch

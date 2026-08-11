/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "debug.h"

#include <cassert>

#include "dtorch/api/cpp/version.h"
#include "logging.h"
#include "stack_track.h"

namespace dtorch {

void AssertFailMsgImp(const std::string& expr, const std::string& msg, const std::string& file, int line) {
    DLogError() << "DTorch assert failed:";
    if (!msg.empty()) {
        DLogError() << "Message: " << msg;
    }
    DLogError() << "File: " << file << ":" << line;
    DLogError() << "DTorch version" << api::cpp::GetVersionString();
    DLogError() << "DTorch compile arguments" << api::cpp::GetCompileArguments();
    DLogError() << "Expression: " << expr;
    DLogError() << "";
    DLogError() << GetStackTrack();
    DLogError() << "";
    DLogError() << "";

    std::abort();
}

void LogUnsupportedImpl(const std::string& file, int line) noexcept {
    DLogError() << "You had reached CAN'T reach code. Please concat DTorch author for more information";
    DLogError() << "File: " << file << ":" << line;
    DLogError() << "DTorch version" << api::cpp::GetVersionString();
    DLogError() << "DTorch compile arguments" << api::cpp::GetCompileArguments();
    DLogError() << "";
    DLogError() << GetStackTrack();
    DLogError() << "";
    std::abort();
}

void LogUnimplemented(const std::string& file, int line) noexcept {
    DLogError() << "You had reached unimplemented code. Please concat DTorch author for more information";
    DLogError() << "File: " << file << ":" << line;
    DLogError() << "DTorch version" << api::cpp::GetVersionString();
    DLogError() << "DTorch compile arguments" << api::cpp::GetCompileArguments();
    DLogError() << "";
    DLogError() << GetStackTrack();
    DLogError() << "";
    std::abort();
}

}  // namespace dtorch

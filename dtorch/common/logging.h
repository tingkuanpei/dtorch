/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <sstream>

#include "dtorch/api/cpp/log.h"
#include "utilities.h"

#define DLogInfo() dtorch::LogMessage(__FILE__, __LINE__, dtorch::LogSeverity::kInfo).Stream()
#define DLogWarning() dtorch::LogMessage(__FILE__, __LINE__, dtorch::LogSeverity::kWarning).Stream()
#define DLogError() dtorch::LogMessage(__FILE__, __LINE__, dtorch::LogSeverity::kError).Stream()
#define DLogFatal() dtorch::LogMessage(__FILE__, __LINE__, dtorch::LogSeverity::kFatal).Stream()

namespace dtorch {

using api::cpp::LogFuncType;
using api::cpp::LogSeverity;

class LogMessage {
public:
    static void SetLogFunc(LogFuncType func);

    static LogFuncType kGlobalLogFunc;

public:
    LogMessage(const char* file, int line, LogSeverity severity);

    ~LogMessage();

    DTORCH_DISABLE_COPY_AND_MOVE(LogMessage);

    DTORCH_FORCEINLINE std::stringstream& Stream() { return mStringStream; };

private:
    LogSeverity mSeverity;
    std::stringstream mStringStream;
    const char* mFile;
    int mLine;
};

}  // namespace dtorch

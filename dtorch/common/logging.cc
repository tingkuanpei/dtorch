/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#ifdef __GNUC__
#pragma GCC diagnostic push
// Disable warning: 'class google::base::CheckOpMessageBuilder' has pointer data members
#pragma GCC diagnostic ignored "-Weffc++"
#endif
#include <glog/logging.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include "logging.h"
#include "stack_track.h"

namespace dtorch {

//---------------------------------------------------- glog ------------------------------------------------------------
LogFuncType kGlogLogFunc = [](LogSeverity severity, const char* file, int line, const std::string& message) {
    switch (severity) {
        case dtorch::LogSeverity::kInfo:
            google::LogMessage(file, line, google::GLOG_INFO).stream() << message;
            break;
        case dtorch::LogSeverity::kWarning:
            google::LogMessage(file, line, google::GLOG_WARNING).stream() << message;
            break;
        case dtorch::LogSeverity::kError:
            google::LogMessage(file, line, google::GLOG_ERROR).stream() << message;
            break;
        case dtorch::LogSeverity::kFatal:
            google::LogMessage(file, line, google::GLOG_ERROR).stream() << message << "\n" << GetStackTrack(6);
            std::abort();
            break;
        default:
            break;
    }
};

class GlobalInitGlog {
public:
    GlobalInitGlog() {
        google::InitGoogleLogging("DTorch");
        FLAGS_logtostderr = 1;
        FLAGS_stderrthreshold = 0;
        FLAGS_minloglevel = 0;
    }
};

GlobalInitGlog kGlobalInitGlog;

//----------------------------------------------------------------------------------------------------------------------

LogFuncType LogMessage::kGlobalLogFunc = kGlogLogFunc;

void LogMessage::SetLogFunc(LogFuncType func) {
    if (func) {
        kGlobalLogFunc = func;
    } else {
        kGlobalLogFunc = kGlogLogFunc;
    }
}

LogMessage::LogMessage(const char* file, int line, LogSeverity severity)
    : mSeverity(severity), mStringStream(), mFile(file), mLine(line) {}

LogMessage::~LogMessage() { kGlobalLogFunc(mSeverity, mFile, mLine, mStringStream.str()); }

}  // namespace dtorch

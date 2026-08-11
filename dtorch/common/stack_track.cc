/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "stack_track.h"

#include "dtorch/common/utilities.h"
#include "logging.h"
#include "utilities.h"

#if DTORCH_PLATFORM_APPLE
#define BOOST_STACKTRACE_GNU_SOURCE_NOT_REQUIRED
#endif
// https://www.boost.org/doc/libs/master/doc/html/stacktrace/configuration_and_build.html
#if DTORCH_PLATFORM_LINUX
#define BOOST_STACKTRACE_USE_BACKTRACE
#endif

#ifdef __GNUC__
#pragma GCC diagnostic push
// Disable warning: 'struct boost::stacktrace::detail::to_string_using_backtrace' has pointer data members
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#include <boost/stacktrace.hpp>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include <sstream>

namespace dtorch {

std::string GetStackTrack(size_t skip, size_t max_depth) {
    std::stringstream ss;
    ss << "Print stack track: \n" << boost::stacktrace::stacktrace(skip + 1, max_depth);
    return ss.str();
}

}  // namespace dtorch

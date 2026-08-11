/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <string>
#include <type_traits>

#include "dtorch/common/config.h"
#include "utilities.h"

#define DAssertMsgImpl(expr, msg, file, line)                 \
    {                                                         \
        if (DTORCH_UNLIKELY(!(expr))) {                       \
            dtorch::AssertFailMsgImp(#expr, msg, file, line); \
        }                                                     \
    }

#define DAlwaysAssert(expr) DAssertMsgImpl(expr, "", __FILE__, __LINE__)
#define DAlwaysAssertMsg(expr, msg) DAssertMsgImpl(expr, msg, __FILE__, __LINE__)
#if DTORCH_DEBUG
#define DDebugAssert(expr) DAssertMsgImpl(expr, "", __FILE__, __LINE__)
#define DDebugAssertMsg(expr, msg) DAssertMsgImpl(expr, msg, __FILE__, __LINE__)
#else
#define DDebugAssert(expr) ((void)(expr))
#define DDebugAssertMsg(expr, msg) ((void)(expr))
#endif

#define DUnsupportedImpl() dtorch::LogUnsupportedImpl(__FILE__, __LINE__)
#define DUnimplemented() dtorch::LogUnimplemented(__FILE__, __LINE__)

namespace dtorch {

template <typename EnumerationType>
DTORCH_FORCEINLINE constexpr auto EnumAsInteger(const EnumerationType value) noexcept ->
    typename std::underlying_type<EnumerationType>::type {
    return static_cast<typename std::underlying_type<EnumerationType>::type>(value);
}

// TODO: add c++14 constexpr
// gnu c++11 not support constexpr with void return type
// https://stackoverflow.com/questions/29261276/constexpr-void-function-rejected
template <typename... Ts>
DTORCH_FORCEINLINE void IgnoreUnused(Ts&&...) {}

template <typename... Ts>
DTORCH_FORCEINLINE void IgnoreUnused() {}

void LogUnsupportedImpl(const std::string& file, int line) noexcept;

void LogUnimplemented(const std::string& file, int line) noexcept;

void AssertFailMsgImp(const std::string& expr, const std::string& msg, const std::string& file, int line);

}  // namespace dtorch

/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <string>

///////////////////////////////////
//
//  DTORCH_API_FORCEINLINE
//
///////////////////////////////////

// Macro to use in place of 'inline' to force a function to be inline
#if !defined(DTORCH_API_FORCEINLINE)
#if defined(_MSC_VER)
#define DTORCH_API_FORCEINLINE __forceinline
#elif defined(__GNUC__) && __GNUC__ > 3
// Clang also defines __GNUC__ (as 4)
#define DTORCH_API_FORCEINLINE inline __attribute__((__always_inline__))
#else
#define DTORCH_API_FORCEINLINE inline
#endif
#endif

///////////////////////////////////
//
//  DTORCH_API_LIKELY & DTORCH_API_UNLIKELY
//
///////////////////////////////////

#if !defined(DTORCH_API_LIKELY)
#define DTORCH_API_LIKELY
#endif

#if !defined(DTORCH_API_UNLIKELY)
#define DTORCH_API_UNLIKELY
#endif

///////////////////////////////////
//
//  DTORCH_API_WARN_UNUSED_RESULT
//
///////////////////////////////////

#if !defined(DTORCH_API_WARN_UNUSED_RESULT)
#define DTORCH_API_WARN_UNUSED_RESULT
#endif

///////////////////////////////////
//
//  DTORCH_API_DISABLE_COPY_AND_MOVE
//
///////////////////////////////////

#if !defined(DTORCH_API_DISABLE_COPY_AND_MOVE)
#define DTORCH_API_DISABLE_COPY_AND_MOVE(type) \
public:                                        \
    type(const type&) = delete;                \
    type(type&&) = delete;                     \
    type& operator=(const type&) = delete;     \
    type& operator=(type&&) = delete
#endif

///////////////////////////////////
//
//  DTORCH_API_DEFAULT_COPY_AND_MOVE
//
///////////////////////////////////

#if !defined(DTORCH_API_DEFAULT_COPY_AND_MOVE)
#define DTORCH_API_DEFAULT_COPY_AND_MOVE(type) \
    type(const type&) = default;               \
    type(type&&) = default;                    \
    type& operator=(const type&) = default;    \
    type& operator=(type&&) = default
#endif

///////////////////////////////////
//
//  DApiAssert & DApiAssertMsg
//
///////////////////////////////////

namespace dtorch {
void AssertFailMsgImp(const std::string& expr, const std::string& msg, const std::string& file, int line);
}

#define DApiAssertMsgImpl(expr, msg, file, line)              \
    {                                                         \
        if (DTORCH_API_UNLIKELY(!(expr))) {                   \
            dtorch::AssertFailMsgImp(#expr, msg, file, line); \
        }                                                     \
    }

#define DApiAssert(expr) DApiAssertMsgImpl(expr, "", __FILE__, __LINE__)
#define DApiAssertMsg(expr, msg) DApiAssertMsgImpl(expr, msg, __FILE__, __LINE__)

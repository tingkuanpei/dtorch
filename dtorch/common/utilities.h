/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

///////////////////////////////////
//
//  DTORCH_FORCEINLINE
//
///////////////////////////////////

// Macro to use in place of 'inline' to force a function to be inline
#if !defined(DTORCH_FORCEINLINE)
#if defined(_MSC_VER)
#define DTORCH_FORCEINLINE __forceinline
#elif defined(__GNUC__) && __GNUC__ > 3
// Clang also defines __GNUC__ (as 4)
#define DTORCH_FORCEINLINE inline __attribute__((__always_inline__))
#else
#define DTORCH_FORCEINLINE inline
#endif
#endif

///////////////////////////////////
//
//  DTORCH_INLINE
//
///////////////////////////////////

// Macro to use in place of 'inline'
#if !defined(DTORCH_INLINE)
#define DTORCH_INLINE inline
#endif

///////////////////////////////////
//
//  DTORCH_NOINLINE
//
///////////////////////////////////

// Macro to use in place of 'inline' to prevent a function to be inlined
#if !defined(DTORCH_NOINLINE)
#if defined(_MSC_VER)
#define DTORCH_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) && __GNUC__ > 3
// Clang also defines __GNUC__ (as 4)
#if defined(__CUDACC__)
// nvcc doesn't always parse __noinline__,
// see: https://svn.boost.org/trac/boost/ticket/9392
#define DTORCH_NOINLINE __attribute__((noinline))
#else
#define DTORCH_NOINLINE __attribute__((__noinline__))
#endif
#else
#define DTORCH_NOINLINE
#endif
#endif

///////////////////////////////////
//
//  DTORCH_LIKELY & DTORCH_UNLIKELY
//
///////////////////////////////////

#if !defined(DTORCH_LIKELY)
#if defined(__GNUC__)
#define DTORCH_LIKELY(x) __builtin_expect(x, 1)
#else
#define DTORCH_LIKELY(x) x
#endif
#endif

#if !defined(DTORCH_UNLIKELY)
#if defined(__GNUC__)
#define DTORCH_UNLIKELY(x) __builtin_expect(x, 0)
#else
#define DTORCH_UNLIKELY(x) x
#endif
#endif

///////////////////////////////////
//
//  DTORCH_STRINGIFY
//
///////////////////////////////////

// Convert the argument to a string
#define DTORCH_STRINGIFY(X) #X

///////////////////////////////////
//
//  DTORCH_DISABLE_COPY_AND_MOVE
//
///////////////////////////////////

#if !defined(DTORCH_DISABLE_COPY_AND_MOVE)
#define DTORCH_DISABLE_COPY_AND_MOVE(type) \
    type(const type&) = delete;            \
    type(type&&) = delete;                 \
    type& operator=(const type&) = delete; \
    type& operator=(type&&) = delete
#endif

///////////////////////////////////
//
//  DTORCH_DISABLE_COPY_AND_DEFAULT_MOVE
//
///////////////////////////////////

#if !defined(DTORCH_DISABLE_COPY_AND_DEFAULT_MOVE)
#define DTORCH_DISABLE_COPY_AND_DEFAULT_MOVE(type) \
    type(const type&) = delete;                    \
    type(type&&) = default;                        \
    type& operator=(const type&) = delete;         \
    type& operator=(type&&) = default
#endif

///////////////////////////////////
//
//  DTORCH_DEFAULT_COPY_AND_MOVE
//
///////////////////////////////////

#if !defined(DTORCH_DEFAULT_COPY_AND_MOVE)
#define DTORCH_DEFAULT_COPY_AND_MOVE(type)  \
    type(const type&) = default;            \
    type(type&&) = default;                 \
    type& operator=(const type&) = default; \
    type& operator=(type&&) = default
#endif

///////////////////////////////////
//
//  DTorch platform definitions
//
///////////////////////////////////

#define DTORCH_PLATFORM_ANDROID 0
#define DTORCH_PLATFORM_LINUX 0
#define DTORCH_PLATFORM_WINDOWS 0
#define DTORCH_PLATFORM_APPLE 0

#if defined(__ANDROID__) || defined(ANDROID)
#undef DTORCH_PLATFORM_ANDROID
#define DTORCH_PLATFORM_ANDROID 1
#elif (defined(linux) || defined(__linux) || defined(__linux__))
#undef DTORCH_PLATFORM_LINUX
#define DTORCH_PLATFORM_LINUX 1
#elif defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
#undef DTORCH_PLATFORM_WINDOWS
#define DTORCH_PLATFORM_WINDOWS 1
#elif defined(__MACH__) || defined(__APPLE__)
#undef DTORCH_PLATFORM_APPLE
#define DTORCH_PLATFORM_APPLE 1
#else
#error Unsupport platform!
#endif

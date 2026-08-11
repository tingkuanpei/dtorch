# Copyright 2026 The DTorch Authors. All rights reserved.
#
# Author: Tingkuan Pei(contact: peitingkuan@163.com)

if(DTORCH_THIRD_PARTY_USE_LOCAL_URL)
    set(GLOG_URL "${DTORCH_LOCAL_URL_DIR}/glog-0.5.0.tar.gz")
    if(NOT EXISTS ${GLOG_URL})
        set(ErrorMessage "Can't find local file: ${GLOG_URL}\n")
        string(APPEND ErrorMessage "Please download it manual or set DTORCH_THIRD_PARTY_USE_LOCAL_URL OFF.")
        message(FATAL_ERROR "${ErrorMessage}")
    endif()
else()
    set(GLOG_URL "https://github.com/google/glog/archive/refs/tags/v0.5.0.tar.gz")
endif()
set(GLOG_URL_SHA256 "eede71f28371bf39aa69b45de23b329d37214016e2055269b3b5e7cfd40b59f5")

# tmp modify: why can't find snprintf in msvc
# include (CheckSymbolExists)
# check_symbol_exists (snprintf cstdio HAVE_SNPRINTF)
# message("HAVE_SNPRINTF: ${HAVE_SNPRINTF}")
set(HAVE_SNPRINTF 1)
set(BUILD_SHARED_LIBS   OFF CACHE BOOL "" FORCE)
set(WITH_GFLAGS         OFF CACHE BOOL "" FORCE)
set(WITH_GTEST          OFF CACHE BOOL "" FORCE)

set(FETCHCONTENT_BASE_DIR   ${CMAKE_BINARY_DIR}/third_party/glog)
FetchContent_Declare(
    glog
    URL ${GLOG_URL}
    URL_HASH SHA256=${GLOG_URL_SHA256}
)
FetchContent_MakeAvailable(glog)

set(GLOG_INCLUDE_DIR ${glog_BINARY_DIR} ${glog_SOURCE_DIR}/src)
set(GLOG_LIBRARY glog::glog)

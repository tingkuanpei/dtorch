# Copyright 2026 The DTorch Authors. All rights reserved.
#
# Author: Tingkuan Pei(contact: peitingkuan@163.com)

if(DTORCH_THIRD_PARTY_USE_LOCAL_URL)
    set(GOOGLETEST_URL "${DTORCH_LOCAL_URL_DIR}/googletest-release-1.11.0.tar.gz")
    if(NOT EXISTS ${GOOGLETEST_URL})
        set(ErrorMessage "Can't find local file: ${GOOGLETEST_URL}\n")
        string(APPEND ErrorMessage "Please download it manual or set DTORCH_THIRD_PARTY_USE_LOCAL_URL OFF.")
        message(FATAL_ERROR "${ErrorMessage}")
    endif()
else()
    set(GOOGLETEST_URL "https://github.com/google/googletest/archive/refs/tags/release-1.11.0.tar.gz")
endif()
set(GOOGLETEST_URL_SHA256 "b4870bf121ff7795ba20d20bcdd8627b8e088f2d1dab299a031c1034eddc93d5")

set(gtest_force_shared_crt  ON CACHE BOOL "" FORCE)
set(BUILD_GMOCK             OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS       OFF CACHE BOOL "" FORCE)

set(FETCHCONTENT_BASE_DIR   ${CMAKE_BINARY_DIR}/third_party/googletest)
FetchContent_Declare(
    googletest
    URL ${GOOGLETEST_URL}
    URL_HASH SHA256=${GOOGLETEST_URL_SHA256}
)
FetchContent_MakeAvailable(googletest)

set(GOOGLETEST_INCLUDE_DIR ${googletest_SOURCE_DIR}/googletest/include)
set(GOOGLETEST_LIBRARY GTest::gtest)

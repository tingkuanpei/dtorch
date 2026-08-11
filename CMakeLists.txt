# Copyright 2026 The DTorch Authors. All rights reserved.
#
# Author: Tingkuan Pei(contact: peitingkuan@163.com)

cmake_minimum_required(VERSION 3.18.0 FATAL_ERROR)

project(DTorch)

set(DTORCH_ROOT_DIR         ${CMAKE_CURRENT_SOURCE_DIR})
set(DTORCH_DLL_DIR          ${CMAKE_BINARY_DIR}/bin)
file(MAKE_DIRECTORY         ${DTORCH_DLL_DIR})

set_property(GLOBAL PROPERTY USE_FOLDERS    ON)

# This option is implemented only by Makefile Generators and the Ninja
# https://cmake.org/cmake/help/latest/variable/CMAKE_EXPORT_COMPILE_COMMANDS.html
if(CMAKE_GENERATOR MATCHES "Makefiles" OR CMAKE_GENERATOR STREQUAL "Ninja" )
    set(CMAKE_EXPORT_COMPILE_COMMANDS   ON)
endif()

list(APPEND CMAKE_MODULE_PATH ${DTORCH_ROOT_DIR}/cmake)
include(CheckLanguage)
include(util)
include(FetchContent)
include(ExternalProject)

DSetValueIfEmpty(DTORCH_LOCAL_URL_DIR    "${DTORCH_ROOT_DIR}/third_party_local_url")
DSetValueIfEmpty(GRPC_INSTALL_DIR        "${DTORCH_ROOT_DIR}/third_party_install/grpc_install")
DSetValueIfEmpty(LIBZMQ_INSTALL_DIR      "${DTORCH_ROOT_DIR}/third_party_install/libzmq_install")
DSetValueIfEmpty(CPPZMQ_INSTALL_DIR      "${DTORCH_ROOT_DIR}/third_party_install/cppzmq_install")

#---------------------------------------------------------------------------------------------------
# Set DTORCH_PLATFORM
set(DTORCH_PLATFORM_IOS             OFF)
set(DTORCH_PLATFORM_ANDROID         OFF)
set(DTORCH_PLATFORM_WINDOWS         OFF)
set(DTORCH_PLATFORM_MACOS           OFF)
set(DTORCH_PLATFORM_LINUX           OFF)

if(IOS)
    set(DTORCH_PLATFORM_IOS         ON)
elseif(ANDROID)
    set(DTORCH_PLATFORM_ANDROID     ON)
elseif(MSVC)
    set(DTORCH_PLATFORM_WINDOWS     ON)
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(DTORCH_PLATFORM_MACOS       ON)
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set(DTORCH_PLATFORM_LINUX       ON)
endif()

#---------------------------------------------------------------------------------------------------
# CMake compile options
option(DTORCH_DEBUG                         "Enbale DTorch debug mode"                          ON)
option(DTORCH_BUILD_TEST                    "Build DTorch with test"                            ON)
if(DTORCH_PLATFORM_WINDOWS OR DTORCH_PLATFORM_LINUX)
    option(DTORCH_WITH_CUDA                 "Build DTorch with CUDA"                            ON)
    option(DTORCH_WITH_TRITON               "Build DTorch with Triton"                          ON)
else()
    option(DTORCH_WITH_CUDA                 "Build DTorch with CUDA"                            OFF)
    option(DTORCH_WITH_TRITON               "Build DTorch with Triton"                          OFF)
endif()
option(DTORCH_WITH_BOOST                    "Build DTorch with Boost"                           ON)
option(DTORCH_THIRD_PARTY_USE_LOCAL_URL     "Third party use local url rather than http url"    ON)

if(DTORCH_WITH_CUDA)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
    else()
        message(FATAL_ERROR "Can't find CUDA support, please check cuda library or set DTORCH_WITH_CUDA OFF")
    endif()

    message(STATUS "")
    message(STATUS "-------------------------- Enable CUDA ----------------------------")
    include(third_party/cuda)
    message(STATUS "CUDAToolkit_FOUND: ${CUDAToolkit_FOUND}")
    message(STATUS "CUDAToolkit_VERSION: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDAToolkit_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")
    message(STATUS "CUDAToolkit_LIBRARY_DIR: ${CUDAToolkit_LIBRARY_DIR}")
    message(STATUS "CUDAToolkit_NVCC_EXECUTABLE: ${CUDAToolkit_NVCC_EXECUTABLE}")
    message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "CUDA_LIBRARY: ${CUDA_LIBRARY}")
    message(STATUS "-------------------------------------------------------------------")
    message(STATUS "")
else()
    set(DTORCH_WITH_TRITON      OFF)
endif()

if(CMAKE_GENERATOR MATCHES "Visual Studio" OR CMAKE_GENERATOR STREQUAL "Xcode")
    set(CMAKE_CONFIGURATION_TYPES   "${CMAKE_BUILD_TYPE}")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(DTORCH_BUILD_DEBUG     ON)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release" OR
       CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo" OR
       CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
    set(DTORCH_BUILD_DEBUG     OFF)
else()
    message(FATAL_ERROR "Unrecognized CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}, candidate is Release or Debug")
endif()

# TODO: close DTORCH_BUILD_DEBUG
# # DTORCH_DEBUG follows CMAKE_BUILD_TYPE: ON for Debug, OFF for Release.
# if(DTORCH_BUILD_DEBUG)
#     set(DTORCH_DEBUG ON CACHE BOOL "Enable DTorch debug mode (runtime assertions, debug logging)" FORCE)
# else()
#     set(DTORCH_DEBUG OFF CACHE BOOL "Enable DTorch debug mode (runtime assertions, debug logging)" FORCE)
# endif()

# LTO: cross-module inlining and optimization
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)

#---------------------------------------------------------------------------------------------------
# Set C/C++ compiler/linker flags
if(NOT DTORCH_PLATFORM_WINDOWS)
    # c11, c++17
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

    set(CMAKE_C_FLAGS_DEBUG "-O0 -g")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
    set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -mtune=native")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native")
else()
    # c11, c++17
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /std:c11")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")

    # Force to always compile with warning level: W4
    if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
        string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
    endif()

    # Disable 4819 warning in visual studio when build cuda src
    # https://forums.developer.nvidia.com/t/how-to-tell-nvcc-exe-disable-to-print-some-warning-information/36684
    if(DTORCH_WITH_CUDA)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"/wd 4819\"")
    endif()

    # Generate .pdb file that contains all the symbolic debugging information for release build
    # https://stackoverflow.com/questions/28178978/how-to-generate-pdb-files-for-release-build-with-cmake-flags/31264946
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /DEBUG /OPT:REF /OPT:ICF")

    # Build with Multiple Processes
    # https://docs.microsoft.com/en-us/cpp/build/reference/mp-build-with-multiple-processes?view=msvc-170
    set(${CMAKE_CXX_FLAGS} "${CMAKE_CXX_FLAGS} /MP")
endif()

#---------------------------------------------------------------------------------------------------
# MSVC runtime library
if(DTORCH_PLATFORM_WINDOWS)
    DSetValueIfEmpty(DTORCH_MSVC_RUNTIME_LIBRARY    "MD")

    if(DTORCH_MSVC_RUNTIME_LIBRARY STREQUAL "MD")
        set(CMAKE_DTORCH_MSVC_RUNTIME_LIBRARY MultiThreaded$<$<CONFIG:Debug>:Debug>DLL)
    elseif(DTORCH_MSVC_RUNTIME_LIBRARY STREQUAL "MT")
        set(CMAKE_DTORCH_MSVC_RUNTIME_LIBRARY MultiThreaded$<$<CONFIG:Debug>:Debug>)
        message(FATAL_ERROR "DTorch only support DTORCH_MSVC_RUNTIME_LIBRARY as MD now")
    else()
        message(FATAL_ERROR "DTORCH_MSVC_RUNTIME_LIBRARY: ${DTORCH_MSVC_RUNTIME_LIBRARY} is not supported")
    endif()
endif()

#---------------------------------------------------------------------------------------------------
# DTORCH_COMPILE_ARGUMENTS
set(DTORCH_COMPILE_ARGUMENTS "${CMAKE_BUILD_TYPE} mode with ${CMAKE_CXX_FLAGS}")

#---------------------------------------------------------------------------------------------------
# Logging compile options
message(STATUS "")
message(STATUS "-------------------- DTorch compile options -----------------------")
message(STATUS "CMAKE_CXX_COMPILER:                            ${CMAKE_CXX_COMPILER}")
message(STATUS "CMAKE_CXX_FLAGS:                               ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_BUILD_TYPE:                              ${CMAKE_BUILD_TYPE}")
message(STATUS "DTORCH_DEBUG:                                  ${DTORCH_DEBUG}")
if(DTORCH_PLATFORM_WINDOWS)
message(STATUS "DTORCH_MSVC_RUNTIME_LIBRARY:                   ${DTORCH_MSVC_RUNTIME_LIBRARY}")
endif()
message(STATUS "")
message(STATUS "DTORCH_BUILD_TEST:                             ${DTORCH_BUILD_TEST}")
message(STATUS "")
message(STATUS "DTORCH_WITH_CUDA:                              ${DTORCH_WITH_CUDA}")
message(STATUS "DTORCH_WITH_TRITON:                            ${DTORCH_WITH_TRITON}")
message(STATUS "DTORCH_WITH_BOOST:                             ${DTORCH_WITH_BOOST}")
message(STATUS "DTORCH_THIRD_PARTY_USE_LOCAL_URL:              ${DTORCH_THIRD_PARTY_USE_LOCAL_URL}")
message(STATUS "DTORCH_INTEL_MAXOS_TORCH_2_2_2:                ${DTORCH_INTEL_MAXOS_TORCH_2_2_2}")
if(DTORCH_THIRD_PARTY_USE_LOCAL_URL)
message(STATUS "DTORCH_LOCAL_URL_DIR:                          ${DTORCH_LOCAL_URL_DIR}")
endif()
message(STATUS "GRPC_INSTALL_DIR:                              ${GRPC_INSTALL_DIR}")
message(STATUS "LIBZMQ_INSTALL_DIR:                            ${LIBZMQ_INSTALL_DIR}")
message(STATUS "CPPZMQ_INSTALL_DIR:                            ${CPPZMQ_INSTALL_DIR}")
message(STATUS "-------------------------------------------------------------------")
message(STATUS "")

#---------------------------------------------------------------------------------------------------
# Find GRPC
#
# include third_party/grpc before third_party/torch to avoid error:
# add_library cannot create ALIAS target "protobuf::libprotobuf" because another target with the same name already exists.
message(STATUS "")
message(STATUS "------------------------- Find GRPC -------------------------------")
include(third_party/grpc)
message(STATUS "GRPC_INCLUDE_DIR: ${GRPC_INCLUDE_DIR}")
message(STATUS "GRPC_LIBRARYS: ${GRPC_LIBRARYS}")

#---------------------------------------------------------------------------------------------------
# Find PyTorch
include(third_party/torch)

if (DTORCH_WITH_CUDA)
    if (NOT TORCH_SUPPORT_CUDA)
        message(FATAL_ERROR "PyTorch not support cuda, but DTORCH_WITH_CUDA = ON. "
                            "Please install PyTorch with CUDA support or set DTORCH_WITH_CUDA = OFF")
    endif()

    # Check main version between TORCH_CUDA_VERSION and CUDAToolkit_VERSION
    string(REGEX MATCH "^([0-9]+)" MAIN_VERSION1 "${TORCH_CUDA_VERSION}")
    string(REGEX MATCH "^([0-9]+)" MAIN_VERSION2 "${CUDAToolkit_VERSION}")
    if(NOT MAIN_VERSION1 STREQUAL MAIN_VERSION2)
        message(FATAL_ERROR "PyTorch with cuda ${TORCH_CUDA_VERSION}, but find NVCC with cuda ${CUDAToolkit_VERSION}")
    endif()
endif()

#---------------------------------------------------------------------------------------------------
# add third party
include(all_third_party)

#---------------------------------------------------------------------------------------------------
# Update C/C++ compiler/linker flags for more stringent code check
set(DTORCH_CXX_FLAGS    "")
if(NOT DTORCH_PLATFORM_WINDOWS)
    # # 函数没返回值报错
    list(APPEND DTORCH_CXX_FLAGS "-Werror=return-type")

    # Force to always compile with the most pedantic warning level
    list(APPEND DTORCH_CXX_FLAGS "-Wall;-Wextra;-Weffc++;-pedantic")

    # Treat warning as error
    list(APPEND DTORCH_CXX_FLAGS "-Werror")
else()
    # Support utf-8
    list(APPEND DTORCH_CXX_FLAGS "/utf-8")

    # Treat warning as error
    list(APPEND DTORCH_CXX_FLAGS "/WX")
endif()

#---------------------------------------------------------------------------------------------------
# compile triton aot kernel
if(DTORCH_WITH_TRITON)
    add_subdirectory(dtorch/external/triton)
endif()

# build library and executor
# dtorch_core library
add_subdirectory(dtorch)

# python _dtorch_py_api library
add_subdirectory(dtorch/api/python)

# Copyright 2026 The DTorch Authors. All rights reserved.
#
# Author: Tingkuan Pei(contact: peitingkuan@163.com)

set(DTORCH_LIB_CXX_SRCS    "")
set(DTORCH_LIB_CUDA_SRCS   "")
set(DTORCH_INCLUDE_DIRS    "")
set(DTORCH_LINK_LIBRARIES  "")
set(DTORCH_LIBRARIES       "")

#---------------------------------------------------------------------------------------------------
# Get DTORCH_COMMIT_HASH
set(DTORCH_COMMIT_HASH "")
DGetAbbrevCommitHash(DTORCH_COMMIT_HASH ${DTORCH_ROOT_DIR})
if(NOT DTORCH_COMMIT_HASH)
    set(DTORCH_COMMIT_HASH "00000000")
endif()

#----------------------------------------------------------------------------------------
# Generate config.h
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/common/config.h.in
    ${CMAKE_BINARY_DIR}/dtorch/common/config.h
)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/common/config.cc.in
    ${CMAKE_BINARY_DIR}/dtorch/common/config.cc
)

list(APPEND DTORCH_LIB_CXX_SRCS "${CMAKE_BINARY_DIR}/dtorch/common/config.h")
list(APPEND DTORCH_LIB_CXX_SRCS "${CMAKE_BINARY_DIR}/dtorch/common/config.cc")

#----------------------------------------------------------------------------------------
# Generate rpc protobuf cpp file
include(rpc_proto2cpp)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/external/rpc/proto)

file(GLOB_RECURSE PROTO_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/external/rpc/proto/*.proto
)
RPC_PROTOBUF_GENERATE_CPP(RPC_PROTO_SRCS RPC_PROTO_HDRS RPC_RPC_SRCS RPC_RPC_HDRS ${PROTO_SRCS})

add_library(dtorch_rpc_generated STATIC
    ${RPC_PROTO_SRCS}
    ${RPC_PROTO_HDRS}
    ${RPC_RPC_SRCS}
    ${RPC_RPC_HDRS}
)
target_include_directories(dtorch_rpc_generated PRIVATE ${GRPC_INCLUDE_DIR})
target_link_libraries(dtorch_rpc_generated ${GRPC_LIBRARYS})

#----------------------------------------------------------------------------------------
# Set DTORCH_LIB_CXX_SRCS DTORCH_LIB_CUDA_SRCS
file(GLOB_RECURSE tmp
    ${CMAKE_CURRENT_SOURCE_DIR}/api/cpp/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/api/cpp/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/common/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/core/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/core/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/external/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/external/*.cc
)
list(APPEND DTORCH_LIB_CXX_SRCS ${tmp})
if(DTORCH_WITH_CUDA)
    file(GLOB_RECURSE tmp ${CMAKE_CURRENT_SOURCE_DIR}/*.cu ${CMAKE_CURRENT_SOURCE_DIR}/*.cuh)
    list(APPEND DTORCH_LIB_CUDA_SRCS ${tmp})
endif()

if(NOT DTORCH_WITH_CUDA)
    file(GLOB_RECURSE tmp
        ${CMAKE_CURRENT_SOURCE_DIR}/core/kernel/cuda/*
        ${CMAKE_CURRENT_SOURCE_DIR}/core/memory/cuda/*
        ${CMAKE_CURRENT_SOURCE_DIR}/core/tensor/cuda/*
        ${CMAKE_CURRENT_SOURCE_DIR}/core/kernel_stream/cuda*
        ${CMAKE_CURRENT_SOURCE_DIR}/external/cuda/*
        ${CMAKE_CURRENT_SOURCE_DIR}/external/device/cuda*
        ${CMAKE_CURRENT_SOURCE_DIR}/external/nccl/*
    )
    list(REMOVE_ITEM DTORCH_LIB_CXX_SRCS ${tmp})
endif()

#----------------------------------------------------------------------------------------
# Set DTORCH_INCLUDE_DIRS DTORCH_LINK_LIBRARIES
list(APPEND DTORCH_INCLUDE_DIRS ${DTORCH_ROOT_DIR})
list(APPEND DTORCH_INCLUDE_DIRS ${CMAKE_BINARY_DIR})

list(APPEND DTORCH_INCLUDE_DIRS ${GRPC_INCLUDE_DIR})
list(APPEND DTORCH_LINK_LIBRARIES ${GRPC_LIBRARYS})
list(APPEND DTORCH_LINK_LIBRARIES dtorch_rpc_generated)

list(APPEND DTORCH_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS})
list(APPEND DTORCH_INCLUDE_DIRS ${NANOBIND_INCLUDE_DIR})
list(APPEND DTORCH_LINK_LIBRARIES ${NANOBIND_LIBRARYS})
list(APPEND DTORCH_LINK_LIBRARIES ${Python_LIBRARIES})
list(APPEND DTORCH_LINK_LIBRARIES ${TORCH_LIBRARIES})

if(DTORCH_WITH_BOOST)
    list(APPEND DTORCH_INCLUDE_DIRS ${BOOST_INCLUDE_DIR})
    list(APPEND DTORCH_LINK_LIBRARIES ${BOOST_LIBRARYS})
endif()

list(APPEND DTORCH_INCLUDE_DIRS ${GHC_FILESYSTEM_INCLUDE_DIR})

list(APPEND DTORCH_INCLUDE_DIRS ${THREAD_SAFE_QUEUE_INCLUDE_DIR})

list(APPEND DTORCH_INCLUDE_DIRS ${GLOG_INCLUDE_DIR})
list(APPEND DTORCH_LINK_LIBRARIES ${GLOG_LIBRARY})

list(APPEND DTORCH_INCLUDE_DIRS ${ZMQ_INCLUDE_DIR})
list(APPEND DTORCH_LINK_LIBRARIES ${ZMQ_LIBRARYS})

if(DTORCH_WITH_CUDA)
    list(APPEND DTORCH_LINK_LIBRARIES  ${CUDA_LIBRARY})
    list(APPEND DTORCH_INCLUDE_DIRS ${NCCL_INCLUDE_DIR})
    list(APPEND DTORCH_LINK_LIBRARIES  ${NCCL_LIBRARY})
    if (DTORCH_WITH_TRITON)
        list(APPEND DTORCH_INCLUDE_DIRS ${DTORCH_TRITON_INCLUDE_DIRS})
        list(APPEND DTORCH_LINK_LIBRARIES  ${DTORCH_TRITON_LINK_LIBRARIES})
    endif()
endif()

#----------------------------------------------------------------------------------------
# Build library
DGroupSourceFiles(DTORCH_LIB_CXX_SRCS)
DGroupSourceFiles(DTORCH_LIB_CUDA_SRCS)

set_source_files_properties(external/torch/torch_stream_guard.cc
    PROPERTIES
    COMPILE_FLAGS "-fvisibility=hidden"
)

dtorch_add_library(dtorch_core STATIC ${DTORCH_LIB_CXX_SRCS} ${DTORCH_LIB_CUDA_SRCS})
target_include_directories(dtorch_core PRIVATE ${DTORCH_INCLUDE_DIRS})
target_include_directories(dtorch_core SYSTEM PRIVATE ${NANOBIND_INCLUDE_DIR})
target_link_libraries(dtorch_core PRIVATE ${DTORCH_LINK_LIBRARIES})

add_dependencies(dtorch_core all_third_party)

#----------------------------------------------------------------------------------------
# Set DTORCH_LIBRARIES
set(DTORCH_LIBRARIES            "dtorch_core")
list(APPEND DTORCH_LIBRARIES    "${DTORCH_LINK_LIBRARIES}")

if(DTORCH_PLATFORM_LINUX)
    # boost backtrace
    # https://www.boost.org/doc/libs/master/doc/html/stacktrace/configuration_and_build.html
    list(APPEND DTORCH_LIBRARIES   "-ldl -lbacktrace")
    list(APPEND DTORCH_LIBRARIES   "-lpthread")
    list(APPEND DTORCH_LIBRARIES   "-fopenmp")
endif()

set(DTORCH_INCLUDE_DIRS ${DTORCH_INCLUDE_DIRS} PARENT_SCOPE)
set(DTORCH_LIBRARIES ${DTORCH_LIBRARIES} PARENT_SCOPE)

message(STATUS "DTORCH_LIBRARIES: ${DTORCH_LIBRARIES}")

#----------------------------------------------------------------------------------------
# test & tools
if(DTORCH_BUILD_TEST)
    add_subdirectory(tests)
endif()

add_subdirectory(tools)

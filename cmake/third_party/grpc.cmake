# Copyright 2026 The DTorch Authors. All rights reserved.
#
# Author: Tingkuan Pei(contact: peitingkuan@163.com)

set(GRPC_CMAKE_PREFIX_PATH ${GRPC_INSTALL_DIR}/lib/cmake)

if(NOT GRPC_INSTALL_DIR STREQUAL "" AND EXISTS "${GRPC_CMAKE_PREFIX_PATH}")
    message(STATUS "Find grpc from ${GRPC_INSTALL_DIR}...")

    list(APPEND CMAKE_PREFIX_PATH   "${GRPC_CMAKE_PREFIX_PATH}")

    option(protobuf_MODULE_COMPATIBLE TRUE)
    find_package(Protobuf CONFIG REQUIRED)

    find_package(gRPC CONFIG REQUIRED)

    set(PROTOBUF_PROTOC_EXECUTABLE  $<TARGET_FILE:protobuf::protoc>)
    if(CMAKE_CROSSCOMPILING)
        find_program(GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
    else()
        set(GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)
    endif()

    set(GRPC_INCLUDE_DIR            ${GRPC_INSTALL_DIR}/include)
    set(GRPC_LIBRARYS               "")
    list(APPEND GRPC_LIBRARYS       gRPC::grpc++_reflection)
    list(APPEND GRPC_LIBRARYS       protobuf::libprotobuf)
    list(APPEND GRPC_LIBRARYS       gRPC::grpc++)
else()
    message(STATUS "Build grpc from source...")

    if(DTORCH_THIRD_PARTY_USE_LOCAL_URL)
        set(GRPC_URL "${DTORCH_LOCAL_URL_DIR}/grpc-1.76.0.tar.gz")
        if(NOT EXISTS ${GRPC_URL})
            set(ErrorMessage "Can't find local file: ${GRPC_URL}\n")
            string(APPEND ErrorMessage "Please download it manual or set DTORCH_THIRD_PARTY_USE_LOCAL_URL OFF.")
            message(FATAL_ERROR "${ErrorMessage}")
        endif()
    else()
        set(GRPC_URL "https://github.com/grpc/grpc/archive/refs/tags/v1.76.0.tar.gz")
    endif()

    # https://github.com/grpc/grpc/issues/34742
    set(protobuf_INSTALL OFF)
    set(utf8_range_ENABLE_INSTALL OFF)
    set(FETCHCONTENT_BASE_DIR   ${CMAKE_BINARY_DIR}/third_party/grpc)
    FetchContent_Declare(
        gRPC
        URL ${GRPC_URL}
    )
    FetchContent_MakeAvailable(gRPC)

    # Fix No rule to make target 'third_party/grpc/grpc-build/third_party/protobuf/libprotobufd.a', needed by 'test/test'.  Stop.
    set_target_properties(libprotobuf PROPERTIES
        DEBUG_POSTFIX ""
    )

    set(PROTOBUF_PROTOC_EXECUTABLE  $<TARGET_FILE:protoc>)
    if(CMAKE_CROSSCOMPILING)
        find_program(GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
    else()
        set(GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:grpc_cpp_plugin>)
    endif()

    set(GRPC_INCLUDE_DIR            ${grpc_SOURCE_DIR}/include)
    set(GRPC_LIBRARYS               "")
    list(APPEND GRPC_LIBRARYS       grpc++_reflection)
    list(APPEND GRPC_LIBRARYS       libprotobuf)
    list(APPEND GRPC_LIBRARYS       grpc++)
endif()

# Copyright 2026 The UGraph Authors. All rights reserved.
#
# Author: Tingkuan Pei(contact: peitingkuan@163.com)

function(RPC_PROTOBUF_GENERATE_CPP PROTO_SRCS PROTO_HDRS RPC_SRCS RPC_HDRS)
    if(NOT ARGN)
        message(SEND_ERROR "Error: RPC_PROTOBUF_GENERATE_CPP() called without any proto files")
        return()
    endif()

    set(${PROTO_SRCS})
    set(${PROTO_HDRS})
    set(${RPC_SRCS})
    set(${RPC_HDRS})

    foreach(FIL ${ARGN})
        set(ABS_FIL "${FIL}")
        file(RELATIVE_PATH REL_FIL ${CMAKE_CURRENT_SOURCE_DIR} ${ABS_FIL})
        get_filename_component(FIL_WE ${REL_FIL} NAME_WE)
        get_filename_component(FIL_DIR ${REL_FIL} PATH)

        set(PROTO_SRC "${CMAKE_CURRENT_BINARY_DIR}/${FIL_DIR}/${FIL_WE}.pb.cc")
        set(PROTO_HDR "${CMAKE_CURRENT_BINARY_DIR}/${FIL_DIR}/${FIL_WE}.pb.h")
        set(RPC_SRC  "${CMAKE_CURRENT_BINARY_DIR}/${FIL_DIR}/${FIL_WE}.grpc.pb.cc")
        set(RPC_HDR  "${CMAKE_CURRENT_BINARY_DIR}/${FIL_DIR}/${FIL_WE}.grpc.pb.h")

        list(APPEND ${PROTO_SRCS} "${PROTO_SRC}")
        list(APPEND ${PROTO_HDRS} "${PROTO_HDR}")
        list(APPEND ${RPC_SRCS}  "${RPC_SRC}")
        list(APPEND ${RPC_HDRS}  "${RPC_HDR}")

        add_custom_command(
            OUTPUT "${PROTO_SRC}"
                   "${PROTO_HDR}"
                   "${RPC_SRC}"
                   "${RPC_HDR}"
            COMMAND  ${PROTOBUF_PROTOC_EXECUTABLE}
            ARGS --grpc_out  ${CMAKE_CURRENT_BINARY_DIR}/${FIL_DIR}
                 --cpp_out  ${CMAKE_CURRENT_BINARY_DIR}/${FIL_DIR}
                 -I ${CMAKE_CURRENT_SOURCE_DIR}/${FIL_DIR}
                 --plugin=protoc-gen-grpc="${GRPC_CPP_PLUGIN_EXECUTABLE}"
                 ${ABS_FIL}
            DEPENDS ${ABS_FIL} ${PROTOBUF_PROTOC_EXECUTABLE}
            COMMENT "Running Protocol Buffer Compiler on ${FIL}")
    endforeach()

    set_source_files_properties(${${PROTO_SRCS}} ${${PROTO_HDRS}} ${${RPC_SRCS}} ${${RPC_HDRS}} PROPERTIES GENERATED TRUE)
    set(${PROTO_SRCS} ${${PROTO_SRCS}} PARENT_SCOPE)
    set(${PROTO_HDRS} ${${PROTO_HDRS}} PARENT_SCOPE)
    set(${RPC_SRCS} ${${RPC_SRCS}} PARENT_SCOPE)
    set(${RPC_HDRS} ${${RPC_HDRS}} PARENT_SCOPE)
endfunction()

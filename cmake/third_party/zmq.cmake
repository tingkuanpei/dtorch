# Copyright 2026 The DTorch Authors. All rights reserved.
#
# Author: Tingkuan Pei(contact: peitingkuan@163.com)

set(LIBZMQ_CMAKE_PREFIX_PATH ${LIBZMQ_INSTALL_DIR}/lib/cmake)
set(CPPZMQ_CMAKE_PREFIX_PATH ${CPPZMQ_INSTALL_DIR}/share/cmake)
list(APPEND CMAKE_PREFIX_PATH   "${LIBZMQ_CMAKE_PREFIX_PATH}")
list(APPEND CMAKE_PREFIX_PATH   "${CPPZMQ_CMAKE_PREFIX_PATH}")

find_package(cppzmq CONFIG REQUIRED)
if(NOT ZeroMQ_FOUND OR NOT cppzmq_FOUND)
    message(FATAL_ERROR "cppzmq not found")
endif()

set(ZMQ_INCLUDE_DIR             "")
list(APPEND ZMQ_INCLUDE_DIR     ${ZeroMQ_INCLUDE_DIR})
list(APPEND ZMQ_INCLUDE_DIR     ${cppzmq_INCLUDE_DIR})
set(ZMQ_LIBRARYS                "")
list(APPEND ZMQ_LIBRARYS        ${ZeroMQ_STATIC_LIBRARY})
list(APPEND ZMQ_LIBRARYS        ${cppzmq_STATIC_LIBRARY})

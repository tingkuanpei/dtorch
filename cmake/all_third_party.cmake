# Copyright 2026 The DTorch Authors. All rights reserved.
#
# Author: Tingkuan Pei(contact: peitingkuan@163.com)

set(DTORCH_THIRD_PARTY_DLLS  "")

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()

message(STATUS "")
message(STATUS "--------------------- DTorch third_party --------------------------")

add_custom_target(all_third_party)
DGetAllTargets(ALL_TARGETS_BEFORE_THIRD_PARTY)

message(STATUS "Generate glog...")
include(third_party/glog)
message(STATUS "GLOG_INCLUDE_DIR: ${GLOG_INCLUDE_DIR}")
message(STATUS "GLOG_LIBRARY: ${GLOG_LIBRARY}")
add_dependencies(all_third_party ${GLOG_LIBRARY})

message(STATUS "------------------------------")
message(STATUS "Find nanobind...")
include(third_party/nanobind)
message(STATUS "NANOBIND_INCLUDE_DIR: ${NANOBIND_INCLUDE_DIR}")
message(STATUS "NANOBIND_LIBRARYS: ${NANOBIND_LIBRARYS}")

if(DTORCH_WITH_BOOST)
    message(STATUS "------------------------------")
    message(STATUS "Find boost...")
    include(third_party/boost)
    message(STATUS "BOOST_INCLUDE_DIR: ${BOOST_INCLUDE_DIR}")
    message(STATUS "BOOST_LIBRARYS:    ${BOOST_LIBRARYS}")
endif()

message(STATUS "------------------------------")
message(STATUS "Generate cameron314 thread safe queue...")
include(third_party/thread_safe_queue)
message(STATUS "THREAD_SAFE_QUEUE_INCLUDE_DIR: ${THREAD_SAFE_QUEUE_INCLUDE_DIR}")

include(third_party/zmq)
message(STATUS "ZMQ_INCLUDE_DIR: ${ZMQ_INCLUDE_DIR}")
message(STATUS "ZMQ_LIBRARYS: ${ZMQ_LIBRARYS}")

if(DTORCH_BUILD_TEST)
    message(STATUS "------------------------------")
    message(STATUS "Build googletest...")
    include(third_party/googletest)
    message(STATUS "GOOGLETEST_INCLUDE_DIR: ${GOOGLETEST_INCLUDE_DIR}")
    message(STATUS "GOOGLETEST_LIBRARY: ${GOOGLETEST_LIBRARY}")
    add_dependencies(all_third_party ${GOOGLETEST_LIBRARY})
endif()

if(DTORCH_WITH_CUDA)
    message(STATUS "------------------------------")
    include(third_party/nccl)
    message(STATUS "NCCL_INCLUDE_DIR: ${NCCL_INCLUDE_DIR}")
    message(STATUS "NCCL_LIBRARY: ${NCCL_LIBRARY}")
endif()

message(STATUS "------------------------------")
# set third party target folder in IDE
if(CMAKE_GENERATOR MATCHES "Visual Studio" OR CMAKE_GENERATOR STREQUAL "Xcode")
    DGetAllTargets(ALL_TARGETS_AFTER_THIRD_PARTY)
    list(REMOVE_ITEM ALL_TARGETS_AFTER_THIRD_PARTY ${ALL_TARGETS_BEFORE_THIRD_PARTY})
    foreach(DTORCH_THIRD_PARTY_TARGET ${ALL_TARGETS_AFTER_THIRD_PARTY})
        set_target_properties(${DTORCH_THIRD_PARTY_TARGET} PROPERTIES FOLDER "ThirdParty")
    endforeach()
endif()

# Copy dll to DTORCH_DLL_DIR
if(DTORCH_PLATFORM_WINDOWS)
    message(STATUS "DTORCH_THIRD_PARTY_DLLS: ${DTORCH_THIRD_PARTY_DLLS}")
    foreach(DTORCH_THIRD_PARTY_DLL ${DTORCH_THIRD_PARTY_DLLS})
        add_custom_command(
            TARGET all_third_party
            POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${DTORCH_THIRD_PARTY_DLL}" "${DTORCH_DLL_DIR}"
        )
    endforeach()
endif()
message(STATUS "-------------------------------------------------------------------")

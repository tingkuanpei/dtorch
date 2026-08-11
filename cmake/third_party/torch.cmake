# Copyright 2026 The DTorch Authors. All rights reserved.
#
# Author: Tingkuan Pei(contact: peitingkuan@163.com)

message(STATUS "")
message(STATUS "------------------------ Find PyTorch -----------------------------")
# PyTorch cpp extension demo: https://github.com/pytorch/extension-cpp

find_package(Python3 COMPONENTS Interpreter Development)
find_package(PythonLibs 3 REQUIRED)
if(Python3_FOUND)
    set(Python_EXECUTABLE ${Python3_EXECUTABLE})
    set(Python_INCLUDE_DIRS ${Python3_INCLUDE_DIRS})
    set(Python_LIBRARIES ${Python3_LIBRARIES})
    message(STATUS "Python_EXECUTABLE: ${Python_EXECUTABLE}")
    message(STATUS "Python_INCLUDE_DIRS: ${Python_INCLUDE_DIRS}")
    message(STATUS "Python_LIBRARIES: ${Python_LIBRARIES}")
else()
    message(FATAL_ERROR
        "Can't find require python enviroment.\n"
        "If you machine have several python interpreter, use Python3_EXECUTABLE to specify python interpreter\n"
        "   ie: cmake -DPython3_EXECUTABLE=\"/usr/bin/python\"")
endif()

DGetTorchCmakePrefixPath(
    ${Python3_EXECUTABLE}
    TORCH_VERSION
    TORCH_CMAKE_PREFIX_PATH
    TORCH_SUPPORT_CUDA
    TORCH_CUDA_VERSION
)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
list(APPEND TORCH_INCLUDE_DIRS "${Python_INCLUDE_DIRS}")
if (DTORCH_PLATFORM_LINUX)
    list(APPEND TORCH_LIBRARIES "${PYTHON_LIBRARIES}")

    # torch_python
    find_library(TORCH_PYTHON_LIBARARY NAMES torch_python PATHS "${TORCH_CMAKE_PREFIX_PATH}/../../lib")
    if(TORCH_PYTHON_LIBARARY)
        list(APPEND TORCH_LIBRARIES "${TORCH_PYTHON_LIBARARY}")
    endif()

    if (TORCH_SUPPORT_CUDA)
        # cudart and cupti
        list(APPEND TORCH_LIBRARIES "${TORCH_CMAKE_PREFIX_PATH}/../../../nvidia/cuda_runtime/lib/libcudart.so.12")
        list(APPEND TORCH_LIBRARIES "${TORCH_CMAKE_PREFIX_PATH}/../../../nvidia/cuda_cupti/lib/libcupti.so.12")
    endif()
endif()

# Set DTORCH_INTEL_MAXOS_TORCH_2_2_2
if(TORCH_VERSION STREQUAL "2.2.2" AND DTORCH_PLATFORM_MACOS AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(DTORCH_INTEL_MAXOS_TORCH_2_2_2 ON)
else()
    set(DTORCH_INTEL_MAXOS_TORCH_2_2_2 OFF)
endif()

message(STATUS "")
message(STATUS "------------------------- Torch options ---------------------------")
message(STATUS "TORCH_FOUND:                                    ${TORCH_FOUND}")
message(STATUS "TORCH_VERSION:                                  ${TORCH_VERSION}")
message(STATUS "TORCH_CMAKE_PREFIX_PATH:                        ${TORCH_CMAKE_PREFIX_PATH}")
message(STATUS "TORCH_SUPPORT_CUDA:                             ${TORCH_SUPPORT_CUDA}")
message(STATUS "TORCH_CUDA_VERSION:                             ${TORCH_CUDA_VERSION}")
message(STATUS "TORCH_CXX_FLAGS:                                ${TORCH_CXX_FLAGS}")
message(STATUS "TORCH_INCLUDE_DIRS:                             ${TORCH_INCLUDE_DIRS}")
message(STATUS "TORCH_LIBRARIES:                                ${TORCH_LIBRARIES}")
message(STATUS "-------------------------------------------------------------------")
message(STATUS "")

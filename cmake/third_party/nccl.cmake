# Copyright 2026 The DTorch Authors. All rights reserved.
#
# Author: Tingkuan Pei(contact: peitingkuan@163.com)

# NCCL_ROOT
get_filename_component(NCCL_ROOT "${TORCH_CMAKE_PREFIX_PATH}/../../../nvidia/nccl" ABSOLUTE)

# NCCL_INCLUDE_DIR
find_path(NCCL_INCLUDE_DIR nccl.h
    PATHS ${NCCL_ROOT}
    PATH_SUFFIXES include
    NO_DEFAULT_PATH
)
if(NOT NCCL_INCLUDE_DIR)
    message(STATUS "Could NOT find nccl.h in ${NCCL_ROOT}")
endif()

# NCCL_LIBNAME
set(NCCL_LIBNAME "libnccl.so.2")
find_library(NCCL_LIBRARY ${NCCL_LIBNAME}
    PATHS ${NCCL_ROOT}
    PATH_SUFFIXES lib
    NO_DEFAULT_PATH
)
if(NOT NCCL_LIBRARY)
    message(STATUS "Could NOT find nccl library in ${NCCL_ROOT}")
endif()

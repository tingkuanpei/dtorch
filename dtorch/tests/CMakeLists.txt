# Copyright 2026 The DTorch Authors. All rights reserved.
#
# Author: Tingkuan Pei(contact: peitingkuan@163.com)

#----------------------------------------------------------------------------------------
# Build DTorch unit test executor
set(DTORCH_TEST_INCLUDE_DIRS     "")
set(DTORCH_TEST_LINK_LIBRARIES   "")

file(GLOB_RECURSE tmp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
)
list(APPEND DTORCH_TEST_CXX_SRCS           ${tmp})

if(DTORCH_WITH_CUDA)
    file(GLOB_RECURSE tmp ${CMAKE_CURRENT_SOURCE_DIR}/*.cu ${CMAKE_CURRENT_SOURCE_DIR}/*.cuh)
    list(APPEND DTORCH_TEST_CUDA_SRCS ${tmp})
endif()

if(NOT DTORCH_WITH_CUDA)
    file(GLOB_RECURSE tmp
        ${CMAKE_CURRENT_SOURCE_DIR}/test_nccl.cc
    )
    list(REMOVE_ITEM DTORCH_TEST_CXX_SRCS ${tmp})
endif()

if(NOT DTORCH_WITH_TRITON)
    file(GLOB_RECURSE tmp
        ${CMAKE_CURRENT_SOURCE_DIR}/*triton_kernel.cc
    )
    list(REMOVE_ITEM DTORCH_TEST_CXX_SRCS ${tmp})
endif()

DGroupSourceFiles(DTORCH_TEST_CXX_SRCS)
DGroupSourceFiles(DTORCH_TEST_CUDA_SRCS)

list(APPEND DTORCH_TEST_INCLUDE_DIRS       "${DTORCH_INCLUDE_DIRS}")
list(APPEND DTORCH_TEST_INCLUDE_DIRS       "${GOOGLETEST_INCLUDE_DIR}")
list(APPEND DTORCH_TEST_LINK_LIBRARIES     "${DTORCH_LIBRARIES}")
list(APPEND DTORCH_TEST_LINK_LIBRARIES     "${GOOGLETEST_LIBRARY}")

dtorch_add_executable(test ${DTORCH_TEST_CXX_SRCS} ${DTORCH_TEST_CUDA_SRCS})
target_include_directories(test PRIVATE ${DTORCH_TEST_INCLUDE_DIRS})
target_link_libraries(test ${DTORCH_TEST_LINK_LIBRARIES})

# GCC 12+ at -O3: false positives in libstdc++ vector<bool> inlined from
# torch/csrc/autograd/custom_function.h (Function::apply). Tests use -Werror project-wide.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(test PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-array-bounds;-Wno-stringop-overflow>)
endif()

add_dependencies(test all_third_party)
add_dependencies(test dtorch_core)

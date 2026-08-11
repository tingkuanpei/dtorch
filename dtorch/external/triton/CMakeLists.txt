# Copyright 2026 The DTorch Authors. All rights reserved.
#
# Author: Tingkuan Pei(contact: peitingkuan@163.com)

set(DTORCH_TRITON_INCLUDE_DIRS    "")
set(DTORCH_TRITON_LINK_LIBRARIES  "")

message(STATUS "")
message(STATUS "------------------- DTorch triton aot compile  ---------------------")
message(STATUS "Generate...")

# Generate triton_aot_kernel.h and libtriton_aot_kernel.so
set(DTORCH_TRITON_COMPILE_DIR          ${CMAKE_BINARY_DIR}/triton_compile_dir)
execute_process(
    COMMAND  ${Python3_EXECUTABLE} "${DTORCH_ROOT_DIR}/dtorch/external/triton/triron_compile.py"
              "${DTORCH_ROOT_DIR}" "${DTORCH_TRITON_COMPILE_DIR}" "${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS}"
    OUTPUT_VARIABLE COMMAND_OUTPUT_VARIABLE
    RESULT_VARIABLE COMMAND_COMMAND_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT COMMAND_COMMAND_RESULT EQUAL 0)
    message(FATAL_ERROR "Run python command, Python3_EXECUTABLE: ${Python3_EXECUTABLE}, "
                        "error code: ${COMMAND_COMMAND_RESULT}, error message: ${COMMAND_OUTPUT_VARIABLE}")
endif()

message(STATUS "Generated triton_aot_kernel.h and libtriton_aot_kernel.so")

list(LENGTH COMMAND_OUTPUT_VARIABLE COMMAND_OUTPUT_VARIABLE_LENGTH)
if(NOT COMMAND_OUTPUT_VARIABLE_LENGTH EQUAL 2)
    message(FATAL_ERROR "Expect command output is string list with lenght == 2")
endif()

# DTORCH_TRITON_INCLUDE_DIRS
list(GET COMMAND_OUTPUT_VARIABLE 0 DTORCH_TRITON_INCLUDE_DIRS)
list(APPEND DTORCH_TRITON_INCLUDE_DIRS    "${DTORCH_TRITON_COMPILE_DIR}")
message(STATUS "DTORCH_TRITON_INCLUDE_DIRS:                     ${DTORCH_TRITON_INCLUDE_DIRS}")

# DTORCH_TRITON_LINK_LIBRARIES
list(GET COMMAND_OUTPUT_VARIABLE 1 DTORCH_TRITON_LINK_LIBRARIES)
set(DTORCH_TRITON_LINK_LIBRARIES "${DTORCH_TRITON_COMPILE_DIR}/libtriton_aot_kernel.so;${DTORCH_TRITON_LINK_LIBRARIES}")
message(STATUS "DTORCH_TRITON_LINK_LIBRARIES:                   ${DTORCH_TRITON_LINK_LIBRARIES}")

message(STATUS "-------------------------------------------------------------------")
message(STATUS "")

set(DTORCH_TRITON_INCLUDE_DIRS ${DTORCH_TRITON_INCLUDE_DIRS} PARENT_SCOPE)
set(DTORCH_TRITON_LINK_LIBRARIES ${DTORCH_TRITON_LINK_LIBRARIES} PARENT_SCOPE)

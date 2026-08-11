# Copyright 2026 The DTorch Authors. All rights reserved.
#
# Author: Tingkuan Pei(contact: peitingkuan@163.com)

#----------------------------------------------------------------------------------------
# Build DTorch python library
set(DTORCH_PY_LIB_CXX_SRCS     "")
set(DTORCH_PY_INCLUDE_DIRS     "")
set(DTORCH_PY_LINK_LIBRARIES   "")

file(GLOB_RECURSE tmp ${CMAKE_CURRENT_SOURCE_DIR}/*.h ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
list(APPEND DTORCH_PY_LIB_CXX_SRCS     ${tmp})
list(APPEND DTORCH_PY_INCLUDE_DIRS     "${DTORCH_INCLUDE_DIRS}")
list(APPEND DTORCH_PY_LINK_LIBRARIES   "${DTORCH_LIBRARIES}")

DGroupSourceFiles(DTORCH_PY_LIB_CXX_SRCS)

# Generate py_bind_functional_generated.h
execute_process(
    COMMAND  ${Python3_EXECUTABLE} "${DTORCH_ROOT_DIR}/dtorch/api/python/functional/generate_py_bind_functional.py"
              "${DTORCH_ROOT_DIR}" "${CMAKE_BINARY_DIR}"
    OUTPUT_VARIABLE COMMAND_OUTPUT_VARIABLE
    RESULT_VARIABLE COMMAND_COMMAND_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT COMMAND_COMMAND_RESULT EQUAL 0)
    message(FATAL_ERROR "Run python command, Python3_EXECUTABLE: ${Python3_EXECUTABLE}, "
                        "error code: ${COMMAND_COMMAND_RESULT}, error message: ${COMMAND_OUTPUT_VARIABLE}")
endif()

message(STATUS "Generate py_bind_functional_generated.h")

# Add target: _dtorch_py_api
nanobind_add_module(_dtorch_py_api ${DTORCH_PY_LIB_CXX_SRCS})
target_include_directories(_dtorch_py_api PRIVATE ${DTORCH_PY_INCLUDE_DIRS})
target_link_libraries(_dtorch_py_api PRIVATE ${DTORCH_PY_LINK_LIBRARIES})
install(TARGETS _dtorch_py_api DESTINATION dtorch)

add_dependencies(_dtorch_py_api dtorch_core)

# Copy dll to ${DTORCH_ROOT_DIR}/python/dtorch
foreach(DTORCH_THIRD_PARTY_DLL ${DTORCH_THIRD_PARTY_DLLS})
    add_custom_command(
        TARGET _dtorch_py_api
        POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${DTORCH_THIRD_PARTY_DLL}" "${DTORCH_ROOT_DIR}/python/dtorch"
    )
endforeach()

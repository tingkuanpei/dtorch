# Copyright 2026 The DTorch Authors. All rights reserved.
#
# Author: Tingkuan Pei(contact: peitingkuan@163.com)

#----------------------------------------------------------------------------------------
# Group source files according to their relative path
function(DGroupSourceFiles)
    foreach(_source IN ITEMS ${${ARGN}})
        if(IS_ABSOLUTE "${_source}")
            file(RELATIVE_PATH _source_rel "${CMAKE_CURRENT_SOURCE_DIR}" "${_source}")
        else()
            set(_source_rel "${_source}")
        endif()
        get_filename_component(_source_path "${_source_rel}" PATH)
        string(REPLACE "../" "" _source_path "${_source_path}")
        string(REPLACE "/" "\\" _source_path "${_source_path}")
        source_group("${_source_path}" FILES "${_source}")
    endforeach()
endfunction(DGroupSourceFiles)

#----------------------------------------------------------------------------------------
# Get the abbreviated commit hash of a specified git repository directory
macro(DGetAbbrevCommitHash resultStr gitRepoDir)
    execute_process(
        COMMAND git log -1 --format=%h --abbrev=8
        WORKING_DIRECTORY ${gitRepoDir}
        OUTPUT_VARIABLE FW_${resultStr}
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(${resultStr} ${FW_${resultStr}})
endmacro()

#----------------------------------------------------------------------------------------
# dtorch_add_executable & dtorch_add_library
function(dtorch_add_executable)
    add_executable(${ARGV})
    target_compile_options(${ARGV0} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${DTORCH_CXX_FLAGS}>)

    set_target_properties(${ARGV0} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${DTORCH_ROOT_DIR}")
    set_target_properties(${ARGV0} PROPERTIES VS_DEBUGGER_ENVIRONMENT "PATH=${DTORCH_DLL_DIR};%PATH%")
endfunction()

function(dtorch_add_library)
    add_library(${ARGV})
    target_compile_options(${ARGV0} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${DTORCH_CXX_FLAGS}>)
endfunction()

#----------------------------------------------------------------------------------------
# Get all cmake target
function(DGetAllTargets var)
    set(targets)
    DGetAllTargetsRecursive(targets ${CMAKE_CURRENT_SOURCE_DIR})
    set(${var} ${targets} PARENT_SCOPE)
endfunction()

macro(DGetAllTargetsRecursive targets dir)
    get_property(subdirectories DIRECTORY ${dir} PROPERTY SUBDIRECTORIES)
    foreach(subdir ${subdirectories})
        DGetAllTargetsRecursive(${targets} ${subdir})
    endforeach()

    get_property(current_targets DIRECTORY ${dir} PROPERTY BUILDSYSTEM_TARGETS)
    list(APPEND ${targets} ${current_targets})
endmacro()

#----------------------------------------------------------------------------------------
# Set default value
function(DSetValueIfEmpty var value)
    if(NOT DEFINED ${var} OR ${var} STREQUAL "")
        set(${var}      "${value}" PARENT_SCOPE)
        message(STATUS  "${var} is empty, set it as ${value} by default")
    endif()
endfunction()

#----------------------------------------------------------------------------------------
# Check for cargo support
function(DCheckSupportRustAndCargo)
    find_program(CARGO cargo)
    if(CARGO)
        message(STATUS "Cargo found: ${CARGO}")
    else()
        set(ERROR_MESSAGE "Rust and Cargo not found, follow this document to install Rust and Cargo: https://doc.rust-lang.org/cargo/getting-started/installation.html")
        message(FATAL_ERROR "${ERROR_MESSAGE}")
    endif()
endfunction()

#----------------------------------------------------------------------------------------
# DGetTorchCmakePrefixPath
function(DGetTorchCmakePrefixPath Python3_EXECUTABLE VERSION CMAKE_PREFIX_PATH SUPPORT_CUDA CUDA_VERSION)
    execute_process(
        COMMAND  ${Python3_EXECUTABLE} -c "
import torch
print(torch.__version__)
print(torch.utils.cmake_prefix_path)
if torch.cuda.is_available():
    print(True)
    print(torch.version.cuda)
else:
    print(False)
    print(None)"
        OUTPUT_VARIABLE COMMAND_OUTPUT_VARIABLE
        RESULT_VARIABLE COMMAND_COMMAND_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(NOT COMMAND_COMMAND_RESULT EQUAL 0)
        message(FATAL_ERROR "Run python command, Python3_EXECUTABLE: ${Python3_EXECUTABLE}, "
                            "error code: ${COMMAND_COMMAND_RESULT}, error message: ${COMMAND_OUTPUT_VARIABLE}")
    endif()

    string(REPLACE "\n" ";" TEMP_STRING_LIST "${COMMAND_OUTPUT_VARIABLE}")
    string(STRIP "${TEMP_STRING_LIST}" TEMP_STRING_LIST)
    list(LENGTH TEMP_STRING_LIST TEMP_STRING_LIST_LENGTH)
    if(TEMP_STRING_LIST_LENGTH EQUAL 4)
        list(GET TEMP_STRING_LIST 0 TMP_TORCH_VERSION)
        list(GET TEMP_STRING_LIST 1 TMP_TORCH_CMAKE_PREFIX_PATH)
        list(GET TEMP_STRING_LIST 2 TMP_TORCH_SUPPORT_CUDA)
        list(GET TEMP_STRING_LIST 3 TMP_TORCH_CUDA_VERSION)
    else()
        message(FATAL_ERROR "Parse python output error, COMMAND_OUTPUT_VARIABLE: ${COMMAND_OUTPUT_VARIABLE}")
    endif()

    if(${TMP_TORCH_SUPPORT_CUDA} STREQUAL "True")
        set(TMP_TORCH_SUPPORT_CUDA ON)
    else()
        set(TMP_TORCH_SUPPORT_CUDA OFF)
    endif()

    set(${VERSION} ${TMP_TORCH_VERSION} PARENT_SCOPE)
    set(${CMAKE_PREFIX_PATH} ${TMP_TORCH_CMAKE_PREFIX_PATH} PARENT_SCOPE)
    set(${SUPPORT_CUDA} ${TMP_TORCH_SUPPORT_CUDA} PARENT_SCOPE)
    set(${CUDA_VERSION} ${TMP_TORCH_CUDA_VERSION} PARENT_SCOPE)
endfunction()

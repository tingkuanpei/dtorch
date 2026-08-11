# Copyright 2026 The DTorch Authors. All rights reserved.
#
# Author: Tingkuan Pei(contact: peitingkuan@163.com)

#----------------------------------------------------------------------------------------
# Build DTorch tool executables (one per .cc file)
set(DTORCH_TOOL_INCLUDE_DIRS     "")
set(DTORCH_TOOL_LINK_LIBRARIES   "")

file(GLOB DTORCH_TOOL_CXX_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.h ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

if(DTORCH_WITH_CUDA)
    file(GLOB DTORCH_TOOL_CUDA_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cu ${CMAKE_CURRENT_SOURCE_DIR}/*.cuh)
endif()

DGroupSourceFiles(DTORCH_TOOL_CXX_SRCS)
DGroupSourceFiles(DTORCH_TOOL_CUDA_SRCS)

list(APPEND DTORCH_TOOL_INCLUDE_DIRS       "${DTORCH_INCLUDE_DIRS}")
list(APPEND DTORCH_TOOL_LINK_LIBRARIES     "${DTORCH_LIBRARIES}")

file(GLOB TOOL_MAIN_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
foreach(TOOL_MAIN ${TOOL_MAIN_FILES})
    get_filename_component(TOOL_NAME ${TOOL_MAIN} NAME_WE)
    dtorch_add_executable(${TOOL_NAME} ${TOOL_MAIN} ${DTORCH_TOOL_CUDA_SRCS})
    target_include_directories(${TOOL_NAME} PRIVATE ${DTORCH_TOOL_INCLUDE_DIRS})
    target_link_libraries(${TOOL_NAME} ${DTORCH_TOOL_LINK_LIBRARIES})
    add_dependencies(${TOOL_NAME} all_third_party)
    add_dependencies(${TOOL_NAME} dtorch_core)
    install(TARGETS ${TOOL_NAME} RUNTIME DESTINATION dtorch/tools)
endforeach()

# Shell launcher
install(PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/dtorch_launcher.sh
        DESTINATION ${SKBUILD_SCRIPTS_DIR}
        RENAME dtorch_launcher)

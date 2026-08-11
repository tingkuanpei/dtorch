# Copyright 2026 The DTorch Authors. All rights reserved.
#
# Author: Tingkuan Pei(contact: peitingkuan@163.com)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
set(Python_EXECUTABLE ${Python_EXECUTABLE})

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
)
find_package(nanobind CONFIG REQUIRED)
set(NANOBIND_INCLUDE_DIR "${nanobind_ROOT}/../include")
set(NANOBIND_LIBRARYS nanobind-static)

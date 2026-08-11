# Copyright 2026 The DTorch Authors. All rights reserved.
#
# Author: Tingkuan Pei(contact: peitingkuan@163.com)

# https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html
# You can set CUDAToolkit_ROOT to cuda path
find_package(CUDAToolkit REQUIRED)

if(NOT CUDAToolkit_VERSION)
    message(FATAL_ERROR "CUDAToolkit_VERSION empty")
endif()

# CMAKE_CUDA_ARCHITECTURES
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "8.0")
        list(APPEND CMAKE_CUDA_ARCHITECTURES 62)
    endif()

    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "9.0")
        list(APPEND CMAKE_CUDA_ARCHITECTURES 70)
    endif()

    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "10.0")
        list(APPEND CMAKE_CUDA_ARCHITECTURES 75)
    endif()

    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.0")
        list(APPEND CMAKE_CUDA_ARCHITECTURES 80)
    endif()

    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.1")
        list(APPEND CMAKE_CUDA_ARCHITECTURES 86)
    endif()

    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.0")
        list(APPEND CMAKE_CUDA_ARCHITECTURES 90)
    endif()
endif()

set(CUDA_LIBRARY        "")
list(APPEND CUDA_LIBRARY CUDA::cuda_driver)
list(APPEND CUDA_LIBRARY CUDA::cudart_static)
list(APPEND CUDA_LIBRARY CUDA::cublas)
list(APPEND CUDA_LIBRARY CUDA::curand)

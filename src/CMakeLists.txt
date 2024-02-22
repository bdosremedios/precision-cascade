cmake_minimum_required(VERSION 3.28)

project(precision-cascade)

find_package(CUDAToolkit)
enable_language(CUDA)

set(CMAKE_EXE_LINKER_FLAGS, "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")

set (CMAKE_CXX_STANDARD 20)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

set (CMAKE_CUDA_STANDARD 20)
set (CMAKE_CUDA_STANDARD_REQUIRED ON)

add_library(
    precision-cascade-gpu_kernels
    STATIC
    types/Scalar/Scalar_kernels.cu
)
set_property(TARGET precision-cascade-gpu_kernels PROPERTY CUDA_ARCHITECTURES 80)

add_library(
    precision-cascade-constants
    STATIC
    types/Scalar/Scalar_constants.cpp
)
target_link_libraries(
    precision-cascade-constants
    PUBLIC
    precision-cascade-headers
)
set_property(TARGET precision-cascade-constants PROPERTY CUDA_ARCHITECTURES 80)

add_library(
    precision-cascade
    STATIC
    tools/cuda_check.cu
    types/Scalar/Scalar_hlf.cu
    types/Scalar/Scalar_sgl.cu
    types/Scalar/Scalar_dbl.cu
    types/Vector/Vector_hlf.cu
    types/Vector/Vector_sgl.cu
    types/Vector/Vector_dbl.cu
    # types/MatrixDense/MatrixDense_hlf.cu
    # types/MatrixDense/MatrixDense_sgl.cu
    # types/MatrixDense/MatrixDense_dbl.cu
)

set_property(TARGET precision-cascade PROPERTY CUDA_ARCHITECTURES 80)

target_link_libraries(
    precision-cascade
    PUBLIC
    precision-cascade-headers
    "$ENV{CUDA_PATH}/lib/x64/cublas.lib"
)
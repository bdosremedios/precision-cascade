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
    types/Scalar/Scalar_gpu_kernels.cu
    types/Vector/Vector_gpu_kernels.cu
    types/GeneralMatrix/GeneralMatrix_gpu_kernels.cu
    types/MatrixDense/MatrixDense_gpu_kernels.cu
)

target_link_options(precision-cascade-gpu_kernels PUBLIC "/NODEFAULTLIB:LIBCMT")

target_link_libraries(
    precision-cascade-gpu_kernels
    PUBLIC
    precision-cascade-headers
    "$ENV{CUDA_PATH}/lib/x64/cublas.lib"
)

set_property(TARGET precision-cascade-gpu_kernels PROPERTY CUDA_ARCHITECTURES 80)

add_library(
    precision-cascade-constants
    STATIC
    types/Scalar/Scalar_constants.cpp
)

target_link_options(precision-cascade-constants PUBLIC "/NODEFAULTLIB:LIBCMT")

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
    tools/cuHandleBundle.cu
    types/Scalar/Scalar_hlf.cu
    types/Scalar/Scalar_sgl.cu
    types/Scalar/Scalar_dbl.cu
    types/Vector/Vector_hlf.cu
    types/Vector/Vector_sgl.cu
    types/Vector/Vector_dbl.cu
    types/MatrixDense/MatrixDense_hlf.cu
    types/MatrixDense/MatrixDense_sgl.cu
    types/MatrixDense/MatrixDense_dbl.cu
    types/MatrixSparse/NoFillMatrixSparse_hlf.cu
    types/MatrixSparse/NoFillMatrixSparse_sgl.cu
    types/MatrixSparse/NoFillMatrixSparse_dbl.cu
)

target_link_options(precision-cascade PUBLIC "/NODEFAULTLIB:LIBCMT")

target_link_libraries(
    precision-cascade
    PUBLIC
    precision-cascade-headers
    precision-cascade-gpu_kernels
    precision-cascade-constants
    "$ENV{CUDA_PATH}/lib/x64/cublas.lib"
)

set_property(TARGET precision-cascade PROPERTY CUDA_ARCHITECTURES 80)
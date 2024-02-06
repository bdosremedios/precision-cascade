cmake_minimum_required(VERSION 3.27.2)

project(precision-cascade LANGUAGES CXX CUDA)

set(CMAKE_EXE_LINKER_FLAGS, "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")

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
    types/MatrixDense/MatrixDense_hlf.cu
    types/MatrixDense/MatrixDense_sgl.cu
    types/MatrixDense/MatrixDense_dbl.cu
)

target_link_libraries(
    precision-cascade
    PUBLIC
    precision-cascade-headers
    "$ENV{CUDA_PATH}/lib/x64/cublas.lib"
)
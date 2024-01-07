cmake_minimum_required(VERSION 3.27.2)

project(precision-cascade LANGUAGES CXX CUDA)

set(CMAKE_EXE_LINKER_FLAGS, "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT /NODEFAULTLIB:MSVCRT")

add_subdirectory(tools)
add_subdirectory(types)

add_library(
    types
    STATIC
    tools/cuda_check.cu
    types/MatrixVector/MatrixVector_hlf.cu
    types/MatrixVector/MatrixVector_sgl.cu
    types/MatrixVector/MatrixVector_dbl.cu
)

target_link_libraries(
    types
    PUBLIC
    precision-cascade
    eigen
    "$ENV{CUDA_PATH}/lib/x64/cublas.lib"
)
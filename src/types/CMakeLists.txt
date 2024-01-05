cmake_minimum_required(VERSION 3.27.2)

project(precision-cascade LANGUAGES CXX CUDA)

add_library(types_lib STATIC MatrixVector.cu)

target_link_libraries(
    types_lib
    PUBLIC
    precision-cascade
    eigen
    "$ENV{CUDA_PATH}/lib/x64/cublas.lib"
)
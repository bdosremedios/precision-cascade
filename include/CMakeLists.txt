cmake_minimum_required(VERSION 3.28)

project(precision-cascade LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

set (CMAKE_CUDA_STANDARD 20)
set (CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXE_LINKER_FLAGS, "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")

add_library(precision-cascade-headers INTERFACE)

set_property(TARGET precision-cascade-headers PROPERTY CUDA_ARCHITECTURES 80)

target_include_directories(
    precision-cascade-headers
    INTERFACE
    .
)

target_link_libraries(
    precision-cascade-headers
    INTERFACE
    precision-cascade-gpu_kernels
    precision-cascade-constants
)
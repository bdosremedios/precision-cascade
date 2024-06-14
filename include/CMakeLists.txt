cmake_minimum_required(VERSION 3.27)

project(precision-cascade LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_library(precision-cascade-headers INTERFACE)

if(WIN32)
    target_link_options(precision-cascade-constants PUBLIC "/NODEFAULTLIB:LIBCMT")
endif()

set_property(
    TARGET precision-cascade-headers
    PROPERTY CUDA_ARCHITECTURES
    ${PROJECT_CUDA_ARCH_REAL} ${PROJECT_CUDA_ARCH_VRTL}
)

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
cmake_minimum_required(VERSION 3.27)

project(precision-cascade LANGUAGES CXX CUDA)

add_library(precision-cascade-headers INTERFACE)

if(WIN32)
    target_link_options(
        precision-cascade-headers
        INTERFACE
        "/NODEFAULTLIB:LIBCMT"
    )
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
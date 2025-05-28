SET(CMAKE_VERSION 3.22)

cmake_minimum_required(VERSION ${CMAKE_VERSION})

project(precision-cascade LANGUAGES CXX CUDA)

set(CMAKE_EXE_LINKER_FLAGS, "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")

find_package(CUDAToolkit REQUIRED)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CUDA_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CUDA_STANDARD_REQUIRED ON)

set (CURRENT_GPU_ARCHITECTURE 70)

set (CMAKE_CUDA_ARCHITECTURES ${CURRENT_GPU_ARCHITECTURE})

set (PROJECT_CUDA_ARCH_REAL ${CURRENT_GPU_ARCHITECTURE}-real)
set (PROJECT_CUDA_ARCH_VRTL ${CURRENT_GPU_ARCHITECTURE}-virtual)

add_subdirectory(lib)
add_subdirectory(include)
add_subdirectory(src)

option(
    BUILD_TEST
    "Whether to build test"
    ON
)

option(
    BUILD_BENCHMARK
    "Whether to build benchmark"
    ON
)

option(
    BUILD_EXPERIMENTATION
    "Whether to build experiment"
    ON
)

if(BUILD_TEST)
    message(STATUS "Building test...")
    add_subdirectory(test)
endif()

if(BUILD_BENCHMARK)
    message(STATUS "Building benchmark...")
    add_subdirectory(benchmark)
endif()

if(BUILD_EXPERIMENTATION)
    message(STATUS "Building experiment...")
    add_subdirectory(experimentation)
endif()
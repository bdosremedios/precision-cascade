cmake_minimum_required(VERSION 3.27.2)

project(precision-cascade LANGUAGES CXX CUDA)

set(CMAKE_EXE_LINKER_FLAGS, "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")

set (CMAKE_CXX_STANDARD 20)
set (CMAKE_CUDA_STANDARD 20)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CUDA_STANDARD_REQUIRED ON)

add_subdirectory(lib)
add_subdirectory(include)
add_subdirectory(src)

option(
    BUILD_MAIN
    "Whether to build main"
    ON
)

option(
    BUILD_TEST
    "Whether to build test"
    ON
)

option(
    BUILD_EXPERIMENTATION
    "Whether to build experiment"
    ON
)

if(BUILD_MAIN)
    message(STATUS "Building main...")
    add_subdirectory(main)
endif()

if(BUILD_TEST)
    message(STATUS "Building test...")
    add_subdirectory(test)
endif()

if(BUILD_EXPERIMENTATION)
    message(STATUS "Building experiment...")
    add_subdirectory(experimentation)
endif()
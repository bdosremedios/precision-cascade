cmake_minimum_required(VERSION 3.22)

project(precision-cascade)

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
    "Whether to build test"
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
    message(STATUS "Building experimentation...")
    add_subdirectory(experimentation)
endif()
cmake_minimum_required(VERSION 3.27.2)

project(precision-cascade LANGUAGES CXX CUDA)

set(CMAKE_EXE_LINKER_FLAGS, "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT /NODEFAULTLIB:MSVCRT")

add_library(precision-cascade INTERFACE)

target_include_directories(precision-cascade INTERFACE .)

target_link_libraries(
    precision-cascade
    INTERFACE
    types
)
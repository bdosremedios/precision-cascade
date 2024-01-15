cmake_minimum_required(VERSION 3.27.2)

set(CMAKE_EXE_LINKER_FLAGS, "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")

project(precision-cascade LANGUAGES CXX CUDA)

# add_subdirectory(eigen)
add_subdirectory(googletest)

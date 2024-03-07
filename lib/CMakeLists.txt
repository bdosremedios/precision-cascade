cmake_minimum_required(VERSION 3.28)

set(CMAKE_EXE_LINKER_FLAGS, "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")

project(precision-cascade LANGUAGES CXX CUDA)

add_compile_definitions(_SILENCE_STDEXT_ARR_ITERS_DEPRECATION_WARNING=1)

# add_subdirectory(eigen)
add_subdirectory(json)
add_subdirectory(googletest)
add_subdirectory(spdlog)
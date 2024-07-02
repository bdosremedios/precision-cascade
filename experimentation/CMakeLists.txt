cmake_minimum_required(VERSION ${CMAKE_VERSION})

project(precision-cascade LANGUAGES CXX CUDA)

add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(main)
add_subdirectory(test)
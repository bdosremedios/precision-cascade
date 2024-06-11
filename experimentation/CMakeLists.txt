cmake_minimum_required(VERSION 3.27)

project(precision-cascade LANGUAGES CXX CUDA)

add_subdirectory(src)
add_subdirectory(main)
add_subdirectory(test)
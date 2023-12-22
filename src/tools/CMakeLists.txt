cmake_minimum_required(VERSION 3.22)

project(precision-cascade)

add_library(math_functions math_functions.cpp)
target_link_libraries(math_functions precision-cascade)
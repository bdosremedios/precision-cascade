cmake_minimum_required(VERSION 3.22)

project(precision-cascade)

add_library(precision-cascade INTERFACE)

target_link_libraries(precision-cascade INTERFACE ArgPkg)

target_include_directories(precision-cascade INTERFACE .)
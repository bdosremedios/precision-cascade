cmake_minimum_required(VERSION 3.27.2)

project(precision-cascade LANGUAGES CXX CUDA)

add_library(precision-cascade INTERFACE)

target_include_directories(precision-cascade INTERFACE .)
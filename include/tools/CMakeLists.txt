cmake_minimum_required(VERSION 3.22)

project(gmres)

add_library(ArgPkg ArgPkg.cpp)

target_link_libraries(
    ArgPkg
    PRIVATE
    eigen
)
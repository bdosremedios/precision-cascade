cmake_minimum_required(VERSION 3.22)

project(gmres)

add_library(matrixReader MatrixReader.cpp)

target_link_libraries(matrixReader eigen)
target_include_directories(
    matrixReader
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../include/"
)
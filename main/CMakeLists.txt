cmake_minimum_required(VERSION 3.22)

project(precision-cascade)

add_executable(main main.cpp)

target_link_libraries(main eigen)

target_link_libraries(main precision-cascade)
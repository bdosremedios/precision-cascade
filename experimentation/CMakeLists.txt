cmake_minimum_required(VERSION 3.22)

project(precision-cascade)

add_executable(experiment experiment.cpp)

target_link_libraries(
    experiment
    PUBLIC
    eigen
    precision-cascade
)
cmake_minimum_required(VERSION 3.28)

project(precision-cascade LANGUAGES CUDA CXX)

add_library(experiment_tools experiment_tools.cpp)

target_link_libraries(
    experiment_tools
    PUBLIC
    precision-cascade
)

add_executable(experiment experiment.cpp)

target_link_libraries(
    experiment
    PUBLIC
    precision-cascade
    experiment_tools
)
cmake_minimum_required(VERSION 3.28)

project(precision-cascade LANGUAGES CUDA CXX)

add_library(experiment_read experiment_read.cpp)

target_link_libraries(
    experiment_read
    PUBLIC
    precision-cascade
    nlohmann_json::nlohmann_json
)

add_library(experiment_record experiment_record.cpp)

target_link_libraries(
    experiment_record
    PUBLIC
    precision-cascade
)

add_executable(experiment experiment.cpp)

target_link_libraries(
    experiment
    PUBLIC
    precision-cascade
    experiment_read
    experiment_record
)
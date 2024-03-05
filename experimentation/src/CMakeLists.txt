cmake_minimum_required(VERSION 3.28)

project(precision-cascade LANGUAGES CUDA CXX)

add_library(experiment_read experiment_read.cpp)

target_link_libraries(
    experiment_read
    PUBLIC
    precision-cascade
    nlohmann_json::nlohmann_json
)

target_include_directories(
    experiment_read
    PUBLIC
    ../include
)

add_library(experiment_record experiment_record.cpp)

target_link_libraries(
    experiment_record
    PUBLIC
    precision-cascade
)

target_include_directories(
    experiment_record
    PUBLIC
    ../include
)
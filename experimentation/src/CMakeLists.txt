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

add_compile_definitions(_SILENCE_STDEXT_ARR_ITERS_DEPRECATION_WARNING=1)

add_library(experiment_log experiment_log.cpp)

target_link_libraries(
    experiment_log
    PUBLIC
    spdlog
)

message(${CMAKE_SOURCE_DIR})

target_include_directories(
    experiment_log
    PUBLIC
    ../include
)
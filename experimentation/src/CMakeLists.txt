cmake_minimum_required(VERSION ${CMAKE_VERSION})

project(precision-cascade LANGUAGES CXX CUDA)

add_compile_definitions(_SILENCE_STDEXT_ARR_ITERS_DEPRECATION_WARNING=1)

set(
    EXPERIMENT_CPPS
    experiment_log.cpp
    experiment_tools.cpp
    experiment_read.cpp
    experiment_recorders.cpp
    experiment_run_record.cpp
)

add_library(
    experiment_lib
    ${EXPERIMENT_CPPS}
)

target_link_libraries(
    experiment_lib
    PUBLIC
    spdlog
    nlohmann_json::nlohmann_json
    precision-cascade
)

target_include_directories(
    experiment_lib
    PUBLIC
    ../include
)
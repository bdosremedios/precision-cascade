cmake_minimum_required(VERSION ${CMAKE_VERSION})

project(precision-cascade LANGUAGES CXX CUDA)

add_library(experiment-headers INTERFACE)

target_include_directories(
    experiment-headers
    INTERFACE
    .
)

target_link_libraries(
    experiment-headers
    INTERFACE
    spdlog
    nlohmann_json::nlohmann_json
    precision-cascade
)
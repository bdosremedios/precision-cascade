cmake_minimum_required(VERSION 3.28)

project(precision-cascade LANGUAGES CUDA CXX)

add_compile_definitions(_SILENCE_STDEXT_ARR_ITERS_DEPRECATION_WARNING=1)

set(
    EXPERIMENT_CPPS
    experiment_log.cpp
    experiment_tools.cpp
    experiment_read.cpp
    experiment_record.cpp
    experiment_run.cpp
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

# add_library(experiment_log experiment_log.cpp)

# target_link_libraries(
#     experiment_log
#     PUBLIC
#     spdlog
# )

# target_include_directories(
#     experiment_log
#     PUBLIC
# )

# add_library(experiment_tools experiment_tools.cpp)

# target_link_libraries(
#     experiment_tools
#     PUBLIC
#     precision-cascade
#     experiment_log
# )

# target_include_directories(
#     experiment_tools
#     PUBLIC
#     ../include
# )

# add_library(experiment_read experiment_read.cpp)

# target_link_libraries(
#     experiment_read
#     PUBLIC
#     nlohmann_json::nlohmann_json
#     precision-cascade
#     experiment_tools
# )

# target_include_directories(
#     experiment_read
#     PUBLIC
#     ../include
# )

# add_library(experiment_record experiment_record.cpp)

# target_link_libraries(
#     experiment_record
#     PUBLIC
#     precision-cascade
#     experiment_tools
# )

# target_include_directories(
#     experiment_record
#     PUBLIC
#     ../include
# )

# add_library(experiment_run experiment_run.cpp)

# target_link_libraries(
#     experiment_run
#     PUBLIC
#     precision-cascade
#     experiment_tools
#     experiment_record
#     experiment_read
# )

# target_include_directories(
#     experiment_run
#     PUBLIC
#     ../include
# )
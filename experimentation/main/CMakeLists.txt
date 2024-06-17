cmake_minimum_required(VERSION ${CMAKE_VERSION})

project(precision-cascade LANGUAGES CUDA CXX)

add_executable(experiment experiment.cpp)

add_compile_definitions(_SILENCE_STDEXT_ARR_ITERS_DEPRECATION_WARNING=1)

target_link_libraries(
    experiment
    PUBLIC
    experiment_lib
    precision-cascade
)
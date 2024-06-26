cmake_minimum_required(VERSION ${CMAKE_VERSION})

project(precision-cascade LANGUAGES CXX CUDA)

add_compile_definitions(_SILENCE_STDEXT_ARR_ITERS_DEPRECATION_WARNING=1)

add_library(
    experimentation_test_headers
    INTERFACE
)

target_include_directories(
    experimentation_test_headers
    INTERFACE
    include
)

set(
    EXPERIMENTATION_TEST_SUITES
    src/test_read.cpp
    src/test_recorders.cpp
    src/test_run_record.cpp
    src/test_tools.cpp
)

add_executable(
    test_experiment
    src/test_experiment.cpp
    ${EXPERIMENTATION_TEST_SUITES}
)

target_link_libraries(
    test_experiment
    PUBLIC
    experiment_lib
    experimentation_test_headers
    test_headers
    precision-cascade
    gtest_main
)
cmake_minimum_required(VERSION 3.28)

project(precision-cascade LANGUAGES CXX CUDA)

add_compile_definitions(_SILENCE_STDEXT_ARR_ITERS_DEPRECATION_WARNING=1)

set(
    TEST_SUITES
    test_read.cpp
    test_run_record.cpp
)

add_executable(
    test_experiment
    test_experiment.cpp
    ${TEST_SUITES}
)

target_link_libraries(
    test_experiment
    PUBLIC
    test_headers
    experiment_lib
    precision-cascade
    gtest_main
)
cmake_minimum_required(VERSION 3.28)

project(precision-cascade LANGUAGES CXX CUDA)

set(
    TEST_SUITES
    test_read.cpp
)

add_executable(
    test_experiment
    test_experiment.cpp
    ${TEST_SUITES}
)

target_link_libraries(
    test_experiment
    PUBLIC
    experiment_read
    experiment_record
    precision-cascade
    test_headers
    gtest_main
)
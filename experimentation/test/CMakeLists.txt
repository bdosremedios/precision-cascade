cmake_minimum_required(VERSION ${CMAKE_VERSION})

project(precision-cascade LANGUAGES CXX CUDA)

add_compile_definitions(_SILENCE_STDEXT_ARR_ITERS_DEPRECATION_WARNING=1)

set(
    EXPERIMENTATION_TEST_SUITES
    src/test_exp_data.cpp
    src/test_exp_read.cpp
    src/test_exp_run.cpp
    src/test_exp_spec.cpp
)

add_executable(
    test_experiment
    main/test_experiment.cpp
    ${EXPERIMENTATION_TEST_SUITES}
)

target_link_libraries(
    test_experiment
    PUBLIC
    experiment-lib
    test_headers
    precision-cascade
    gtest_main
)

target_include_directories(
    test_experiment
    PUBLIC
    include
)
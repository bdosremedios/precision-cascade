cmake_minimum_required(VERSION 3.22)

project(gmres)

set(
    TEST_SUITE_CPPS
    test_MatrixReader.cpp
    test_Jacobi.cpp
    test_GMRES.cpp
)

add_executable(
    test
    test.cpp
    ${TEST_SUITE_CPPS}
)

target_include_directories(
    test
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"
)

target_link_libraries(
    test
    PRIVATE
    gtest_main matrixReader
)
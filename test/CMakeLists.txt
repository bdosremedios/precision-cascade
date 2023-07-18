cmake_minimum_required(VERSION 3.22)

project(gmres)

set(
    TEST_SUITE_CPPS
    test_MatrixReader.cpp
    test_Jacobi.cpp
)

set(
    TEST_GMRES_CPPS
    test_GMRES/test_GMRESComponents.cpp
    test_GMRES/test_GMRESDouble.cpp
    test_GMRES/test_GMRESSingle.cpp
    test_GMRES/test_GMRESHalf.cpp
)

add_executable(
    test
    test.cpp
    ${TEST_SUITE_CPPS}
    ${TEST_GMRES_CPPS}
)


target_include_directories(
    test
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"
)

target_link_libraries(
    test
    PRIVATE
    gtest_main eigen
)
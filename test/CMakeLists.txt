cmake_minimum_required(VERSION 3.22)

project(gmres)

set(
    TEST_TOOLS_CPPS
    test_Substitution.cpp
)

set(
    TEST_PRECONDITIONER_CPPS
    test_Preconditioner/test_Preconditioner.cpp
    test_Preconditioner/test_ILU.cpp
)

set(
    TEST_ITERATIVESOLVE_CPPS
    test_IterativeSolve/test_IterativeSolve.cpp
)

set(
    TEST_SOR_CPPS
    test_SOR/test_GaussSeidel.cpp
    test_SOR/test_SOR.cpp
)

set(
    TEST_GMRES_CPPS
    test_GMRES/test_GMRES_components.cpp
    test_GMRES/test_GMRES_solve_dbl.cpp
    test_GMRES/test_GMRES_solve_sgl.cpp
    test_GMRES/test_GMRES_solve_hlf.cpp
    test_GMRES/test_PGMRES.cpp
)

set(
    TEST_SUITE_CPPS
    test_MatrixReader.cpp
    ${TEST_TOOLS_CPPS}
    ${TEST_ITERATIVESOLVE_CPPS}
    test_Jacobi.cpp
    ${TEST_SOR_CPPS}
    ${TEST_GMRES_CPPS}
    ${TEST_PRECONDITIONER_CPPS}
    test_FP_GMRES_IR_Solve.cpp
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
    gtest_main eigen
)
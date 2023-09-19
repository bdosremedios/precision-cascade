cmake_minimum_required(VERSION 3.22)

project(precision-cascade)

set(
    TEST_TOOLS_CPPS
    test_tools/test_MatrixReader.cpp
    test_tools/test_SolveArgPkg.cpp
    # test_tools/test_PrecondArgPkg.cpp
    # test_tools/test_Substitution.cpp
)

set(
    TEST_PRECONDITIONER_CPPS
    # test_preconditioners/test_Preconditioner.cpp
    # test_preconditioners/test_ILU.cpp
)

set(
    TEST_SOLVERS

    # test_solvers/test_IterativeSolve/test_IterativeSolve.cpp

    # test_solvers/test_stationary/test_Jacobi.cpp
    # test_solvers/test_stationary/test_SOR/test_GaussSeidel.cpp
    # test_solvers/test_stationary/test_SOR/test_SOR.cpp

    # test_solvers/test_krylov/test_GMRES/test_GMRES_components.cpp
    # test_solvers/test_krylov/test_GMRES/test_GMRES_solve_dbl.cpp
    # test_solvers/test_krylov/test_GMRES/test_GMRES_solve_sgl.cpp
    # test_solvers/test_krylov/test_GMRES/test_GMRES_solve_hlf.cpp
    # test_solvers/test_krylov/test_GMRES/test_PGMRES.cpp

    # test_solvers/test_nested/test_FP_GMRES_IR_Solve.cpp
    # test_solvers/test_nested/test_MP_GMRES_IR_Solve.cpp
)

set(
    TEST_SUITE_CPPS
    ${TEST_TOOLS_CPPS}
    # ${TEST_PRECONDITIONER_CPPS}
    ${TEST_SOLVERS}
)

add_executable(
    test
    test.cpp
    ${TEST_SUITE_CPPS}
)

target_link_libraries(
    test
    PRIVATE
    gtest_main
    eigen
    precision-cascade
)
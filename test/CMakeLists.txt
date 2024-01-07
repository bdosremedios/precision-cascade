cmake_minimum_required(VERSION 3.27.2)

project(precision-cascade LANGUAGES CXX CUDA)

set(
    TEST_TYPES_CPPS
    test_types/test_MatrixVector.cu
    test_types/test_MatrixDense.cu
    # test_types/test_MatrixSparse.cpp
    # test_types/test_TypeConversion.cpp
)

set(
    TEST_TOOLS_CPPS
    test_tools/test_GenericLinearSystem.cpp
    test_tools/test_TypedLinearSystem.cpp
    test_tools/test_SolveArgPkg.cpp
    test_tools/test_PrecondArgPkg.cpp
    test_tools/test_Substitution.cpp
    test_tools/test_VectorSort.cpp
)

set(
    TEST_PRECONDITIONER_CPPS
    test_preconditioners/test_Preconditioner.cpp
    test_preconditioners/test_ILU.cpp
    test_preconditioners/test_ILU0.cpp
    test_preconditioners/test_ILUTP.cpp
)

set(
    TEST_SOLVERS

    test_solvers/test_IterativeSolve/test_IterativeSolve.cpp

    test_solvers/test_stationary/test_Jacobi_solve.cpp
    test_solvers/test_stationary/test_SOR/test_GaussSeidel_solve.cpp
    test_solvers/test_stationary/test_SOR/test_SOR_solve.cpp

    test_solvers/test_krylov/test_GMRES/test_GMRES_components.cpp
    test_solvers/test_krylov/test_GMRES/test_GMRES_solve_dbl.cpp
    test_solvers/test_krylov/test_GMRES/test_GMRES_solve_sgl.cpp
    test_solvers/test_krylov/test_GMRES/test_GMRES_solve_hlf.cpp
    test_solvers/test_krylov/test_GMRES/test_PGMRES.cpp

    test_solvers/test_nested/test_FP_GMRES_IR_solve.cpp
    test_solvers/test_nested/test_MP_GMRES_IR_component.cpp
    test_solvers/test_nested/test_MP_GMRES_IR_solve.cpp
)

set(
    TEST_SUITE_CPPS
    test_tools/test_MatrixRead.cu
    ${TEST_TYPES_CPPS}
    # ${TEST_TOOLS_CPPS}
    # ${TEST_PRECONDITIONER_CPPS}
    # ${TEST_SOLVERS}
)

add_executable(
    test
    test.cu
    ${TEST_SUITE_CPPS}
)

target_link_libraries(
    test
    PUBLIC
    gtest_main
    eigen
    precision-cascade
    math_functions
    types_lib
    "$ENV{CUDA_PATH}/lib/x64/cublas.lib"
)
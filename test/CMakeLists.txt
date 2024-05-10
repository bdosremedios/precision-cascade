cmake_minimum_required(VERSION 3.28)

project(precision-cascade LANGUAGES CXX CUDA)

set(
    TEST_TYPES_CPPS
    test_types/test_Scalar.cpp
    test_types/test_Vector.cpp
    test_types/test_MatrixDense.cpp
    test_types/test_NoFillMatrixSparse.cpp
    test_types/test_TypeConversion.cpp
)

set(
    TEST_TOOLS_CPPS
    test_tools/test_read_matrix.cpp
    test_tools/test_Heap.cpp
    test_tools/test_Sort.cpp
)

set(
    TEST_ARG_PKGS
    test_tools/test_arg_pkgs/test_GenericLinearSystem.cpp
    test_tools/test_arg_pkgs/test_TypedLinearSystem.cpp
    test_tools/test_arg_pkgs/test_SolveArgPkg.cpp
    test_tools/test_arg_pkgs/test_PrecondArgPkg.cpp
)

set(
    TEST_PRECONDITIONER_CPPS
    test_preconditioners/test_Preconditioner.cpp
    test_preconditioners/test_ILU.cpp
    test_preconditioners/test_ILU0.cpp
    test_preconditioners/test_ILUTP.cpp
)

set(
    TEST_STATIONARY_SOLVERS
    test_solvers/test_stationary/test_JacobiSolve.cpp
    test_solvers/test_stationary/test_GaussSeidelSolve.cpp
    test_solvers/test_stationary/test_SORSolve.cpp
)

set(
    TEST_GMRES_SOLVERS
    test_solvers/test_GMRES/test_GMRESSolve_components.cpp
    test_solvers/test_GMRES/test_GMRESSolve_solve_hlf.cpp
    test_solvers/test_GMRES/test_GMRESSolve_solve_sgl.cpp
    test_solvers/test_GMRES/test_GMRESSolve_solve_dbl.cpp
    test_solvers/test_GMRES/test_PGMRES.cpp
)

set(
    TEST_NESTED_SOLVERS
    test_solvers/test_nested/test_FP_GMRES_IR_solve.cpp
    test_solvers/test_nested/test_MP_GMRES_IR_component.cpp
    test_solvers/test_nested/test_MP_GMRES_IR_solve.cpp
)

set(
    TEST_SOLVERS
    test_solvers/test_IterativeSolve/test_IterativeSolve.cpp
    ${TEST_STATIONARY_SOLVERS}
    ${TEST_GMRES_SOLVERS}
    ${TEST_NESTED_SOLVERS}
)

set(
    TEST_BENCHMARK
    # benchmarking/benchmark_MatrixDense.cpp
    # benchmarking/benchmark_NoFillMatrixSparse.cpp
    # benchmarking/benchmark_GMRESSolve.cpp
    # benchmarking/benchmark_FP_GMRES_IR.cpp
    benchmarking/benchmark_MP_GMRES_IR.cpp
)

set(
    TEST_SUITE_CPPS
    ${TEST_TYPES_CPPS}
    ${TEST_TOOLS_CPPS}
    ${TEST_PRECONDITIONER_CPPS}
    ${TEST_ARG_PKGS}
    ${TEST_SOLVERS}
    ${TEST_BENCHMARK}
)

add_library(
    test_toolkit
    test_toolkit.cpp
)

target_link_libraries(
    test_toolkit
    PUBLIC
    precision-cascade
)

add_executable(
    test
    test.cpp
    benchmarking/src/benchmark.cpp
    ${TEST_SUITE_CPPS}
)

target_compile_options(
    test
    PRIVATE
    /MDd
)

target_link_libraries(
    test
    PUBLIC
    test_toolkit
    gtest_main
    precision-cascade
)

add_library(
    test_headers
    INTERFACE
)

target_include_directories(
    test_headers
    INTERFACE
    .
)
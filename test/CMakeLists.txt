cmake_minimum_required(VERSION 3.28)

project(precision-cascade LANGUAGES CXX CUDA)

set(
    TEST_TYPES_CPPS
    src/test_types/test_Scalar.cpp
    src/test_types/test_Vector.cpp
    src/test_types/test_MatrixDense.cpp
    src/test_types/test_NoFillMatrixSparse.cpp
    src/test_types/test_TypeConversion.cpp
)

set(
    TEST_TOOLS_CPPS
    src/test_tools/test_read_matrixCSV.cpp
    src/test_tools/test_read_matrixMTX.cpp
    src/test_tools/test_Heap.cpp
    src/test_tools/test_Sort.cpp
)

set(
    TEST_ARG_PKGS
    src/test_tools/test_arg_pkgs/test_GenericLinearSystem.cpp
    src/test_tools/test_arg_pkgs/test_TypedLinearSystem.cpp
    src/test_tools/test_arg_pkgs/test_SolveArgPkg.cpp
    src/test_tools/test_arg_pkgs/test_PrecondArgPkg.cpp
)

set(
    TEST_PRECONDITIONER_CPPS
    src/test_preconditioners/test_NoPreconditioner.cpp
    src/test_preconditioners/test_MatrixInversePreconditioner.cpp
    src/test_preconditioners/test_JacobiPreconditioner.cpp
    src/test_preconditioners/test_ILUPreconditioner.cpp
    src/test_preconditioners/test_ILU0Preconditioner.cpp
    src/test_preconditioners/test_ILUTPPreconditioner.cpp
)

set(
    TEST_STATIONARY_SOLVERS
    src/test_solvers/test_stationary/test_JacobiSolve.cpp
    src/test_solvers/test_stationary/test_GaussSeidelSolve.cpp
    src/test_solvers/test_stationary/test_SORSolve.cpp
)

set(
    TEST_GMRES_SOLVERS
    src/test_solvers/test_GMRES/test_GMRESSolve_components.cpp
    src/test_solvers/test_GMRES/test_GMRESSolve_solve_hlf.cpp
    src/test_solvers/test_GMRES/test_GMRESSolve_solve_sgl.cpp
    src/test_solvers/test_GMRES/test_GMRESSolve_solve_dbl.cpp
    src/test_solvers/test_GMRES/test_PGMRES.cpp
)

set(
    TEST_NESTED_SOLVERS
    src/test_solvers/test_nested/test_FP_GMRES_IR_solve.cpp
    src/test_solvers/test_nested/test_Preconditioned_FP_GMRES_IR_solve.cpp
    src/test_solvers/test_nested/test_MP_GMRES_IR_component.cpp
    src/test_solvers/test_nested/test_MP_GMRES_IR_solve.cpp
    src/test_solvers/test_nested/test_Preconditioned_MP_GMRES_IR_solve.cpp
)

set(
    TEST_SOLVERS
    src/test_solvers/test_IterativeSolve/test_IterativeSolve.cpp
    ${TEST_STATIONARY_SOLVERS}
    ${TEST_GMRES_SOLVERS}
    ${TEST_NESTED_SOLVERS}
)

set(
    TEST_SUITE_CPPS
    ${TEST_TYPES_CPPS}
    ${TEST_TOOLS_CPPS}
    ${TEST_PRECONDITIONER_CPPS}
    ${TEST_ARG_PKGS}
    ${TEST_SOLVERS}
)


add_library(
    test_headers
    INTERFACE
)

target_include_directories(
    test_headers
    INTERFACE
    include
)


add_library(
    test_toolkit
    src/test_toolkit.cpp
)

target_link_libraries(
    test_toolkit
    PUBLIC
    precision-cascade
    test_headers
)

add_executable(
    test
    src/test.cpp
    ${TEST_SUITE_CPPS}
)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")

target_link_libraries(
    test
    PUBLIC
    test_headers
    test_toolkit
    gtest_main
    precision-cascade
)
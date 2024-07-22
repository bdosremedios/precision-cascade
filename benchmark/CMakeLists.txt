cmake_minimum_required(VERSION ${CMAKE_VERSION})

project(precision-cascade LANGUAGES CXX CUDA)

add_library(
    benchmarking_headers_lib
    INTERFACE
)

target_include_directories(
    benchmarking_headers_lib
    INTERFACE
    include
)

add_library(
    benchmark_tools_lib
    src/benchmark_tools.cpp
)

target_include_directories(
    benchmark_tools_lib
    PUBLIC
    include
)

set(
    BENCHMARK_SUITE_CPPS
    # src/benchmark_types/benchmark_MatrixDense.cpp
    src/benchmark_types/benchmark_NoFillMatrixSparse.cpp
    src/benchmark_solvers/benchmark_GMRES.cpp
    src/benchmark_solvers/benchmark_FP_GMRES_IR.cpp
    src/benchmark_solvers/benchmark_MP_GMRES_IR.cpp
    src/benchmark_preconditioners/benchmark_ILUPreconditioner.cpp
    src/benchmark_solvers/benchmark_Precond_FP_GMRES_IR.cpp
    src/benchmark_solvers/benchmark_Precond_MP_GMRES_IR.cpp
)

add_executable(
    benchmark
    main/benchmark.cpp
    ${BENCHMARK_SUITE_CPPS}
)

target_link_libraries(
    benchmark
    benchmarking_headers_lib
    benchmark_tools_lib
    precision-cascade
    gtest_main
)
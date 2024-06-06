cmake_minimum_required(VERSION 3.28)

project(precision-cascade LANGUAGES CXX CUDA)

set(
    BENCHMARK_SUITE_CPPS
    src/benchmark_MatrixDense.cpp
    src/benchmark_NoFillMatrixSparse.cpp
    src/benchmark_GMRES.cpp
    src/benchmark_FP_GMRES_IR.cpp
    src/benchmark_MP_GMRES_IR.cpp
    src/benchmark_ILUPreconditioner.cpp
    src/benchmark_Precond_FP_GMRES_IR.cpp
    src/benchmark_Precond_MP_GMRES_IR.cpp
)

add_executable(
    benchmark
    main/benchmark.cpp
    BENCHMARK_SUITE_CPPS
)
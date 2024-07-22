#include "benchmark_MP_GMRES_IR.h"

TEST_F(Benchmark_MP_GMRES_IR, MP_GMRES_IR_RestartCount_BENCHMARK) {
    
    std::function<void (Benchmark_AccumClock &, NoFillMatrixSparse<double> &)> execute_func = [this] (
        Benchmark_AccumClock &clock, NoFillMatrixSparse<double> &A
    ) {
        execute_mp_gmres_ir<RestartCount>(clock, A);
    };

    benchmark_exec_func<NoFillMatrixSparse, double>(
        sparse_dims, make_norm_A, execute_func, "restart_count"
    );

}

TEST_F(
    Benchmark_MP_GMRES_IR,
    MP_GMRES_IR_SimpleConstantThreshold_BENCHMARK
) {

    std::function<void (Benchmark_AccumClock &, NoFillMatrixSparse<double> &)> execute_func = [this] (
        Benchmark_AccumClock &clock, NoFillMatrixSparse<double> &A
    ) {
        execute_mp_gmres_ir<SimpleConstantThreshold>(clock, A);
    };

    benchmark_exec_func<NoFillMatrixSparse, double>(
        sparse_dims, make_norm_A, execute_func, "simple_constant_threshold"
    );

}
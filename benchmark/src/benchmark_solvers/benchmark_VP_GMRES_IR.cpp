#include "benchmark_VP_GMRES_IR.h"

TEST_F(Benchmark_VP_GMRES_IR, VP_GMRES_IR_OuterRestartCount_BENCHMARK) {
    
    std::function<void (Benchmark_AccumClock &, NoFillMatrixSparse<double> &)> execute_func = [this] (
        Benchmark_AccumClock &clock, NoFillMatrixSparse<double> &A
    ) {
        execute_vp_gmres_ir<NoProgress_OuterRestartCount>(clock, A);
    };

    benchmark_exec_func<NoFillMatrixSparse, double>(
        sparse_dims, make_norm_A, execute_func, "outer_restart_count"
    );

}
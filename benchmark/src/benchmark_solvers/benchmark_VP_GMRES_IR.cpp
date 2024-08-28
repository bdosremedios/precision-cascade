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

TEST_F(
    Benchmark_VP_GMRES_IR,
    VP_GMRES_IR_RelativeResidualThreshold_BENCHMARK
) {

    std::function<void (Benchmark_AccumClock &, NoFillMatrixSparse<double> &)> execute_func = [this] (
        Benchmark_AccumClock &clock, NoFillMatrixSparse<double> &A
    ) {
        execute_vp_gmres_ir<NoProgress_RelativeResidualThreshold>(clock, A);
    };

    benchmark_exec_func<NoFillMatrixSparse, double>(
        sparse_dims, make_norm_A, execute_func, "relative_residual_threshold"
    );

}

TEST_F(
    Benchmark_VP_GMRES_IR,
    VP_GMRES_IR_CheckStagnation_BENCHMARK
) {

    std::function<void (Benchmark_AccumClock &, NoFillMatrixSparse<double> &)> execute_func = [this] (
        Benchmark_AccumClock &clock, NoFillMatrixSparse<double> &A
    ) {
        execute_vp_gmres_ir<NoProgress_CheckStagnation>(clock, A);
    };

    benchmark_exec_func<NoFillMatrixSparse, double>(
        sparse_dims, make_norm_A, execute_func, "check_stagnation"
    );

}

TEST_F(
    Benchmark_VP_GMRES_IR,
    VP_GMRES_IR_ThresholdToStagnation_BENCHMARK
) {

    std::function<void (Benchmark_AccumClock &, NoFillMatrixSparse<double> &)> execute_func = [this] (
        Benchmark_AccumClock &clock, NoFillMatrixSparse<double> &A
    ) {
        execute_vp_gmres_ir<NoProgress_ThresholdToStagnation>(clock, A);
    };

    benchmark_exec_func<NoFillMatrixSparse, double>(
        sparse_dims,
        make_norm_A,
        execute_func,
        "project_threshold_after_stagnation"
    );

}
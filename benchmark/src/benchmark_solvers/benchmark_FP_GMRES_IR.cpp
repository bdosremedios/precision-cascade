#include "benchmark_FP_GMRES_IR.h"

TEST_F(Benchmark_FP_GMRES_IR, FP_GMRES_IR_Double_BENCHMARK) {

    std::function<void (Benchmark_AccumClock &, NoFillMatrixSparse<double> &)> execute_func = [this] (
        Benchmark_AccumClock &clock, NoFillMatrixSparse<double> &A
    ) {
        execute_fp_gmres_ir<double>(clock, A);
    };

    benchmark_exec_func<NoFillMatrixSparse>(
        // sparse_dims, make_norm_A, execute_func, "fp_gmres_ir_dbl"
        std::vector<int>({167554, 185664}),
        make_norm_A,
        execute_func,
        "fp_gmres_ir_dbl"
    );

}

TEST_F(Benchmark_FP_GMRES_IR, FP_GMRES_IR_Single_BENCHMARK) {

    std::function<void (Benchmark_AccumClock &, NoFillMatrixSparse<double> &)> execute_func = [this] (
        Benchmark_AccumClock &clock, NoFillMatrixSparse<double> &A
    ) {
        execute_fp_gmres_ir<float>(clock, A);
    };

    benchmark_exec_func<NoFillMatrixSparse>(
        sparse_dims, make_norm_A, execute_func, "fp_gmres_ir_sgl"
    );

}

TEST_F(Benchmark_FP_GMRES_IR, FP_GMRES_IR_Half_BENCHMARK) {

    std::function<void (Benchmark_AccumClock &, NoFillMatrixSparse<double> &)> execute_func = [this] (
        Benchmark_AccumClock &clock, NoFillMatrixSparse<double> &A
    ) {
        execute_fp_gmres_ir<__half>(clock, A);
    };

    benchmark_exec_func<NoFillMatrixSparse>(
        sparse_dims, make_norm_A, execute_func, "fp_gmres_ir_hlf"
    );

}
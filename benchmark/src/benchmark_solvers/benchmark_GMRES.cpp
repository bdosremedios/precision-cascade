#include "benchmark_GMRES.h"

TEST_F(Benchmark_GMRES, GMRESSolve_Double_BENCHMARK) {

    std::function<void (Benchmark_AccumClock &, NoFillMatrixSparse<double> &)> execute_func = [this] (
        Benchmark_AccumClock &clock, NoFillMatrixSparse<double> &A
    ) {
        execute_GMRES<double>(clock, A);
    };

    benchmark_exec_func<NoFillMatrixSparse, double>(
        sparse_start, sparse_stop, sparse_incr,
        make_norm_A, execute_func, "gmressolve_dbl"
    );

}

TEST_F(Benchmark_GMRES, GMRESSolve_Single_BENCHMARK) {

    std::function<void (Benchmark_AccumClock &, NoFillMatrixSparse<double> &)> execute_func = [this] (
        Benchmark_AccumClock &clock, NoFillMatrixSparse<double> &A
    ) {
        execute_GMRES<float>(clock, A);
    };

    benchmark_exec_func<NoFillMatrixSparse, double>(
        sparse_start, sparse_stop, sparse_incr,
        make_norm_A, execute_func, "gmressolve_sgl"
    );

}

TEST_F(Benchmark_GMRES, GMRESSolve_Half_BENCHMARK) {

    std::function<void (Benchmark_AccumClock &, NoFillMatrixSparse<double> &)> execute_func = [this] (
        Benchmark_AccumClock &clock, NoFillMatrixSparse<double> &A
    ) {
        execute_GMRES<__half>(clock, A);
    };

    benchmark_exec_func<NoFillMatrixSparse, double>(
        sparse_start, sparse_stop, sparse_incr,
        make_norm_A, execute_func, "gmressolve_hlf"
    );

}

TEST_F(Benchmark_GMRES, GetExtrapolationData_Double) {

    std::function<MatrixDense<double> (int, int)> make_A_wrapper = [this] (
        int m, int n
    ) -> MatrixDense<double> {
        return make_extrap_A<double>(m, n);
    };

    std::function<void (Benchmark_AccumClock &, MatrixDense<double> &)> execute_func = [this] (
        Benchmark_AccumClock &clock, MatrixDense<double> &A
    ) {
        gmres_extrap_data_exec_func<double>(clock, A);
    };

    benchmark_exec_func<MatrixDense, double>(
        sparse_start, sparse_stop, sparse_incr,
        make_A_wrapper, execute_func, "gmressolveextrapdata_dbl"
    );


}

TEST_F(Benchmark_GMRES, GetExtrapolationData_Single) {

    std::function<MatrixDense<float> (int, int)> make_A_wrapper = [this] (
        int m, int n
    ) -> MatrixDense<float> {
        return make_extrap_A<float>(m, n);
    };

    std::function<void (Benchmark_AccumClock &, MatrixDense<float> &)> execute_func = [this] (
        Benchmark_AccumClock &clock, MatrixDense<float> &A
    ) {
        gmres_extrap_data_exec_func<float>(clock, A);
    };

    benchmark_exec_func<MatrixDense, float>(
        sparse_start, sparse_stop, sparse_incr,
        make_A_wrapper, execute_func, "gmressolveextrapdata_sgl"
    );

}

TEST_F(Benchmark_GMRES, GetExtrapolationData_Half) {

    std::function<MatrixDense<__half> (int, int)> make_A_wrapper = [this] (
        int m, int n
    ) -> MatrixDense<__half> {
        return make_extrap_A<__half>(m, n);
    };

    std::function<void (Benchmark_AccumClock &, MatrixDense<__half> &)> execute_func = [this] (
        Benchmark_AccumClock &clock, MatrixDense<__half> &A
    ) {
        gmres_extrap_data_exec_func<__half>(clock, A);
    };

    benchmark_exec_func<MatrixDense, __half>(
        sparse_start, sparse_stop, sparse_incr,
        make_A_wrapper, execute_func, "gmressolveextrapdata_hlf"
    );

}
#include "benchmark_Sparse.h"

#include "types/types.h"

#include <functional>
#include <fstream>
#include <cmath>

class Benchmark_NoFillMatrixSparse: public Benchmark_Sparse {};

TEST_F(Benchmark_NoFillMatrixSparse, MatrixVectorMult_Double_BENCHMARK) {

    std::function<void (Benchmark_AccumClock &, NoFillMatrixSparse<double> &)> execute_func = [this] (
        Benchmark_AccumClock &clock, NoFillMatrixSparse<double> &A
    ) {
        Vector<double> b = Vector<double>::Random(
            BenchmarkBase::bundle, A.rows()
        );
        clock.clock_start();
        for (int k=0; k<run_fast_tests_count; ++k) { A*b; }
        clock.clock_stop();
    };

    benchmark_exec_func<NoFillMatrixSparse, double>(
        sparse_dims, make_A_dbl, execute_func, "matsparse_mv_dbl"
    );

}

TEST_F(Benchmark_NoFillMatrixSparse, MatrixVectorMult_Single_BENCHMARK) {

    std::function<void (Benchmark_AccumClock &, NoFillMatrixSparse<float> &)> execute_func = [this] (
        Benchmark_AccumClock &clock, NoFillMatrixSparse<float> &A
    ) {
        Vector<float> b = Vector<float>::Random(
            BenchmarkBase::bundle, A.rows()
        );
        clock.clock_start();
        for (int k=0; k<run_fast_tests_count; ++k) { A*b; }
        clock.clock_stop();
    };

    benchmark_exec_func<NoFillMatrixSparse, float>(
        sparse_dims, make_A_sgl, execute_func, "matsparse_mv_sgl"
    );

}

TEST_F(Benchmark_NoFillMatrixSparse, MatrixVectorMult_Half_BENCHMARK) {

    std::function<void (Benchmark_AccumClock &, NoFillMatrixSparse<__half> &)> execute_func = [this] (
        Benchmark_AccumClock &clock, NoFillMatrixSparse<__half> &A
    ) {
        Vector<__half> b = Vector<__half>::Random(
            BenchmarkBase::bundle, A.rows()
        );
        clock.clock_start();
        for (int k=0; k<run_fast_tests_count; ++k) { A*b; }
        clock.clock_stop();
    };

    benchmark_exec_func<NoFillMatrixSparse, __half>(
        sparse_dims, make_A_hlf, execute_func, "matsparse_mv_hlf"
    );

}

TEST_F(Benchmark_NoFillMatrixSparse, TransposeMatrixVectorMult_Double_BENCHMARK) {

    std::function<void (Benchmark_AccumClock &, NoFillMatrixSparse<double> &)> execute_func = [this] (
        Benchmark_AccumClock &clock, NoFillMatrixSparse<double> &A
    ) {
        Vector<double> b = Vector<double>::Random(
            BenchmarkBase::bundle, A.rows()
        );
        clock.clock_start();
        for (int k=0; k<run_fast_tests_count; ++k) { A.transpose_prod(b); }
        clock.clock_stop();
    };

    benchmark_exec_func<NoFillMatrixSparse, double>(
        sparse_dims, make_A_dbl, execute_func, "matsparse_tmv_dbl"
    );

}

TEST_F(Benchmark_NoFillMatrixSparse, TransposeMatrixVectorMult_Single_BENCHMARK) {

    std::function<void (Benchmark_AccumClock &, NoFillMatrixSparse<float> &)> execute_func = [this] (
        Benchmark_AccumClock &clock, NoFillMatrixSparse<float> &A
    ) {
        Vector<float> b = Vector<float>::Random(
            BenchmarkBase::bundle, A.rows()
        );
        clock.clock_start();
        for (int k=0; k<run_fast_tests_count; ++k) { A.transpose_prod(b); }
        clock.clock_stop();
    };

    benchmark_exec_func<NoFillMatrixSparse, float>(
        sparse_dims, make_A_sgl, execute_func, "matsparse_tmv_sgl"
    );

}

TEST_F(Benchmark_NoFillMatrixSparse, TransposeMatrixVectorMult_Half_BENCHMARK) {

    std::function<void (Benchmark_AccumClock &, NoFillMatrixSparse<__half> &)> execute_func = [this] (
        Benchmark_AccumClock &clock, NoFillMatrixSparse<__half> &A
    ) {
        Vector<__half> b = Vector<__half>::Random(
            BenchmarkBase::bundle, A.rows()
        );
        clock.clock_start();
        for (int k=0; k<run_fast_tests_count; ++k) { A.transpose_prod(b); }
        clock.clock_stop();
    };

    benchmark_exec_func<NoFillMatrixSparse, __half>(
        sparse_dims, make_A_hlf, execute_func, "matsparse_tmv_hlf"
    );

}

TEST_F(Benchmark_NoFillMatrixSparse, ForwardSubstitution_Double_BENCHMARK) {

    std::function<void (Benchmark_AccumClock &, NoFillMatrixSparse<double> &)> execute_func = [] (
        Benchmark_AccumClock &clock, NoFillMatrixSparse<double> &A
    ) {
        Vector<double> x = Vector<double>::Random(
            BenchmarkBase::bundle, A.rows()
        );
        Vector<double> b = A*x;
        clock.clock_start();
        A.frwd_sub(b);
        clock.clock_stop();
    };

    benchmark_exec_func<NoFillMatrixSparse, double>(
        sparse_dims, make_low_tri_A_dbl, execute_func, "matsparse_frwdsub_dbl"
    );

}

TEST_F(Benchmark_NoFillMatrixSparse, ForwardSubstitution_Single_BENCHMARK) {

    std::function<void (Benchmark_AccumClock &, NoFillMatrixSparse<float> &)> execute_func = [] (
        Benchmark_AccumClock &clock, NoFillMatrixSparse<float> &A
    ) {
        Vector<float> x = Vector<float>::Random(
            BenchmarkBase::bundle, A.rows()
        );
        Vector<float> b = A*x;
        clock.clock_start();
        A.frwd_sub(b);
        clock.clock_stop();
    };

    benchmark_exec_func<NoFillMatrixSparse, float>(
        sparse_dims, make_low_tri_A_sgl, execute_func, "matsparse_frwdsub_sgl"
    );

}

TEST_F(Benchmark_NoFillMatrixSparse, ForwardSubstitution_Half_BENCHMARK) {

    std::function<void (Benchmark_AccumClock &, NoFillMatrixSparse<__half> &)> execute_func = [] (
        Benchmark_AccumClock &clock, NoFillMatrixSparse<__half> &A
    ) {
        Vector<__half> x = Vector<__half>::Random(
            BenchmarkBase::bundle, A.rows()
        );
        Vector<__half> b = A*x;
        clock.clock_start();
        A.frwd_sub(b);
        clock.clock_stop();
    };

    benchmark_exec_func<NoFillMatrixSparse, __half>(
        sparse_dims, make_low_tri_A_hlf, execute_func, "matsparse_frwdsub_hlf"
    );

}

TEST_F(Benchmark_NoFillMatrixSparse, BackwardSubstitution_Double_BENCHMARK) {

    std::function<void (Benchmark_AccumClock &, NoFillMatrixSparse<double> &)> execute_func = [] (
        Benchmark_AccumClock &clock, NoFillMatrixSparse<double> &A
    ) {
        Vector<double> x = Vector<double>::Random(
            BenchmarkBase::bundle, A.rows()
        );
        Vector<double> b = A*x;
        clock.clock_start();
        A.back_sub(b);
        clock.clock_stop();
    };

    benchmark_exec_func<NoFillMatrixSparse, double>(
        sparse_dims, make_upp_tri_A_dbl, execute_func, "matsparse_backsub_dbl"
    );

}

TEST_F(Benchmark_NoFillMatrixSparse, BackwardSubstitution_Single_BENCHMARK) {

    std::function<void (Benchmark_AccumClock &, NoFillMatrixSparse<float> &)> execute_func = [] (
        Benchmark_AccumClock &clock, NoFillMatrixSparse<float> &A
    ) {
        Vector<float> x = Vector<float>::Random(
            BenchmarkBase::bundle, A.rows()
        );
        Vector<float> b = A*x;
        clock.clock_start();
        A.back_sub(b);
        clock.clock_stop();
    };

    benchmark_exec_func<NoFillMatrixSparse, float>(
        sparse_dims, make_upp_tri_A_sgl, execute_func, "matsparse_backsub_sgl"
    );

}

TEST_F(Benchmark_NoFillMatrixSparse, BackwardSubstitution_Half_BENCHMARK) {

    std::function<void (Benchmark_AccumClock &, NoFillMatrixSparse<__half> &)> execute_func = [] (
        Benchmark_AccumClock &clock, NoFillMatrixSparse<__half> &A
    ) {
        Vector<__half> x = Vector<__half>::Random(
            BenchmarkBase::bundle, A.rows()
        );
        Vector<__half> b = A*x;
        clock.clock_start();
        A.back_sub(b);
        clock.clock_stop();
    };

    benchmark_exec_func<NoFillMatrixSparse, __half>(
        sparse_dims, make_upp_tri_A_hlf, execute_func, "matsparse_backsub_hlf"
    );

}
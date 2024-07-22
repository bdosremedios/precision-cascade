#include "benchmark_Dense.h"

#include "types/types.h"

class Benchmark_MatrixDense: public Benchmark_Dense {};

TEST_F(Benchmark_MatrixDense, MatrixVectorMult_Double_BENCHMARK) {

    std::function<void (Benchmark_AccumClock &, MatrixDense<double> &)> execute_func = [this] (
        Benchmark_AccumClock &clock, MatrixDense<double> &A
    ) {
        Vector<double> b = Vector<double>::Random(
            BenchmarkBase::bundle, A.rows()
        );
        clock.clock_start();
        for (int k=0; k<run_fast_tests_count; ++k) { A*b; }
        clock.clock_stop();
    };

    benchmark_exec_func<MatrixDense, double>(
        dense_dims, make_A_dbl, execute_func, "matdense_mv_dbl"
    );

}

TEST_F(Benchmark_MatrixDense, MatrixVectorMult_Single_BENCHMARK) {

    std::function<void (Benchmark_AccumClock &, MatrixDense<float> &)> execute_func = [this] (
        Benchmark_AccumClock &clock, MatrixDense<float> &A
    ) {
        Vector<float> b = Vector<float>::Random(
            BenchmarkBase::bundle, A.rows()
        );
        clock.clock_start();
        for (int k=0; k<run_fast_tests_count; ++k) { A*b; }
        clock.clock_stop();
    };

    benchmark_exec_func<MatrixDense, float>(
        dense_dims, make_A_sgl, execute_func, "matdense_mv_sgl"
    );

}

TEST_F(Benchmark_MatrixDense, MatrixVectorMult_Half_BENCHMARK) {

    std::function<void (Benchmark_AccumClock &, MatrixDense<__half> &)> execute_func = [this] (
        Benchmark_AccumClock &clock, MatrixDense<__half> &A
    ) {
        Vector<__half> b = Vector<__half>::Random(
            BenchmarkBase::bundle, A.rows()
        );
        clock.clock_start();
        for (int k=0; k<run_fast_tests_count; ++k) { A*b; }
        clock.clock_stop();
    };

    benchmark_exec_func<MatrixDense, __half>(
        dense_dims, make_A_hlf, execute_func, "matdense_mv_hlf"
    );

}

TEST_F(Benchmark_MatrixDense, TransposeMatrixVectorMult_Double_BENCHMARK) {

    std::function<void (Benchmark_AccumClock &, MatrixDense<double> &)> execute_func = [this] (
        Benchmark_AccumClock &clock, MatrixDense<double> &A
    ) {
        Vector<double> b = Vector<double>::Random(
            BenchmarkBase::bundle, A.rows()
        );
        clock.clock_start();
        for (int k=0; k<run_fast_tests_count; ++k) { A.transpose_prod(b); }
        clock.clock_stop();
    };

    benchmark_exec_func<MatrixDense, double>(
        dense_dims, make_A_dbl, execute_func, "matdense_tmv_dbl"
    );

}

TEST_F(Benchmark_MatrixDense, TransposeMatrixVectorMult_Single_BENCHMARK) {

    std::function<void (Benchmark_AccumClock &, MatrixDense<float> &)> execute_func = [this] (
        Benchmark_AccumClock &clock, MatrixDense<float> &A
    ) {
        Vector<float> b = Vector<float>::Random(
            BenchmarkBase::bundle, A.rows()
        );
        clock.clock_start();
        for (int k=0; k<run_fast_tests_count; ++k) { A.transpose_prod(b); }
        clock.clock_stop();
    };

    benchmark_exec_func<MatrixDense, float>(
        dense_dims, make_A_sgl, execute_func, "matdense_tmv_sgl"
    );

}

TEST_F(Benchmark_MatrixDense, TransposeMatrixVectorMult_Half_BENCHMARK) {

    std::function<void (Benchmark_AccumClock &, MatrixDense<__half> &)> execute_func = [this] (
        Benchmark_AccumClock &clock, MatrixDense<__half> &A
    ) {
        Vector<__half> b = Vector<__half>::Random(
            BenchmarkBase::bundle, A.rows()
        );
        clock.clock_start();
        for (int k=0; k<run_fast_tests_count; ++k) { A.transpose_prod(b); }
        clock.clock_stop();
    };

    benchmark_exec_func<MatrixDense, __half>(
        dense_dims, make_A_hlf, execute_func, "matdense_tmv_hlf"
    );

}

TEST_F(Benchmark_MatrixDense, SubsetcolsMatrixVectorMult_Double_BENCHMARK) {

    std::function<void (Benchmark_AccumClock &, MatrixDense<double> &)> execute_func = [this] (
        Benchmark_AccumClock &clock, MatrixDense<double> &A
    ) {
        Vector<double> b = Vector<double>::Random(BenchmarkBase::bundle, dense_subset_cols);
        clock.clock_start();
        for (int k=0; k<run_fast_tests_count; ++k) {
            A.mult_subset_cols(0, dense_subset_cols, b);
        }
        clock.clock_stop();
    };

    benchmark_exec_func<MatrixDense, double>(
        dense_dims, make_A_dbl, execute_func, "matdense_subsetcolsmv_dbl"
    );

}

TEST_F(Benchmark_MatrixDense, SubsetcolsMatrixVectorMult_Single_BENCHMARK) {

    std::function<void (Benchmark_AccumClock &, MatrixDense<float> &)> execute_func = [this] (
        Benchmark_AccumClock &clock, MatrixDense<float> &A
    ) {
        Vector<float> b = Vector<float>::Random(BenchmarkBase::bundle, dense_subset_cols);
        clock.clock_start();
        for (int k=0; k<run_fast_tests_count; ++k) {
            A.mult_subset_cols(0, dense_subset_cols, b);
        }
        clock.clock_stop();
    };

    benchmark_exec_func<MatrixDense, float>(
        dense_dims, make_A_sgl, execute_func, "matdense_subsetcolsmv_sgl"
    );

}

TEST_F(Benchmark_MatrixDense, SubsetcolsMatrixVectorMult_Half_BENCHMARK) {

    std::function<void (Benchmark_AccumClock &, MatrixDense<__half> &)> execute_func = [this] (
        Benchmark_AccumClock &clock, MatrixDense<__half> &A
    ) {
        Vector<__half> b = Vector<__half>::Random(BenchmarkBase::bundle, dense_subset_cols);
        clock.clock_start();
        for (int k=0; k<run_fast_tests_count; ++k) {
            A.mult_subset_cols(0, dense_subset_cols, b);
        }
        clock.clock_stop();
    };

    benchmark_exec_func<MatrixDense, __half>(
        dense_dims, make_A_hlf, execute_func, "matdense_subsetcolsmv_hlf"
    );

}

TEST_F(Benchmark_MatrixDense, SubsetcolsTransposeMatrixVectorMult_Double_BENCHMARK) {

    std::function<void (Benchmark_AccumClock &, MatrixDense<double> &)> execute_func = [this] (
        Benchmark_AccumClock &clock, MatrixDense<double> &A
    ) {
        Vector<double> b = Vector<double>::Random(BenchmarkBase::bundle, A.rows());
        clock.clock_start();
        for (int k=0; k<run_fast_tests_count; ++k) {
            A.transpose_prod_subset_cols(0, dense_subset_cols, b);
        }
        clock.clock_stop();
    };

    benchmark_exec_func<MatrixDense, double>(
        dense_dims, make_A_dbl, execute_func, "matdense_subsetcolstmv_dbl"
    );

}

TEST_F(Benchmark_MatrixDense, SubsetcolsTransposeMatrixVectorMult_Single_BENCHMARK) {

    std::function<void (Benchmark_AccumClock &, MatrixDense<float> &)> execute_func = [this] (
        Benchmark_AccumClock &clock, MatrixDense<float> &A
    ) {
        Vector<float> b = Vector<float>::Random(BenchmarkBase::bundle, A.rows());
        clock.clock_start();
        for (int k=0; k<run_fast_tests_count; ++k) {
            A.transpose_prod_subset_cols(0, dense_subset_cols, b);
        }
        clock.clock_stop();
    };

    benchmark_exec_func<MatrixDense, float>(
        dense_dims, make_A_sgl, execute_func, "matdense_subsetcolstmv_sgl"
    );

}

TEST_F(Benchmark_MatrixDense, SubsetcolsTransposeMatrixVectorMult_Half_BENCHMARK) {

    std::function<void (Benchmark_AccumClock &, MatrixDense<__half> &)> execute_func = [this] (
        Benchmark_AccumClock &clock, MatrixDense<__half> &A
    ) {
        Vector<__half> b = Vector<__half>::Random(BenchmarkBase::bundle, A.rows());
        clock.clock_start();
        for (int k=0; k<run_fast_tests_count; ++k) {
            A.transpose_prod_subset_cols(0, dense_subset_cols, b);
        }
        clock.clock_stop();
    };

    benchmark_exec_func<MatrixDense, __half>(
        dense_dims, make_A_hlf, execute_func, "matdense_subsetcolstmv_hlf"
    );

}

TEST_F(Benchmark_MatrixDense, ForwardSubstitution_Double_BENCHMARK) {

    std::function<void (Benchmark_AccumClock &, MatrixDense<double> &)> execute_func = [] (
        Benchmark_AccumClock &clock, MatrixDense<double> &A
    ) {
        Vector<double> x = Vector<double>::Random(BenchmarkBase::bundle, A.rows());
        Vector<double> b = A*x;
        clock.clock_start();
        A.frwd_sub(b);
        clock.clock_stop();
    };

    benchmark_exec_func<MatrixDense, double>(
        dense_dims, make_low_tri_A_dbl, execute_func, "matdense_frwdsub_dbl"
    );

}



TEST_F(Benchmark_MatrixDense, ForwardSubstitution_Single_BENCHMARK) {

    std::function<void (Benchmark_AccumClock &, MatrixDense<float> &)> execute_func = [] (
        Benchmark_AccumClock &clock, MatrixDense<float> &A
    ) {
        Vector<float> x = Vector<float>::Random(BenchmarkBase::bundle, A.rows());
        Vector<float> b = A*x;
        clock.clock_start();
        A.frwd_sub(b);
        clock.clock_stop();
    };

    benchmark_exec_func<MatrixDense, float>(
        dense_dims, make_low_tri_A_sgl, execute_func, "matdense_frwdsub_sgl"
    );

}

TEST_F(Benchmark_MatrixDense, ForwardSubstitution_Half_BENCHMARK) {

    std::function<void (Benchmark_AccumClock &, MatrixDense<__half> &)> execute_func = [] (
        Benchmark_AccumClock &clock, MatrixDense<__half> &A
    ) {
        Vector<__half> x = Vector<__half>::Random(BenchmarkBase::bundle, A.rows());
        Vector<__half> b = A*x;
        clock.clock_start();
        A.frwd_sub(b);
        clock.clock_stop();
    };

    benchmark_exec_func<MatrixDense, __half>(
        dense_dims, make_low_tri_A_hlf, execute_func, "matdense_frwdsub_hlf"
    );

}

TEST_F(Benchmark_MatrixDense, BackwardSubstitution_Double_BENCHMARK) {

    std::function<void (Benchmark_AccumClock &, MatrixDense<double> &)> execute_func = [] (
        Benchmark_AccumClock &clock, MatrixDense<double> &A
    ) {
        Vector<double> x = Vector<double>::Random(BenchmarkBase::bundle, A.rows());
        Vector<double> b = A*x;
        clock.clock_start();
        A.back_sub(b);
        clock.clock_stop();
    };

    benchmark_exec_func<MatrixDense, double>(
        dense_dims, make_upp_tri_A_dbl, execute_func, "matdense_backsub_dbl"
    );

}

TEST_F(Benchmark_MatrixDense, BackwardSubstitution_Single_BENCHMARK) {

    std::function<void (Benchmark_AccumClock &, MatrixDense<float> &)> execute_func = [] (
        Benchmark_AccumClock &clock, MatrixDense<float> &A
    ) {
        Vector<float> x = Vector<float>::Random(BenchmarkBase::bundle, A.rows());
        Vector<float> b = A*x;
        clock.clock_start();
        A.back_sub(b);
        clock.clock_stop();
    };

    benchmark_exec_func<MatrixDense, float>(
        dense_dims, make_upp_tri_A_sgl, execute_func, "matdense_backsub_sgl"
    );

}

TEST_F(Benchmark_MatrixDense, BackwardSubstitution_Half_BENCHMARK) {

    std::function<void (Benchmark_AccumClock &, MatrixDense<__half> &)> execute_func = [] (
        Benchmark_AccumClock &clock, MatrixDense<__half> &A
    ) {
        Vector<__half> x = Vector<__half>::Random(BenchmarkBase::bundle, A.rows());
        Vector<__half> b = A*x;
        clock.clock_start();
        A.back_sub(b);
        clock.clock_stop();
    };

    benchmark_exec_func<MatrixDense, __half>(
        dense_dims, make_upp_tri_A_hlf, execute_func, "matdense_backsub_hlf"
    );

}
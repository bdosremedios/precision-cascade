#include "benchmark_Matrix.h"

#include "types/types.h"

class MatrixDense_Benchmark: public Benchmark_Matrix
{
public:

    std::function<MatrixDense<double> (int, int)> make_A_dbl = [this] (
        int m, int n
    ) -> MatrixDense<double> {
        return MatrixDense<double>::Random(BenchmarkBase::bundle, m, m);
    };

    std::function<MatrixDense<float> (int, int)> make_A_sgl = [this] (
        int m, int n
    ) -> MatrixDense<float> {
        return MatrixDense<float>::Random(BenchmarkBase::bundle, m, m);
    };

    std::function<MatrixDense<__half> (int, int)> make_A_hlf = [this] (
        int m, int n
    ) -> MatrixDense<__half> {
        return MatrixDense<__half>::Random(BenchmarkBase::bundle, m, m);
    };

    std::function<MatrixDense<double> (int, int)> make_low_tri_A_dbl = [this] (
        int m, int n
    ) -> MatrixDense<double> {
        return MatrixDense<double>::Random_LT(BenchmarkBase::bundle, m, m);
    };

    std::function<MatrixDense<float> (int, int)> make_low_tri_A_sgl = [this] (
        int m, int n
    ) -> MatrixDense<float> {
        return MatrixDense<float>::Random_LT(BenchmarkBase::bundle, m, m);
    };

    std::function<MatrixDense<__half> (int, int)> make_low_tri_A_hlf = [this] (
        int m, int n
    ) -> MatrixDense<__half> {
        return MatrixDense<__half>::Random_LT(BenchmarkBase::bundle, m, m);
    };
    
    std::function<MatrixDense<double> (int, int)> make_upp_tri_A_dbl = [this] (
        int m, int n
    ) -> MatrixDense<double> {
        return MatrixDense<double>::Random_UT(BenchmarkBase::bundle, m, m);
    };
    
    std::function<MatrixDense<float> (int, int)> make_upp_tri_A_sgl = [this] (
        int m, int n
    ) -> MatrixDense<float> {
        return MatrixDense<float>::Random_UT(BenchmarkBase::bundle, m, m);
    };
    
    std::function<MatrixDense<__half> (int, int)> make_upp_tri_A_hlf = [this] (
        int m, int n
    ) -> MatrixDense<__half> {
        return MatrixDense<__half>::Random_UT(BenchmarkBase::bundle, m, m);
    };

};

TEST_F(MatrixDense_Benchmark, MatrixVectorMult_Double_BENCHMARK) {

    std::function<void (Benchmark_AccumulatingClock &, MatrixDense<double> &)> execute_func = [] (
        Benchmark_AccumulatingClock &clock, MatrixDense<double> &A
    ) {
        Vector<double> b = Vector<double>::Random(BenchmarkBase::bundle, A.rows());
        clock.clock_start();
        A*b;
        clock.clock_stop();
    };

    benchmark_exec_func<MatrixDense, double>(
        dense_start, dense_stop, dense_incr,
        make_A_dbl, execute_func, "matdense_mv_dbl"
    );

}

TEST_F(MatrixDense_Benchmark, MatrixVectorMult_Single_BENCHMARK) {

    std::function<void (Benchmark_AccumulatingClock &, MatrixDense<float> &)> execute_func = [] (
        Benchmark_AccumulatingClock &clock, MatrixDense<float> &A
    ) {
        Vector<float> b = Vector<float>::Random(BenchmarkBase::bundle, A.rows());
        clock.clock_start();
        A*b;
        clock.clock_stop();
    };

    benchmark_exec_func<MatrixDense, float>(
        dense_start, dense_stop, dense_incr,
        make_A_sgl, execute_func, "matdense_mv_sgl"
    );

}

TEST_F(MatrixDense_Benchmark, MatrixVectorMult_Half_BENCHMARK) {

    std::function<void (Benchmark_AccumulatingClock &, MatrixDense<__half> &)> execute_func = [] (
        Benchmark_AccumulatingClock &clock, MatrixDense<__half> &A
    ) {
        Vector<__half> b = Vector<__half>::Random(BenchmarkBase::bundle, A.rows());
        clock.clock_start();
        A*b;
        clock.clock_stop();
    };

    benchmark_exec_func<MatrixDense, __half>(
        dense_start, dense_stop, dense_incr,
        make_A_hlf, execute_func, "matdense_mv_hlf"
    );

}

TEST_F(MatrixDense_Benchmark, TransposeMatrixVectorMult_Double_BENCHMARK) {

    std::function<void (Benchmark_AccumulatingClock &, MatrixDense<double> &)> execute_func = [] (
        Benchmark_AccumulatingClock &clock, MatrixDense<double> &A
    ) {
        Vector<double> b = Vector<double>::Random(BenchmarkBase::bundle, A.rows());
        clock.clock_start();
        A.transpose_prod(b);
        clock.clock_stop();
    };

    benchmark_exec_func<MatrixDense, double>(
        dense_start, dense_stop, dense_incr,
        make_A_dbl, execute_func, "matdense_tmv_dbl"
    );

}

TEST_F(MatrixDense_Benchmark, TransposeMatrixVectorMult_Single_BENCHMARK) {

    std::function<void (Benchmark_AccumulatingClock &, MatrixDense<float> &)> execute_func = [] (
        Benchmark_AccumulatingClock &clock, MatrixDense<float> &A
    ) {
        Vector<float> b = Vector<float>::Random(BenchmarkBase::bundle, A.rows());
        clock.clock_start();
        A.transpose_prod(b);
        clock.clock_stop();
    };

    benchmark_exec_func<MatrixDense, float>(
        dense_start, dense_stop, dense_incr,
        make_A_sgl, execute_func, "matdense_tmv_sgl"
    );

}

TEST_F(MatrixDense_Benchmark, TransposeMatrixVectorMult_Half_BENCHMARK) {

    std::function<void (Benchmark_AccumulatingClock &, MatrixDense<__half> &)> execute_func = [] (
        Benchmark_AccumulatingClock &clock, MatrixDense<__half> &A
    ) {
        Vector<__half> b = Vector<__half>::Random(BenchmarkBase::bundle, A.rows());
        clock.clock_start();
        A.transpose_prod(b);
        clock.clock_stop();
    };

    benchmark_exec_func<MatrixDense, __half>(
        dense_start, dense_stop, dense_incr,
        make_A_hlf, execute_func, "matdense_tmv_hlf"
    );

}

TEST_F(MatrixDense_Benchmark, SubsetcolsMatrixVectorMult_Double_BENCHMARK) {

    std::function<void (Benchmark_AccumulatingClock &, MatrixDense<double> &)> execute_func = [this] (
        Benchmark_AccumulatingClock &clock, MatrixDense<double> &A
    ) {
        Vector<double> b = Vector<double>::Random(BenchmarkBase::bundle, dense_subset_cols);
        clock.clock_start();
        A.mult_subset_cols(0, dense_subset_cols, b);
        clock.clock_stop();
    };

    benchmark_exec_func<MatrixDense, double>(
        dense_start, dense_stop, dense_incr,
        make_A_dbl, execute_func, "matdense_subsetcolsmv_dbl"
    );

}

TEST_F(MatrixDense_Benchmark, SubsetcolsMatrixVectorMult_Single_BENCHMARK) {

    std::function<void (Benchmark_AccumulatingClock &, MatrixDense<float> &)> execute_func = [this] (
        Benchmark_AccumulatingClock &clock, MatrixDense<float> &A
    ) {
        Vector<float> b = Vector<float>::Random(BenchmarkBase::bundle, dense_subset_cols);
        clock.clock_start();
        A.mult_subset_cols(0, dense_subset_cols, b);
        clock.clock_stop();
    };

    benchmark_exec_func<MatrixDense, float>(
        dense_start, dense_stop, dense_incr,
        make_A_sgl, execute_func, "matdense_subsetcolsmv_sgl"
    );

}

TEST_F(MatrixDense_Benchmark, SubsetcolsMatrixVectorMult_Half_BENCHMARK) {

    std::function<void (Benchmark_AccumulatingClock &, MatrixDense<__half> &)> execute_func = [this] (
        Benchmark_AccumulatingClock &clock, MatrixDense<__half> &A
    ) {
        Vector<__half> b = Vector<__half>::Random(BenchmarkBase::bundle, dense_subset_cols);
        clock.clock_start();
        A.mult_subset_cols(0, dense_subset_cols, b);
        clock.clock_stop();
    };

    benchmark_exec_func<MatrixDense, __half>(
        dense_start, dense_stop, dense_incr,
        make_A_hlf, execute_func, "matdense_subsetcolsmv_hlf"
    );

}

TEST_F(MatrixDense_Benchmark, SubsetcolsTransposeMatrixVectorMult_Double_BENCHMARK) {

    std::function<void (Benchmark_AccumulatingClock &, MatrixDense<double> &)> execute_func = [this] (
        Benchmark_AccumulatingClock &clock, MatrixDense<double> &A
    ) {
        Vector<double> b = Vector<double>::Random(BenchmarkBase::bundle, A.rows());
        clock.clock_start();
        A.transpose_prod_subset_cols(0, dense_subset_cols, b);
        clock.clock_stop();
    };

    benchmark_exec_func<MatrixDense, double>(
        dense_start, dense_stop, dense_incr,
        make_A_dbl, execute_func, "matdense_subsetcolstmv_dbl"
    );

}

TEST_F(MatrixDense_Benchmark, SubsetcolsTransposeMatrixVectorMult_Single_BENCHMARK) {

    std::function<void (Benchmark_AccumulatingClock &, MatrixDense<float> &)> execute_func = [this] (
        Benchmark_AccumulatingClock &clock, MatrixDense<float> &A
    ) {
        Vector<float> b = Vector<float>::Random(BenchmarkBase::bundle, A.rows());
        clock.clock_start();
        A.transpose_prod_subset_cols(0, dense_subset_cols, b);
        clock.clock_stop();
    };

    benchmark_exec_func<MatrixDense, float>(
        dense_start, dense_stop, dense_incr,
        make_A_sgl, execute_func, "matdense_subsetcolstmv_sgl"
    );

}

TEST_F(MatrixDense_Benchmark, SubsetcolsTransposeMatrixVectorMult_Half_BENCHMARK) {

    std::function<void (Benchmark_AccumulatingClock &, MatrixDense<__half> &)> execute_func = [this] (
        Benchmark_AccumulatingClock &clock, MatrixDense<__half> &A
    ) {
        Vector<__half> b = Vector<__half>::Random(BenchmarkBase::bundle, A.rows());
        clock.clock_start();
        A.transpose_prod_subset_cols(0, dense_subset_cols, b);
        clock.clock_stop();
    };

    benchmark_exec_func<MatrixDense, __half>(
        dense_start, dense_stop, dense_incr,
        make_A_hlf, execute_func, "matdense_subsetcolstmv_hlf"
    );

}

TEST_F(MatrixDense_Benchmark, ForwardSubstitution_Double_BENCHMARK) {

    std::function<void (Benchmark_AccumulatingClock &, MatrixDense<double> &)> execute_func = [] (
        Benchmark_AccumulatingClock &clock, MatrixDense<double> &A
    ) {
        Vector<double> x = Vector<double>::Random(BenchmarkBase::bundle, A.rows());
        Vector<double> b = A*x;
        clock.clock_start();
        A.frwd_sub(b);
        clock.clock_stop();
    };

    benchmark_exec_func<MatrixDense, double>(
        dense_start, dense_stop, dense_incr,
        make_low_tri_A_dbl, execute_func, "matdense_frwdsub_dbl"
    );

}



TEST_F(MatrixDense_Benchmark, ForwardSubstitution_Single_BENCHMARK) {

    std::function<void (Benchmark_AccumulatingClock &, MatrixDense<float> &)> execute_func = [] (
        Benchmark_AccumulatingClock &clock, MatrixDense<float> &A
    ) {
        Vector<float> x = Vector<float>::Random(BenchmarkBase::bundle, A.rows());
        Vector<float> b = A*x;
        clock.clock_start();
        A.frwd_sub(b);
        clock.clock_stop();
    };

    benchmark_exec_func<MatrixDense, float>(
        dense_start, dense_stop, dense_incr,
        make_low_tri_A_sgl, execute_func, "matdense_frwdsub_sgl"
    );

}

TEST_F(MatrixDense_Benchmark, ForwardSubstitution_Half_BENCHMARK) {

    std::function<void (Benchmark_AccumulatingClock &, MatrixDense<__half> &)> execute_func = [] (
        Benchmark_AccumulatingClock &clock, MatrixDense<__half> &A
    ) {
        Vector<__half> x = Vector<__half>::Random(BenchmarkBase::bundle, A.rows());
        Vector<__half> b = A*x;
        clock.clock_start();
        A.frwd_sub(b);
        clock.clock_stop();
    };

    benchmark_exec_func<MatrixDense, __half>(
        dense_start, dense_stop, dense_incr,
        make_low_tri_A_hlf, execute_func, "matdense_frwdsub_hlf"
    );

}

TEST_F(MatrixDense_Benchmark, BackwardSubstitution_Double_BENCHMARK) {

    std::function<void (Benchmark_AccumulatingClock &, MatrixDense<double> &)> execute_func = [] (
        Benchmark_AccumulatingClock &clock, MatrixDense<double> &A
    ) {
        Vector<double> x = Vector<double>::Random(BenchmarkBase::bundle, A.rows());
        Vector<double> b = A*x;
        clock.clock_start();
        A.back_sub(b);
        clock.clock_stop();
    };

    benchmark_exec_func<MatrixDense, double>(
        dense_start, dense_stop, dense_incr,
        make_upp_tri_A_dbl, execute_func, "matdense_backsub_dbl"
    );

}

TEST_F(MatrixDense_Benchmark, BackwardSubstitution_Single_BENCHMARK) {

    std::function<void (Benchmark_AccumulatingClock &, MatrixDense<float> &)> execute_func = [] (
        Benchmark_AccumulatingClock &clock, MatrixDense<float> &A
    ) {
        Vector<float> x = Vector<float>::Random(BenchmarkBase::bundle, A.rows());
        Vector<float> b = A*x;
        clock.clock_start();
        A.back_sub(b);
        clock.clock_stop();
    };

    benchmark_exec_func<MatrixDense, float>(
        dense_start, dense_stop, dense_incr,
        make_upp_tri_A_sgl, execute_func, "matdense_backsub_sgl"
    );

}

TEST_F(MatrixDense_Benchmark, BackwardSubstitution_Half_BENCHMARK) {

    std::function<void (Benchmark_AccumulatingClock &, MatrixDense<__half> &)> execute_func = [] (
        Benchmark_AccumulatingClock &clock, MatrixDense<__half> &A
    ) {
        Vector<__half> x = Vector<__half>::Random(BenchmarkBase::bundle, A.rows());
        Vector<__half> b = A*x;
        clock.clock_start();
        A.back_sub(b);
        clock.clock_stop();
    };

    benchmark_exec_func<MatrixDense, __half>(
        dense_start, dense_stop, dense_incr,
        make_upp_tri_A_hlf, execute_func, "matdense_backsub_hlf"
    );

}
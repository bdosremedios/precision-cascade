#include <functional>
#include <fstream>

#include <cmath>

#include "types/types.h"

#include "../test.h"

#include "include/benchmark_Sparse.h"

class NoFillMatrixSparse_Benchmark: public Benchmark_Sparse
{
public:

    std::function<NoFillMatrixSparse<double> (int, int)> make_low_tri_A = [this] (
        int m, int n
    ) -> NoFillMatrixSparse<double> {
        return make_lower_tri(make_A_dbl(m, n));
    };
    
    std::function<NoFillMatrixSparse<double> (int, int)> make_upp_tri_A = [this] (
        int m, int n
    ) -> NoFillMatrixSparse<double> {
        return make_upper_tri(make_A_dbl(m, n));
    };

};

TEST_F(NoFillMatrixSparse_Benchmark, MatrixVectorMult_Double_BENCHMARK) {

    std::function<void (Benchmark_AccumulatingClock &, NoFillMatrixSparse<double> &)> execute_func = [] (
        Benchmark_AccumulatingClock &clock, NoFillMatrixSparse<double> &A
    ) {
        Vector<double> b = Vector<double>::Random(TestBase::bundle, A.rows());
        clock.clock_start();
        A*b;
        clock.clock_stop();
    };

    benchmark_exec_func<NoFillMatrixSparse, double>(
        sparse_start, sparse_stop, sparse_incr, make_A_dbl, execute_func, "matsparse_mv_dbl"
    );

}

TEST_F(NoFillMatrixSparse_Benchmark, MatrixVectorMult_Single_BENCHMARK) {

    std::function<void (Benchmark_AccumulatingClock &, NoFillMatrixSparse<float> &)> execute_func = [] (
        Benchmark_AccumulatingClock &clock, NoFillMatrixSparse<float> &A
    ) {
        Vector<float> b = Vector<float>::Random(TestBase::bundle, A.rows());
        clock.clock_start();
        A*b;
        clock.clock_stop();
    };

    benchmark_exec_func<NoFillMatrixSparse, float>(
        sparse_start, sparse_stop, sparse_incr, make_A_sgl, execute_func, "matsparse_mv_sgl"
    );

}

TEST_F(NoFillMatrixSparse_Benchmark, MatrixVectorMult_Half_BENCHMARK) {

    std::function<void (Benchmark_AccumulatingClock &, NoFillMatrixSparse<__half> &)> execute_func = [] (
        Benchmark_AccumulatingClock &clock, NoFillMatrixSparse<__half> &A
    ) {
        Vector<__half> b = Vector<__half>::Random(TestBase::bundle, A.rows());
        clock.clock_start();
        A*b;
        clock.clock_stop();
    };

    benchmark_exec_func<NoFillMatrixSparse, __half>(
        sparse_start, sparse_stop, sparse_incr, make_A_hlf, execute_func, "matsparse_mv_hlf"
    );

}

TEST_F(NoFillMatrixSparse_Benchmark, TransposeMatrixVectorMult_Double_BENCHMARK) {

    std::function<void (Benchmark_AccumulatingClock &, NoFillMatrixSparse<double> &)> execute_func = [] (
        Benchmark_AccumulatingClock &clock, NoFillMatrixSparse<double> &A
    ) {
        Vector<double> b = Vector<double>::Random(TestBase::bundle, A.rows());
        clock.clock_start();
        A.transpose_prod(b);
        clock.clock_stop();
    };

    benchmark_exec_func<NoFillMatrixSparse, double>(
        sparse_start, sparse_stop, sparse_incr, make_A_dbl, execute_func, "matsparse_tmv_dbl"
    );

}

TEST_F(NoFillMatrixSparse_Benchmark, TransposeMatrixVectorMult_Single_BENCHMARK) {

    std::function<void (Benchmark_AccumulatingClock &, NoFillMatrixSparse<float> &)> execute_func = [] (
        Benchmark_AccumulatingClock &clock, NoFillMatrixSparse<float> &A
    ) {
        Vector<float> b = Vector<float>::Random(TestBase::bundle, A.rows());
        clock.clock_start();
        A.transpose_prod(b);
        clock.clock_stop();
    };

    benchmark_exec_func<NoFillMatrixSparse, float>(
        sparse_start, sparse_stop, sparse_incr, make_A_sgl, execute_func, "matsparse_tmv_sgl"
    );

}

TEST_F(NoFillMatrixSparse_Benchmark, TransposeMatrixVectorMult_Half_BENCHMARK) {

    std::function<void (Benchmark_AccumulatingClock &, NoFillMatrixSparse<__half> &)> execute_func = [] (
        Benchmark_AccumulatingClock &clock, NoFillMatrixSparse<__half> &A
    ) {
        Vector<__half> b = Vector<__half>::Random(TestBase::bundle, A.rows());
        clock.clock_start();
        A.transpose_prod(b);
        clock.clock_stop();
    };

    benchmark_exec_func<NoFillMatrixSparse, __half>(
        sparse_start, sparse_stop, sparse_incr, make_A_hlf, execute_func, "matsparse_tmv_hlf"
    );

}

TEST_F(NoFillMatrixSparse_Benchmark, MatrixBlockVectorMult_BENCHMARK) {
}

TEST_F(NoFillMatrixSparse_Benchmark, TransposeMatrixBlockVectorMult_BENCHMARK) {
}

TEST_F(NoFillMatrixSparse_Benchmark, ForwardSubstitution_BENCHMARK) {

    std::function<void (Benchmark_AccumulatingClock &, NoFillMatrixSparse<double> &)> execute_func = [] (
        Benchmark_AccumulatingClock &clock, NoFillMatrixSparse<double> &A
    ) {
        Vector<double> x = Vector<double>::Random(TestBase::bundle, A.rows());
        Vector<double> b = A*x;
        clock.clock_start();
        A.frwd_sub(b);
        clock.clock_stop();
    };

    benchmark_exec_func<NoFillMatrixSparse, double>(
        sparse_start, sparse_stop, sparse_incr, make_low_tri_A, execute_func, "matsparse_frwdsub"
    );

}

TEST_F(NoFillMatrixSparse_Benchmark, BackwardSubstitution_BENCHMARK) {

    std::function<void (Benchmark_AccumulatingClock &, NoFillMatrixSparse<double> &)> execute_func = [] (
        Benchmark_AccumulatingClock &clock, NoFillMatrixSparse<double> &A
    ) {
        Vector<double> x = Vector<double>::Random(TestBase::bundle, A.rows());
        Vector<double> b = A*x;
        clock.clock_start();
        A.back_sub(b);
        clock.clock_stop();
    };

    benchmark_exec_func<NoFillMatrixSparse, double>(
        sparse_start, sparse_stop, sparse_incr, make_upp_tri_A, execute_func, "matsparse_backsub"
    );

}
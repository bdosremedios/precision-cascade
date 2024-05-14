#include "../test.h"
#include "include/benchmark_Matrix.h"

#include "types/types.h"

class MatrixDense_Benchmark: public Benchmark_Matrix
{
public:

    std::function<MatrixDense<double> (int, int)> make_A = [this] (
        int m, int n
    ) -> MatrixDense<double> {
        return MatrixDense<double>::Random(TestBase::bundle, m, m);
    };

    std::function<MatrixDense<double> (int, int)> make_low_tri_A = [this] (
        int m, int n
    ) -> MatrixDense<double> {
        return MatrixDense<double>::Random_LT(TestBase::bundle, m, m);
    };
    
    std::function<MatrixDense<double> (int, int)> make_upp_tri_A = [this] (
        int m, int n
    ) -> MatrixDense<double> {
        return MatrixDense<double>::Random_UT(TestBase::bundle, m, m);
    };

};

TEST_F(MatrixDense_Benchmark, MatrixVectorMult_BENCHMARK) {

    std::function<void (Benchmark_AccumulatingClock &, MatrixDense<double> &)> execute_func = [] (
        Benchmark_AccumulatingClock &clock, MatrixDense<double> &A
    ) {
        Vector<double> b = Vector<double>::Random(TestBase::bundle, A.rows());
        clock.clock_start();
        A*b;
        clock.clock_stop();
    };

    benchmark_exec_func<MatrixDense, double>(
        dense_start, dense_stop, dense_incr,
        make_A, execute_func, "matdense_mv"
    );

}

TEST_F(MatrixDense_Benchmark, TransposeMatrixVectorMult_BENCHMARK) {

    std::function<void (Benchmark_AccumulatingClock &, MatrixDense<double> &)> execute_func = [] (
        Benchmark_AccumulatingClock &clock, MatrixDense<double> &A
    ) {
        Vector<double> b = Vector<double>::Random(TestBase::bundle, A.rows());
        clock.clock_start();
        A.transpose_prod(b);
        clock.clock_stop();
    };

    benchmark_exec_func<MatrixDense, double>(
        dense_start, dense_stop, dense_incr,
        make_A, execute_func, "matdense_tmv"
    );

}

TEST_F(MatrixDense_Benchmark, SubsetcolsMatrixVectorMult_BENCHMARK) {

}

TEST_F(MatrixDense_Benchmark, SubsetcolsTransposeMatrixVectorMult_BENCHMARK) {

}

TEST_F(MatrixDense_Benchmark, ForwardSubstitution_BENCHMARK) {

    std::function<void (Benchmark_AccumulatingClock &, MatrixDense<double> &)> execute_func = [] (
        Benchmark_AccumulatingClock &clock, MatrixDense<double> &A
    ) {
        Vector<double> x = Vector<double>::Random(TestBase::bundle, A.rows());
        Vector<double> b = A*x;
        clock.clock_start();
        A.frwd_sub(b);
        clock.clock_stop();
    };

    benchmark_exec_func<MatrixDense, double>(
        dense_start, dense_stop, dense_incr,
        make_low_tri_A, execute_func, "matdense_frwdsub"
    );

}

TEST_F(MatrixDense_Benchmark, BackwardSubstitution_BENCHMARK) {

    std::function<void (Benchmark_AccumulatingClock &, MatrixDense<double> &)> execute_func = [] (
        Benchmark_AccumulatingClock &clock, MatrixDense<double> &A
    ) {
        Vector<double> x = Vector<double>::Random(TestBase::bundle, A.rows());
        Vector<double> b = A*x;
        clock.clock_start();
        A.back_sub(b);
        clock.clock_stop();
    };

    benchmark_exec_func<MatrixDense, double>(
        dense_start, dense_stop, dense_incr,
        make_upp_tri_A, execute_func, "matdense_backsub"
    );

}
#include "../test.h"
#include "include/benchmark_Matrix.h"

#include <functional>
#include <fstream>

#include <cmath>

#include "types/types.h"

class MatrixDense_Benchmark_Test: public Benchmark_Matrix_Test
{
public:

    std::function<MatrixDense<double> (int, int)> make_A = [this] (
        int m, int n
    ) -> MatrixDense<double> {
        return MatrixDense<double>::Random(TestBase::bundle, m, m);
    };

};

TEST_F(MatrixDense_Benchmark_Test, MatrixVectorMult_BENCHMARK) {

    std::function<void (MatrixDense<double> &, Vector<double> &)> execute_func = [] (
        MatrixDense<double> &A, Vector<double> &b
    ) {
        A*b;
    };

    matrix_mult_func_benchmark<MatrixDense>(10, 14, make_A, execute_func, "matdense_mv");

}

TEST_F(MatrixDense_Benchmark_Test, TransposeMatrixVectorMult_BENCHMARK) {

    std::function<void (MatrixDense<double> &, Vector<double> &)> execute_func = [] (
        MatrixDense<double> &A, Vector<double> &b
    ) {
        A.transpose_prod(b);
    };

    matrix_mult_func_benchmark<MatrixDense>(10, 14, make_A, execute_func, "matdense_tmv");

}

TEST_F(MatrixDense_Benchmark_Test, MatrixBlockVectorMult_BENCHMARK) {

}

TEST_F(MatrixDense_Benchmark_Test, TransposeMatrixBlockVectorMult_BENCHMARK) {

}

TEST_F(MatrixDense_Benchmark_Test, ForwardSubstitution_BENCHMARK) {

}

TEST_F(MatrixDense_Benchmark_Test, BackwardSubstitution_BENCHMARK) {

}


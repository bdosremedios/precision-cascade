#include "../test.h"
#include "include/benchmark_Matrix.h"

#include <functional>
#include <fstream>

#include <cmath>

#include "types/types.h"

class NoFillMatrixSparse_Benchmark_Test: public Benchmark_Matrix_Test
{
public:

    const double col_non_zeros = 1000.;

    std::function<NoFillMatrixSparse<double> (int, int)> make_A = [this] (
        int m, int n
    ) -> NoFillMatrixSparse<double> {
        return NoFillMatrixSparse<double>::Random(
            TestBase::bundle, m, m,
            (col_non_zeros/static_cast<double>(m) > 1) ? 1 : col_non_zeros/static_cast<double>(m)
        );
    };

};

TEST_F(NoFillMatrixSparse_Benchmark_Test, MatrixVectorMult_BENCHMARK)
{

    std::function<void (NoFillMatrixSparse<double> &, Vector<double> &)> execute_func = [] (
        NoFillMatrixSparse<double> &A, Vector<double> &b
    ) {
        A*b;
    };

    matrix_mult_func_benchmark<NoFillMatrixSparse>(10, 15, make_A, execute_func, "matsparse_mv");

}

TEST_F(NoFillMatrixSparse_Benchmark_Test, TransposeMatrixVectorMult_BENCHMARK)
{

    std::function<void (NoFillMatrixSparse<double> &, Vector<double> &)> execute_func = [] (
        NoFillMatrixSparse<double> &A, Vector<double> &b
    ) {
        A.transpose_prod(b);
    };

    matrix_mult_func_benchmark<NoFillMatrixSparse>(10, 15, make_A, execute_func, "matsparse_tmv");

}

TEST_F(NoFillMatrixSparse_Benchmark_Test, MatrixBlockVectorMult_BENCHMARK)
{
}

TEST_F(NoFillMatrixSparse_Benchmark_Test, TransposeMatrixBlockVectorMult_BENCHMARK)
{
}

TEST_F(NoFillMatrixSparse_Benchmark_Test, ForwardSubstitution_BENCHMARK)
{
}

TEST_F(NoFillMatrixSparse_Benchmark_Test, BackwardSubstitution_BENCHMARK)
{
}
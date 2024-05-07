#include "../test.h"
#include "include/benchmark_Matrix.h"

#include <functional>
#include <fstream>

#include <cmath>

#include "types/types.h"

class MatrixDense_Benchmark_Test: public Benchmark_Matrix_Test
{
public:

    int min_n_mult = 8;
    int max_n_mult = 14-prototying_n_speed_up;

    int min_n_substitution = 7;
    int max_n_substitution = 12-prototying_n_speed_up;

    std::function<MatrixDense<double> (int, int)> make_A = [this] (
        int m, int n
    ) -> MatrixDense<double> {
        return MatrixDense<double>::Random(TestBase::bundle, m, m);
    };

    std::function<MatrixDense<double> (int, int)> make_low_tri_A = [this] (
        int m, int n
    ) -> MatrixDense<double> {
        return make_lower_tri(make_A(m, n));
    };
    
    std::function<MatrixDense<double> (int, int)> make_upp_tri_A = [this] (
        int m, int n
    ) -> MatrixDense<double> {
        return make_upper_tri(make_A(m, n));
    };

};

TEST_F(MatrixDense_Benchmark_Test, MatrixVectorMult_BENCHMARK) {

    std::function<void (MatrixDense<double> &, Vector<double> &)> execute_func = [] (
        MatrixDense<double> &A, Vector<double> &b
    ) {
        A*b;
    };

    basic_matrix_func_benchmark<MatrixDense>(
        min_n_mult, max_n_mult, make_A, execute_func, "matdense_mv"
    );

}

TEST_F(MatrixDense_Benchmark_Test, TransposeMatrixVectorMult_BENCHMARK) {

    std::function<void (MatrixDense<double> &, Vector<double> &)> execute_func = [] (
        MatrixDense<double> &A, Vector<double> &b
    ) {
        A.transpose_prod(b);
    };

    basic_matrix_func_benchmark<MatrixDense>(
        min_n_mult, max_n_mult, make_A, execute_func, "matdense_tmv"
    );

}

TEST_F(MatrixDense_Benchmark_Test, MatrixBlockVectorMult_BENCHMARK) {

}

TEST_F(MatrixDense_Benchmark_Test, TransposeMatrixBlockVectorMult_BENCHMARK) {

}

TEST_F(MatrixDense_Benchmark_Test, ForwardSubstitution_BENCHMARK) {

    std::function<void (MatrixDense<double> &, Vector<double> &)> execute_func = [] (
        MatrixDense<double> &A, Vector<double> &b
    ) {
        A.frwd_sub(b);
    };

    basic_matrix_func_benchmark<MatrixDense>(
        min_n_substitution, max_n_substitution, make_low_tri_A, execute_func, "matdense_frwdsub"
    );

}

TEST_F(MatrixDense_Benchmark_Test, BackwardSubstitution_BENCHMARK) {

    std::function<void (MatrixDense<double> &, Vector<double> &)> execute_func = [] (
        MatrixDense<double> &A, Vector<double> &b
    ) {
        A.back_sub(b);
    };

    basic_matrix_func_benchmark<MatrixDense>(
        min_n_substitution, max_n_substitution, make_upp_tri_A, execute_func, "matdense_backsub"
    );

}


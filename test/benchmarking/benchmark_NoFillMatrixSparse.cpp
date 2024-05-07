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

    int min_n_mult = 8;
    int max_n_mult = 15-prototying_n_speed_up;

    int min_n_substitution = 7;
    int max_n_substitution = 13-prototying_n_speed_up;

    std::function<NoFillMatrixSparse<double> (int, int)> make_A = [this] (
        int m, int n
    ) -> NoFillMatrixSparse<double> {
        return NoFillMatrixSparse<double>::Random(
            TestBase::bundle, m, m,
            (col_non_zeros/static_cast<double>(m) > 1) ? 1 : col_non_zeros/static_cast<double>(m)
        );
    };

    std::function<NoFillMatrixSparse<double> (int, int)> make_low_tri_A = [this] (
        int m, int n
    ) -> NoFillMatrixSparse<double> {
        return make_lower_tri(make_A(m, n));
    };
    
    std::function<NoFillMatrixSparse<double> (int, int)> make_upp_tri_A = [this] (
        int m, int n
    ) -> NoFillMatrixSparse<double> {
        return make_upper_tri(make_A(m, n));
    };

};

TEST_F(NoFillMatrixSparse_Benchmark_Test, MatrixVectorMult_BENCHMARK) {

    std::function<void (NoFillMatrixSparse<double> &, Vector<double> &)> execute_func = [] (
        NoFillMatrixSparse<double> &A, Vector<double> &b
    ) {
        A*b;
    };

    basic_matrix_func_benchmark<NoFillMatrixSparse>(
        min_n_mult, max_n_mult, make_A, execute_func, "matsparse_mv"
    );

}

TEST_F(NoFillMatrixSparse_Benchmark_Test, TransposeMatrixVectorMult_BENCHMARK) {

    std::function<void (NoFillMatrixSparse<double> &, Vector<double> &)> execute_func = [] (
        NoFillMatrixSparse<double> &A, Vector<double> &b
    ) {
        A.transpose_prod(b);
    };

    basic_matrix_func_benchmark<NoFillMatrixSparse>(
        min_n_mult, max_n_mult, make_A, execute_func, "matsparse_tmv"
    );

}

TEST_F(NoFillMatrixSparse_Benchmark_Test, MatrixBlockVectorMult_BENCHMARK) {
}

TEST_F(NoFillMatrixSparse_Benchmark_Test, TransposeMatrixBlockVectorMult_BENCHMARK) {
}

TEST_F(NoFillMatrixSparse_Benchmark_Test, ForwardSubstitution_BENCHMARK) {

    std::function<void (NoFillMatrixSparse<double> &, Vector<double> &)> execute_func = [] (
        NoFillMatrixSparse<double> &A, Vector<double> &b
    ) {
        A.frwd_sub(b);
    };

    basic_matrix_func_benchmark<NoFillMatrixSparse>(
        min_n_substitution, max_n_substitution, make_low_tri_A, execute_func, "matsparse_frwdsub"
    );

}

TEST_F(NoFillMatrixSparse_Benchmark_Test, BackwardSubstitution_BENCHMARK) {

    std::function<void (NoFillMatrixSparse<double> &, Vector<double> &)> execute_func = [] (
        NoFillMatrixSparse<double> &A, Vector<double> &b
    ) {
        A.back_sub(b);
    };

    basic_matrix_func_benchmark<NoFillMatrixSparse>(
        min_n_substitution, max_n_substitution, make_upp_tri_A, execute_func, "matsparse_backsub"
    );

}
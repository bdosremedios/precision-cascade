#ifndef BENCHMARK_SPARSE_NESTED_H
#define BENCHMARK_SPARSE_NESTED_H

#include "benchmark_Sparse.h"

class Benchmark_Sparse_Nested: public Benchmark_Sparse
{
private:

    const double col_non_zeros = 1000.;

public:

    const int nested_outer_iter = 100;
    const int nested_inner_iter = 50;

    int nested_n_min = 11;
    int nested_n_max = 17-prototying_n_speed_up;

    std::function<NoFillMatrixSparse<double> (int, int)> make_norm_A = [this] (
        int m, int n
    ) -> NoFillMatrixSparse<double> {
        NoFillMatrixSparse<double> mat = NoFillMatrixSparse<double>::Random(
            TestBase::bundle, m, m,
            (col_non_zeros/static_cast<double>(m) > 1) ? 1 : col_non_zeros/static_cast<double>(m)
        );
        mat.normalize_magnitude();
        return mat;
    };

};

#endif
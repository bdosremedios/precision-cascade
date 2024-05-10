#ifndef BENCHMARK_SPARSE_H
#define BENCHMARK_SPARSE_H

#include "types/types.h"

#include "benchmark_Matrix.h"

class Benchmark_Sparse: public Benchmark_Matrix
{
private:

    const double col_non_zeros = 1000.;

public:

    std::function<NoFillMatrixSparse<double> (int, int)> make_A_dbl = [this] (
        int m, int n
    ) -> NoFillMatrixSparse<double> {
        return NoFillMatrixSparse<double>::Random(
            TestBase::bundle, m, m,
            (col_non_zeros/static_cast<double>(m) > 1) ? 1 : col_non_zeros/static_cast<double>(m)
        );
    };

    std::function<NoFillMatrixSparse<float> (int, int)> make_A_sgl = [this] (
        int m, int n
    ) -> NoFillMatrixSparse<float> {
        return NoFillMatrixSparse<float>::Random(
            TestBase::bundle, m, m,
            (static_cast<float>(col_non_zeros)/static_cast<float>(m) > 1) ?
             static_cast<float>(1) : static_cast<float>(col_non_zeros)/static_cast<float>(m)
        );
    };

    std::function<NoFillMatrixSparse<__half> (int, int)> make_A_hlf = [this] (
        int m, int n
    ) -> NoFillMatrixSparse<__half> {
        return NoFillMatrixSparse<__half>::Random(
            TestBase::bundle, m, m,
            (static_cast<__half>(col_non_zeros)/static_cast<__half>(m) > static_cast<__half>(1)) ?
             static_cast<__half>(1) : static_cast<__half>(col_non_zeros)/static_cast<__half>(m)
        );
    };

};

#endif
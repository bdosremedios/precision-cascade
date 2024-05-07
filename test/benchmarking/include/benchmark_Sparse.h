#ifndef BENCHMARK_SPARSE_H
#define BENCHMARK_SPARSE_H

#include "types/types.h"

#include "benchmark_Matrix.h"

class Benchmark_Sparse: public Benchmark_Matrix
{
private:

    const double col_non_zeros = 1000.;

public:

    std::function<NoFillMatrixSparse<double> (int, int)> make_A = [this] (
        int m, int n
    ) -> NoFillMatrixSparse<double> {
        return NoFillMatrixSparse<double>::Random(
            TestBase::bundle, m, m,
            (col_non_zeros/static_cast<double>(m) > 1) ? 1 : col_non_zeros/static_cast<double>(m)
        );
    };

};

#endif
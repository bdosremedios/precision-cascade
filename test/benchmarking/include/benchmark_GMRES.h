#ifndef BENCHMARK_GMRES_H
#define BENCHMARK_GMRES_H

#include "benchmark.h"

class Benchmark_GMRES: public BenchmarkBase
{
private:

    const double nested_col_non_zeros = 1000.;

public:
    
    double gmressolve_iters = 30;

    std::function<NoFillMatrixSparse<double> (int, int)> make_norm_A = [this] (
        int m, int n
    ) -> NoFillMatrixSparse<double> {
        NoFillMatrixSparse<double> mat = NoFillMatrixSparse<double>::Random(
            TestBase::bundle, m, m,
            (nested_col_non_zeros/static_cast<double>(m) > 1) ? 1 : nested_col_non_zeros/static_cast<double>(m)
        );
        mat.normalize_magnitude();
        return mat;
    };

};

#endif
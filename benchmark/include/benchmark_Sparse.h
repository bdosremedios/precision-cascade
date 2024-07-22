#ifndef BENCHMARK_SPARSE_H
#define BENCHMARK_SPARSE_H

#include "benchmark.h"

#include "types/types.h"

class Benchmark_Sparse: public BenchmarkBase
{
public:

    std::function<NoFillMatrixSparse<double> (int, int)> make_A_dbl = [this] (
        int m, int n
    ) -> NoFillMatrixSparse<double> {

        if (m != n) {
            throw std::runtime_error(
                "make_A_dbl: require square dimensions"
            );
        }

        return NoFillMatrixSparse<double>::Random(
            BenchmarkBase::bundle, m, m,
            ((sqrt(static_cast<double>(m))/static_cast<double>(m) > 1) ?
             1 :
             sqrt(static_cast<double>(m))/static_cast<double>(m))
        );

    };

    std::function<NoFillMatrixSparse<float> (int, int)> make_A_sgl = [this] (
        int m, int n
    ) -> NoFillMatrixSparse<float> {
        return make_A_dbl(m, n).cast<float>();
    };

    std::function<NoFillMatrixSparse<__half> (int, int)> make_A_hlf = [this] (
        int m, int n
    ) -> NoFillMatrixSparse<__half> {
        return make_A_dbl(m, n).cast<__half>();
    };

    std::function<NoFillMatrixSparse<double> (int, int)> make_low_tri_A_dbl = [this] (
        int m, int n
    ) -> NoFillMatrixSparse<double> {

        if (m != n) {
            throw std::runtime_error(
                "make_low_tri_A_dbl: require square dimensions"
            );
        }

        return NoFillMatrixSparse<double>::Random_LT(
            BenchmarkBase::bundle, m, m,
            (sqrt(static_cast<double>(m))/static_cast<double>(m) > 1) ?
            1 :
            sqrt(static_cast<double>(m))/static_cast<double>(m)
        );

    };

    std::function<NoFillMatrixSparse<float> (int, int)> make_low_tri_A_sgl = [this] (
        int m, int n
    ) -> NoFillMatrixSparse<float> {
        return make_low_tri_A_dbl(m, n).cast<float>();
    };

    std::function<NoFillMatrixSparse<__half> (int, int)> make_low_tri_A_hlf = [this] (
        int m, int n
    ) -> NoFillMatrixSparse<__half> {
        return make_low_tri_A_dbl(m, n).cast<__half>();
    };
    
    std::function<NoFillMatrixSparse<double> (int, int)> make_upp_tri_A_dbl = [this] (
        int m, int n
    ) -> NoFillMatrixSparse<double> {

        if (m != n) {
            throw std::runtime_error(
                "make_upp_tri_A_dbl: require square dimensions"
            );
        }

        return NoFillMatrixSparse<double>::Random_UT(
            BenchmarkBase::bundle, m, m,
            (sqrt(static_cast<double>(m))/static_cast<double>(m) > 1) ?
            1 :
            sqrt(static_cast<double>(m))/static_cast<double>(m)
        );

    };

    std::function<NoFillMatrixSparse<float> (int, int)> make_upp_tri_A_sgl = [this] (
        int m, int n
    ) -> NoFillMatrixSparse<float> {
        return make_upp_tri_A_dbl(m, n).cast<float>();
    };

    std::function<NoFillMatrixSparse<__half> (int, int)> make_upp_tri_A_hlf = [this] (
        int m, int n
    ) -> NoFillMatrixSparse<__half> {
        return make_upp_tri_A_dbl(m, n).cast<__half>();
    };

};

#endif
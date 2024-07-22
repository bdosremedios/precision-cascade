#ifndef BENCHMARK_DENSE_H
#define BENCHMARK_DENSE_H

#include "benchmark.h"

#include <filesystem>
#include <fstream>
#include <functional>

namespace fs = std::filesystem;

class Benchmark_Dense: public BenchmarkBase
{
public:

    std::function<MatrixDense<double> (int, int)> make_A_dbl = [this] (
        int m, int n
    ) -> MatrixDense<double> {

        if (m != n) {
            throw std::runtime_error(
                "make_A_dbl: require square dimensions"
            );
        }

        return MatrixDense<double>::Random(BenchmarkBase::bundle, m, m);

    };

    std::function<MatrixDense<float> (int, int)> make_A_sgl = [this] (
        int m, int n
    ) -> MatrixDense<float> {
        return make_A_dbl(m, m).cast<float>();
    };

    std::function<MatrixDense<__half> (int, int)> make_A_hlf = [this] (
        int m, int n
    ) -> MatrixDense<__half> {
        return make_A_dbl(m, m).cast<__half>();
    };

    std::function<MatrixDense<double> (int, int)> make_low_tri_A_dbl = [this] (
        int m, int n
    ) -> MatrixDense<double> {

        if (m != n) {
            throw std::runtime_error(
                "make_low_tri_A_dbl: require square dimensions"
            );
        }

        return MatrixDense<double>::Random_LT(BenchmarkBase::bundle, m, m);

    };

    std::function<MatrixDense<float> (int, int)> make_low_tri_A_sgl = [this] (
        int m, int n
    ) -> MatrixDense<float> {
        return make_low_tri_A_dbl(m, m).cast<float>();
    };

    std::function<MatrixDense<__half> (int, int)> make_low_tri_A_hlf = [this] (
        int m, int n
    ) -> MatrixDense<__half> {
        return make_low_tri_A_dbl(m, m).cast<__half>();
    };
    
    std::function<MatrixDense<double> (int, int)> make_upp_tri_A_dbl = [this] (
        int m, int n
    ) -> MatrixDense<double> {

        if (m != n) {
            throw std::runtime_error(
                "make_upp_tri_A_dbl: require square dimensions"
            );
        }

        return MatrixDense<double>::Random_UT(BenchmarkBase::bundle, m, m);

    };
    
    std::function<MatrixDense<float> (int, int)> make_upp_tri_A_sgl = [this] (
        int m, int n
    ) -> MatrixDense<float> {
        return make_upp_tri_A_dbl(m, m).cast<float>();
    };
    
    std::function<MatrixDense<__half> (int, int)> make_upp_tri_A_hlf = [this] (
        int m, int n
    ) -> MatrixDense<__half> {
        return make_upp_tri_A_dbl(m, m).cast<__half>();
    };

};

#endif
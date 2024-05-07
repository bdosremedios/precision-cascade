#ifndef BENCHMARK_MATRIX_H
#define BENCHMARK_MATRIX_H

#include <filesystem>
#include <fstream>

#include "benchmark.h"

namespace fs = std::filesystem;

class Benchmark_Matrix: public BenchmarkBase
{
public:

    template <template <typename> typename M>
    M<double> make_lower_tri(M<double> mat) {

        MatrixDense<double> temp_dense(mat);

        for (int j=0; j<mat.cols(); ++j) {
            for (int i=0; i<j; ++i) {
                temp_dense.set_elem(i, j, SCALAR_ZERO<double>::get());
            }
        }

        return M<double>(mat);

    }

    template <template <typename> typename M>
    M<double> make_upper_tri(M<double> mat) {

        MatrixDense<double> temp_dense(mat);

        for (int j=0; j<mat.cols(); ++j) {
            for (int i=j+1; i<mat.rows(); ++i) {
                temp_dense.set_elem(i, j, SCALAR_ZERO<double>::get());
            }
        }

        return M<double>(mat);

    }

};

#endif
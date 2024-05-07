#ifndef BENCHMARK_MATRIX_H
#define BENCHMARK_MATRIX_H

#include <filesystem>
#include <fstream>

#include "benchmark.h"

namespace fs = std::filesystem;

class Benchmark_Matrix_Test: public BenchmarkTestBase
{
private:

    const int n_runs = 10;

protected:

    const int prototying_n_speed_up = 1;

public:

    template <template <typename> typename M>
    void basic_matrix_func_benchmark(
        int two_pow_n_start,
        int two_pow_n_end,
        std::function<M<double> (int, int)> make_A,
        std::function<void (M<double> &, Vector<double> &)> execute_func,
        std::string label
    ) {

        fs::path file_path = data_dir / fs::path(label + ".csv");
        std::ofstream f_out;
        f_out.open(file_path);

        if (!(f_out.is_open())) {
            throw std::runtime_error(
                std::format(
                    "basic_matrix_func_benchmark: {} did not open",
                    file_path.string()
                )
            );
        }

        for (int n=two_pow_n_start; n<=two_pow_n_end; ++n) {

            Benchmark_AccumulatingClock curr_clock;

            int m = std::pow(2, n);

            M<double> A = make_A(m, m);

            std::function<void(Benchmark_AccumulatingClock &)> test_func = [m, execute_func, &A](
                Benchmark_AccumulatingClock &arg_clock
            ) {
                Vector<double> b = Vector<double>::Random(TestBase::bundle, m);
                arg_clock.clock_start();
                execute_func(A, b);
                arg_clock.clock_stop();
            };

            benchmark_n_runs(
                n_runs,
                test_func,
                curr_clock,
                label + std::format("_2^{:d}", n)
            );

            f_out << std::format("{},{}", m, curr_clock.get_median().count()) << std::endl; 

        }

        f_out.close();

    }

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
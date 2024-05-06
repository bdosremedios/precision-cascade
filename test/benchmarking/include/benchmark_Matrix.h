#ifndef BENCHMARK_MATRIX_H
#define BENCHMARK_MATRIX_H

#include "benchmark.h"

class Benchmark_Matrix_Test: public BenchmarkTestBase
{
private:

    const int n_runs = 10;

public:

    template <template <typename> typename M>
    void matrix_mult_func_benchmark(
        int two_pow_n_start,
        int two_pow_n_end,
        std::function<M<double> (int, int)> make_A,
        std::function<void (M<double> &, Vector<double> &)> execute_func,
        std::string label
    ) {

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

        }

    }

};

#endif
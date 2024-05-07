#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "../../test.h"

#include <filesystem>
#include <fstream>

#include <chrono>
#include <functional>

namespace fs = std::filesystem;

class Benchmark_AccumulatingClock
{
private:

    bool clock_ticking = false;

    using acc_clock_duration = std::chrono::microseconds;

    std::chrono::steady_clock clock;
    std::chrono::time_point<std::chrono::steady_clock> start;
    std::chrono::time_point<std::chrono::steady_clock> stop;

    std::vector<acc_clock_duration> prev_durations;

public:

    Benchmark_AccumulatingClock() {}

    void clock_start();
    void clock_stop();

    int get_count();
    acc_clock_duration get_avg();
    acc_clock_duration get_median();
    acc_clock_duration get_total();

};

class BenchmarkBase: public TestBase
{
protected:

    const int n_runs = 10;
    const int prototying_n_speed_up = 1;
    const fs::path data_dir = (
        fs::current_path() / fs::path("..") /
        fs::path("test") / fs::path("benchmarking") / fs::path("data")
    );

public:

    void SetUp() {
        TestBase::SetUp();
    }

    void TearDown() {
        TestBase::TearDown();
    }

    void benchmark_n_runs(
        int n,
        std::function<void (Benchmark_AccumulatingClock &)> f,
        Benchmark_AccumulatingClock &benchmark_clock,
        std::string label=""
    ) {
        for (int i=0; i<n; ++i) {
            f(benchmark_clock);
        }
        std::string label_formatted = (label == "") ? "" : std::format(" {}", label);
        std::cout << std::format(
                        "[Benchmark{}] runs: {} | avg: {} | median: {} | total: {}",
                        label_formatted,
                        benchmark_clock.get_count(),
                        benchmark_clock.get_avg(),
                        benchmark_clock.get_median(),
                        benchmark_clock.get_total()
                     )
                  << std::endl;
    }

    template <typename T, typename W>
    void record_data_row(std::ofstream &os, T indepedent_var, W dependent_var) {
        os << std::format("{},{}", indepedent_var, dependent_var) << std::endl;
    }

    template <template <typename> typename M>
    void basic_func_benchmark(
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

};

#endif
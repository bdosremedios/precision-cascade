#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "benchmark_tools.h"

#include "tools/cuHandleBundle.h"
#include "types/types.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <functional>

namespace fs = std::filesystem;

class BenchmarkBase: public testing::Test
{
protected:

    const int n_runs = 5;
    bool prototyping_speed_up = false;

public:

    static cuHandleBundle bundle;
    static fs::path data_dir;

    int dense_start = (prototyping_speed_up) ? 1024 : 2500;
    int dense_stop = 20001;
    int dense_incr = (
        (prototyping_speed_up) ? (dense_stop-dense_start) : 2500
    );

    int sparse_start = (prototyping_speed_up) ? 1024 : 25000;
    int sparse_stop = 200001;
    int sparse_incr = (
        (prototyping_speed_up) ? (sparse_stop-sparse_start) : 25000
    );

    int ilu_start = (prototyping_speed_up) ? 1024 : 10000;
    int ilu_stop = 80001;
    int ilu_incr = (
        (prototyping_speed_up) ? (sparse_stop-sparse_start) : 10000
    );

    const bool pivot_ilu = false;
    const int gmressolve_iters = 200;
    const int nested_outer_iter = (prototyping_speed_up) ? 5 : 300;
    const int nested_inner_iter = gmressolve_iters;
    const int dense_subset_cols = gmressolve_iters;

    void benchmark_n_runs(
        int n,
        std::function<void (Benchmark_AccumClock &)> f,
        Benchmark_AccumClock &benchmark_clock,
        std::string label=""
    ) {

        for (int i=0; i<n; ++i) {
            f(benchmark_clock);
        }

        std::string label_formatted = (
            (label == "") ? "" : (" " + label)
        );
        std::cout << "[Benchmark" << label_formatted << "] runs: "
                  << std::to_string(benchmark_clock.get_count())
                  << " | avg: "
                  << std::to_string(benchmark_clock.get_avg().count())
                  << " | median: "
                  << std::to_string(benchmark_clock.get_median().count())
                  << " | total: "
                  << std::to_string(benchmark_clock.get_total().count())
                  << std::endl;

    }

    template <template <typename> typename TMatrix, typename TPrecision>
    void benchmark_exec_func(
        int m_start_incl,
        int m_stop_excl,
        int m_incr,
        std::function<TMatrix<TPrecision> (int, int)> make_A,
        std::function<void (Benchmark_AccumClock &, TMatrix<TPrecision> &)> exec_func,
        std::string label
    ) {

        fs::path file_path = data_dir / fs::path(label + ".csv");
        std::ofstream f_out;
        f_out.open(file_path);

        if (!(f_out.is_open())) {
            throw std::runtime_error(
                "basic_matrix_func_benchmark: " +
                file_path.string() +
                " did not open"
            );
        }

        std::vector<int> exp_m_values;
        for (int m = m_start_incl; m < m_stop_excl; m += m_incr) {
            exp_m_values.push_back(m);
        }

        for (int m : exp_m_values) {

            Benchmark_AccumClock curr_clock;

            TMatrix<TPrecision> A = make_A(m, m);
            std::function<void(Benchmark_AccumClock &)> test_func = (
                [exec_func, &A](Benchmark_AccumClock &arg_clock) {
                    exec_func(arg_clock, A);
                }
            );

            benchmark_n_runs(
                n_runs,
                test_func,
                curr_clock,
                label + "_" + std::to_string(m)
            );

            f_out << m << "," << curr_clock.get_median().count()
                  << std::endl; 

        }

        f_out.close();

    }

};

#endif
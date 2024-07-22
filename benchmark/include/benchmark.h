#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "benchmark_tools.h"

#include "tools/cuHandleBundle.h"
#include "types/types.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <functional>
#include <cmath>
#include <vector>

namespace fs = std::filesystem;

using namespace cascade;

class BenchmarkBase: public testing::Test
{
protected:

    const int n_runs = 7;
    bool prototyping_speed_up = false;

public:

    static cuHandleBundle bundle;
    static fs::path data_dir;

    std::vector<int> dense_dims = (
        (prototyping_speed_up) ?
        std::vector<int>({1024}) :
        std::vector<int>({
            2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000
        })
    );

    std::vector<int> sparse_dims = (
        (prototyping_speed_up) ?
        std::vector<int>({1024}) :
        std::vector<int>({
            2154, 50991, 80708, 105655, 127930, 148406, 167554, 185664
        })
    );

    std::vector<int> ilu_dims = (
        (prototyping_speed_up) ?
        std::vector<int>({1024}) :
        std::vector<int>({
            2154, 20536, 32228, 42069, 50864, 58954, 66522, 73681
        })
    );

    const bool pivot_ilu = true;

    // const int sparse_col_non_zeros = 200;
    const int run_fast_tests_count = 200;
    const int dense_subset_cols = 200;

    const int gmres_iters = 200;
    const int nested_gmres_inner_iters = gmres_iters;
    const int nested_gmres_outer_iters = (prototyping_speed_up) ? 5 : 300;

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
        std::vector<int> m_dimensions,
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

        for (int m : m_dimensions) {

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
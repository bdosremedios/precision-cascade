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

    acc_clock_duration total;
    int window_count;

public:

    Benchmark_AccumulatingClock(): total(0), window_count(0) {}

    void clock_start();
    void clock_stop();

    int get_count();
    acc_clock_duration get_avg();
    acc_clock_duration get_total();

};

class BenchmarkTestBase: public TestBase
{
public:

    const fs::path data_dir = (fs::current_path() / fs::path("..") / fs::path("data"));

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
                        "[Benchmark{}] runs: {} | avg: {} | total: {}",
                        label_formatted,
                        benchmark_clock.get_count(),
                        benchmark_clock.get_avg(),
                        benchmark_clock.get_total()
                     )
                  << std::endl;
    }

    template <typename T, typename W>
    void record_data_row(std::ofstream &os, T indepedent_var, W dependent_var) {
        os << std::format("{},{}", indepedent_var, dependent_var) << std::endl;
    }

};

#endif
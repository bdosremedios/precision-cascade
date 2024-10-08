#ifndef BENCHMARK_TOOLS_H
#define BENCHMARK_TOOLS_H

#include <vector>
#include <chrono>

class Benchmark_AccumClock
{
private:

    bool clock_ticking = false;

    using acc_clock_duration = std::chrono::microseconds;

    std::chrono::high_resolution_clock clock;
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> stop;

    std::vector<acc_clock_duration> prev_durations;

public:

    Benchmark_AccumClock() {}

    void clock_start();
    void clock_stop();

    int get_count();
    acc_clock_duration get_avg();
    acc_clock_duration get_median();
    acc_clock_duration get_total();
    acc_clock_duration get_min();
    acc_clock_duration get_max();

};

#endif
#include "../include/benchmark.h"

void Benchmark_AccumulatingClock::clock_start() {
    if (!(clock_ticking)) {
        clock_ticking = true;
        start = clock.now();
    } else {
        throw std::runtime_error("BenchMark_AccumulatingClock: clock_start error clock already ticking");
    }
}

void Benchmark_AccumulatingClock::clock_stop() {
    if (clock_ticking) {
        stop = clock.now();
        clock_ticking = false;
        total += std::chrono::duration_cast<acc_clock_duration>(stop-start);
        ++window_count;
    } else {
        throw std::runtime_error("BenchMark_AccumulatingClock: clock_stop error clock not ticking");
    }
}

int Benchmark_AccumulatingClock::get_count() {
    return window_count;
}

Benchmark_AccumulatingClock::acc_clock_duration Benchmark_AccumulatingClock::get_avg() {
    return total/window_count;
}

Benchmark_AccumulatingClock::acc_clock_duration Benchmark_AccumulatingClock::get_total() {
    return total;
}
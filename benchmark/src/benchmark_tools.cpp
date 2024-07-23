#include "benchmark_tools.h"

#include <stdexcept>
#include <algorithm>

void Benchmark_AccumClock::clock_start() {
    if (!(clock_ticking)) {
        clock_ticking = true;
        start = clock.now();
    } else {
        throw std::runtime_error(
            "BenchMark_AccumulatingClock: clock_start error clock already "
            "ticking"
        );
    }
}

void Benchmark_AccumClock::clock_stop() {
    if (clock_ticking) {
        stop = clock.now();
        clock_ticking = false;
        prev_durations.push_back(
            std::chrono::duration_cast<acc_clock_duration>(stop-start)
        );
    } else {
        throw std::runtime_error(
            "BenchMark_AccumulatingClock: clock_stop error clock not ticking"
        );
    }
}

int Benchmark_AccumClock::get_count() {
    return prev_durations.size();
}

Benchmark_AccumClock::acc_clock_duration Benchmark_AccumClock::get_avg() {
    return get_total()/get_count();
}

Benchmark_AccumClock::acc_clock_duration Benchmark_AccumClock::get_median() {

    std::vector<acc_clock_duration> to_sort(prev_durations);
    std::sort(to_sort.begin(), to_sort.end());

    return to_sort[get_count()/2];

}

Benchmark_AccumClock::acc_clock_duration Benchmark_AccumClock::get_total() {
    acc_clock_duration total(0);
    for (acc_clock_duration dur : prev_durations) { total += dur; }
    return total;
}

Benchmark_AccumClock::acc_clock_duration Benchmark_AccumClock::get_min() {
    acc_clock_duration min_val(prev_durations[0]);
    for (acc_clock_duration dur : prev_durations) {
        if (dur < min_val) { min_val = dur; }
    }
    return min_val;
}

Benchmark_AccumClock::acc_clock_duration Benchmark_AccumClock::get_max() {
    acc_clock_duration max_val(prev_durations[0]);
    for (acc_clock_duration dur : prev_durations) {
        if (dur > max_val) { max_val = dur; }
    }
    return max_val;
}
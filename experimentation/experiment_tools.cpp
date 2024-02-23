#include "experiment_tools.h"

void Experiment_Clock::start_clock_experiment() {
    if (!clock_ticking) {
        start = clock.now();
        clock_ticking = true;
    } else {
        throw std::runtime_error("Experiment_Clock: start_clock_experiment clock already ticking");
    } 
}
    
void Experiment_Clock::stop_clock_experiment() {
    if (clock_ticking) {
        stop = clock.now();
        clock_ticking = false;
        time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start);
    } else {
        throw std::runtime_error("Experiment_Clock: stop_clock_experiment clock not ticking");
    }
}

int Experiment_Clock::get_elapsed_time_ms() const {
    return time_ms.count();
}

std::string Experiment_Clock::get_info_string() const {
    return std::format("Elapsed time (ms): {}", get_elapsed_time_ms());
}
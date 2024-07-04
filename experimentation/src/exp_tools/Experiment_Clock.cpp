#include "exp_tools/Experiment_Clock.h"

#include <stdexcept>

void Experiment_Clock::start_clock_experiment() {
    if (!clock_ticking) {
        start = clock.now();
        clock_ticking = true;
        completed = false;
    } else {
        throw std::runtime_error(
            "Experiment_Clock: start_clock_experiment clock already ticking"
        );
    } 
}
    
void Experiment_Clock::stop_clock_experiment() {
    if (clock_ticking) {
        stop = clock.now();
        clock_ticking = false;
        completed = true;
        time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            stop-start
        );
    } else {
        throw std::runtime_error(
            "Experiment_Clock: stop_clock_experiment clock not ticking"
        );
    }
}

int Experiment_Clock::get_elapsed_time_ms() const {
    return time_ms.count();
}

bool Experiment_Clock::check_completed() const {
    return completed;
}

std::string Experiment_Clock::get_info_string() const {
    return "Elapsed time (ms): " + std::to_string(get_elapsed_time_ms());
}
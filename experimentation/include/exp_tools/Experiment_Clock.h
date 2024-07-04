#ifndef EXPERIMENT_CLOCK_H
#define EXPERIMENT_CLOCK_H

#include <chrono>
#include <string>

class Experiment_Clock 
{
private:

    std::chrono::steady_clock clock;
    std::chrono::time_point<std::chrono::steady_clock> start;
    std::chrono::time_point<std::chrono::steady_clock> stop;
    std::chrono::milliseconds time_ms;
    bool clock_ticking = false;
    bool completed = false;

public:

    Experiment_Clock() = default;

    ~Experiment_Clock() = default;

    Experiment_Clock(const Experiment_Clock &other) {
        *this = other;
    }

    void operator=(const Experiment_Clock &other) {
        clock = other.clock;
        start = other.start;
        stop = other.stop;
        time_ms = other.time_ms;
        clock_ticking = other.clock_ticking;
        completed = other.completed;
    }

    void start_clock_experiment();
    
    void stop_clock_experiment();

    int get_elapsed_time_ms() const;

    bool check_completed() const;

    std::string get_info_string() const;

};

#endif
#ifndef EXPERIMENT_TOOLS_H
#define EXPERIMENT_TOOLS_H

#include <chrono>
#include <iostream>

#include "solvers/IterativeSolve.h"

class Experiment_Clock 
{
public:

    std::chrono::steady_clock clock;
    std::chrono::time_point<std::chrono::steady_clock> start;
    std::chrono::time_point<std::chrono::steady_clock> stop;
    std::chrono::milliseconds time_ms;
    bool clock_ticking = false;

    void start_clock_experiment();
    
    void stop_clock_experiment();

    int get_elapsed_time_ms();

    void print_elapsed_time();

};

template <template <typename> typename M>
struct Experiment_Data
{
public:
    
    Experiment_Clock clock;
    std::shared_ptr<GenericIterativeSolve<M>> solver_ptr;

    Experiment_Data(
        Experiment_Clock arg_clock,
        std::shared_ptr<GenericIterativeSolve<M>> arg_solver_ptr
    ):
        clock(arg_clock), solver_ptr(arg_solver_ptr) 
    {}

    Experiment_Data(const Experiment_Data &other) = default;
    Experiment_Data & operator=(const Experiment_Data &other) = default;

};

#endif
#ifndef EXPERIMENT_TOOLS_H
#define EXPERIMENT_TOOLS_H

#include <format>
#include <filesystem>

#include "tools/arg_pkgs/SolveArgPkg.h"

namespace fs = std::filesystem;

void check_dir_exists(fs::path dir);

struct Solve_Group
{
public:
    
    const std::string id;
    const int experiment_iterations;
    const std::vector<std::string> solvers_to_use;
    const std::string matrix_type;
    const SolveArgPkg solver_args;
    const std::string preconditioning;
    const std::vector<std::string> matrices_to_test;

    Solve_Group(
        std::string arg_id,
        std::vector<std::string> arg_solvers_to_use,
        std::string arg_matrix_type,
        int arg_experiment_iterations,
        int arg_solver_max_outer_iterations,
        int arg_solver_max_inner_iterations,
        double arg_solver_target_relres,
        std::string preconditioning,
        std::vector<std::string> arg_matrices_to_test
    );

};

struct Experiment_Specification
{
public:
    
    const std::string id;
    std::vector<Solve_Group> solve_groups;

    Experiment_Specification(std::string arg_id);

    void add_solve_group(Solve_Group solve_group);

};

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

    int get_elapsed_time_ms() const;

    std::string get_info_string() const;

};

template <template <template <typename> typename> typename Solver, template <typename> typename M>
struct Experiment_Data
{
public:
    
    Experiment_Clock clock;
    std::shared_ptr<Solver<M>> solver_ptr;

    Experiment_Data(
        Experiment_Clock arg_clock,
        std::shared_ptr<Solver<M>> arg_solver_ptr
    ):
        clock(arg_clock), solver_ptr(arg_solver_ptr) 
    {}

    Experiment_Data(const Experiment_Data &other) = default;
    Experiment_Data & operator=(const Experiment_Data &other) = default;

    std::string get_info_string() const {
        return std::format(
            "{} | {}",
            clock.get_info_string(),
            solver_ptr->get_info_string()
        );
    }

};

#endif
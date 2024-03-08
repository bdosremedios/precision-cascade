#include "experiment_tools.h"

void check_dir_exists(fs::path dir) {
    if (!fs::exists(dir)) {
        throw std::runtime_error(
            std::format("Directory {} does not exist", dir.string())
        );
    }
}

Solve_Group::Solve_Group(
    std::string arg_id,
    std::string arg_solver_suite_type,
    std::string arg_matrix_type,
    int arg_experiment_iterations,
    int arg_solver_max_outer_iterations,
    int arg_solver_max_inner_iterations,
    double arg_solver_target_relres,
    std::string arg_preconditioning,
    std::vector<std::string> arg_matrices_to_test
): 
    id(arg_id),
    solver_suite_type(arg_solver_suite_type),
    matrix_type(arg_matrix_type),
    experiment_iterations(arg_experiment_iterations),
    solver_args(
        arg_solver_max_outer_iterations,
        arg_solver_max_inner_iterations,
        arg_solver_target_relres
    ),
    preconditioning(arg_preconditioning),
    matrices_to_test(arg_matrices_to_test)
{}

Experiment_Specification::Experiment_Specification(std::string arg_id):
    id(arg_id)
{}

void Experiment_Specification::add_solve_group(Solve_Group solve_group) {
    solve_groups.push_back(solve_group);
}

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
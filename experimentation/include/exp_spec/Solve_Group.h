#ifndef SOLVE_GROUP_H
#define SOLVE_GROUP_H

#include "Preconditioner_Spec.h"

#include "tools/arg_pkgs/SolveArgPkg.h"

#include <vector>
#include <string>
#include <unordered_set>

struct Solve_Group {

    const std::string id;
    const int experiment_iterations;
    const std::vector<std::string> solvers_to_use;
    const std::string matrix_type;
    const cascade::SolveArgPkg solver_args;
    const Preconditioner_Spec precond_specs;
    const std::vector<std::string> matrices_to_test;

    static const std::unordered_set<std::string> valid_fp_solver_ids;
    static const std::unordered_set<std::string> valid_mp_solver_ids;

    Solve_Group(
        std::string arg_id,
        std::vector<std::string> arg_solvers_to_use,
        std::string arg_matrix_type,
        int arg_experiment_iterations,
        int arg_solver_max_outer_iterations,
        int arg_solver_max_inner_iterations,
        double arg_solver_target_relres,
        Preconditioner_Spec arg_precond_specs,
        std::vector<std::string> arg_matrices_to_test
    );

};

#endif
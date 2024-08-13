#include "exp_spec/Solve_Group.h"

#include <filesystem>

namespace fs = std::filesystem;

const std::unordered_set<std::string> Solve_Group::valid_fp_solver_ids {
    "FP16", "FP32", "FP64"
};

const std::unordered_set<std::string> Solve_Group::valid_mp_solver_ids {
    "OuterRestartCount",
    "RelativeResidualThreshold",
    "CheckStagnation",
    "ProjectThresholdAfterStagnation"
};

Solve_Group::Solve_Group(
    std::string arg_id,
    std::vector<std::string> arg_solvers_to_use,
    std::string arg_matrix_type,
    int arg_experiment_iterations,
    int arg_solver_max_outer_iterations,
    int arg_solver_max_inner_iterations,
    double arg_solver_target_relres,
    Preconditioner_Spec arg_precond_specs,
    std::vector<std::string> arg_matrices_to_test
): 
    id(arg_id),
    solvers_to_use(arg_solvers_to_use),
    matrix_type(arg_matrix_type),
    experiment_iterations(arg_experiment_iterations),
    solver_args(
        arg_solver_max_outer_iterations,
        arg_solver_max_inner_iterations,
        arg_solver_target_relres
    ),
    precond_specs(arg_precond_specs),
    matrices_to_test(arg_matrices_to_test)
{
    // Check validity of solve_group parameters
    std::unordered_set<std::string> solvers_seen;
    for (std::string solver_id : solvers_to_use) {
        if (
            (valid_fp_solver_ids.count(solver_id) == 0) &&
            (valid_mp_solver_ids.count(solver_id) == 0)
        ) {
            throw std::runtime_error(
                "Solve_Group: invalid solver encountered in solvers_to_use "
                "\"" + solver_id + "\""
            );
        } else if (solvers_seen.count(solver_id) == 1) {
            throw std::runtime_error(
                "Solve_Group: repeated solver encountered in solvers_to_use "
                "\"" + solver_id + "\""
            );
        } else {
            solvers_seen.insert(solver_id);
        }
    }
    if (!((matrix_type == "dense") || (matrix_type == "sparse"))) {
        throw std::runtime_error(
            "Solve_Group: invalid matrix_type \"" + matrix_type + "\""
        );
    }
    if (experiment_iterations <= 0) {
        throw std::runtime_error(
            "Solve_Group: invalid experiment_iterations \"" +
            std::to_string(experiment_iterations) + "\""
        );
    }
    if (
        (arg_solver_max_outer_iterations <= 0) ||
        (arg_solver_max_inner_iterations <= 0) ||
        (arg_solver_target_relres < 0.)
    ) {
        throw std::runtime_error(
            "Solve_Group: invalid nested solver arguments"
        );
    }
    if (matrices_to_test.size() == 0) {
        throw std::runtime_error(
            "Solve_Group: empty matrices_to_test"
        );
    }
    for (std::string mat_file_name : matrices_to_test) {
        fs::path file_path(mat_file_name);
        if (
            !((file_path.extension() == ".mtx") ||
              (file_path.extension() == ".csv"))
        ) {
            throw std::runtime_error(
                "Solve_Group: invalid matrix in matrices_to_test \"" +
                mat_file_name + "\""
            );
        }
    }

}
#ifndef SOLVE_GROUP_H
#define SOLVE_GROUP_H

#include "Preconditioner_Spec.h"

#include "tools/arg_pkgs/SolveArgPkg.h"
#include "exp_tools/write_json.h"
#include "exp_tools/Experiment_Log.h"

#include <filesystem>
#include <vector>
#include <string>
#include <unordered_set>

namespace fs = std::filesystem;

struct Solve_Group {

    const std::string id;
    const int experiment_iterations;
    const std::vector<std::string> solvers_to_use;
    const std::string matrix_type;
    const cascade::SolveArgPkg solver_args;
    const Preconditioner_Spec precond_specs;
    const std::vector<std::string> matrices_to_test;

    static const std::unordered_set<std::string> valid_fp_solver_ids;
    static const std::unordered_set<std::string> valid_vp_solver_ids;

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

    void record_json(
        std::string file_name,
        fs::path output_data_dir,
        Experiment_Log logger
    ) const {

        std::ofstream file_out = write_json::open_json_ofstream(
            file_name, output_data_dir, logger
        );

        write_json::start_json(file_out);

        file_out << "\t\"id\" : \"" << id << "\",\n";
        file_out << "\t\"matrix_type\" : \"" << matrix_type << "\",\n";
        file_out << "\t\"experiment_iterations\" : "
                 << experiment_iterations << ",\n";
        file_out << "\t\"max_outer_iterations\" : "
                 << solver_args.max_iter << ",\n";
        file_out << "\t\"max_inner_iterations\" : "
                 << solver_args.max_inner_iter << ",\n";
        file_out << "\t\"target_rel_res\" : "
                 << solver_args.target_rel_res << ",\n";
        file_out << "\t\"precond_specs\" : \""
                 << precond_specs.get_spec_string() << "\",\n";
        file_out << "\t\"solver_ids\" : "
                 << write_json::str_vector_to_jsonarray_str(solvers_to_use, 0)
                 << ",\n";

        // Remove file extensions from ids before adding
        std::vector<std::string> matrix_ids;
        for (std::string matrix_file : matrices_to_test) {
            matrix_ids.push_back(fs::path(matrix_file).stem().string());
        }
        file_out << "\t\"matrix_ids\" : "
                 << write_json::str_vector_to_jsonarray_str(matrix_ids, 0)
                 << "\n";

        write_json::end_json(file_out);

    }

};

#endif
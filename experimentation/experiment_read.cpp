#include "experiment_read.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>

#include <string>
#include <tuple>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

Solve_Group::Solve_Group(
    std::string arg_id,
    std::string arg_matrix_type,
    int arg_experiment_iterations,
    int arg_solver_max_outer_iterations,
    int arg_solver_max_inner_iterations,
    double arg_solver_target_relres,
    std::vector<std::string> arg_matrices_to_test
): 
    id(arg_id),
    matrix_type(arg_matrix_type),
    experiment_iterations(arg_experiment_iterations),
    solver_args(
        arg_solver_max_outer_iterations,
        arg_solver_max_inner_iterations,
        arg_solver_target_relres
    ),
    matrices_to_test(arg_matrices_to_test)
{}

Experiment_Specification::Experiment_Specification(std::string arg_id):
    id(arg_id)
{}

void Experiment_Specification::add_solve_group(Solve_Group solve_group) {
    solve_groups.push_back(solve_group);
}

int extract_integer(json::iterator member) {
    if (member->is_number_integer()) {
        return *member;
    } else {
        throw std::runtime_error(
            std::format(
                "extract_solve_group: extract_integer invalid key \"{}\" value",
                member.key()
            )
        );
    }
}

std::string extract_matrix_type(json::iterator member) {
    if ((member->is_string()) && ((*member == "dense") || (*member == "sparse"))) {
        return *member;
    } else {
        throw std::runtime_error(
            std::format(
                "extract_solve_group: extract_matrix_type invalid key \"{}\" value",
                member.key()
            )
        );
    }
}

double extract_double(json::iterator member) {
    if (member->is_number_float()) {
        return *member;
    } else {
        throw std::runtime_error(
            std::format(
                "extract_solve_group: extract_double invalid key \"{}\" value",
                member.key()
            )
        );
    }
}

std::vector<std::string> extract_string_vector(json::iterator member) {
    std::vector<std::string> matrix_names;
    if (member->is_array()) {
        for (json::iterator it = member->begin(); it != member->end(); ++it) {
            if (it->is_string()) {
                matrix_names.push_back(*it);
            } else {
                throw std::runtime_error(
                    std::format(
                        "extract_solve_group: extract_string_vector in id \"{}\" non string member",
                        member.key()
                    )
                );
            }
        }
        return matrix_names;
    } else {
        throw std::runtime_error(
            std::format(
                "extract_solve_group: extract_string_vector invalid key \"{}\" value",
                member.key()
            )
        );
    }
}

Solve_Group extract_solve_group(std::string id, json cand_obj) {

    int member_count = 0;
    int experiment_iterations = -1;
    std::string matrix_type = "";
    int solver_max_outer_iterations = -1;
    int solver_max_inner_iterations = -1;
    double solver_target_relres = -1.;
    std::vector<std::string> matrices_to_test;
    for (json::iterator it = cand_obj.begin(); (it != cand_obj.end()) && (member_count < 6); ++it) {

        if (it.key() == "experiment_iterations") {
            experiment_iterations = extract_integer(it);
        } else if (it.key() == "matrix_type") {
            matrix_type = extract_matrix_type(it);
        } else if (it.key() == "solver_max_outer_iterations") {
            solver_max_outer_iterations = extract_integer(it);
        } else if (it.key() == "solver_max_inner_iterations") {
            solver_max_inner_iterations = extract_integer(it);
        } else if (it.key() == "solver_target_relres") {
            solver_target_relres = extract_double(it);
        } else if (it.key() == "matrices_to_test") {
            matrices_to_test = extract_string_vector(it);
        } else {
            throw std::runtime_error(
                std::format(
                    "extract_solve_group: in solve group id \"{}\" invalid key \"{}\"",
                    id,
                    it.key()
                )
            );
        }
        ++member_count;

    }

    if (
        (member_count != 6) ||
        (experiment_iterations == -1) ||
        (matrix_type == "") ||
        (solver_max_outer_iterations == -1) ||
        (solver_max_inner_iterations == -1) ||
        (solver_target_relres == -1.) ||
        (matrices_to_test.size() == 0)
    ) {
        throw std::runtime_error(
            std::format("extract_solve_group: incorrect members for solve group id \"{}\"", id)
        );
    }

    return Solve_Group(
        id,
        matrix_type,
        experiment_iterations,
        solver_max_outer_iterations,
        solver_max_inner_iterations,
        solver_target_relres,
        matrices_to_test
    );

}

Experiment_Specification parse_experiment_spec(fs::path exp_spec_path) {

    std::ifstream exp_spec_stream(exp_spec_path);

    if (!exp_spec_stream.is_open()) {
        throw std::runtime_error("parse_experiment_spec: failed to open " + exp_spec_path.string());
    }
    
    std::string exp_spec_id = exp_spec_path.stem().string();
    Experiment_Specification exp_spec = Experiment_Specification(exp_spec_id);

    std::cout << std::format("\nParsing experiment specification: {}", exp_spec_id) << std::endl;
    json exp_spec_json;
    try {
        exp_spec_json = json::parse(exp_spec_stream);
    } catch (json::parse_error e) {
        throw std::runtime_error(
            std::format(
                "parse_experiment_spec: error in json::parse of experiment specification id \"{}\" - {}",
                exp_spec_id,
                e.what()
            )
        );
    }

    for (json::iterator iter = exp_spec_json.begin(); iter != exp_spec_json.end(); ++iter) {

        if (iter->is_object()) {

            Solve_Group solve_group = extract_solve_group(iter.key(), *iter);
            exp_spec.add_solve_group(solve_group);

        } else {

            throw std::runtime_error(
                std::format(
                    "parse_experiment_spec: solve group with id \"{}\" is not json_object",
                    iter.key()
                )
            );

        }

    }

    return exp_spec;

}
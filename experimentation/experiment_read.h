#ifndef EXPERIMENT_READ_H
#define EXPERIMENT_READ_H

#include <nlohmann/json.hpp>

#include <filesystem>

#include "tools/arg_pkgs/SolveArgPkg.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

struct Solve_Group
{
public:
    
    const std::string id;
    const int experiment_iterations;
    const std::string matrix_type;
    const SolveArgPkg solver_args;
    const std::vector<std::string> matrices_to_test;

    Solve_Group(
        std::string arg_id,
        std::string arg_matrix_type,
        int arg_experiment_iterations,
        int arg_solver_max_outer_iterations,
        int arg_solver_max_inner_iterations,
        double arg_solver_target_relres,
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

int extract_integer(json::iterator member);

std::string extract_matrix_type(json::iterator member);

double extract_double(json::iterator member);

std::vector<std::string> extract_string_vector(json::iterator member);

Solve_Group extract_solve_group(std::string id, json cand_obj);

Experiment_Specification parse_experiment_spec(fs::path exp_spec_path);

#endif
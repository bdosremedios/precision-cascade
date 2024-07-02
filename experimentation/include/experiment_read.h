#ifndef EXPERIMENT_READ_H
#define EXPERIMENT_READ_H

#include "exp_spec/exp_spec.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_set>

namespace fs = std::filesystem;
using json = nlohmann::json;

int extract_integer(json::iterator member);
std::vector<std::string> extract_solvers_to_use(json::iterator member);
std::string extract_matrix_type(json::iterator member);
Preconditioner_Spec extract_solve_group_precond_specs(
    json::iterator member
);
double extract_double(json::iterator member);
std::vector<std::string> extract_string_vector(json::iterator member);

Solve_Group extract_solve_group(std::string id, json cand_obj);

Experiment_Spec parse_experiment_spec(fs::path exp_spec_path);

#endif
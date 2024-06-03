#ifndef EXPERIMENT_READ_H
#define EXPERIMENT_READ_H

#include <nlohmann/json.hpp>

#include "experiment_tools.h"

#include <filesystem>
#include <fstream>

#include <format>
#include <string>
#include <vector>
#include <unordered_set>

namespace fs = std::filesystem;
using json = nlohmann::json;

int extract_integer(json::iterator member);
std::vector<std::string> extract_solvers_to_use(json::iterator member);
std::string extract_matrix_type(json::iterator member);
Solve_Group_Precond_Specs extract_solve_group_precond_specs(json::iterator member);
double extract_double(json::iterator member);
std::vector<std::string> extract_string_vector(json::iterator member);

Solve_Group extract_solve_group(std::string id, json cand_obj);

Experiment_Specification parse_experiment_spec(fs::path exp_spec_path);

#endif
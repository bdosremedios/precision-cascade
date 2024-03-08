#ifndef EXPERIMENT_READ_H
#define EXPERIMENT_READ_H

#include <nlohmann/json.hpp>

#include "experiment_tools.h"

#include <filesystem>
#include <fstream>

#include <format>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

int extract_integer(json::iterator member);
std::string extract_solver_suite_type(json::iterator member);
std::string extract_matrix_type(json::iterator member);
std::string extract_preconditioning(json::iterator member);
double extract_double(json::iterator member);
std::vector<std::string> extract_string_vector(json::iterator member);

Solve_Group extract_solve_group(std::string id, json cand_obj);

Experiment_Specification parse_experiment_spec(fs::path exp_spec_path);

#endif
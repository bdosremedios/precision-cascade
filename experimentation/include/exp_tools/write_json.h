#ifndef WRITE_JSON_H
#define WRITE_JSON_H

#include "exp_tools/Experiment_Log.h"

#include <vector>
#include <filesystem>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>

namespace fs = std::filesystem;

namespace write_json {
    
std::ofstream open_json_ofstream(
    std::string file_name, fs::path save_dir, Experiment_Log logger
);

void start_json(std::ofstream &file_out);

void end_json(std::ofstream &file_out);

std::string bool_to_string(bool b);

std::string dbl_vector_to_jsonarray_str(
    std::vector<double> vec, int padding_level
);

std::string int_vector_to_jsonarray_str(
    std::vector<int> vec, int padding_level
);

std::string str_vector_to_jsonarray_str(
    std::vector<std::string> vec, int padding_level
);

}

#endif
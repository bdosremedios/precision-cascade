#include <memory>

#include <filesystem>
#include <iostream>
#include <sstream>

#include <cmath>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "experiment_read.h"
#include "experiment_run.h"
#include "experiment_log.h"

namespace fs = std::filesystem;

int main(int argc, char *argv[]) {

    fs::path data_dir_path("C:\\Users\\dosre\\dev\\numerical_experimentation\\data");
    fs::path input_dir_path("C:\\Users\\dosre\\dev\\numerical_experimentation\\input");
    fs::path output_dir_path("C:\\Users\\dosre\\dev\\numerical_experimentation\\output");

    Experiment_Log experiment_logger(
        "experiment", output_dir_path / fs::path("experiment.log"), true
    );

    experiment_logger.info(std::format("Start numerical experiment: {}", argv[0]));
    experiment_logger.info("Data directory for experiment matrices: " + data_dir_path.string());
    experiment_logger.info("Input directory for experiment specifications: " + input_dir_path.string());
    experiment_logger.info("Output directory for experiment data: " + output_dir_path.string());

    fs::directory_iterator dir_iter = fs::directory_iterator(input_dir_path);

    // Find candidate experimental spec files in input directory (all jsons)
    std::vector<fs::path> candidate_exp_specs;
    experiment_logger.info(std::format("Searching {} for experimental spec files", input_dir_path.string()));
    for (auto curr = begin(dir_iter); curr != end(dir_iter); ++curr) {
        if (curr->path().extension() == ".json") {
            candidate_exp_specs.push_back(curr->path());
        }
    }
    experiment_logger.info(std::format("Found {} experimental spec files", candidate_exp_specs.size()));

    // Validate found experimental specs
    int count_val_success = 0;
    int count_val_fail = 0;
    experiment_logger.info("Validating experimental spec files");
    for (fs::path cand_exp_spec_path : candidate_exp_specs) {
        try {
            experiment_logger.info("Validating: " + cand_exp_spec_path.filename().string());
            Experiment_Specification loaded_exp_spec = parse_experiment_spec(cand_exp_spec_path);
            ++count_val_success;
        } catch (std::runtime_error e) {
            experiment_logger.warn("Failed validation for: " + cand_exp_spec_path.filename().string());
            experiment_logger.warn("Skipping: " + cand_exp_spec_path.filename().string());
            ++count_val_fail;
        }
    }
    experiment_logger.info(
        std::format("Completed validation: {} passed | {} fail", count_val_success, count_val_fail)
    );


    // Execute valid experimental specs


    // fs::path input_dir_path("C:\\Users\\dosre\\dev\\numerical_experimentation\\input\\experiment_matrices");
    // std::cout << "Input directory: " << input_dir_path << std::endl;

    // fs::path output_dir_path("C:\\Users\\dosre\\dev\\numerical_experimentation\\output");
    // std::cout << "Output directory: " << output_dir_path << std::endl;

    // std::ifstream csv_load_order;
    // fs::path csv_load_order_path(input_dir_path / fs::path("csv_load_order.txt"));
    // csv_load_order.open(csv_load_order_path);
    // std::cout << "csv load order file: " << csv_load_order_path << std::endl << std::endl;

    // if (!csv_load_order.is_open()) {
    //     throw std::runtime_error("csv_load_order did not load correctly");
    // }

    // cublasHandle_t handle;
    // cublasCreate(&handle);
    // cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

    // std::string temp_str;
    // std::getline(csv_load_order, temp_str);

    // bool show_plots = false;

    // run_all_solves<MatrixDense>(
    //     handle, input_dir_path, temp_str, output_dir_path, 10, 5, //200, 20,
    //     std::pow(10., -10.), show_plots
    // );

    // std::cout << "\n*** Finish Numerical Experimentation ***" << std::endl;
    
    return 0;

}
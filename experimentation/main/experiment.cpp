#include <memory>

#include <filesystem>
#include <iostream>
#include <sstream>

#include <cmath>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "experiment_read.h"
#include "experiment_run.h"

namespace fs = std::filesystem;

int main() {

    fs::path data_dir_path("C:\\Users\\dosre\\dev\\numerical_experimentation\\data");
    std::cout << "Data directory for experiment matrices: " << data_dir_path << std::endl;

    fs::path input_dir_path("C:\\Users\\dosre\\dev\\numerical_experimentation\\input");
    std::cout << "Input directory for experiment specifications: " << input_dir_path << std::endl;

    fs::path output_dir_path("C:\\Users\\dosre\\dev\\numerical_experimentation\\output");
    std::cout << "Output directory for experiment data: " << output_dir_path << std::endl;
    std::cout << std::endl;

    fs::directory_iterator dir_iter = fs::directory_iterator(input_dir_path);

    // Validate experimental specs to ensure they can be loaded before testing
    for (auto curr = begin(dir_iter); curr != end(dir_iter); ++curr) {
        try {
            std::cout << "Validating experiment specification: " << curr->path().string() << std::endl;
            Experiment_Specification loaded_exp_spec = parse_experiment_spec(curr->path());

        } catch (std::runtime_error e) {
            std::cout << e.what() << std::endl;
        }
    }

    // Execute experiment accoridng to experimental specs

    // std::cout << "*** Start Numerical Experimentation: experiment.cpp ***\n" << std::endl;

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
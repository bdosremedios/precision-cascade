#include <memory>

#include <filesystem>
#include <iostream>
#include <sstream>

#include <cmath>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "run_experiment.h"

namespace fs = std::filesystem;

int main() {

    std::cout << "*** Start Numerical Experimentation: experiment.cpp ***\n" << std::endl;

    fs::path input_dir_path("C:\\Users\\dosre\\dev\\numerical_experimentation\\input\\experiment_matrices");
    std::cout << "Input directory: " << input_dir_path << std::endl;

    fs::path output_dir_path("C:\\Users\\dosre\\dev\\numerical_experimentation\\output");
    std::cout << "Output directory: " << output_dir_path << std::endl;

    std::ifstream csv_load_order;
    fs::path csv_load_order_path(input_dir_path / fs::path("csv_load_order.txt"));
    csv_load_order.open(csv_load_order_path);
    std::cout << "csv load order file: " << csv_load_order_path << std::endl << std::endl;

    if (!csv_load_order.is_open()) {
        throw std::runtime_error("csv_load_order did not load correctly");
    }

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

    std::string temp_str;
    std::getline(csv_load_order, temp_str);

    bool show_plots = false;

    run_all_solves<MatrixDense>(
        handle, input_dir_path, temp_str, output_dir_path, 10, 5, //200, 20,
        std::pow(10., -10.), show_plots
    );

    std::cout << "\n*** Finish Numerical Experimentation ***" << std::endl;
    
    return 0;

}
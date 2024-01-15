#include <gtest/gtest.h>
// #include <Eigen/Dense>

#include <cmath>
#include <string>
#include <iostream>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include "test.h"

#include "types/types.h"
// #include "tools/argument_pkgs.h"
// #include "tools/math_functions.h"

// using Eigen::Matrix, Eigen::Dynamic;

bool *TestBase::show_plots = new bool;
cublasHandle_t *TestBase::handle_ptr = new cublasHandle_t;
bool *TestBase::print_errors = new bool;

int main(int argc, char **argv) {

    testing::InitGoogleTest();

    // Check if should run long tests
    bool run_long_tests = false;
    for (int i=0; i<argc; ++i) {
        if ((string(argv[i]) == "--run_long_tests") || (string(argv[i]) == "-rlt")) { run_long_tests = true; }
    }
    if (run_long_tests) {
        cout << "Running long tests..." << endl;
    } else {
        cout << "Skipping long tests..." << endl;
        testing::GTEST_FLAG(filter) = "-*LONGRUNTIME";
    }

    // Check if should run long tests
    bool only_new = false;
    for (int i=0; i<argc; ++i) {
        if ((string(argv[i]) == "--only-new") || (string(argv[i]) == "-on")) { only_new = true; }
    }
    if (only_new) {
        cout << "Running only new tests..." << endl;
        testing::GTEST_FLAG(filter) = "*NEW";
    } else {
        cout << "Running all tests..." << endl;
    }

    // Check if should show plots
    bool show_plots = false;
    for (int i=0; i<argc; ++i) {
        if ((string(argv[i]) == "--show_plots") || (string(argv[i]) == "-sp")) { show_plots = true; }
    }
    if (show_plots) {
        cout << "Showing plots..." << endl;
        *(TestBase::show_plots) = true;
    } else {
        cout << "Not showing plots..." << endl;
        *(TestBase::show_plots) = false;
    }

    // Check if should print errors
    bool print_errors = false;
    for (int i=0; i<argc; ++i) {
        if ((string(argv[i]) == "--print_errors") || (string(argv[i]) == "-pe")) { print_errors = true; }
    }
    if (print_errors) {
        cout << "Printing expected errors..." << endl;
        *(TestBase::print_errors) = true;
    } else {
        cout << "Not printing expected errors..." << endl;
        *(TestBase::print_errors) = false;
    }

    cublasCreate(TestBase::handle_ptr);
    int return_status = RUN_ALL_TESTS();
    cublasDestroy(*TestBase::handle_ptr);

    // Free dynamically allocated test variables
    free(TestBase::show_plots);
    free(TestBase::handle_ptr);
    free(TestBase::print_errors);

    return return_status;

}
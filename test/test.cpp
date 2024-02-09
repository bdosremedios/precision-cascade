#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <iostream>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include "test.h"

#include "types/types.h"
// #include "tools/argument_pkgs.h"

bool *TestBase::show_plots = new bool;
cublasHandle_t *TestBase::handle_ptr = new cublasHandle_t;
bool *TestBase::print_errors = new bool;

int main(int argc, char **argv) {

    testing::InitGoogleTest();

    // Check if should run long tests
    bool run_long_tests = false;
    for (int i=0; i<argc; ++i) {
        if ((std::string(argv[i]) == "--run_long_tests") || (std::string(argv[i]) == "-rlt")) {
            run_long_tests = true;
        }
    }
    if (run_long_tests) {
        std::cout << "Running long tests..." << std::endl;
    } else {
        std::cout << "Skipping long tests..." << std::endl;
        testing::GTEST_FLAG(filter) = "-*LONGRUNTIME";
    }

    // Check if should run long tests
    bool only_new = false;
    for (int i=0; i<argc; ++i) {
        if ((std::string(argv[i]) == "--only_new") || (std::string(argv[i]) == "-on")) {
            only_new = true;
        }
    }
    if (only_new) {
        std::cout << "Running only new tests..." << std::endl;
        testing::GTEST_FLAG(filter) = "*NEW";
    } else {
        std::cout << "Running all tests..." << std::endl;
    }

    // Check if should show plots
    bool show_plots = false;
    for (int i=0; i<argc; ++i) {
        if ((std::string(argv[i]) == "--show_plots") || (std::string(argv[i]) == "-sp")) {
            show_plots = true;
        }
    }
    if (show_plots) {
        std::cout << "Showing plots..." << std::endl;
        *(TestBase::show_plots) = true;
    } else {
        std::cout << "Not showing plots..." << std::endl;
        *(TestBase::show_plots) = false;
    }

    // Check if should print errors
    bool print_errors = false;
    for (int i=0; i<argc; ++i) {
        if ((std::string(argv[i]) == "--print_errors") || (std::string(argv[i]) == "-pe")) {
            print_errors = true;
        }
    }
    if (print_errors) {
        std::cout << "Printing expected errors..." << std::endl;
        *(TestBase::print_errors) = true;
    } else {
        std::cout << "Not printing expected errors..." << std::endl;
        *(TestBase::print_errors) = false;
    }

    cublasCreate(TestBase::handle_ptr);
    cublasSetPointerMode(*TestBase::handle_ptr, CUBLAS_POINTER_MODE_DEVICE);
    int return_status = RUN_ALL_TESTS();
    cublasDestroy(*TestBase::handle_ptr);

    // Free dynamically allocated test variables
    free(TestBase::show_plots);
    free(TestBase::handle_ptr);
    free(TestBase::print_errors);

    return return_status;

}
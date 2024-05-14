#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <iostream>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include "test.h"

#include "types/types.h"
#include "tools/arg_pkgs/argument_pkgs.h"

bool *TestBase::show_plots = new bool;
bool *TestBase::print_errors = new bool;
cuHandleBundle TestBase::bundle;

int main(int argc, char **argv) {

    testing::InitGoogleTest();

    std::string filter_include = "";
    std::string filter_exclude = "";

    // Check if should run exclusively benchmark tests
    bool benchmark = false;
    for (int i=0; i<argc; ++i) {
        if ((std::string(argv[i]) == "--benchmark") || (std::string(argv[i]) == "-bm")) {
            benchmark = true;
        }
    }
    if (benchmark) {

        std::cout << "Running benchmark tests..." << std::endl;
        if (filter_include != "") { filter_include += ":"; }
        filter_include += "*BENCHMARK*";

    } else {

        std::cout << "Excluding benchmark tests..." << std::endl;
        if (filter_exclude != "") { filter_exclude += ":"; }
        filter_exclude += "*BENCHMARK";

        // Check if should run exclusively MPGMRES tests
        bool only_mp_gmres = false;
        for (int i=0; i<argc; ++i) {
            if ((std::string(argv[i]) == "--mpgmres") || (std::string(argv[i]) == "-mp")) {
                only_mp_gmres = true;
            }
        }
        if (only_mp_gmres) {
            std::cout << "Running exclusively MP_GMRES_IR_Solve related tests..." << std::endl;
            if (filter_include != "") { filter_include += ":"; }
            filter_include += "*MP_GMRES_IR*";
        } else {
            std::cout << "Running all tests..." << std::endl;
        }

        // Check if should run preconditioner tests
        bool no_preconditioners = false;
        for (int i=0; i<argc; ++i) {
            if ((std::string(argv[i]) == "--no_preconditioners") || (std::string(argv[i]) == "-np")) {
                no_preconditioners = true;
            }
        }
        if (no_preconditioners) {
            std::cout << "Skipping preconditioner tests..." << std::endl;
            if (filter_exclude != "") { filter_exclude += ":"; }
            filter_exclude += "*_PRECONDITIONER*";
        } else {
            std::cout << "Running preconditioner tests..." << std::endl;
        }

        // Check if should run solver tests
        bool no_solvers = false;
        for (int i=0; i<argc; ++i) {
            if ((std::string(argv[i]) == "--no_solvers") || (std::string(argv[i]) == "-ns")) {
                no_solvers = true;
            }
        }
        if (no_solvers) {
            std::cout << "Skipping solver tests..." << std::endl;
            if (filter_exclude != "") { filter_exclude += ":"; }
            filter_exclude += "*_SOLVER*";
        } else {
            std::cout << "Running solver tests..." << std::endl;
        }

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
            if (filter_exclude != "") { filter_exclude += ":"; }
            filter_exclude += "*_LONGRUNTIME*";
        }

    }

    if (filter_include == "") {
        testing::GTEST_FLAG(filter) = std::format("*:-{}", filter_exclude);
    } else {
        testing::GTEST_FLAG(filter) = std::format("{}:-{}", filter_include, filter_exclude);
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

    TestBase::bundle.create();
    int return_status = RUN_ALL_TESTS();
    TestBase::bundle.destroy();

    // Free dynamically allocated test variables
    free(TestBase::show_plots);
    free(TestBase::print_errors);

    return return_status;

}
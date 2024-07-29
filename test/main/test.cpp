#include "test.h"

#include "types/types.h"
#include "tools/arg_pkgs/argument_pkgs.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <cmath>
#include <string>
#include <iostream>

#ifdef WIN32
#include <windows.h>
#endif

bool *TestBase::show_plots = new bool;
bool *TestBase::print_errors = new bool;
cuHandleBundle TestBase::bundle;
fs::path TestBase::data_dir;
fs::path TestBase::read_matrix_dir;
fs::path TestBase::solve_matrix_dir;

int main(int argc, char **argv) {

    #ifdef WIN32
        CHAR path[MAX_PATH];
        GetModuleFileNameA(NULL, path, MAX_PATH);
        TestBase::data_dir = fs::path(path).parent_path() / fs::path("data");
    #else
        TestBase::data_dir = (
            fs::canonical("/proc/self/exe").parent_path() /
            fs::path("data")
        );
    #endif
    TestBase::read_matrix_dir = (
        TestBase::data_dir / fs::path("read_matrices")
    );
    TestBase::solve_matrix_dir = (
        TestBase::data_dir / fs::path("solve_matrices")
    );

    testing::InitGoogleTest();

    #ifdef _DEBUG
    std::cout << "In debug checking memory allocations" << std::endl;
    #else
    std::cout << "Skip debug not checking memory allocations" << std::endl;
    #endif

    std::string filter_include = "";
    std::string filter_exclude = "";

    // Check if should run exclusively MPGMRES tests
    bool only_mp_gmres = false;
    for (int i=0; i<argc; ++i) {
        if (
            (std::string(argv[i]) == "--mpgmres") ||
            (std::string(argv[i]) == "-mp")
        ) {
            only_mp_gmres = true;
        }
    }
    if (only_mp_gmres) {
        std::cout << "Running exclusively MP_GMRES_IR_Solve related tests..."
                  << std::endl;
        if (filter_include != "") { filter_include += ":"; }
        filter_include += "*MP_GMRES_IR*";
    } else {
        std::cout << "Running all tests..." << std::endl;
    }

    // Check if should run preconditioner tests
    bool no_preconditioners = false;
    for (int i=0; i<argc; ++i) {
        if (
            (std::string(argv[i]) == "--no_preconditioners") ||
            (std::string(argv[i]) == "-np")
        ) {
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
        if (
            (std::string(argv[i]) == "--no_solvers") ||
            (std::string(argv[i]) == "-ns")
        ) {
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
        if (
            (std::string(argv[i]) == "--run_long_tests") ||
            (std::string(argv[i]) == "-rlt")
        ) {
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

    if (filter_include == "") {
        testing::GTEST_FLAG(filter) = "*:-" + filter_exclude;
    } else {
        testing::GTEST_FLAG(filter) = filter_include + ":-" + filter_exclude;
    }

    // Check if should show plots
    bool show_plots = false;
    for (int i=0; i<argc; ++i) {
        if (
            (std::string(argv[i]) == "--show_plots") ||
            (std::string(argv[i]) == "-sp")
        ) {
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
        if (
            (std::string(argv[i]) == "--print_errors") ||
            (std::string(argv[i]) == "-pe")
        ) {
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

    // Delete dynamically allocated test variables
    delete TestBase::show_plots;
    delete TestBase::print_errors;

    return return_status;

}
#include "gtest/gtest.h"
#include "Eigen/Dense"

#include "test.h"

#include <cmath>
#include <string>
#include <iostream>

using Eigen::Matrix, Eigen::Dynamic;
using std::abs;
using std::string;
using std::cout, std::endl;

double gamma(int n, double u) { return n*u/(1-n*u); }

bool *TestBase::show_plots = new bool;

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

    return RUN_ALL_TESTS();

}
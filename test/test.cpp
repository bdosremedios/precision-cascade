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

int main(int argc, char **argv) {

    testing::InitGoogleTest();
    if ((argc > 1) && ((string(argv[1]) == "--run_long_tests") || (string(argv[1]) == "-rlt"))) {
        cout << "Running long tests..." << endl;
    } else {
        cout << "Skipping long tests..." << endl;
        testing::GTEST_FLAG(filter) = "-*LONGRUNTIME";
    }
    return RUN_ALL_TESTS();

}
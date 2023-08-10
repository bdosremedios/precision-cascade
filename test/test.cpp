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

bool matrix_near(Matrix<double, Dynamic, Dynamic> A, Matrix<double, Dynamic, Dynamic> B, double tol) {

    if ((A.cols() != B.cols()) && (A.rows() != B.rows())) {
        return false;
    } else {
        for (int i=0; i<A.rows(); ++i) {
            for (int j=0; j<A.cols(); ++j) {
                if (abs(A(i, j) - B(i, j)) > tol) { return false; }
            }
        }
        return true;
    }

}

bool matrix_zero(Matrix<double, Dynamic, Dynamic> A, double tol) {

    return matrix_near(A, Matrix<double, Dynamic, Dynamic>::Zero(A.rows(), A.cols()), tol);

}

bool matrix_eye(Matrix<double, Dynamic, Dynamic> A, double tol) {

    return matrix_near(A, Matrix<double, Dynamic, Dynamic>::Identity(A.rows(), A.cols()), tol);

}

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
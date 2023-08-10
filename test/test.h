#ifndef TEST_H
#define TEST_H

#include "gtest/gtest.h"
#include "Eigen/Dense"

#include <cmath>
#include <string>

using Eigen::Matrix, Eigen::Dynamic;
using std::pow;
using std::string;

double gamma(int n, double u);

bool matrix_near(Matrix<double, Dynamic, Dynamic>, Matrix<double, Dynamic, Dynamic>, double);
bool matrix_zero(Matrix<double, Dynamic, Dynamic>, double);
bool matrix_eye(Matrix<double, Dynamic, Dynamic>, double);

class TestBase: public testing::Test {

    public:

        const double u_hlf = pow(2, -10);
        const double u_sgl = pow(2, -23);
        const double u_dbl = pow(2, -52);

        const double conv_tol_hlf = pow(10, -02);
        const double conv_tol_sgl = pow(10, -05);
        const double conv_tol_dbl = pow(10, -10);

        const string read_matrix_dir = "/home/bdosremedios/dev/gmres/test/read_matrices/";
        const string solve_matrix_dir = "/home/bdosremedios/dev/gmres/test/solve_matrices/";

};

#endif
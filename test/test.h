#ifndef TEST_H
#define TEST_H

#include "gtest/gtest.h"

#include <cmath>
#include <string>

using std::pow;
using std::string;

double gamma_u(int n, double u);

class TestBase: public testing::Test {

    public:

        double u_hlf = pow(2, -10);
        double u_sgl = pow(2, -23);
        double u_dbl = pow(2, -52);

        double conv_tol_hlf = pow(10, -02);
        double conv_tol_sgl = pow(10, -05);
        double conv_tol_dbl = pow(10, -10);

        string read_matrix_dir = "/home/bdosremedios/dev/gmres/test/read_matrices/";
        string solve_matrix_dir = "/home/bdosremedios/dev/gmres/test/solve_matrices/";

};

#endif
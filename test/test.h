#ifndef TEST_H
#define TEST_H

#include "gtest/gtest.h"
#include "Eigen/Dense"

#include "tools/MatrixReader.h"

#include <cmath>
#include <string>
#include <memory>
#include <iostream>

using read_matrix::read_matrix_csv;

using Eigen::Matrix, Eigen::Dynamic;
using Eigen::half;
using MatrixXh = Eigen::Matrix<Eigen::half, Dynamic, Dynamic>;
using Eigen::MatrixXf;
using Eigen::MatrixXd;

using std::pow;
using std::string;
using std::shared_ptr, std::make_shared;
using std::cout, std::endl;

double gamma(int n, double u);

class TestBase: public testing::Test {

    public:

        const double u_hlf = pow(2, -10);
        const double u_sgl = pow(2, -23);
        const double u_dbl = pow(2, -52);

        // Error tolerance allowed in an entry for a double precision calculation after
        // accumulation of errors where prediction of error bound is difficult
        const double dbl_error_acc = pow(10, -10);

        const double conv_tol_hlf = pow(10, -02);
        const double conv_tol_sgl = pow(10, -05);
        const double conv_tol_dbl = pow(10, -10);

        const string read_matrix_dir = "/home/bdosremedios/dev/precision-cascade/test/read_matrices/";
        const string solve_matrix_dir = "/home/bdosremedios/dev/precision-cascade/test/solve_matrices/";

        const bool show_plots = false;

};

#endif
#ifndef TEST_H
#define TEST_H

#include "gtest/gtest.h"
#include "Eigen/Dense"

#include "types/types.h"
#include "tools/MatrixReader.h"
#include "tools/argument_pkgs.h"

#include <cmath>
#include <string>
#include <memory>
#include <iostream>
#include <filesystem>

using Eigen::Matrix;
using Eigen::Dynamic;
using Eigen::half;
using MatrixXh = Eigen::Matrix<Eigen::half, Dynamic, Dynamic>;
using Eigen::MatrixXf;
using Eigen::MatrixXd;

namespace fs = std::filesystem;
using std::pow;
using std::string;
using std::shared_ptr, std::make_shared;
using std::cout, std::endl;

double gamma(int n, double u);

template<template <typename> typename M, typename T>
int count_zeros(M<T> A, double zero_tol) {

    int count = 0;
    for (int i=0; i<A.rows(); ++i) {
        for (int j=0; j<A.cols(); ++j) {
            if (abs(A.coeff(i, j)) <= zero_tol) { ++count; }
        }
    }
    return count;

}

class TestBase: public testing::Test
{
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

    const fs::path read_matrix_dir = (
        fs::current_path() / fs::path("..") / fs::path("test") / fs::path("read_matrices")
    );
    const fs::path solve_matrix_dir = (
        fs::current_path() / fs::path("..") / fs::path("test") / fs::path("solve_matrices")
    );

    SolveArgPkg default_args;

    static bool *show_plots;

};

#endif
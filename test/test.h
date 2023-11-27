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

template <template <typename> typename M, typename T>
T mat_max_mag(const M<T> &A) {

    T abs_max = static_cast<T>(0);
    for (int i=0; i<A.rows(); ++i) {
        for (int j=0; j<A.cols(); ++j) {
            if (std::abs(A.coeff(i, j)) > abs_max) { abs_max = abs(A.coeff(i, j)); }
        }
    }
    return abs_max;

}

template <typename T>
int count_zeros(MatrixVector<T> vec, double zero_tol) {

    int count = 0;
    for (int i=0; i<vec.rows(); ++i) {
        if (abs(vec(i)) <= zero_tol) { ++count; }
    }
    return count;

}

template <template <typename> typename M, typename T>
int count_zeros(M<T> A, double zero_tol) {

    int count = 0;
    for (int i=0; i<A.rows(); ++i) {
        for (int j=0; j<A.cols(); ++j) {
            if (abs(A.coeff(i, j)) <= zero_tol) { ++count; }
        }
    }
    return count;

}

template <typename T>
void ASSERT_VECTOR_NEAR(MatrixVector<T> test, MatrixVector<T> target, T tol) {

    ASSERT_EQ(test.rows(), target.rows());

    for (int i=0; i<target.rows(); ++i) {
        ASSERT_NEAR(test(i), target(i), tol);
    }

}

template <typename T>
void ASSERT_VECTOR_EQ(MatrixVector<T> test, MatrixVector<T> target) {
    ASSERT_VECTOR_NEAR(test, target, static_cast<T>(0));
}

template <template <typename> typename M, typename T>
void ASSERT_MATRIX_NEAR(M<T> test, M<T> target, T tol) {

    ASSERT_EQ(test.rows(), target.rows());
    ASSERT_EQ(test.cols(), target.cols());

    for (int i=0; i<target.rows(); ++i) {
        for (int j=0; j<target.cols(); ++j) {
            ASSERT_NEAR(test.coeff(i, j), target.coeff(i, j), tol);
        }
    }

}

template <template <typename> typename M, typename T>
void ASSERT_MATRIX_EQ(M<T> test, M<T> target) {
    ASSERT_MATRIX_NEAR(test, target, static_cast<T>(0));
}

template <template <typename> typename M, typename T>
void ASSERT_MATRIX_SAMESPARSITY(M<T> test, M<T> target, T zero_tol) {

    ASSERT_EQ(test.rows(), target.rows());
    ASSERT_EQ(test.cols(), target.cols());

    for (int i=0; i<target.rows(); ++i) {
        for (int j=0; j<target.cols(); ++j) {
            if (abs(target.coeff(i, j)) <= zero_tol) {
                ASSERT_LE(abs(test.coeff(i, j)), zero_tol);
            }
        }
    }

}

template <template <typename> typename M, typename T>
void ASSERT_MATRIX_ZERO(M<T> test, T tol) {
    ASSERT_MATRIX_NEAR(test, M<T>::Zero(test.rows(), test.cols()), tol);
}

template <template <typename> typename M, typename T>
void ASSERT_MATRIX_IDENTITY(M<T> test, T tol) {
    ASSERT_MATRIX_NEAR(test, M<T>::Identity(test.rows(), test.cols()), tol);
}

template <template <typename> typename M, typename T>
void ASSERT_MATRIX_LOWTRI(M<T> test, T tol) {

    for (int i=0; i<test.rows(); ++i) {
        for (int j=i+1; j<test.cols(); ++j) {
            ASSERT_NEAR(test.coeff(i, j), static_cast<T>(0), tol);
        }
    }

}

template <template <typename> typename M, typename T>
void ASSERT_MATRIX_UPPTRI(M<T> test, T tol) {

    for (int i=0; i<test.rows(); ++i) {
        for (int j=0; j<i; ++j) {
            ASSERT_NEAR(test.coeff(i, j), static_cast<T>(0), tol);
        }
    }

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

    const double matlab_impl_diff = pow(10, -8);

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
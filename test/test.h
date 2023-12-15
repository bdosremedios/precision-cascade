#ifndef TEST_H
#define TEST_H

#include "gtest/gtest.h"
#include "Eigen/Dense"

#include "types/types.h"
#include "tools/MatrixReader.h"
#include "tools/argument_pkgs.h"

#include <cmath>
#include <memory>
#include <iostream>
#include <filesystem>

using Eigen::half;

namespace fs = std::filesystem;
using std::pow;
using std::shared_ptr, std::make_shared;
using std::cout, std::endl;

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
void ASSERT_MATRIX_LT(M<T> test, M<T> target) {

    ASSERT_EQ(test.rows(), target.rows());
    ASSERT_EQ(test.cols(), target.cols());

    for (int i=0; i<target.rows(); ++i) {
        for (int j=0; j<target.cols(); ++j) {
            ASSERT_LT(test.coeff(i, j), target.coeff(i, j));
        }
    }

}

template <template <typename> typename M, typename T>
void ASSERT_MATRIX_GT(M<T> test, M<T> target) {

    ASSERT_EQ(test.rows(), target.rows());
    ASSERT_EQ(test.cols(), target.cols());

    for (int i=0; i<target.rows(); ++i) {
        for (int j=0; j<target.cols(); ++j) {
            ASSERT_GT(test.coeff(i, j), target.coeff(i, j));
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

template <typename T>
class Tol
{
public:
    static double roundoff() { assert(false); return -1.; }
    static T roundoff_T() { return static_cast<T>(roundoff()); }
    static double gamma(int n) { 
        return static_cast<T>(n)*roundoff()/(static_cast<T>(1)-static_cast<T>(n)*roundoff());
    }
    static T gamma_T(int n) { return static_cast<T>(gamma(n)); }
};

template<>
static double Tol<half>::roundoff() { return static_cast<half>(pow(2, -10)); }

template<>
static double Tol<float>::roundoff() { return static_cast<float>(pow(2, -23)); }

template<>
static double Tol<double>::roundoff() { return static_cast<double>(pow(2, -52)); }

class TestBase: public testing::Test
{
public:

    const double conv_tol_hlf = 5*pow(10, -02);
    const double conv_tol_sgl = 5*pow(10, -06);
    const double conv_tol_dbl = pow(10, -10);

    const double precond_error_tol = pow(10, -10);

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
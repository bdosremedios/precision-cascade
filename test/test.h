#ifndef TEST_H
#define TEST_H

#include <cmath>
#include <memory>
#include <iostream>
#include <filesystem>

#include "gtest/gtest.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include "types/types.h"
#include "tools/argument_pkgs.h"
#include "tools/math_functions.h"
#include "tools/MatrixReader.h"

namespace fs = std::filesystem;
using std::shared_ptr, std::make_shared;
using std::cout, std::endl;

template <template <typename> typename M, typename T>
T mat_max_mag(const M<T> &A) {

    T abs_max = static_cast<T>(0);
    for (int i=0; i<A.rows(); ++i) {
        for (int j=0; j<A.cols(); ++j) {
            if (std::abs(A.coeff(i, j)) > abs_max) { abs_max = std::abs(A.coeff(i, j)); }
        }
    }
    return abs_max;

}

template <typename T>
T vec_max_mag(const MatrixVector<T> &vec) {

    T abs_max = static_cast<T>(0);
    for (int i=0; i<vec.rows(); ++i) {
            if (std::abs(vec(i)) > abs_max) { abs_max = std::abs(vec(i)); }
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
void ASSERT_VECTOR_NEAR(MatrixVector<T> &test, MatrixVector<T> &target, T tol) {

    ASSERT_EQ(test.rows(), target.rows());

    for (int i=0; i<target.rows(); ++i) {
        ASSERT_NEAR(test.get_elem(i), target.get_elem(i), tol);
    }

}

template <typename T>
void ASSERT_VECTOR_EQ(MatrixVector<T> &test, MatrixVector<T> &target) {
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

    // Core values
    static double roundoff() { assert(false); return -1.; }
    static double gamma(int n) { return n*roundoff()/(1-n*roundoff()); }

    // Special case values
    static double dbl_loss_of_ortho_tol() { return std::pow(10, 2)*roundoff(); }
    static double dbl_substitution_tol() { return std::pow(10, 2)*roundoff(); }
    static double matlab_dbl_near() { return std::pow(10, -14); }

    // Preconditioner error test tolerance
    static double dbl_inv_elem_tol() { return std::pow(10, -12); }
    static double dbl_ilu_elem_tol() { return std::pow(10, -12); }

    // Iterative solver convergence tolerance
    static double Tol<T>::stationary_conv_tol() { assert(false); return -1.; }
    static double Tol<T>::krylov_conv_tol() { assert(false); return -1.; }
    static double Tol<T>::nested_krylov_conv_tol() { assert(false); return -1.; }

};

// Half Constants
template<>
double Tol<__half>::roundoff() { return std::pow(2, -10); }
template<>
double Tol<__half>::stationary_conv_tol() { return 5*std::pow(10, -02); }
template<>
double Tol<__half>::krylov_conv_tol() { return 5*std::pow(10, -02); }
template<>
double Tol<__half>::nested_krylov_conv_tol() { return 5*std::pow(10, -02); }

// Single Constants
template<>
double Tol<float>::roundoff() { return std::pow(2, -23); }
template<>
double Tol<float>::stationary_conv_tol() { return 5*std::pow(10, -06); }
template<>
double Tol<float>::krylov_conv_tol() { return 5*std::pow(10, -06); }
template<>
double Tol<float>::nested_krylov_conv_tol() { return 5*std::pow(10, -06); }

// Double Constants
template<>
double Tol<double>::roundoff() { return std::pow(2, -52); }
template<>
double Tol<double>::stationary_conv_tol() { return std::pow(10, -10); }
template<>
double Tol<double>::krylov_conv_tol() { return std::pow(10, -10); }
template<>
double Tol<double>::nested_krylov_conv_tol() { return std::pow(10, -10); }

class TestBase: public testing::Test
{
public:

    const fs::path read_matrix_dir = (
        fs::current_path() / fs::path("..") / fs::path("test") / fs::path("read_matrices")
    );
    const fs::path solve_matrix_dir = (
        fs::current_path() / fs::path("..") / fs::path("test") / fs::path("solve_matrices")
    );

    SolveArgPkg default_args;
    static bool *show_plots;
    static cublasHandle_t *handle_ptr;

};

#endif
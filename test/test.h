#ifndef TEST_H
#define TEST_H

#include <cmath>
#include <filesystem>
#include <memory>
#include <iostream>

#include "gtest/gtest.h"

#include "types/types.h"

#include "test_assertions.h"

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

    // SolveArgPkg default_args;
    static bool *show_plots;
    static cublasHandle_t *handle_ptr;

};

#endif
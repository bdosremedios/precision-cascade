#ifndef TEST_TOOLS_H
#define TEST_TOOLS_H

#include <cmath>

#include "types/types.h"

template <template <typename> typename M, typename T>
T mat_max_mag(const M<T> &A) {

    T abs_max = static_cast<T>(0);
    for (int i=0; i<A.rows(); ++i) {
        for (int j=0; j<A.cols(); ++j) {
            if (std::abs(A.get_elem(i, j).get_scalar()) > abs_max) {
                abs_max = std::abs(A.get_elem(i, j).get_scalar());
            }
        }
    }
    return abs_max;

}

template <typename T>
T min_1_mag(const T &val) {
    return static_cast<T>(std::max(std::abs(static_cast<double>(val)), 1.));
}

template <template <typename> typename M, typename T>
int count_zeros(M<T> A, double zero_tol) {

    int count = 0;
    for (int i=0; i<A.rows(); ++i) {
        for (int j=0; j<A.cols(); ++j) {
            if (abs(A.get_elem(i, j).get_scalar()) <= zero_tol) { ++count; }
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
    static T roundoff_T() { return static_cast<T>(roundoff()); }
    static double gamma(int n) { return n*roundoff()/(1-n*roundoff()); }

    // Special case values
    static double substitution_tol(int n) { return gamma(n); }
    static T substitution_tol_T(int n) { return static_cast<T>(substitution_tol(n)); }
    static double dbl_loss_of_ortho_tol(int n) { return gamma(n); }
    static double matlab_dbl_near() { return std::pow(10, -14); }

    // Preconditioner error test tolerance
    static T inv_elem_tol() { return static_cast<T>(std::pow(10, 3)*roundoff()); }
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

#endif
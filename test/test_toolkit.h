#ifndef TEST_TOOLKIT_H
#define TEST_TOOLKIT_H

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

    // Basic error tolerance
    static double roundoff();
    static T roundoff_T() { return static_cast<T>(roundoff()); }
    static double gamma(int n) { return n*roundoff()/(1-n*roundoff()); }  // 2002 Higham Ch.3

    // Algorithm accumulation error tolerance
    static double substitution_tol(double cond, int n) {
        return cond*gamma(n)/(1-cond*gamma(n)); // 2002 Higham Ch.8
    }
    static T substitution_tol_T(double cond, int n) {
        return static_cast<T>(substitution_tol(cond, n));
    }
    static double loss_of_ortho_tol(double cond, int n) { // 1967 Giraud
        double scale_c_on_n = 1.74*std::sqrt(static_cast<double>(n))*(static_cast<double>(n)+1);
        return scale_c_on_n*cond*roundoff();
    }
    static T loss_of_ortho_tol_T(int n) { return static_cast<T>(loss_of_ortho_tol(n)); }

    // Special case values
    static double matlab_dbl_near() { return std::pow(10, -14); }

    // Preconditioner error test tolerance
    static T inv_elem_tol() { return static_cast<T>(std::pow(10, 3)*roundoff()); }
    static double dbl_ilu_elem_tol() { return std::pow(10, -12); }

    // Iterative solver convergence tolerance
    static double stationary_conv_tol();
    static double krylov_conv_tol();
    static double nested_krylov_conv_tol();

};

#endif
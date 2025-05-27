#ifndef TEST_TOOLKIT_H
#define TEST_TOOLKIT_H

#include "tools/abs.h"
#include "types/types.h"

#include <cmath>

using namespace cascade;

template <template <typename> typename TMatrix, typename TPrecision>
TPrecision mat_max_mag(const TMatrix<TPrecision> &A) {

    TPrecision abs_max = static_cast<TPrecision>(0);
    for (int i=0; i<A.rows(); ++i) {
        for (int j=0; j<A.cols(); ++j) {
            if (abs_ns::abs(A.get_elem(i, j).get_scalar()) > abs_max) {
                abs_max = abs_ns::abs(A.get_elem(i, j).get_scalar());
            }
        }
    }
    return abs_max;

}

template <typename TPrecision>
class Tol
{
public:

    // Basic roundoff unit
    static double roundoff();
    static TPrecision roundoff_T() {
        return static_cast<TPrecision>(roundoff());
    }

    // Gamma accumulation coefficient 2002 Higham Ch.3
    static double gamma(int n) {
        return n*roundoff()/(1-n*roundoff());
    }
    static TPrecision gamma_T(int n) {
        return static_cast<TPrecision>(gamma(n));
    }

    // Gamma tilde for Given's rotation QR error Higham 2002 Ch.19
    static double gamma_tilde(int r) {
        return std::pow((1+std::sqrt(2)*gamma(6)), r)-1.;
    }
    static double gamma_tilde_T(int r) {
        return static_cast<TPrecision>(gamma_tilde(r));
    }

    // Triangular substitution error tolerance 2002 Higham Ch.8
    static double substitution_tol(double cond, int n) {
        return cond*gamma(n)/(1-cond*gamma(n));
    }
    static TPrecision substitution_tol_T(double cond, int n) {
        return static_cast<TPrecision>(substitution_tol(cond, n));
    }

    // Preconditioner error test tolerance
    static TPrecision inv_elem_tol() {
        return static_cast<TPrecision>(std::pow(10, 3)*roundoff());
    }
    static double dbl_ilu_elem_tol() { return std::pow(10, -10); }

    // Iterative solver convergence tolerance
    static double stationary_conv_tol();
    static double krylov_conv_tol();
    static double nested_krylov_conv_tol();

};

template <template <typename> typename TMatrix, typename TPrecision>
class CommonMatRandomInterface
{
public:

    static TMatrix<TPrecision> rand_matrix(
        cuHandleBundle cu_handles, int arg_m_rows, int arg_n_cols
    ) {
        throw std::runtime_error(
            "CommonRandomInterface: reached unimplemented default "
            "implementation in generate_rand_matrix"
        );
    }

};

template <typename TPrecision>
class CommonMatRandomInterface<MatrixDense, TPrecision>
{
public:

    static MatrixDense<TPrecision> rand_matrix(
        cuHandleBundle cu_handles, int arg_m_rows, int arg_n_cols
    ) {
        return MatrixDense<TPrecision>::Random(
            cu_handles, arg_m_rows, arg_n_cols
        );
    }

};

template <typename TPrecision>
class CommonMatRandomInterface<NoFillMatrixSparse, TPrecision>
{
private:

    inline static double sparse_fill_ratio = 0.67;

public:

    static NoFillMatrixSparse<TPrecision> rand_matrix(
        cuHandleBundle cu_handles, int arg_m_rows, int arg_n_cols
    ) {
        return NoFillMatrixSparse<TPrecision>::Random(
            cu_handles, arg_m_rows, arg_n_cols, sparse_fill_ratio
        );
    }

};

#endif
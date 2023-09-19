#ifndef IMPLEMENTED_PRECONDITIONERS_H
#define IMPLEMENTED_PRECONDITIONERS_H

#include "Eigen/Dense"

#include <cmath>
#include <functional>

#include "Preconditioner.h"
#include "../tools/Substitution.h"

using Eigen::Matrix, Eigen::Dynamic;

using std::abs;

template <template <typename> typename M, typename T>
class NoPreconditioner: public Preconditioner<M, T> {

    public:

    using Preconditioner<M, T>::Preconditioner;

    MatrixVector<T> action_inv_M(MatrixVector<T> const &vec) const override {
        return vec;
    }

    bool check_compatibility_left(int const &arg_m) const override { return true; };
    bool check_compatibility_right(int const &arg_n) const override { return true; };

};

template <template <typename> typename M, typename T>
class MatrixInverse: public Preconditioner<M, T> {

    public:

    M<T> inv_M;

    MatrixInverse(M<T> const &arg_inv_M): inv_M(arg_inv_M) {}

    bool check_compatibility_left(int const &arg_m) const override {
        return ((inv_M.cols() == arg_m) && (inv_M.rows() == arg_m));
    };

    bool check_compatibility_right(int const &arg_n) const override {
        return ((inv_M.cols() == arg_n) && (inv_M.rows() == arg_n));
    };

    MatrixVector<T> action_inv_M(MatrixVector<T> const &vec) const override {
        return inv_M*vec;
    }

};

template <template <typename> typename M, typename T>
class ILU: public Preconditioner<M, T> {

    public:

    M<T> L;
    M<T> U;
    int m;

    // ILU(0)
    ILU(M<T> const &A, T const &zero_tol) {

        std::function<bool(T const &, T const &, int &, int &)> drop_if_orig_0 = [zero_tol] (
            T const &entry, T const &orig_entry, int &i, int&j
        ) -> bool { return (abs(orig_entry) <= zero_tol); };

        constructionHelper(A, zero_tol, drop_if_orig_0, drop_if_orig_0);

    }

    // ILUT(eps)
    ILU(M<T> const &A, T const &zero_tol, T const &eps) {

        std::function<bool(T const &, T const &, int &, int &)> drop_if_lt = [eps] (
            T const &entry, T const &orig_entry, int &i, int&j
        ) -> bool { return (abs(entry) <= eps); };

        // Guard against removing diagonal entries to maintain non-singularity
        std::function<bool(T const &, T const &, int &, int &)> drop_if_lt_skip_diag = [eps] (
            T const &entry, T const &orig_entry, int &i, int&j
        ) -> bool {
            if (i != j) {
                return (abs(entry) <= eps);
            } else {
                return false;
            }
        };

        constructionHelper(A, zero_tol, drop_if_lt, drop_if_lt_skip_diag);

    }

    void constructionHelper(
        M<T> const &A,
        T const &zero_tol,
        std::function<bool(T const &, T const &, int &, int &)> drop_rule_rho,
        std::function<bool(T const &, T const &, int &, int &)> drop_rule_tau
    ) {
        
        if (A.rows() != A.cols()) { throw runtime_error("Non square matrix A"); }

        m = A.rows();
        L = M<T>::Identity(m, m);
        U = M<T>::Zero(m, m);

        // Set U to be A and perform ILU
        U = A;

        // Use IKJ variant for better predictability of modification
        for (int i=0; i<m; ++i) {

            for (int k=0; k<=i-1; ++k) {

                // Ensure pivot is non-zero
                if (abs(U.coeffRef(k, k)) > zero_tol) {
    
                    T coeff = U.coeff(i, k)/U.coeff(k, k);
                    // Apply rho dropping rule to zero-ing val, skipping subsequent
                    // calcs in row if dropped
                    if (!drop_rule_rho(coeff, A.coeff(i, k), i, k)) {
                        L.coeffRef(i, k) = coeff;
                        for (int j=k+1; j<m; ++j) {
                            U.coeffRef(i, j) = U.coeff(i, j) - coeff*U.coeff(k, j);
                        }
                    } else {
                        L.coeffRef(i, k) = static_cast<T>(0);
                    }
                    U.coeffRef(i, k) = static_cast<T>(0);

                } else {
                    throw runtime_error("ILU encountered zero diagonal entry");
                }

            }
            
            // Iterate through the row again to ensure enforcement of 2nd drop rule
            for (int j=0; j<m; ++j) {
                if (drop_rule_tau(L.coeff(i, j), A.coeff(i, j), i, j)) { L(i, j) = 0; }
                if (drop_rule_tau(U.coeff(i, j), A.coeff(i, j), i, j)) { U(i, j) = 0; }
            }

        }

    }

    MatrixVector<T> action_inv_M(MatrixVector<T> const &vec) const override {
        return frwd_substitution(L, back_substitution(U, vec));
    }

    M<T> get_L() const { return L; }
    M<T> get_U() const { return U; }

    bool check_compatibility_left(int const &arg_m) const override { return arg_m == m; };
    bool check_compatibility_right(int const &arg_n) const override { return arg_n == m; };

};

#endif
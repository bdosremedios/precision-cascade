#ifndef IMPLEMENTED_PRECONDITIONERS_H
#define IMPLEMENTED_PRECONDITIONERS_H

#include "Eigen/Dense"

#include <cmath>
#include <functional>

#include "Preconditioner.h"
#include "../tools/Substitution.h"

using Eigen::Matrix, Eigen::Dynamic;

using std::abs;

template <template <typename> typename M, typename W>
class NoPreconditioner: public Preconditioner<M, W>
{
public:

    using Preconditioner<M, W>::Preconditioner;

    MatrixVector<W> action_inv_M(MatrixVector<W> const &vec) const override {
        return vec;
    }

    bool check_compatibility_left(int const &arg_m) const override { return true; };
    bool check_compatibility_right(int const &arg_n) const override { return true; };

};

template <template <typename> typename M, typename W>
class MatrixInverse: public Preconditioner<M, W>
{
public:

    M<W> inv_M;

    MatrixInverse(M<W> const &arg_inv_M): inv_M(arg_inv_M) {}

    bool check_compatibility_left(int const &arg_m) const override {
        return ((inv_M.cols() == arg_m) && (inv_M.rows() == arg_m));
    };

    bool check_compatibility_right(int const &arg_n) const override {
        return ((inv_M.cols() == arg_n) && (inv_M.rows() == arg_n));
    };

    MatrixVector<W> action_inv_M(MatrixVector<W> const &vec) const override {
        return inv_M*vec;
    }

};

template <template <typename> typename M, typename W>
class ILU: public Preconditioner<M, W>
{
protected:

    M<W> L;
    M<W> U;
    int m;

public:

    // ILU taking premade L and U
    ILU(M<W> arg_L, M<W> arg_U) {

        if (arg_L.rows() != arg_L.cols()) { throw runtime_error("Non square matrix L"); }
        if (arg_U.rows() != arg_U.cols()) { throw runtime_error("Non square matrix U"); }
        if (arg_L.rows() != arg_U.rows()) { throw runtime_error("L and U mismatch"); }

        L = arg_L;
        U = arg_U;

        L.makeCompressed();
        U.makeCompressed();

    }

    // ILU(0)
    ILU(const M<W> &A, W zero_tol) {

        std::function<bool(W const &, W const &, int &, int &)> drop_if_orig_0 = [zero_tol] (
            W const &entry, W const &orig_entry, int &i, int&j
        ) -> bool { return (abs(orig_entry) <= zero_tol); };

        constructionHelper(A, zero_tol, drop_if_orig_0, drop_if_orig_0);

    }

    // ILUT(eps)
    ILU(M<W> const &A, W zero_tol, W eps) {

        std::function<bool(W const &, W const &, int &, int &)> drop_if_lt = [eps] (
            W const &entry, W const &orig_entry, int &i, int&j
        ) -> bool { return (abs(entry) <= eps); };

        // Guard against removing diagonal entries to maintain non-singularity
        std::function<bool(W const &, W const &, int &, int &)> drop_if_lt_skip_diag = [eps] (
            W const &entry, W const &orig_entry, int &i, int&j
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
        M<W> const &A,
        W const &zero_tol,
        std::function<bool(W const &, W const &, int &, int &)> drop_rule_rho,
        std::function<bool(W const &, W const &, int &, int &)> drop_rule_tau
    ) {
        
        if (A.rows() != A.cols()) { throw runtime_error("Non square matrix A"); }

        m = A.rows();
        L = M<W>::Identity(m, m);
        U = M<W>::Zero(m, m);

        // Set U to be A and perform ILU
        U = A;

        // Use IKJ variant for better predictability of modification
        for (int i=0; i<m; ++i) {

            for (int k=0; k<=i-1; ++k) {

                // Ensure pivot is non-zero
                if (abs(U.coeffRef(k, k)) > zero_tol) {
    
                    W coeff = U.coeff(i, k)/U.coeff(k, k);
                    // Apply rho dropping rule to zero-ing val, skipping subsequent
                    // calcs in row if dropped
                    if (!drop_rule_rho(coeff, A.coeff(i, k), i, k)) {
                        L.coeffRef(i, k) = coeff;
                        for (int j=k+1; j<m; ++j) {
                            U.coeffRef(i, j) = U.coeff(i, j) - coeff*U.coeff(k, j);
                        }
                    } else {
                        L.coeffRef(i, k) = static_cast<W>(0);
                    }
                    U.coeffRef(i, k) = static_cast<W>(0);

                } else {
                    throw runtime_error("ILU encountered zero diagonal entry");
                }

            }
            
            // Iterate through the row again to ensure enforcement of 2nd drop rule
            for (int j=0; j<m; ++j) {
                if (drop_rule_tau(L.coeff(i, j), A.coeff(i, j), i, j)) { L.coeffRef(i, j) = 0; }
                if (drop_rule_tau(U.coeff(i, j), A.coeff(i, j), i, j)) { U.coeffRef(i, j) = 0; }
            }

        }

        L.makeCompressed();
        U.makeCompressed();

    }

    MatrixVector<W> action_inv_M(const MatrixVector<W> &vec) const override {
        return back_substitution<W>(U, frwd_substitution<W>(L, vec));
    }

    M<W> get_L() const { return L; }
    M<W> get_U() const { return U; }

    template <typename T>
    M<T> get_L_cast() const { return L.template cast<T>(); }

    template <typename T>
    M<T> get_U_cast() const { return U.template cast<T>(); }

    bool check_compatibility_left(const int &arg_m) const override { return arg_m == m; };
    bool check_compatibility_right( const int &arg_n) const override { return arg_n == m; };

};

#endif
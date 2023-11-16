#ifndef IMPLEMENTED_PRECONDITIONERS_H
#define IMPLEMENTED_PRECONDITIONERS_H

#include "Preconditioner.h"
#include "../tools/Substitution.h"
#include "../tools/VectorSort.h"

#include <cmath>
#include <functional>

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
private:

    void reduce_matrices() {
        L.reduce();
        U.reduce();
        P.reduce();
    }

protected:

    int m;
    M<W> L;
    M<W> U;
    M<W> P;

    void construction_helper(
        const M<W> &A, const W &zero_tol, const bool &pivot, 
        std::function<bool (const W &curr_val, const W &zero_tol)> drop_rule_tau,
        std::function<void (MatrixVector<W> &vec, const W &zero_tol)> apply_drop_rule_p
    ) {

        if (A.rows() != A.cols()) { throw runtime_error("Non square matrix A"); }

        m = A.rows();
        L = M<W>::Identity(m, m);
        U = A;
        P = M<W>::Identity(m, m);

        // Use IKJ variant for better predictability in execution
        for (int i=0; i<m; ++i) {

            // Perform row elimination
            for (int k=0; k<i; ++k) {
                W pivot = U.coeff(k, k);
                if (abs(pivot) > zero_tol) {
                    W val_to_zero = U.coeff(i, k);
                    if (abs(val_to_zero) > zero_tol) {
                        W l_ik = val_to_zero/pivot;
                        L.coeffRef(i, k) = l_ik;
                        for (int j=k+1; j<m; ++j) {
                            if (drop_rule_tau(U.coeff(i, j), zero_tol)) {
                                U.coeffRef(i, j) = static_cast<W>(0);
                            } else {
                                U.coeffRef(i, j) -= l_ik*U.coeff(k, j);
                            }
                        }
                    }
                    U.coeffRef(i, k) = static_cast<W>(0);
                } else { throw runtime_error("ILU encountered zero pivot in elimination"); }
            }

            // Apply drop rule to row
            MatrixVector<W> U_vec(U.cols());
            MatrixVector<W> L_vec(L.cols());
            for (int l=i+1; l<U.cols(); ++l) { U_vec(l) = U.coeff(i, l); }
            for (int l=0; l<i; ++l) { L_vec(l) = L.coeff(i, l); }
            apply_drop_rule_p(U_vec, zero_tol);
            apply_drop_rule_p(L_vec, zero_tol);
            for (int l=i+1; l<U.cols(); ++l) { U.coeffRef(i, l) = U_vec(l); }
            for (int l=0; l<i; ++l) { L.coeffRef(i, l) = L_vec(l); }

            // Pivot columns and throw error if no column entry large enough. Do after such that
            // row elimination has already occured so that largest pivot right now can be found
            if (pivot) {
                int pivot = i;
                W abs_max_val = zero_tol;
                for (int ind=i; ind<m; ++ind) {
                    W temp = abs(U.coeff(i, ind));
                    if (abs(temp) > abs_max_val) {
                        abs_max_val = temp;
                        pivot = ind;
                    }
                }
                if (abs_max_val <= zero_tol) { throw runtime_error("ILU encountered not large enough pivot error"); }
                if (i != pivot) {
                    const MatrixVector<W> U_i = U.col(i);
                    U.col(i) = U.col(pivot);
                    U.col(pivot) = U_i;

                    const MatrixVector<W> P_i = P.col(i);
                    P.col(i) = P.col(pivot);
                    P.col(pivot) = P_i;
                }
            }
    
        }

        reduce_matrices();

    }

public:

    // ILU constructor taking premade L and U and no permutation
    ILU(const M<W> &arg_L, const M<W> &arg_U):
        ILU(arg_L, arg_U, M<W>::Identity(arg_L.rows(), arg_L.rows()))
    {}

    // ILU constructor taking premade L, U, and P
    ILU(const M<W> &arg_L, const M<W> &arg_U, const M<W> &arg_P) {

        if (arg_L.rows() != arg_L.cols()) { throw runtime_error("Non square matrix L"); }
        if (arg_U.rows() != arg_U.cols()) { throw runtime_error("Non square matrix U"); }
        if (arg_P.rows() != arg_P.cols()) { throw runtime_error("Non square matrix P"); }
        if (arg_L.rows() != arg_U.rows()) { throw runtime_error("L and U mismatch"); }
        if (arg_L.rows() != arg_P.rows()) { throw runtime_error("L and U mismatch"); }

        m = arg_L.rows();
        L = arg_L;
        U = arg_U;
        P = arg_P;

        reduce_matrices();

    }

    // ILU(0)
    ILU(M<W> const &A, const W &zero_tol, const bool &pivot) {
        
        std::function<bool (W const &, W const &)> drop_rule_tau = (
            [] (W const &curr_val, W const &_zero_tol) -> bool { return (abs(curr_val) <= _zero_tol); }
        );

        std::function<void (MatrixVector<W> &, const W &)> apply_drop_rule_p = (
            [] (MatrixVector<W> &_1, W const &_2) -> void {}
        );

        construction_helper(A, zero_tol, pivot, drop_rule_tau, apply_drop_rule_p);

    }

    // ILUT(tau, p), tau threshold to drop and p number of entries to keep
    ILU(const M<W> &A, const W &tau, const int &p, const W &zero_tol, const bool &pivot) {

        std::function<bool (const W &, const W &)> drop_rule_tau = (
            [tau] (const W &curr_val, const W &_zero_tol) -> bool { return (abs(curr_val) <= tau); }
        );

        std::function<void (MatrixVector<W> &, const W &)> apply_drop_rule_p = (
            [p] (MatrixVector<W> &vec, const W &_) -> void { 
                if (p < vec.rows()) {
                    MatrixVector<int> sorted_indices = sort_indices(vec);
                    for (int i=vec.rows()-1; i > p-1; --i) { 
                        vec(sorted_indices(i)) = static_cast<W>(0);
                    }
                };
            }
        );

        construction_helper(A, zero_tol, pivot, drop_rule_tau, apply_drop_rule_p);

    }

    MatrixVector<W> action_inv_M(const MatrixVector<W> &vec) const override {
        return P*(back_substitution<W>(U, frwd_substitution<W>(L, vec)));
    }

    M<W> get_L() const { return L; }
    M<W> get_U() const { return U; }
    M<W> get_P() const { return P; }

    template <typename T>
    M<T> get_L_cast() const { return L.template cast<T>(); }

    template <typename T>
    M<T> get_U_cast() const { return U.template cast<T>(); }

    bool check_compatibility_left(const int &arg_m) const override { return arg_m == m; };
    bool check_compatibility_right( const int &arg_n) const override { return arg_n == m; };

};

#endif
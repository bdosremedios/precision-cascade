#ifndef IMPLEMENTED_PRECONDITIONERS_H
#define IMPLEMENTED_PRECONDITIONERS_H

#include "Preconditioner.h"
#include "../tools/Substitution.h"
#include "../tools/VectorSort.h"

#include <cmath>
#include <functional>
#include <iostream>

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
        std::function<void (const int &row, const W &zero_tol)> apply_drop_rule_vec
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
                            // Apply drop_rule_tau but dont remove row pivot regardless of drop_rule_tau
                            if ((i != j) && drop_rule_tau(U.coeff(i, j), zero_tol)) {
                                U.coeffRef(i, j) = static_cast<W>(0);
                            } else {
                                U.coeffRef(i, j) -= l_ik*U.coeff(k, j);
                            }
                        }
                    }
                    U.coeffRef(i, k) = static_cast<W>(0);
                } else {
                    throw runtime_error("ILU encountered zero pivot in elimination"); }
            }

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

            apply_drop_rule_vec(i, zero_tol);
    
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
            [] (W const &curr_val, W const &_zero_tol) -> bool {
                return (abs(curr_val) <= _zero_tol);
            }
        );

        std::function<void (const int &, const W &)> apply_drop_rule_vec = (
            [] (const int &, const W &) -> void {}
        );

        construction_helper(A, zero_tol, pivot, drop_rule_tau, apply_drop_rule_vec);

    }

    // ILUT(tau, p), tau threshold to drop and p number of entries to keep
    ILU(const M<W> &A, const W &tau, const int &p, const W &zero_tol, const bool &pivot) {

        std::function<bool (const W &, const W &)> drop_rule_tau = (
            [tau] (const W &curr_val, const W &_zero_tol) -> bool {
                return (abs(curr_val) <= tau);
            }
        );

        std::function<void (const int &, const W &)> apply_drop_rule_vec = (
            [this, p, drop_rule_tau] (const int &row_i, const W &_zero_tol) -> void { 

                int row_mat = row_i+1;
                int U_size = U.cols()-row_mat;
                int L_size = row_mat-1;

                // Modify U vec (skip diagonal)
                MatrixVector<W> U_vec = MatrixVector<W>::Zero(U_size);
                for (int l=0; l<U_size; ++l) { // Apply tau drop rule
                    if (!drop_rule_tau(U.coeff(row_i, l), _zero_tol)) {
                        U_vec(l) = U.coeff(row_i, l);
                    }
                }
                if (p < U_size) { // Drop p largest elements
                    MatrixVector<int> U_sorted_indices = sort_indices(U_vec);
                    for (int i=0; i<U_size-p; ++i) {
                        U_vec(U_sorted_indices(i)) = static_cast<W>(0);
                    }
                }

                // Modify L vec (skip diagonal)
                MatrixVector<W> L_vec = MatrixVector<W>::Zero(L_size);
                for (int l=0; l<L_size; ++l) { // Apply tau drop rule
                    if (!drop_rule_tau(L.coeff(row_i, l), _zero_tol)) {
                        L_vec(l) = L.coeff(row_i, l);
                    }
                }
                if (p < L_size) { // Drop p largest elements
                    MatrixVector<int> L_sorted_indices = sort_indices(L_vec);
                    for (int i=0; i<L_size-p; ++i) {
                        L_vec(L_sorted_indices(i)) = static_cast<W>(0);
                    }
                }

                for (int l=0; l<U_size; ++l) { U.coeffRef(row_i, row_i+1+l) = U_vec(l); }
                for (int l=0; l<L_size; ++l) { L.coeffRef(row_i, l) = L_vec(l); }

            }

        );

        construction_helper(A, zero_tol, pivot, drop_rule_tau, apply_drop_rule_vec);

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
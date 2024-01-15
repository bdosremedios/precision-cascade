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

    int m;
    M<W> L;
    M<W> U;
    M<W> P;
    std::function<bool (const W &curr_val, const int &row, const int &col, const W &zero_tol)> drop_rule_tau;
    std::function<void (const int &col, const W &zero_tol)> apply_drop_rule_vec;

    void reduce_matrices() {
        L.reduce();
        U.reduce();
        P.reduce();
    }

    void instantiate_vars(const M<W> &A) {
        m = A.rows();
        L = M<W>::Identity(m, m);
        U = A;
        P = M<W>::Identity(m, m);
    }

    // Left looking LU factorization for better memory access
    void execute_leftlook_col_elimination(const int &col_j, const W &zero_tol, bool to_pivot_row) {

        // Apply previous kth column's zeroing effects and apply drop_rule_tau
        for (int k=0; k<col_j; ++k) {
            
            W propogating_val = U.coeff(k, col_j);
            if ((abs(propogating_val) <= zero_tol) || (drop_rule_tau(propogating_val, k, col_j, zero_tol))) {
                U.coeffRef(k, col_j) = static_cast<W>(0);
            } else {
                for (int i=k+1; i<m; ++i) {
                    U.coeffRef(i, col_j) -= L.coeff(i, k)*propogating_val;
                }
            }

        }

        // Permute row to get largest pivot
        if (to_pivot_row) { pivot_row(col_j); }

        // Zero elements below diagonal in col_j
        W pivot = U.coeff(col_j, col_j);
        if (abs(pivot) > zero_tol) {

            for (int i=col_j+1; i<m; ++i) {

                W val_to_zero = U.coeff(i, col_j);
                if (abs(val_to_zero) > zero_tol) {

                    W l_ij = val_to_zero/pivot;
                    if (abs(l_ij) > zero_tol) {
                        L.coeffRef(i, col_j) = l_ij;
                    }
                    U.coeffRef(i, col_j) = static_cast<W>(0);

                }

            }

        } else { throw runtime_error("ILU has zero pivot in elimination"); }

        // Apply drop rule to column
        apply_drop_rule_vec(col_j, zero_tol);

    }

    void pivot_row(const int &beg_j) {

        // Find largest pivot
        int pivot_i = beg_j;
        W largest_val = abs(U.coeff(beg_j, beg_j));
        for (int i=beg_j; i<m; ++i) {
            W temp = abs(U.coeff(i, beg_j));
            if (abs(temp) > largest_val) {
                largest_val = temp;
                pivot_i = i;
            }
        }

        // Pivot row element by element
        if (beg_j != pivot_i) {
            for (int j=beg_j; j<m; ++j) {
                W temp = U.coeff(beg_j, j);
                U.coeffRef(beg_j, j) = U.coeff(pivot_i, j);
                U.coeffRef(pivot_i, j) = temp;
            }
            for (int j=0; j<beg_j; ++j) {
                W temp = L.coeff(beg_j, j);
                L.coeffRef(beg_j, j) = L.coeff(pivot_i, j);
                L.coeffRef(pivot_i, j) = temp;
            }
            for (int j=0; j<m; ++j) {
                if ((P.coeff(beg_j, j) != static_cast<W>(0)) || (P.coeff(pivot_i, j) != static_cast<W>(0))) {
                    W temp = P.coeff(beg_j, j);
                    P.coeffRef(beg_j, j) = P.coeff(pivot_i, j);
                    P.coeffRef(pivot_i, j) = temp;
                }
            }
        }

    }

    void construction_helper(const M<W> &A, const W &zero_tol, const bool &pivot) {

        if (A.rows() != A.cols()) { throw runtime_error("Non square matrix A"); }
        instantiate_vars(A);

        // Execute left-looking ILU
        for (int j=0; j<m; ++j) { execute_leftlook_col_elimination(j, zero_tol, pivot); }

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
        
        drop_rule_tau = [A] (
            W const &curr_val, const int &i, const int &j, W const &_zero_tol
        ) -> bool {
            return (abs(A.coeff(i, j)) <= _zero_tol);
        };

        apply_drop_rule_vec = [this] (const int &col_j, const W &_zero_tol) -> void { 

            int U_size = col_j;
            int L_size = U.rows()-(col_j+1);

            for (int l=0; l<U_size; ++l) { // Apply tau drop rule
                if (drop_rule_tau(U.coeff(l, col_j), l, col_j, _zero_tol)) {
                    U.coeffRef(l, col_j) = static_cast<W>(0);
                }
            }
            for (int l=col_j+1; l<L_size+col_j+1; ++l) { // Apply tau drop rule
                if (drop_rule_tau(L.coeff(l, col_j), l, col_j, _zero_tol)) {
                    L.coeffRef(l, col_j) = static_cast<W>(0);
                }
            }

        };

        construction_helper(A, zero_tol, pivot);

    }

    // ILUT(tau, p), tau threshold to drop and p number of entries to keep
    ILU(const M<W> &A, const W &tau, const int &p, const W &zero_tol, const bool &pivot) {

        // Calculate original A col norm for threshold comparison
        MatrixVector<W> A_j_norm(A.cols());
        for (int j=0; j<A.cols(); ++j) {
            W norm = static_cast<W>(0);
            for (int i=0; i<A.rows(); ++i) { norm += A.coeff(i, j)*A.coeff(i, j); }
            A_j_norm(j) = std::sqrt(norm);
        }

        drop_rule_tau = [A_j_norm, tau] (
            const W &curr_val, const int &i, const int &col_j, const W &_zero_tol
        ) -> bool {
            return (abs(curr_val) <= tau*A_j_norm(col_j));
        };

        apply_drop_rule_vec = [this, p] (const int &col_j, const W &_zero_tol) -> void { 

            int U_size = col_j;
            int L_size = U.rows()-(col_j+1);

            // Modify col_j in U
            for (int l=0; l<U_size; ++l) { // Apply tau drop rule
                if (drop_rule_tau(U.coeff(l, col_j), l, col_j, _zero_tol)) {
                    U.coeffRef(l, col_j) = static_cast<W>(0);
                }
            }
            if (p < U_size) { // Drop all but p largest elements
                MatrixVector<W> U_j(U_size);
                for (int i=0; i<U_size; ++i) { U_j(i) = U.coeff(i, col_j); }
                MatrixVector<int> U_sorted_indices = sort_indices(U_j);
                for (int i=0; i<U_size-p; ++i) {
                    U.coeffRef(U_sorted_indices(i), col_j) = static_cast<W>(0);
                }
            }

            // Modify col_j in L
            for (int l=col_j+1; l<L_size+col_j+1; ++l) { // Apply tau drop rule
                if (drop_rule_tau(L.coeff(l, col_j), l, col_j, _zero_tol)) {
                    L.coeffRef(l, col_j) = static_cast<W>(0);
                }
            }
            if (p < L_size) { // Drop all but p largest elements
                MatrixVector<W> L_j(L_size);
                for (int i=0; i<L_size; ++i) { L_j(i) = L.coeff(i+col_j+1, col_j); }
                MatrixVector<int> L_sorted_indices = sort_indices(L_j);
                for (int i=0; i<L_size-p; ++i) {
                    L.coeffRef(L_sorted_indices(i)+col_j+1, col_j) = static_cast<W>(0);
                }
            }

        };

        construction_helper(A, zero_tol, pivot);

    }

    MatrixVector<W> action_inv_M(const MatrixVector<W> &vec) const override {
        return back_substitution<W>(U, frwd_substitution<W>(L, P*vec));
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
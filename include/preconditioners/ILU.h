#ifndef ILU_H
#define ILU_H

#include "Preconditioner.h"

#include <cmath>
#include <vector>
#include <functional>

template <template <typename> typename M, typename W>
class ILU: public Preconditioner<M, W>
{
private:

    int m;
    M<W> L = M<W>(NULL);
    M<W> U = M<W>(NULL);
    M<W> P = M<W>(NULL);
    std::function<bool (const W &curr_val, const int &row, const int &col, const W &zero_tol)> drop_rule_tau;
    std::function<void (const int &col, const W &zero_tol)> apply_drop_rule_vec;

    void reduce_matrices() {
        L.reduce();
        U.reduce();
        P.reduce();
    }

    void instantiate_vars(const M<W> &A) {
        m = A.rows();
        L = M<W>::Identity(A.get_handle(), m, m);
        U = A;
        P = M<W>::Identity(A.get_handle(), m, m);
    }

    // Left looking LU factorization for better memory access
    void execute_leftlook_col_elimination(const int &col_j, const W &zero_tol, bool to_pivot_row) {

        // Apply previous k columns' zeroing effects to column j and apply drop_rule_tau
        for (int k=0; k<col_j; ++k) {
            
            W U_k_j = U.get_elem(k, col_j);
            if ((std::abs(U_k_j) <= zero_tol) || (drop_rule_tau(U_k_j, k, col_j, zero_tol)))
            {
                U.set_elem(k, col_j, static_cast<W>(0));
            } else {
                for (int i=k+1; i<m; ++i) {
                    U.set_elem(i, col_j, U.get_elem(i, col_j)-L.get_elem(i, k)*U_k_j);
                }
            }

        }

        // Permute row to get largest pivot
        if (to_pivot_row) { pivot_row(col_j); }

        // Zero elements below diagonal in col_j
        W pivot = U.get_elem(col_j, col_j);
        if (std::abs(pivot) > zero_tol) {

            for (int i=col_j+1; i<m; ++i) {

                W val_to_zero = U.get_elem(i, col_j);
                if (std::abs(val_to_zero) > zero_tol) {

                    W l_ij = val_to_zero/pivot;
                    if (std::abs(l_ij) > zero_tol) {
                        L.set_elem(i, col_j, l_ij);
                    }
                    U.set_elem(i, col_j, static_cast<W>(0));

                }

            }

        } else { throw std::runtime_error("ILU has zero pivot in elimination"); }

        // Apply drop rule to column
        apply_drop_rule_vec(col_j, zero_tol);

    }

    void pivot_row(const int &beg_j) {

        // Find largest pivot
        int pivot_i = beg_j;
        W largest_val = std::abs(U.get_elem(beg_j, beg_j));
        for (int i=beg_j; i<m; ++i) {
            W temp = std::abs(U.get_elem(i, beg_j));
            if (std::abs(temp) > largest_val) {
                largest_val = temp;
                pivot_i = i;
            }
        }

        // Pivot row element by element
        if (beg_j != pivot_i) {
            for (int j=beg_j; j<m; ++j) {
                W temp = U.get_elem(beg_j, j);
                U.set_elem(beg_j, j, U.get_elem(pivot_i, j));
                U.set_elem(pivot_i, j, temp);
            }
            for (int j=0; j<beg_j; ++j) {
                W temp = L.get_elem(beg_j, j);
                L.set_elem(beg_j, j, L.get_elem(pivot_i, j));
                L.set_elem(pivot_i, j, temp);
            }
            for (int j=0; j<m; ++j) {
                if ((P.get_elem(beg_j, j) != static_cast<W>(0)) || (P.get_elem(pivot_i, j) != static_cast<W>(0))) {
                    W temp = P.get_elem(beg_j, j);
                    P.set_elem(beg_j, j, P.get_elem(pivot_i, j));
                    P.set_elem(pivot_i, j, temp);
                }
            }
        }

    }

    void construction_helper(const M<W> &A, const W &zero_tol, const bool &pivot) {

        if (A.rows() != A.cols()) { throw std::runtime_error("Non square matrix A"); }
        instantiate_vars(A);

        // Execute left-looking ILU
        for (int j=0; j<m; ++j) { execute_leftlook_col_elimination(j, zero_tol, pivot); }

        reduce_matrices();

    }

public:

    // ILU constructor taking premade L and U and no permutation
    ILU(const M<W> &arg_L, const M<W> &arg_U):
        ILU(arg_L, arg_U, M<W>::Identity(arg_L.get_handle(), arg_L.rows(), arg_L.rows()))
    {}

    // ILU constructor taking premade L, U, and P
    ILU(const M<W> &arg_L, const M<W> &arg_U, const M<W> &arg_P) {

        if (arg_L.rows() != arg_L.cols()) { throw std::runtime_error("ILU(L, U, P): Non square matrix L"); }
        if (arg_U.rows() != arg_U.cols()) { throw std::runtime_error("ILU(L, U, P): Non square matrix U"); }
        if (arg_P.rows() != arg_P.cols()) { throw std::runtime_error("ILU(L, U, P): Non square matrix P"); }
        if (arg_L.rows() != arg_U.rows()) { throw std::runtime_error("ILU(L, U, P): L and U mismatch"); }
        if (arg_L.rows() != arg_P.rows()) { throw std::runtime_error("ILU(L, U, P): L and U mismatch"); }

        m = arg_L.rows();
        L = arg_L;
        U = arg_U;
        P = arg_P;

        reduce_matrices();

    }

    // ILU(0)   
    ILU(const M<W> &A, const W &zero_tol, const bool &pivot) {

        drop_rule_tau = [A] (
            const W &curr_val, const int &i, const int &j, W const &_zero_tol
        ) -> bool {
            return (std::abs(A.get_elem(i, j)) <= _zero_tol);
        };

        apply_drop_rule_vec = [this] (const int &col_j, const W &_zero_tol) -> void { 

            int U_size = col_j;
            int L_size = U.rows()-(col_j+1);

            for (int l=0; l<U_size; ++l) { // Apply tau drop rule
                if (drop_rule_tau(U.get_elem(l, col_j), l, col_j, _zero_tol)) {
                    U.set_elem(l, col_j, static_cast<W>(0));
                }
            }
            for (int l=col_j+1; l<L_size+col_j+1; ++l) { // Apply tau drop rule
                if (drop_rule_tau(L.get_elem(l, col_j), l, col_j, _zero_tol)) {
                    L.set_elem(l, col_j, static_cast<W>(0));
                }
            }

        };

        construction_helper(A, zero_tol, pivot);

    }

    // ILUT(tau, p), tau threshold to drop and p number of entries to keep
    ILU(const M<W> &A, const W &tau, const int &p, const W &zero_tol, const bool &pivot) {

        // Calculate original A col norm for threshold comparison
        MatrixVector<W> A_j_norm(A.get_handle(), A.cols());
        for (int j=0; j<A.cols(); ++j) {
            W norm = static_cast<W>(0);
            for (int i=0; i<A.rows(); ++i) { norm += A.get_elem(i, j)*A.get_elem(i, j); }
            A_j_norm.set_elem(j, std::sqrt(norm));
        }

        drop_rule_tau = [A_j_norm, tau] (
            const W &curr_val, const int &i, const int &col_j, const W &_zero_tol
        ) -> bool {
            return (std::abs(curr_val) <= tau*A_j_norm.get_elem(col_j));
        };

        apply_drop_rule_vec = [this, p, &A] (const int &col_j, const W &_zero_tol) -> void { 

            int U_size = col_j;
            int L_size = U.rows()-(col_j+1);

            // Modify col_j in U
            for (int l=0; l<U_size; ++l) { // Apply tau drop rule
                if (drop_rule_tau(U.get_elem(l, col_j), l, col_j, _zero_tol)) {
                    U.set_elem(l, col_j, static_cast<W>(0));
                }
            }
            if (p < U_size) { // Drop all but p largest elements
                MatrixVector<W> U_j(A.get_handle(), U_size);
                for (int i=0; i<U_size; ++i) { U_j.set_elem(i, U.get_elem(i, col_j)); }
                std::vector<int> U_sorted_indices = U_j.sort_indices();
                for (int i=0; i<U_size-p; ++i) {
                    U.set_elem(U_sorted_indices[i], col_j, static_cast<W>(0));
                }
            }

            // Modify col_j in L
            for (int l=col_j+1; l<L_size+col_j+1; ++l) { // Apply tau drop rule
                if (drop_rule_tau(L.get_elem(l, col_j), l, col_j, _zero_tol)) {
                    L.set_elem(l, col_j, static_cast<W>(0));
                }
            }
            if (p < L_size) { // Drop all but p largest elements
                MatrixVector<W> L_j(A.get_handle(), L_size);
                for (int i=0; i<L_size; ++i) { L_j.set_elem(i, L.get_elem(i+col_j+1, col_j)); }
                std::vector<int> L_sorted_indices = L_j.sort_indices();
                for (int i=0; i<L_size-p; ++i) {
                    L.set_elem(L_sorted_indices[i]+col_j+1, col_j, static_cast<W>(0));
                }
            }

        };

        construction_helper(A, zero_tol, pivot);

    }

    MatrixVector<W> action_inv_M(const MatrixVector<W> &vec) const override {
        return U.back_sub(L.frwd_sub(P*vec));
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
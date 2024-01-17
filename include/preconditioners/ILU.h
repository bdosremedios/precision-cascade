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
    std::function<void (const int &col, const W &zero_tol, W *U_mat, W *L_mat, const int &m)> apply_drop_rule_vec;

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

    void swap_row_elem(W *mat, const int &col, const int &row_1, const int &row_2) {
        W temp = mat[row_1+col*m];
        mat[row_1+col*m] = mat[row_2+col*m];
        mat[row_2+col*m] = temp;
    }

    void pivot_row(const int &beg_col, W *U_mat, W *L_mat, W *P_mat) {

        // Find largest pivot
        int pivot_row = beg_col;
        W largest_val = std::abs(U_mat[beg_col+beg_col*m]);
        for (int i=beg_col; i<m; ++i) {
            W temp = std::abs(U_mat[i+beg_col*m]);
            if (temp > largest_val) {
                largest_val = temp;
                pivot_row = i;
            }
        }

        // Pivot row element by element
        if (beg_col != pivot_row) {
            for (int j=beg_col; j<m; ++j) {
                swap_row_elem(U_mat, j, beg_col, pivot_row);
            }
            for (int j=0; j<beg_col; ++j) {
                swap_row_elem(L_mat, j, beg_col, pivot_row);
            }
            for (int j=0; j<m; ++j) {
                if ((P_mat[beg_col+j*m] != static_cast<W>(0)) || (P_mat[pivot_row+j*m] != static_cast<W>(0))) {
                    swap_row_elem(P_mat, j, beg_col, pivot_row);
                }
            }
        }

    }

    // Left looking LU factorization for better memory access
    void execute_leftlook_col_elimination(
        const int &col_ind, const W &zero_tol, bool to_pivot_row, W *U_mat, W *L_mat, W *P_mat
    ) {

        // Apply previous columns' zeroing effects to col and apply drop_rule_tau
        for (int prev_col_ind=0; prev_col_ind<col_ind; ++prev_col_ind) {
            W earlier_col_val = U_mat[prev_col_ind+col_ind*m];
            if (
                (std::abs(earlier_col_val) <= zero_tol) ||
                (drop_rule_tau(earlier_col_val, prev_col_ind, col_ind, zero_tol))
            ) {
                U_mat[prev_col_ind+col_ind*m] = static_cast<W>(0);
            } else {
                for (int i=prev_col_ind+1; i<m; ++i) {
                    U_mat[i+col_ind*m] -= L_mat[i+prev_col_ind*m]*earlier_col_val;
                }
            }
        }

        // Permute row to get largest pivot
        if (to_pivot_row) { pivot_row(col_ind, U_mat, L_mat, P_mat); }

        // Zero elements below diagonal in col_ind
        W pivot = U_mat[col_ind+col_ind*m];
        if (std::abs(pivot) >= zero_tol) {

            for (int row_ind=col_ind+1; row_ind<m; ++row_ind) {

                W val_to_zero = U_mat[row_ind+col_ind*m];
                if (std::abs(val_to_zero) >= zero_tol) {

                    W l_ij = val_to_zero/pivot;
                    if (std::abs(l_ij) >= zero_tol) {
                        L_mat[row_ind+col_ind*m] = l_ij;
                    }
                    U_mat[row_ind+col_ind*m] = static_cast<W>(0);

                }

            }
        } else {
            throw std::runtime_error("ILU has zero pivot in elimination");
        }

        // Apply drop rule to column
        apply_drop_rule_vec(col_ind, zero_tol, U_mat, L_mat, m);

    }

    void print_mat(W *mat, const int &m, const int &n) {
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                std::cout << static_cast<double>(mat[i+j*m]) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    void perform_left_looking_ILU(
        const W &zero_tol, const bool &pivot, const M<W> &A
    ) {

        W *U_mat = static_cast<W *>(malloc(A.rows()*A.cols()*sizeof(W)));
        W *L_mat = static_cast<W *>(malloc(A.rows()*A.cols()*sizeof(W)));
        W *P_mat = static_cast<W *>(malloc(A.rows()*A.cols()*sizeof(W)));

        U.copy_data_to_ptr(U_mat, A.rows(), A.cols());
        L.copy_data_to_ptr(L_mat, A.rows(), A.cols());
        P.copy_data_to_ptr(P_mat, A.rows(), A.cols());
        for (int j=0; j<A.cols(); ++j) {
            execute_leftlook_col_elimination(j, zero_tol, pivot, U_mat, L_mat, P_mat);
        }

        U = M<W>(A.get_handle(), U_mat, A.rows(), A.cols());
        L = M<W>(A.get_handle(), L_mat, A.rows(), A.cols());
        P = M<W>(A.get_handle(), P_mat, A.rows(), A.cols());
        reduce_matrices();

        free(U_mat);
        free(L_mat);
        free(P_mat);
    
    }

    void construction_helper(const M<W> &A, const W &zero_tol, const bool &pivot) {

        if (A.rows() != A.cols()) { throw std::runtime_error("Non square matrix A"); }
        instantiate_vars(A);
        perform_left_looking_ILU(zero_tol, pivot, A);

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

        apply_drop_rule_vec = [this] (
            const int &col_j, const W &_zero_tol, W *U_mat, W *L_mat, const int &m
        ) -> void { 

            int U_size = col_j;
            int L_size = m-(col_j+1);

            for (int l=0; l<U_size; ++l) { // Apply tau drop rule
                if (drop_rule_tau(U_mat[l+col_j*m], l, col_j, _zero_tol)) {
                    U_mat[l+col_j*m] = static_cast<W>(0);
                }
            }
            for (int l=col_j+1; l<L_size+col_j+1; ++l) { // Apply tau drop rule
                if (drop_rule_tau(L_mat[l+col_j*m], l, col_j, _zero_tol)) {
                    L_mat[l+col_j*m] = static_cast<W>(0);
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

        apply_drop_rule_vec = [this, p, &A] (
            const int &col_j, const W &_zero_tol, W *U_mat, W *L_mat, const int &m
        ) -> void { 

            int U_size = col_j;
            int L_size = m-(col_j+1);

            // Modify col_j in U
            for (int l=0; l<U_size; ++l) { // Apply tau drop rule
                if (drop_rule_tau(U_mat[l+col_j*m], l, col_j, _zero_tol)) {
                    U_mat[l+col_j*m] = static_cast<W>(0);
                }
            }
            if (p < U_size) { // Drop all but p largest elements
                MatrixVector<W> U_j(A.get_handle(), U_size);
                for (int i=0; i<U_size; ++i) { U_j.set_elem(i, U_mat[i+col_j*m]); }
                std::vector<int> U_sorted_indices = U_j.sort_indices();
                for (int i=0; i<U_size-p; ++i) {
                    U_mat[U_sorted_indices[i]+col_j*m] = static_cast<W>(0);
                }
            }

            // Modify col_j in L
            for (int l=col_j+1; l<L_size+col_j+1; ++l) { // Apply tau drop rule
                if (drop_rule_tau(L_mat[l+col_j*m], l, col_j, _zero_tol)) {
                    L_mat[l+col_j*m] = static_cast<W>(0);
                }
            }
            if (p < L_size) { // Drop all but p largest elements
                MatrixVector<W> L_j(A.get_handle(), L_size);
                for (int i=0; i<L_size; ++i) { L_j.set_elem(i, L_mat[i+col_j+1+col_j*m]); }
                std::vector<int> L_sorted_indices = L_j.sort_indices();
                for (int i=0; i<L_size-p; ++i) {
                    L_mat[L_sorted_indices[i]+col_j+1+col_j*m] = static_cast<W>(0);
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
#ifndef ILU_H
#define ILU_H

#include "Preconditioner.h"

#include <cmath>
#include <vector>
#include <functional>

namespace ilu
{

template <typename W>
void apply_prev_zeroing_col(
    const int &col_ind, const W &zero_tol,
    std::function<bool (const W &curr_val, const int &row, const int &col, const W &zero_tol)> drop_rule_tau,
    const int &m_dim,
    W *U_mat, W *L_mat
) {
    for (int i=0; i<col_ind; ++i) {
        W prev_col_val = U_mat[i+col_ind*m_dim];
        if ((std::abs(prev_col_val) <= zero_tol) || (drop_rule_tau(prev_col_val, i, col_ind, zero_tol))) {
            U_mat[i+col_ind*m_dim] = static_cast<W>(0);
        } else {
            for (int k=i+1; k<m_dim; ++k) {
                U_mat[k+col_ind*m_dim] -= L_mat[k+i*m_dim]*prev_col_val;
            }
        }
    }
}

template <typename W>
int find_largest_pivot(
    const int &col_ind,
    const int &m_dim,
    W *mat
) {

    int pivot_ind = col_ind;
    W largest_val = std::abs(mat[col_ind+col_ind*m_dim]);
    for (int i=col_ind+1; i<m_dim; ++i) {
        W temp = std::abs(mat[i+col_ind*m_dim]);
        if (temp > largest_val) {
            pivot_ind = i;
            largest_val = temp;
        }
    }

    return pivot_ind;

}

template <typename W>
void partial_swap_row(
    const int &row_1, const int &row_2, const int &beg, const int &end,
    const int &m_dim,
    W *mat
) {
    for (int j=beg; j<end; ++j) {
        W temp = mat[row_1+j*m_dim];
        mat[row_1+j*m_dim] = mat[row_2+j*m_dim];
        mat[row_2+j*m_dim] = temp;
    }
}

template <typename W>
void pivot_rows_U_L_P(
    const int &pivot_ind, const int &diag_ind,
    const int &m_dim,
    W *U_mat, W *L_mat, W *P_mat
) {
    if (diag_ind != pivot_ind) {
        partial_swap_row(diag_ind, pivot_ind, diag_ind, m_dim, m_dim, U_mat);
        partial_swap_row(diag_ind, pivot_ind, 0, diag_ind, m_dim, L_mat);
        partial_swap_row(diag_ind, pivot_ind, 0, m_dim, m_dim, P_mat);
    }
}

template <typename W>
void zero_below_diag_col_update_U_L(
    const int &col_ind, const W &zero_tol,
    const int &m_dim,
    W *U_mat, W *L_mat
) {

    W pivot_val = U_mat[col_ind+col_ind*m_dim];
    if (std::abs(pivot_val) >= zero_tol) {
        for (int i=col_ind+1; i<m_dim; ++i) {
            W val_to_zero = U_mat[i+col_ind*m_dim];
            if (std::abs(val_to_zero) >= zero_tol) {
                W l_ij = val_to_zero/pivot_val;
                if (std::abs(l_ij) >= zero_tol) { L_mat[i+col_ind*m_dim] = l_ij; }
                U_mat[i+col_ind*m_dim] = static_cast<W>(0);
            }
        }
    } else {
        throw std::runtime_error("ILU has zero pivot in elimination");
    }

}

// Left looking LU factorization for better memory access
template <typename W>
void execute_leftlook_col_elimination(
    const int &col_ind,
    const W &zero_tol, bool to_pivot_row,
    std::function<bool (const W &curr_val, const int &row, const int &col, const W &zero_tol)> drop_rule_tau,
    std::function<void (const int &col, const W &zero_tol, const int &m, W *U_mat, W *L_mat)> apply_drop_rule_col,
    const int &m_dim,
    W *U_mat, W *L_mat, W *P_mat
) {

    apply_prev_zeroing_col(col_ind, zero_tol, drop_rule_tau, m_dim, U_mat, L_mat);

    if (to_pivot_row) {
        int pivot_ind = find_largest_pivot(col_ind, m_dim, U_mat);
        pivot_rows_U_L_P(pivot_ind, col_ind, m_dim, U_mat, L_mat, P_mat);
    }

    zero_below_diag_col_update_U_L(col_ind, zero_tol, m_dim, U_mat, L_mat);
    apply_drop_rule_col(col_ind, zero_tol, m_dim, U_mat, L_mat);

}

template <template <typename> typename M, typename W>
void dynamic_construct_leftlook_square_ILU(
    const W &zero_tol, const bool &pivot,
    std::function<bool (const W &curr_val, const int &row, const int &col, const W &zero_tol)> drop_rule_tau,
    std::function<void (const int &col, const W &zero_tol, const int &m, W *U_mat, W *L_mat)> apply_drop_rule_col,
    const M<W> &A,
    M<W> &U, M<W> &L, M<W> &P
) {

    if (A.rows() != A.cols()) { throw std::runtime_error("Non square matrix A"); }
    const int m_dim(A.rows());

    W *U_mat = static_cast<W *>(malloc(m_dim*m_dim*sizeof(W)));
    W *L_mat = static_cast<W *>(malloc(m_dim*m_dim*sizeof(W)));
    W *P_mat = static_cast<W *>(malloc(m_dim*m_dim*sizeof(W)));

    U = A;
    L = M<W>::Identity(A.get_handle(), m_dim, m_dim);
    P = M<W>::Identity(A.get_handle(), m_dim, m_dim);

    U.copy_data_to_ptr(U_mat, m_dim, m_dim);
    L.copy_data_to_ptr(L_mat, m_dim, m_dim);
    P.copy_data_to_ptr(P_mat, m_dim, m_dim);

    for (int j=0; j<m_dim; ++j) {
        execute_leftlook_col_elimination(
            j,
            zero_tol, pivot,
            drop_rule_tau,
            apply_drop_rule_col,
            m_dim,
            U_mat, L_mat, P_mat
        );
    }

    U = M<W>(A.get_handle(), U_mat, m_dim, m_dim);
    L = M<W>(A.get_handle(), L_mat, m_dim, m_dim);
    P = M<W>(A.get_handle(), P_mat, m_dim, m_dim);

    free(U_mat);
    free(L_mat);
    free(P_mat);

}

}

template <template <typename> typename M, typename W>
class ILUPreconditioner: public Preconditioner<M, W>
{
private:

    int m;
    M<W> L = M<W>(NULL);
    M<W> U = M<W>(NULL);
    M<W> P = M<W>(NULL);
    std::function<bool (const W &curr_val, const int &row, const int &col, const W &zero_tol)> drop_rule_tau;
    std::function<void (const int &col, const W &zero_tol, const int &m, W *U_mat, W *L_mat)> apply_drop_rule_col;

    void reduce_matrices() {
        L.reduce();
        U.reduce();
        P.reduce();
    }

    void construct_ILU(const M<W> &A, const W &zero_tol, const bool &pivot) {
        ilu::dynamic_construct_leftlook_square_ILU(
            zero_tol, pivot, drop_rule_tau, apply_drop_rule_col, A, U, L, P
        );
        reduce_matrices();
    }

public:

    // ILU constructor taking premade L and U and no permutation
    ILUPreconditioner(const M<W> &arg_L, const M<W> &arg_U):
        ILUPreconditioner(
            arg_L, arg_U, M<W>::Identity(arg_L.get_handle(), arg_L.rows(), arg_L.rows())
        )
    {}

    // ILU constructor taking premade L, U, and P
    ILUPreconditioner(const M<W> &arg_L, const M<W> &arg_U, const M<W> &arg_P):
        m(arg_L.rows()), L(arg_L), U(arg_U), P(arg_P)
    {
        if (arg_L.rows() != arg_L.cols()) {
            throw std::runtime_error("ILU(L, U, P): Non square matrix L");
        }
        if (arg_U.rows() != arg_U.cols()) {
            throw std::runtime_error("ILU(L, U, P): Non square matrix U");
        }
        if (arg_P.rows() != arg_P.cols()) {
            throw std::runtime_error("ILU(L, U, P): Non square matrix P");
        }
        if (arg_L.rows() != arg_U.rows()) {
            throw std::runtime_error("ILU(L, U, P): L and U dim mismatch");
        }
        if (arg_L.rows() != arg_P.rows()) {
            throw std::runtime_error("ILU(L, U, P): L and P dim mismatch");
        }
        reduce_matrices();
    }

    // ILU(0)   
    ILUPreconditioner(const M<W> &A, const W &zero_tol, const bool &pivot):
        m(A.rows())
    {

        drop_rule_tau = [A] (
            const W &curr_val, const int &i, const int &j, W const &_zero_tol
        ) -> bool {
            return (std::abs(A.get_elem(i, j)) <= _zero_tol);
        };

        apply_drop_rule_col = [this] (
            const int &col, const W &_zero_tol, const int &m, W *U_mat, W *L_mat
        ) -> void { 

            int U_size = col;
            int L_size = m-(col+1);

            for (int l=0; l<U_size; ++l) { // Apply tau drop rule
                if (drop_rule_tau(U_mat[l+col*m], l, col, _zero_tol)) {
                    U_mat[l+col*m] = static_cast<W>(0);
                }
            }
            for (int l=col+1; l<L_size+col+1; ++l) { // Apply tau drop rule
                if (drop_rule_tau(L_mat[l+col*m], l, col, _zero_tol)) {
                    L_mat[l+col*m] = static_cast<W>(0);
                }
            }

        };

        construct_ILU(A, zero_tol, pivot);

    }

    // ILUT(tau, p), tau threshold to drop and p number of entries to keep
    ILUPreconditioner(M<W> &A, const W &tau, const int &p, const W &zero_tol, const bool &pivot):
        m(A.rows())
    {

        // Calculate original A col norm for threshold comparison
        MatrixVector<W> A_col_norms(A.get_handle(), A.cols());
        for (int j=0; j<A.cols(); ++j) {
            A_col_norms.set_elem(j, A.get_col(j).copy_to_vec().norm());
        }

        drop_rule_tau = [A_col_norms, tau] (
            const W &curr_val, const int &row, const int &col, const W &_zero_tol
        ) -> bool {
            return (std::abs(curr_val) <= tau*A_col_norms.get_elem(col));
        };

        apply_drop_rule_col = [this, p, &A] (
            const int &col, const W &_zero_tol, const int &m, W *U_mat, W *L_mat
        ) -> void {

            int U_size = col;
            int L_size = m-(col+1);

            // Modify col in U
            for (int l=0; l<U_size; ++l) { // Apply tau drop rule
                if (drop_rule_tau(U_mat[l+col*m], l, col, _zero_tol)) {
                    U_mat[l+col*m] = static_cast<W>(0);
                }
            }
            if (p < U_size) { // Drop all but p largest elements
                MatrixVector<W> U_col(A.get_handle(), U_mat+col*m, U_size);
                std::vector<int> U_sorted_indices = U_col.sort_indices();
                for (int i=0; i<U_size-p; ++i) {
                    U_mat[U_sorted_indices[i]+col*m] = static_cast<W>(0);
                }
            }

            // Modify col in L
            for (int l=col+1; l<L_size+col+1; ++l) { // Apply tau drop rule
                if (drop_rule_tau(L_mat[l+col*m], l, col, _zero_tol)) {
                    L_mat[l+col*m] = static_cast<W>(0);
                }
            }
            if (p < L_size) { // Drop all but p largest elements
                MatrixVector<W> L_col(A.get_handle(), L_mat+col+1+col*m, L_size);
                std::vector<int> L_sorted_indices = L_col.sort_indices();
                for (int i=0; i<L_size-p; ++i) {
                    L_mat[L_sorted_indices[i]+col+1+col*m] = static_cast<W>(0);
                }
            }

        };

        construct_ILU(A, zero_tol, pivot);

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
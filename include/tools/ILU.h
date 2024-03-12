#ifndef ILU_H
#define ILU_H

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
    const int &col_ind, const int &m_dim, W *mat
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
    L = M<W>::Identity(A.get_cu_handles(), m_dim, m_dim);
    P = M<W>::Identity(A.get_cu_handles(), m_dim, m_dim);

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

    U = M<W>(A.get_cu_handles(), U_mat, m_dim, m_dim);
    L = M<W>(A.get_cu_handles(), L_mat, m_dim, m_dim);
    P = M<W>(A.get_cu_handles(), P_mat, m_dim, m_dim);

    free(U_mat);
    free(L_mat);
    free(P_mat);

}

}

#endif
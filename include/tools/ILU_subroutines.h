#ifndef ILU_SUBROUTINES_H
#define ILU_SUBROUTINES_H

// #include <functional>

#include "types/types.h"

namespace ilu
{

// template <typename W>
// void apply_prev_zeroing_col(
//     const int &col_ind, const W &zero_tol,
//     std::function<bool (const W &curr_val, const int &row, const int &col, const W &zero_tol)> drop_rule_tau,
//     const int &m_dim,
//     W *U_mat, W *L_mat
// ) {
//     for (int i=0; i<col_ind; ++i) {
//         W prev_col_val = U_mat[i+col_ind*m_dim];
//         if ((std::abs(prev_col_val) <= zero_tol) || (drop_rule_tau(prev_col_val, i, col_ind, zero_tol))) {
//             U_mat[i+col_ind*m_dim] = static_cast<W>(0);
//         } else {
//             for (int k=i+1; k<m_dim; ++k) {
//                 U_mat[k+col_ind*m_dim] -= L_mat[k+i*m_dim]*prev_col_val;
//             }
//         }
//     }
// }

// template <typename W>
// int find_largest_pivot(
//     const int &col_ind, const int &m_dim, W *mat
// ) {

//     int pivot_ind = col_ind;
//     W largest_val = std::abs(mat[col_ind+col_ind*m_dim]);
//     for (int i=col_ind+1; i<m_dim; ++i) {
//         W temp = std::abs(mat[i+col_ind*m_dim]);
//         if (temp > largest_val) {
//             pivot_ind = i;
//             largest_val = temp;
//         }
//     }

//     return pivot_ind;

// }

// template <typename W>
// void partial_swap_row(
//     const int &row_1, const int &row_2, const int &beg, const int &end,
//     const int &m_dim,
//     W *mat
// ) {
//     for (int j=beg; j<end; ++j) {
//         W temp = mat[row_1+j*m_dim];
//         mat[row_1+j*m_dim] = mat[row_2+j*m_dim];
//         mat[row_2+j*m_dim] = temp;
//     }
// }

// template <typename W>
// void pivot_rows_U_L_P(
//     const int &pivot_ind, const int &diag_ind,
//     const int &m_dim,
//     W *U_mat, W *L_mat, W *P_mat
// ) {
//     if (diag_ind != pivot_ind) {
//         partial_swap_row(diag_ind, pivot_ind, diag_ind, m_dim, m_dim, U_mat);
//         partial_swap_row(diag_ind, pivot_ind, 0, diag_ind, m_dim, L_mat);
//         partial_swap_row(diag_ind, pivot_ind, 0, m_dim, m_dim, P_mat);
//     }
// }

// template <typename W>
// void zero_below_diag_col_update_U_L(
//     const int &col_ind, const W &zero_tol,
//     const int &m_dim,
//     W *U_mat, W *L_mat
// ) {

//     W pivot_val = U_mat[col_ind+col_ind*m_dim];
//     if (std::abs(pivot_val) >= zero_tol) {
//         for (int i=col_ind+1; i<m_dim; ++i) {
//             W val_to_zero = U_mat[i+col_ind*m_dim];
//             if (std::abs(val_to_zero) >= zero_tol) {
//                 W l_ij = val_to_zero/pivot_val;
//                 if (std::abs(l_ij) >= zero_tol) { L_mat[i+col_ind*m_dim] = l_ij; }
//                 U_mat[i+col_ind*m_dim] = static_cast<W>(0);
//             }
//         }
//     } else {
//         throw std::runtime_error("ILU has zero pivot in elimination");
//     }

// }

// // Left looking LU factorization for better memory access
// template <typename W>
// void execute_leftlook_col_elimination(
//     const int &col_ind,
//     const W &zero_tol, bool to_pivot_row,
//     std::function<bool (const W &curr_val, const int &row, const int &col, const W &zero_tol)> drop_rule_tau,
//     std::function<void (const int &col, const W &zero_tol, const int &m, W *U_mat, W *L_mat)> apply_drop_rule_col,
//     const int &m_dim,
//     W *U_mat, W *L_mat, W *P_mat
// ) {

//     apply_prev_zeroing_col(col_ind, zero_tol, drop_rule_tau, m_dim, U_mat, L_mat);

//     if (to_pivot_row) {
//         int pivot_ind = find_largest_pivot(col_ind, m_dim, U_mat);
//         pivot_rows_U_L_P(pivot_ind, col_ind, m_dim, U_mat, L_mat, P_mat);
//     }

//     zero_below_diag_col_update_U_L(col_ind, zero_tol, m_dim, U_mat, L_mat);
//     apply_drop_rule_col(col_ind, zero_tol, m_dim, U_mat, L_mat);

// }

// template <template <typename> typename M, typename W>
// void dynamic_construct_leftlook_square_ILU(
//     const W &zero_tol, const bool &pivot,
//     std::function<bool (const W &curr_val, const int &row, const int &col, const W &zero_tol)> drop_rule_tau,
//     std::function<void (const int &col, const W &zero_tol, const int &m, W *U_mat, W *L_mat)> apply_drop_rule_col,
//     const M<W> &A,
//     M<W> &U, M<W> &L, M<W> &P
// ) {

//     if (A.rows() != A.cols()) { throw std::runtime_error("Non square matrix A"); }
//     const int m_dim(A.rows());

//     W *U_mat = static_cast<W *>(malloc(m_dim*m_dim*sizeof(W)));
//     W *L_mat = static_cast<W *>(malloc(m_dim*m_dim*sizeof(W)));
//     W *P_mat = static_cast<W *>(malloc(m_dim*m_dim*sizeof(W)));

//     U = A;
//     L = M<W>::Identity(A.get_cu_handles(), m_dim, m_dim);
//     P = M<W>::Identity(A.get_cu_handles(), m_dim, m_dim);

//     U.copy_data_to_ptr(U_mat, m_dim, m_dim);
//     L.copy_data_to_ptr(L_mat, m_dim, m_dim);
//     P.copy_data_to_ptr(P_mat, m_dim, m_dim);

//     for (int j=0; j<m_dim; ++j) {
//         execute_leftlook_col_elimination(
//             j,
//             zero_tol, pivot,
//             drop_rule_tau,
//             apply_drop_rule_col,
//             m_dim,
//             U_mat, L_mat, P_mat
//         );
//     }

//     U = M<W>(A.get_cu_handles(), U_mat, m_dim, m_dim);
//     L = M<W>(A.get_cu_handles(), L_mat, m_dim, m_dim);
//     P = M<W>(A.get_cu_handles(), P_mat, m_dim, m_dim);

//     free(U_mat);
//     free(L_mat);
//     free(P_mat);

//

template <typename T> class ColValHeadManager;

template <typename T>
class ColVal_DblLink
{
public:

    ColVal_DblLink<T> *prev_col_val;
    ColVal_DblLink<T> *next_col_val;
    const int row;
    const int col;
    const T val;
    ColValHeadManager<T> *manager_ptr;

    ColVal_DblLink(int arg_row, int arg_col, T arg_val, ColValHeadManager<T> *arg_manager_ptr):
        row(arg_row), col(arg_col), val(arg_val),
        prev_col_val(nullptr), next_col_val(nullptr), manager_ptr(arg_manager_ptr)
    {
        arg_manager_ptr->validate_col_val(this);
    }

    ~ColVal_DblLink() {
        manager_ptr->update_manager_on_deletion(this);
        if (prev_col_val != nullptr) {
            prev_col_val->next_col_val = next_col_val;
        }
        if (next_col_val != nullptr) {
            next_col_val->prev_col_val = prev_col_val;
        }
    }

    ColVal_DblLink(const ColVal_DblLink &) = delete;
    ColVal_DblLink &operator=(const ColVal_DblLink &) = delete;

    bool operator>(const ColVal_DblLink &other) const {
        return val > other.val;
    }

    ColVal_DblLink * connect(ColVal_DblLink *other) {
        next_col_val = other;
        other->prev_col_val = this;
        return this;
    }

};

template <typename T>
class ColValHeadManager
{
private:

    const int n_cols;

public:

    std::vector<ColVal_DblLink<T> *> heads;

    ColValHeadManager(int arg_n_cols): n_cols(arg_n_cols) {
        heads.resize(n_cols);
        for (int i=0; i<n_cols; ++i) { heads[i] = nullptr; }
    }

    ColValHeadManager(const ColValHeadManager &) = delete;
    ColValHeadManager &operator=(const ColValHeadManager &) = delete;

    void update_manager_on_deletion(const ColVal_DblLink<T> *col_val) {

        if ((heads[col_val->col] != nullptr) && (heads[col_val->col] == col_val)) {
            heads[col_val->col] = col_val->next_col_val;
        }

    }

    void validate_col_val(ColVal_DblLink<T> *col_val) {
        if ((col_val->col < 0) || (col_val->col >= n_cols)) {
            throw std::runtime_error(
                "ColValHeadManager: col_val col invalid for ColValHeadManager"
            );
        }
    }

    void add_head(ColVal_DblLink<T> *col_val) {
        if ((heads[col_val->col] == nullptr) && (col_val->prev_col_val == nullptr)) {
            heads[col_val->col] = col_val;
        } else {
            throw std::runtime_error(
                "ColValHeadManager: invalid use of add_head either filled or non head val"
            );
        }
    }

};

template <typename T>
class PSizeRowHeap
{
private:

    const int p;
    const int row;
    int count = 0;

    int calc_parent_ind(int curr_ind) const {
        return (curr_ind-1)/2;
    }

    int calc_child_ind_L(int curr_ind) const {
        return 2*curr_ind + 1;
    }

    void heap_swap(int i, int j) {
        ColVal_DblLink<T> *temp = heap[i];
        heap[i] = heap[j];
        heap[j] = temp;
    }

    void no_replace_push(ColVal_DblLink<T> *new_val, int end_ind) {

        heap[end_ind] = new_val;

        int curr_ind = end_ind;
        int parent_ind = calc_parent_ind(curr_ind);
        while ((curr_ind > 0) && (*heap[parent_ind] > *heap[curr_ind])) {
            heap_swap(parent_ind, curr_ind);
            curr_ind = parent_ind;
            parent_ind = calc_parent_ind(curr_ind);
        }

    }

    void replace_min_push(ColVal_DblLink<T> *new_val) {

        heap[0] = new_val;

        int curr_ind = 0;
        int child_ind_L = calc_child_ind_L(curr_ind);
        while (child_ind_L < p) {

            int min_child = child_ind_L;
            if (((child_ind_L+1) < p) && (*heap[child_ind_L] > *heap[child_ind_L+1])) {
                min_child = child_ind_L+1;
            }

            if (*heap[curr_ind] > *heap[min_child]) {
                heap_swap(curr_ind, min_child);
                curr_ind = min_child;
                child_ind_L = calc_child_ind_L(curr_ind);
            } else {
                break;
            }

        }
    
    }

public:

    std::vector<ColVal_DblLink<T> *> heap;

    PSizeRowHeap(int arg_row, int arg_p):
        row(arg_row), p(arg_p)
    {
        if (arg_p <= 0) { throw std::runtime_error("PSizeRowHeap: invalid row size"); }
        heap.resize(p);
        for (int i=0; i<p; ++i) { heap[i] = nullptr; }
    }

    void attempt_add_and_delete_min_val(ColVal_DblLink<T> *new_val) {
        
        if (new_val->row != row) {
            throw std::runtime_error("PSizeRowHeap: unmatched row in ColVal_DblLink");
        }

        if (count < p) {
            no_replace_push(new_val, count);
            count++;
        } else {
            if (*new_val > *heap[0]) {
                delete heap[0];
                replace_min_push(new_val);
            } else {
                delete new_val;
            }
        }

    }

    PSizeRowHeap(const PSizeRowHeap &other):
        PSizeRowHeap(other.row, other.p)
    {}

    PSizeRowHeap &operator=(const PSizeRowHeap &other) = delete;

};

template <typename T>
void sparse_construct_square_ILUTP(
    double tau, int p, bool to_pivot,
    const NoFillMatrixSparse<T> &A,
    NoFillMatrixSparse<T> &L, NoFillMatrixSparse<T> &U, NoFillMatrixSparse<T> &P
) {

    if (A.rows() != A.cols()) { throw std::runtime_error("Non square matrix A"); }

    int m = A.rows();

    int *h_U_col_offsets = static_cast<T>(malloc(m*sizeof(T)));
    int *h_L_col_offsets = static_cast<T>(malloc(m*sizeof(T)));
    int *h_P_col_offsets = static_cast<T>(malloc(m*sizeof(T)));

    

    free(h_U_col_offsets);
    free(h_L_col_offsets);
    free(h_P_col_offsets);

}

}

#endif
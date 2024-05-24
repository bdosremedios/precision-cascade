#ifndef ILU_SUBROUTINES_H
#define ILU_SUBROUTINES_H

#include <functional>

#include "types/types.h"

#include "tools/abs.h"
#include "tools/Sort.h"
#include "tools/Heap.h"

namespace ilu_subroutines
{

template <template <typename> typename M, typename T>
struct ILUTriplet {
    M<T> L = M<T>(cuHandleBundle());
    M<T> U = M<T>(cuHandleBundle());
    M<T> P = M<T>(cuHandleBundle());
};

namespace
{

template <typename T>
using DropRuleTauFunc = std::function<bool (T curr_val, int row, int col)>;

template <typename T>
using DropRulePFunc = std::function<void (int col_ind, int pivot_ind, T *col_ptr, int m_dim)>;

template <typename T>
struct SparseMatrixDescrPtrs {
    int *col_offsets;
    int *row_indices;
    T *vals;
};

template <typename T>
NoFillMatrixSparse<T> convert_SparseMatrixDescrPtrs_to_NoFillMatrixSparse(
    const cuHandleBundle &arg_cu_handles, int m_dim, SparseMatrixDescrPtrs<T> mat_descrp_ptrs
) {
    return NoFillMatrixSparse<T>(
        arg_cu_handles,
        mat_descrp_ptrs.col_offsets,
        mat_descrp_ptrs.row_indices,
        mat_descrp_ptrs.vals,
        m_dim, m_dim, mat_descrp_ptrs.col_offsets[m_dim]
    );
}

template <typename T>
T * get_new_U_col_from_left_look(
    int col_ind,
    int m_dim,
    DropRuleTauFunc<T> drop_rule_tau_U,
    SparseMatrixDescrPtrs<T> h_A,
    SparseMatrixDescrPtrs<T> h_L,
    int *pivot_row_hist
) {
    
    // Instantiate new dense U col with A column
    T *new_col_U = static_cast<T *>(malloc(m_dim*sizeof(T)));
    for (int i=0; i<m_dim; ++i) {
        new_col_U[i] = static_cast<T>(0.);
    }
    int col_beg_A = h_A.col_offsets[col_ind];
    int col_end_A = h_A.col_offsets[col_ind+1];
    for (int offset_A=col_beg_A; offset_A<col_end_A; ++offset_A) {
        new_col_U[h_A.row_indices[offset_A]] = h_A.vals[offset_A];
    }

    // Apply effects from previous column updates
    for (int k=0; k<col_ind; ++k) {

        int col_beg_L = h_L.col_offsets[k];
        int col_end_L = h_L.col_offsets[k+1];

        if (drop_rule_tau_U(new_col_U[pivot_row_hist[k]], pivot_row_hist[k], col_ind)) { // Drop pivot U if matches rule
            new_col_U[pivot_row_hist[k]] = static_cast<T>(0.);
        }
        T effecting_U_val = new_col_U[pivot_row_hist[k]];

        if (effecting_U_val != static_cast<T>(0.)) { // Skip if no effect from U value
            for (int offset_L=col_beg_L; offset_L<col_end_L; ++offset_L) {
                int row_L = h_L.row_indices[offset_L];
                if (row_L != pivot_row_hist[k]) { // Don't apply to own pivot row
                    T val_L = h_L.vals[offset_L];
                    if (val_L != static_cast<T>(0.)) { // Skip computation if there's no L modification
                        new_col_U[row_L] -= val_L*effecting_U_val;
                    }
                }
            }
        }
    
    }

    return new_col_U;

}

template <typename T>
void single_elem_swap(int i, int j, T *vec) {
    if (i != j) {
        T temp = vec[i];
        vec[i] = vec[j];
        vec[j] = temp;
    }
}

template <typename T>
int find_pivot_loc_in_perm_map(
    int col_ind, bool to_pivot, int m_dim, T *new_U_col, int *row_permutation_map, bool *row_finished
) {

    int ret_val = col_ind;
    if (to_pivot) {
        for (int i=0; i<m_dim; ++i) {
            if (
                !row_finished[row_permutation_map[i]] &&
                (abs_ns::abs(new_U_col[row_permutation_map[i]]) >
                 abs_ns::abs(new_U_col[row_permutation_map[ret_val]]))
            ) {
                ret_val = i;
            }
        }
    }

    if (new_U_col[row_permutation_map[ret_val]] == static_cast<T>(0.)) {
        throw std::runtime_error("find_pivot: zero pivot encountered");
    }

    return ret_val;

}

template <typename T>
T * get_new_L_col_from_zeroing(
    int col_ind,
    int pivot_ind,
    int m_dim,
    T *U_col,
    bool *row_finished
) {

    T *new_L_col = static_cast<T *>(malloc(m_dim*sizeof(T)));
    for (int i=0; i<m_dim; ++i) {
        if (i == pivot_ind) {
            new_L_col[i] = static_cast<T>(1.);
        } else {
            new_L_col[i] = static_cast<T>(0.);
        }
    }

    T pivot_val = U_col[pivot_ind];
    for (int i=0; i<m_dim; ++i) {
        if (!row_finished[i] && (i != pivot_ind)) {
            new_L_col[i] = U_col[i]/pivot_val;
            U_col[i] = static_cast<T>(0.);
        }
    }

    return new_L_col;

}

template <typename T>
void update_SparseMatrixDescrPtrs_w_col(
    int col_ind, int m_dim, SparseMatrixDescrPtrs<T> mat_descrp_ptrs, T *col_ptr
) {

    int col_nnz = 0;
    int init_offset = mat_descrp_ptrs.col_offsets[col_ind];
    for (int i=0; i<m_dim; ++i) {
        if (col_ptr[i] != static_cast<T>(0.)) {
            mat_descrp_ptrs.row_indices[init_offset + col_nnz] = i;
            mat_descrp_ptrs.vals[init_offset + col_nnz] = col_ptr[i];
            ++col_nnz;
        }
    }
    mat_descrp_ptrs.col_offsets[col_ind+1] = init_offset + col_nnz;

}

template <typename T>
void left_looking_col_elim_delay_perm(
    int col_ind,
    bool to_pivot,
    DropRuleTauFunc<T> drop_rule_tau_U,
    DropRuleTauFunc<T> drop_rule_tau_L,
    DropRulePFunc<T> drop_rule_p,
    int m_dim,
    SparseMatrixDescrPtrs<T> h_A,
    SparseMatrixDescrPtrs<T> h_U,
    SparseMatrixDescrPtrs<T> h_L,
    int *row_permutation_map,
    int *pivot_row_hist,
    bool *row_finished
) {

    // Determine new column of U
    T *new_U_col = get_new_U_col_from_left_look<T>(
        col_ind, m_dim, drop_rule_tau_U, h_A, h_L, pivot_row_hist
    );

    // Determine pivot row
    int pivot_loc;
    try {
        pivot_loc = find_pivot_loc_in_perm_map<T>(
            col_ind, to_pivot, m_dim, new_U_col, row_permutation_map, row_finished
        );
    } catch (std::runtime_error e) {
        free(new_U_col);
        throw e;
    }
    int pivot_ind = row_permutation_map[pivot_loc];
    single_elem_swap(col_ind, pivot_loc, row_permutation_map);
    pivot_row_hist[col_ind] = pivot_ind;
    row_finished[pivot_ind] = true;

    // Determine new column of L
    T *new_L_col = get_new_L_col_from_zeroing<T>(
        col_ind, pivot_ind, m_dim, new_U_col, row_finished
    );

    // Apply drop rules to new columns ensuring skip of pivot
    for (int i=0; i<m_dim; ++i) {
        if ((i != pivot_ind) && drop_rule_tau_U(new_U_col[i], i, col_ind)) {
            new_U_col[i] = static_cast<T>(0.);
        }
        if ((i != pivot_ind) && drop_rule_tau_L(new_L_col[i], i, col_ind)) {
            new_L_col[i] = static_cast<T>(0.);
        }
    }
    drop_rule_p(col_ind, pivot_ind, new_U_col, m_dim);
    drop_rule_p(col_ind, pivot_ind, new_L_col, m_dim);

    // Update h_U and h_L
    update_SparseMatrixDescrPtrs_w_col<T>(col_ind, m_dim, h_U, new_U_col);
    update_SparseMatrixDescrPtrs_w_col<T>(col_ind, m_dim, h_L, new_L_col);

    // Free columns
    free(new_U_col);
    free(new_L_col);

}

template <typename T>
void exec_SparseMatrixDescrPtrs_perm(
    int m_dim, SparseMatrixDescrPtrs<T> mat_descrp_ptrs, int *row_permutation_dict
) {

    for (int j=0; j<m_dim; ++j) {

        int col_begin = mat_descrp_ptrs.col_offsets[j];
        int col_end = mat_descrp_ptrs.col_offsets[j+1];

        for (int offset = col_begin; offset < col_end; ++offset) {
            mat_descrp_ptrs.row_indices[offset] = row_permutation_dict[mat_descrp_ptrs.row_indices[offset]];
        }

        sort::in_place_passengered_sort<int, T>(
            col_begin, col_end, mat_descrp_ptrs.row_indices, mat_descrp_ptrs.vals
        );

    }

}

template <typename T>
NoFillMatrixSparse<T> form_permutation_matrix(
    const cuHandleBundle &arg_cu_handles, int m_dim, int *row_permutation_map
) {

    int *col_offsets = static_cast<int *>(malloc((m_dim+1)*sizeof(int)));
    int *row_indices = static_cast<int *>(malloc(m_dim*sizeof(int)));
    T *vals = static_cast<T *>(malloc(m_dim*sizeof(T)));

    for (int i=0; i<(m_dim+1); ++i) { col_offsets[i] = i; }
    for (int i=0; i<m_dim; ++i) {
        row_indices[row_permutation_map[i]] = i;
        vals[i] = static_cast<T>(1.);
    }

    NoFillMatrixSparse<T> ret_val(
        arg_cu_handles,
        col_offsets,
        row_indices,
        vals,
        m_dim, m_dim, m_dim
    );

    free(col_offsets);
    free(row_indices);
    free(vals);

    return ret_val;

}

template <template <typename> typename M, typename T>
ILUTriplet<M, T> sparse_construct_drop_rule_ILU(
    bool to_pivot,
    DropRuleTauFunc<T> drop_rule_tau_U,
    DropRuleTauFunc<T> drop_rule_tau_L,
    DropRulePFunc<T> drop_rule_p,
    int max_output_nnz,
    const NoFillMatrixSparse<T> &A
) {

    if (A.rows() != A.cols()) { throw std::runtime_error("Non square matrix A"); }

    int m_dim = A.rows();
    int input_nnz = A.non_zeros();

    // Load data of A into host memory
    SparseMatrixDescrPtrs<T> h_A;
    h_A.col_offsets = static_cast<int *>(malloc((m_dim+1)*sizeof(int)));
    h_A.row_indices = static_cast<int *>(malloc(input_nnz*sizeof(int)));
    h_A.vals = static_cast<T *>(malloc(input_nnz*sizeof(T)));

    A.copy_data_to_ptr(
        h_A.col_offsets, h_A.row_indices, h_A.vals,
        m_dim, m_dim, input_nnz
    );

    // Instantiate storage for L and U
    SparseMatrixDescrPtrs<T> h_U;
    h_U.col_offsets = static_cast<int *>(malloc((m_dim+1)*sizeof(int)));
    h_U.col_offsets[0] = 0;
    h_U.row_indices = static_cast<int *>(malloc(max_output_nnz*sizeof(int)));
    h_U.vals = static_cast<T *>(malloc(max_output_nnz*sizeof(T)));

    SparseMatrixDescrPtrs<T> h_L;
    h_L.col_offsets = static_cast<int *>(malloc((m_dim+1)*sizeof(int)));
    h_L.col_offsets[0] = 0;
    h_L.row_indices = static_cast<int *>(malloc(max_output_nnz*sizeof(int)));
    h_L.vals = static_cast<T *>(malloc(max_output_nnz*sizeof(T)));

    // Instantiate trackers for permutations, previous pivots, and for completed rows
    // information for delayed permutation
    int *row_permutation_map = static_cast<int *>(malloc(m_dim*sizeof(int)));
    for (int i=0; i<m_dim; ++i) { row_permutation_map[i] = i; }

    int *pivot_row_hist = static_cast<int *>(malloc(m_dim*sizeof(int)));

    bool *row_finished = static_cast<bool *>(malloc(m_dim*sizeof(bool)));
    for (int i=0; i<m_dim; ++i) { row_finished[i] = false; }

    // Eliminate columns of U with partial pivoting and drop rules
    for (int j=0; j<m_dim; ++j) {

        try {
            left_looking_col_elim_delay_perm(
                j,
                to_pivot,
                drop_rule_tau_U,
                drop_rule_tau_L,
                drop_rule_p,
                m_dim,
                h_A, h_U, h_L,
                row_permutation_map,
                pivot_row_hist,
                row_finished
            );
        } catch (std::runtime_error e) {
            free(h_A.col_offsets);
            free(h_A.row_indices);
            free(h_A.vals);
            free(h_U.col_offsets);
            free(h_U.row_indices);
            free(h_U.vals);
            free(h_L.col_offsets);
            free(h_L.row_indices);
            free(h_L.vals);
            free(row_permutation_map);
            free(pivot_row_hist);
            free(row_finished);
            throw e;
        }
    }

    // Create permutation dict such that every access where it should be
    int *row_permutation_dict = static_cast<int *>(malloc(m_dim*sizeof(int)));
    for (int i=0; i<m_dim; ++i) {
        row_permutation_dict[row_permutation_map[i]] = i;
    }

    // Execute permutations from elimination
    exec_SparseMatrixDescrPtrs_perm<T>(m_dim, h_L, row_permutation_dict);
    exec_SparseMatrixDescrPtrs_perm<T>(m_dim, h_U, row_permutation_dict);

    // Form matrices
    ILUTriplet<M, T> ret_val;
    ret_val.U = M<T>(
        convert_SparseMatrixDescrPtrs_to_NoFillMatrixSparse(A.get_cu_handles(), m_dim, h_U)
    );
    ret_val.L = M<T>(
        convert_SparseMatrixDescrPtrs_to_NoFillMatrixSparse(A.get_cu_handles(), m_dim, h_L)
    );
    ret_val.P = M<T>(
        form_permutation_matrix<T>(A.get_cu_handles(), m_dim, row_permutation_map)
    );

    // Free allocated memory
    free(h_A.col_offsets);
    free(h_A.row_indices);
    free(h_A.vals);

    free(h_U.col_offsets);
    free(h_U.row_indices);
    free(h_U.vals);

    free(h_L.col_offsets);
    free(h_L.row_indices);
    free(h_L.vals);

    free(row_permutation_map);
    free(row_permutation_dict);
    free(pivot_row_hist);
    free(row_finished);

    return ret_val;

}

}

template <template <typename> typename M, typename T>
ILUTriplet<M, T> construct_square_ILU_0(
   const NoFillMatrixSparse<T> &A, bool to_pivot
) {

    DropRuleTauFunc<T> drop_rule_tau_U = [&A] (T _, int row, int col) -> bool {
        return A.get_elem(row, col) == SCALAR_ZERO<T>::get();
    };

    DropRuleTauFunc<T> drop_rule_tau_L = [&A] (T _, int row, int col) -> bool {
        return A.get_elem(row, col) == SCALAR_ZERO<T>::get();
    };

    DropRulePFunc<T> drop_rule_p = [] (int col_ind, int pivot_ind, T *col_ptr, int m_dim) { ; };

    return sparse_construct_drop_rule_ILU<M ,T>(
        to_pivot,
        drop_rule_tau_U,
        drop_rule_tau_L,
        drop_rule_p,
        A.non_zeros(),
        A
    );

}

template <typename T>
T get_largest_magnitude_in_col(int col_ind, int *col_offsets, T *vals) {
    T largest_mag = 0.;
    for (int offset = col_offsets[col_ind]; offset < col_offsets[col_ind+1]; ++offset) {
        if (std::abs(vals[offset]) > largest_mag) {
            largest_mag = std::abs(vals[offset]);
        }
    }
    return largest_mag;
}

template <template <typename> typename M, typename T>
ILUTriplet<M, T> construct_square_ILUTP(
    const NoFillMatrixSparse<T> &A, double tau, int p, bool to_pivot
) {

    if (p < 1) {
        throw std::runtime_error(
            "construct_square_ILUTP: must at least keep pivot for p drop rule"
        );
    }

    int *col_offsets = static_cast<int *>(malloc((A.cols()+1)*sizeof(int)));
    int *row_indices = static_cast<int *>(malloc(A.non_zeros()*sizeof(int)));
    T *vals = static_cast<T *>(malloc(A.non_zeros()*sizeof(T)));

    A.copy_data_to_ptr(
        col_offsets, row_indices, vals,
        A.rows(), A.cols(), A.non_zeros()
    );

    std::vector<T> col_inf_norm(A.cols());
    for (int j=0; j<A.cols(); ++j) {
        col_inf_norm[j] = get_largest_magnitude_in_col(j, col_offsets, vals);
    }

    free(col_offsets);
    free(row_indices);
    free(vals);

    DropRuleTauFunc<T> drop_rule_tau_U = [col_inf_norm, tau] (T val, int _, int col) -> bool {
        return std::abs(val) <= tau*col_inf_norm[col];
    };

    DropRuleTauFunc<T> drop_rule_tau_L = [tau] (T val, int _, int col) -> bool {
        return std::abs(val) <= tau;
    };

    DropRulePFunc<T> drop_rule_p = [p] (int col_ind, int pivot_ind, T *col_ptr, int m_dim) {
        
        heap::PSizeHeap<T> heap(p-1);

        for (int i=0; i<m_dim; ++i) {
            if ((i != pivot_ind) && (col_ptr[i] != static_cast<T>(0.))) {
                heap.push(col_ptr[i], i);
                col_ptr[i] = static_cast<T>(0.);
            }
        }

        for (int k=0; k<heap.count; ++k) {
            col_ptr[heap.heap[k].row] = heap.heap[k].orig_val;
        }

    };

    return sparse_construct_drop_rule_ILU<M, T>(
        to_pivot,
        drop_rule_tau_U,
        drop_rule_tau_L,
        drop_rule_p,
        p*A.rows(),
        A
    );

}

}

#endif
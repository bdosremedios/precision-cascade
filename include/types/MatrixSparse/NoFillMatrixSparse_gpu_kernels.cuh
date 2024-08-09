#ifndef NOFILLMATRIXSPARSE_GPU_KERNELS_CUH
#define NOFILLMATRIXSPARSE_GPU_KERNELS_CUH

#include <cuda_runtime.h>

namespace cascade::nofillmatrixsparse_kernels {

template <typename TPrecision>
__global__ void fast_back_sub_solve_level_set(
    int *d_level_set, TPrecision *d_soln,
    int *d_row_offsets, int *d_col_indices, TPrecision *d_values
);

template <typename TPrecision>
__global__ void update_row_pivot(
    int row, int pivot_offset, TPrecision *d_vals, TPrecision *x_soln
);

template <typename TPrecision>
__global__ void upptri_update_right_of_pivot(
    int row,
    int row_start_offset,
    int row_elem_count,
    int *d_col_indices,
    TPrecision *d_values,
    TPrecision *x_soln
);

template <typename TPrecision>
__global__ void lowtri_update_left_of_pivot(
    int row,
    int row_start_offset,
    int row_elem_count,
    int *d_col_indices,
    TPrecision *d_values,
    TPrecision *x_soln
);

}

#endif
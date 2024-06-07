#include "types/MatrixSparse/NoFillMatrixSparse_gpu_kernels.cuh"

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cuda_fp16.h>

template <typename TPrecision>
__global__ void nofillmatrixsparse_kernels::update_pivot(
    int pivot_offset, int *d_row_indices, TPrecision *d_vals, TPrecision *x_soln
) {
    int row = d_row_indices[pivot_offset];
    x_soln[row] /= d_vals[pivot_offset];
}

template __global__ void nofillmatrixsparse_kernels::update_pivot<__half>(
    int, int *, __half *, __half *
);
template __global__ void nofillmatrixsparse_kernels::update_pivot<float>(
    int, int *, float *, float *
);
template __global__ void nofillmatrixsparse_kernels::update_pivot<double>(
    int, int *, double *, double *
);

template <typename TPrecision>
__global__ void nofillmatrixsparse_kernels::lowtri_update_remaining_col(
    int col_start_offset,
    int col_size,
    int *d_row_indices,
    TPrecision *d_vals, 
    TPrecision *x_soln
) {

    __shared__ TPrecision xs;

    xs = x_soln[d_row_indices[col_start_offset]];

    if (threadIdx.x+1 < col_size) {
        int coeff_offset = col_start_offset+threadIdx.x+1;
        int updating_row = d_row_indices[coeff_offset];
        x_soln[updating_row] -= d_vals[coeff_offset]*xs;
    }

}

template __global__ void nofillmatrixsparse_kernels::lowtri_update_remaining_col<__half>(
    int, int, int *, __half *, __half *
);
template __global__ void nofillmatrixsparse_kernels::lowtri_update_remaining_col<float>(
    int, int, int *, float *, float *
);
template __global__ void nofillmatrixsparse_kernels::lowtri_update_remaining_col<double>(
    int, int, int *, double *, double *
);

template <typename TPrecision>
__global__ void nofillmatrixsparse_kernels::upptri_update_remaining_col(
    int col_start_offset,
    int col_size,
    int *d_row_indices,
    TPrecision *d_vals,
    TPrecision *x_soln
) {

    __shared__ TPrecision xs;

    xs = x_soln[d_row_indices[col_start_offset+col_size-1]];

    if (threadIdx.x < col_size-1) {
        int coeff_offset = col_start_offset+threadIdx.x;
        int updating_row = d_row_indices[coeff_offset];
        x_soln[updating_row] -= d_vals[coeff_offset]*xs;
    }

}

template __global__ void nofillmatrixsparse_kernels::upptri_update_remaining_col<__half>(
    int, int, int *, __half *, __half *
);
template __global__ void nofillmatrixsparse_kernels::upptri_update_remaining_col<float>(
    int, int, int *, float *, float *
);
template __global__ void nofillmatrixsparse_kernels::upptri_update_remaining_col<double>(
    int, int, int *, double *, double *
);
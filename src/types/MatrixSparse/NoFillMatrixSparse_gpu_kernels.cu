#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "types/MatrixSparse/NoFillMatrixSparse_gpu_kernels.cuh"

template <typename T>
__global__ void nofillmatrixsparse_kernels::update_pivot(
    int pivot_offset, int *d_row_indices, T *d_vals, T *x_soln
) {
    int row = d_row_indices[pivot_offset];
    x_soln[row] /= d_vals[pivot_offset];
}

template __global__ void nofillmatrixsparse_kernels::update_pivot<__half>(int, int *, __half *, __half *);
template __global__ void nofillmatrixsparse_kernels::update_pivot<float>(int, int *, float *, float *);
template __global__ void nofillmatrixsparse_kernels::update_pivot<double>(int, int *, double *, double *);

template <typename T>
__global__ void nofillmatrixsparse_kernels::lowtri_update_remaining_col(
    int pivot_offset, int max_offset, int *d_row_indices, T *d_vals, T *x_soln
) {
    __shared__ T xs;

    xs = x_soln[d_row_indices[pivot_offset]];

    if (pivot_offset+threadIdx.x+1 < max_offset) {
        int row = d_row_indices[pivot_offset+threadIdx.x+1];
        x_soln[row] -= d_vals[pivot_offset+threadIdx.x+1]*xs;
    }

}

template __global__ void nofillmatrixsparse_kernels::lowtri_update_remaining_col<__half>(int, int, int *, __half *, __half *);
template __global__ void nofillmatrixsparse_kernels::lowtri_update_remaining_col<float>(int, int, int *, float *, float *);
template __global__ void nofillmatrixsparse_kernels::lowtri_update_remaining_col<double>(int, int, int *, double *, double *);

template <typename T>
__global__ void nofillmatrixsparse_kernels::upptri_update_remaining_col(
    int col_start_offset, int col_size, int *d_row_indices, T *d_vals, T *x_soln
) {
    __shared__ T xs;
    __shared__ int pivot_offset;
    pivot_offset = col_start_offset+col_size-1;

    xs = x_soln[d_row_indices[pivot_offset]];

    if (threadIdx.x < col_size-1) {
        int val_offset = col_start_offset+threadIdx.x;
        int row = d_row_indices[val_offset];
        x_soln[row] -= d_vals[val_offset]*xs;
    }

}

template __global__ void nofillmatrixsparse_kernels::upptri_update_remaining_col<__half>(int, int, int *, __half *, __half *);
template __global__ void nofillmatrixsparse_kernels::upptri_update_remaining_col<float>(int, int, int *, float *, float *);
template __global__ void nofillmatrixsparse_kernels::upptri_update_remaining_col<double>(int, int, int *, double *, double *);
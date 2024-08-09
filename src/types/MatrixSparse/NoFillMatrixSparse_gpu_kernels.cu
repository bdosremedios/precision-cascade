#include "types/GeneralMatrix/GeneralMatrix_gpu_constants.cuh"
#include "types/MatrixSparse/NoFillMatrixSparse_gpu_kernels.cuh"

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cuda_fp16.h>

namespace cascade::nofillmatrixsparse_kernels {

template <typename TPrecision>
__global__ void fast_back_sub_solve_level_set(
    int *d_level_set, TPrecision *d_soln,
    int *d_row_offsets, int *d_col_indices, TPrecision *d_values
) {

    __shared__ int row_id;
    if (threadIdx.x == 0) {
        row_id = d_level_set[blockIdx.x];
    }

    __shared__ int row_offset;
    if (threadIdx.x == 0) {
        row_offset = d_row_offsets[row_id];
    }

    __shared__ int row_offset_end;
    if (threadIdx.x == 0) {
        row_offset_end = d_row_offsets[row_id+1];
    }

    __shared__ int row_elem;
    if (threadIdx.x == 0) {
        row_elem = d_row_offsets[row_id+1]-d_row_offsets[row_id]-1; 
    }

    __shared__ TPrecision left_sum[genmat_gpu_const::WARPSIZE];
    left_sum[threadIdx.x] = static_cast<TPrecision>(0.);

    __shared__ int max_iter;
    if (threadIdx.x == 0) {
        max_iter = __float2int_ru(
            ceilf(
                __int2float_rd(row_elem) /
                __int2float_rd(genmat_gpu_const::WARPSIZE)
            )
        );
    }

    __syncwarp();
    
    for (int iter=0; iter<max_iter; ++iter) {

        int offset = (
            row_offset +
            iter * genmat_gpu_const::WARPSIZE +
            threadIdx.x +
            1
        );

        if (offset < row_offset_end) {
            left_sum[threadIdx.x] += (
                d_values[offset]*d_soln[d_col_indices[offset]]
            );
        }

    }
    
    for (int limit=genmat_gpu_const::WARPSIZE; limit >= 2; limit /= 2) {

        __shared__ int half_limit;
        if (threadIdx.x == 0) { half_limit = limit/2; }

        __syncwarp();

        if ((half_limit <= threadIdx.x) && (threadIdx.x < limit)) {
            left_sum[threadIdx.x-half_limit] += left_sum[threadIdx.x];
        }

    }

    __syncwarp();

    if (threadIdx.x == 0) {
        d_soln[row_id] = (d_soln[row_id] - left_sum[0])/d_values[row_offset];
    }

}

template __global__ void fast_back_sub_solve_level_set(
    int *d_level_set, __half *d_soln,
    int *d_row_offsets, int *d_col_indices, __half *d_values
);
template __global__ void fast_back_sub_solve_level_set(
    int *d_level_set, float *d_soln,
    int *d_row_offsets, int *d_col_indices, float *d_values
);
template __global__ void fast_back_sub_solve_level_set(
    int *d_level_set, double *d_soln,
    int *d_row_offsets, int *d_col_indices, double *d_values
);

template <typename TPrecision>
__global__ void fast_frwd_sub_solve_level_set(
    int *d_level_set, TPrecision *d_soln,
    int *d_row_offsets, int *d_col_indices, TPrecision *d_values
) {

    __shared__ int row_id;
    if (threadIdx.x == 0) {
        row_id = d_level_set[blockIdx.x];
    }

    __shared__ int row_offset;
    if (threadIdx.x == 0) {
        row_offset = d_row_offsets[row_id];
    }

    __shared__ int row_offset_end;
    if (threadIdx.x == 0) {
        row_offset_end = d_row_offsets[row_id+1];
    }

    __shared__ int row_elem;
    if (threadIdx.x == 0) {
        row_elem = d_row_offsets[row_id+1]-d_row_offsets[row_id]-1; 
    }

    __shared__ TPrecision left_sum[genmat_gpu_const::WARPSIZE];
    left_sum[threadIdx.x] = static_cast<TPrecision>(0.);

    __shared__ int max_iter;
    if (threadIdx.x == 0) {
        max_iter = __float2int_ru(
            ceilf(
                __int2float_rd(row_elem) /
                __int2float_rd(genmat_gpu_const::WARPSIZE)
            )
        );
    }

    __syncwarp();
    
    for (int iter=0; iter<max_iter; ++iter) {

        int offset = (
            row_offset +
            iter * genmat_gpu_const::WARPSIZE +
            threadIdx.x
        );

        if (offset < row_offset_end-1) {
            left_sum[threadIdx.x] += (
                d_values[offset]*d_soln[d_col_indices[offset]]
            );
        }

    }
    
    for (int limit=genmat_gpu_const::WARPSIZE; limit >= 2; limit /= 2) {

        __shared__ int half_limit;
        if (threadIdx.x == 0) { half_limit = limit/2; }

        __syncwarp();

        if ((half_limit <= threadIdx.x) && (threadIdx.x < limit)) {
            left_sum[threadIdx.x-half_limit] += left_sum[threadIdx.x];
        }

    }

    __syncwarp();

    if (threadIdx.x == 0) {
        d_soln[row_id] = (
            (d_soln[row_id] - left_sum[0]) / d_values[row_offset_end-1]
        );
    }

}

template __global__ void fast_frwd_sub_solve_level_set(
    int *d_level_set, __half *d_soln,
    int *d_row_offsets, int *d_col_indices, __half *d_values
);
template __global__ void fast_frwd_sub_solve_level_set(
    int *d_level_set, float *d_soln,
    int *d_row_offsets, int *d_col_indices, float *d_values
);
template __global__ void fast_frwd_sub_solve_level_set(
    int *d_level_set, double *d_soln,
    int *d_row_offsets, int *d_col_indices, double *d_values
);

template <typename TPrecision>
__global__ void update_row_pivot(
    int row, int pivot_offset, TPrecision *d_vals, TPrecision *x_soln
) {
    x_soln[row] /= d_vals[pivot_offset];
}

template __global__ void update_row_pivot<__half>(
    int, int, __half *, __half *
);
template __global__ void update_row_pivot<float>(
    int, int, float *, float *
);
template __global__ void update_row_pivot<double>(
    int, int, double *, double *
);

template <typename TPrecision>
__global__ void upptri_update_right_of_pivot(
    int row,
    int row_start_offset,
    int row_elem_count,
    int *d_col_indices,
    TPrecision *d_values,
    TPrecision *x_soln
) {

    __shared__ TPrecision xs[genmat_gpu_const::MAXTHREADSPERBLOCK];

    // Skip first element of row (pivot)
    int row_elem_idx = (blockIdx.x * blockDim.x) + threadIdx.x + 1; 
    int xs_idx = threadIdx.x;
    bool inside_row = row_elem_idx < row_elem_count;

    // Calc modified solution values
    if (inside_row) {
        int offset = row_start_offset + row_elem_idx;
        int col_idx = d_col_indices[offset];
        xs[xs_idx] = d_values[offset]*x_soln[col_idx]; 
    }

    __syncthreads();

    // Accumulate in O(log(blockDim.x)) fashion to first element of xs inspired
    // from https://developer.download.nvidia.com/compute/cuda/1.1-Beta/
    // x86_website/projects/reduction/doc/reduction.pdf but for from outermost
    // to innermost such that outer can skip itself if it is not inside row
    for (unsigned int s=blockDim.x; s>=2; s/=2) {
        int half_s = s/2;
        if (inside_row && (half_s <= xs_idx) && (xs_idx < s)) {
            xs[xs_idx-half_s] += xs[xs_idx];
        }
        __syncthreads();
    }

    // Subtract accumulation from pivot
    if (xs_idx == 0) {
        atomicAdd(x_soln + row, -xs[0]);
    }

}

template __global__ void upptri_update_right_of_pivot<__half>(
    int, int, int, int *, __half *, __half *
);
template __global__ void upptri_update_right_of_pivot<float>(
    int, int, int, int *, float *, float *
);
template __global__ void upptri_update_right_of_pivot<double>(
    int, int, int, int *, double *, double *
);

template <typename TPrecision>
__global__ void lowtri_update_left_of_pivot(
    int row,
    int row_start_offset,
    int row_elem_count,
    int *d_col_indices,
    TPrecision *d_values,
    TPrecision *x_soln
) {

    __shared__ TPrecision xs[genmat_gpu_const::MAXTHREADSPERBLOCK];

    // Skip last element of row (pivot)
    int row_elem_idx = (blockIdx.x * blockDim.x) + threadIdx.x; 
    int xs_idx = threadIdx.x;
    bool inside_row = row_elem_idx < (row_elem_count - 1);

    // Calc modified solution values
    if (inside_row) {
        int offset = row_start_offset + row_elem_idx;
        int col_idx = d_col_indices[offset];
        xs[xs_idx] = d_values[offset]*x_soln[col_idx]; 
    }

    __syncthreads();

    // Accumulate in O(log(blockDim.x)) fashion to first element of xs inspired
    // from https://developer.download.nvidia.com/compute/cuda/1.1-Beta/
    // x86_website/projects/reduction/doc/reduction.pdf but for from outermost
    // to innermost such that outer can skip itself if it is not inside row
    for (unsigned int s=blockDim.x; s>=2; s/=2) {
        int half_s = s/2;
        if (inside_row && (half_s <= xs_idx) && (xs_idx < s)) {
            xs[xs_idx-half_s] += xs[xs_idx];
        }
        __syncthreads();
    }

    // Subtract accumulation from pivot
    if (xs_idx == 0) {
        atomicAdd(x_soln + row, -xs[0]);
    }

}

template __global__ void lowtri_update_left_of_pivot<__half>(
    int, int, int, int *, __half *, __half *
);
template __global__ void lowtri_update_left_of_pivot<float>(
    int, int, int, int *, float *, float *
);
template __global__ void lowtri_update_left_of_pivot<double>(
    int, int, int, int *, double *, double *
);

}
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "types/MatrixSparse/NoFillMatrixSparse_gpu_kernels.cuh"

template <typename T>
__global__ void nofillmatrixsparse_kernels::solve_column(
    int col_offset, int32_t *d_row_indices, T *d_vals
) {
    
    

}

template __global__ void nofillmatrixsparse_kernels::solve_column<__half>(int, int32_t *, __half *);
template __global__ void nofillmatrixsparse_kernels::solve_column<float>(int, int32_t *, float *);
template __global__ void nofillmatrixsparse_kernels::solve_column<double>(int, int32_t *, double *);
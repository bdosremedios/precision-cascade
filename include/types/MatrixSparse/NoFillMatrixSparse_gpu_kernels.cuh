#ifndef NOFILLMATRIXSPARSE_GPU_KERNELS_CUH
#define NOFILLMATRIXSPARSE_GPU_KERNELS_CUH

#include <cuda_runtime.h>

namespace nofillmatrixsparse_kernels
{

    template <typename T>
    __global__ void update_pivot(
        int pivot_offset, int32_t *d_row_indices, T *d_vals, T *x_soln
    );

    template <typename T>
    __global__ void lowtri_update_remaining_col(
        int pivot_offset, int max_offset, int32_t *d_row_indices, T *d_vals, T *x_soln
    );

}

#endif
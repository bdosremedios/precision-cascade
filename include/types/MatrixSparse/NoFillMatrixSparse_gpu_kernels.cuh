#ifndef NOFILLMATRIXSPARSE_GPU_KERNELS_CUH
#define NOFILLMATRIXSPARSE_GPU_KERNELS_CUH

#include <cuda_runtime.h>

namespace nofillmatrixsparse_kernels
{

    template <typename T>
    __global__ void solve_column(int col_offset, int32_t *d_row_indices, T *d_vals);

}

#endif
#ifndef MATRIXDENSE_GPU_KERNELS_CUH
#define MATRIXDENSE_GPU_KERNELS_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace matrixdense_kernels
{

    template <typename T>
    __global__ void upptri_blk_solve_warp(const T *U, int m_rows, int diag_offset, T *x_soln);

    template <typename T>
    __global__ void upptri_rect_update_warp(const T *U, int m_rows, int diag_offset, T *x_soln);

    template <typename T>
    __global__ void lowtri_blk_solve_warp(const T *L, int m_rows, int diag_offset, T *x_soln);

    template <typename T>
    __global__ void lowtri_rect_update_warp(const T *L, int m_rows, int diag_offset, T *x_soln);

}

namespace matrixdense_dbl_kernels
{
    __global__ void solve_pivot_and_find_alpha(double *rhs, double *diag, double *alpha);
}

namespace matrixdense_sgl_kernels
{
    __global__ void solve_pivot_and_find_alpha(float *rhs, float *diag, float *alpha);
}

namespace matrixdense_hlf_kernels
{
    __global__ void solve_pivot_and_find_alpha(__half *rhs, __half *diag, float *alpha);
}

#endif
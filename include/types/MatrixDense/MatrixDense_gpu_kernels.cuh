#ifndef MATRIXDENSE_GPU_KERNELS_CUH
#define MATRIXDENSE_GPU_KERNELS_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace cascade::matrixdense_kernels
{

    template <typename TPrecision>
    __global__ void upptri_blk_solve_warp(
        const TPrecision *U, int m_rows, int diag_offset, TPrecision *x_soln
    );

    template <typename TPrecision>
    __global__ void upptri_rect_update_warp(
        const TPrecision *U, int m_rows, int diag_offset, TPrecision *x_soln
    );

    template <typename TPrecision>
    __global__ void lowtri_blk_solve_warp(
        const TPrecision *L, int m_rows, int diag_offset, TPrecision *x_soln
    );

    template <typename TPrecision>
    __global__ void lowtri_rect_update_warp(
        const TPrecision *L, int m_rows, int diag_offset, TPrecision *x_soln
    );

}

#endif
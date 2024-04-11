#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "types/GeneralMatrix/GeneralMatrix_gpu_constants.cuh"

#include "types/MatrixDense/MatrixDense_gpu_kernels.cuh"

template <typename T>
__global__ void matrixdense_kernels::upptri_blk_solve_warp(
    const T *U, int m_rows, int diag_offset, T *x_soln
) {

    __shared__ T xs[genmat_gpu_const::WARPSIZE];
    xs[threadIdx.x] = x_soln[diag_offset+threadIdx.x];

    #pragma unroll
    for (int i=genmat_gpu_const::WARPSIZE-1; i>=0; --i) {

        if ((diag_offset+threadIdx.x < m_rows) && (diag_offset+i < m_rows)) {

            if (threadIdx.x == i) {
                xs[threadIdx.x] /= U[(diag_offset+threadIdx.x)+(diag_offset+threadIdx.x)*m_rows];
            }

            if ((i != 0) && (threadIdx.x <= i-1)) {
                xs[threadIdx.x] -= U[(diag_offset+threadIdx.x)+(diag_offset+i)*m_rows]*xs[i];
            }

        }

    }

    x_soln[diag_offset+threadIdx.x] = xs[threadIdx.x];

}

template __global__ void matrixdense_kernels::upptri_blk_solve_warp<__half>(const __half *, int , int , __half *);
template __global__ void matrixdense_kernels::upptri_blk_solve_warp<float>(const float *, int , int , float *);
template __global__ void matrixdense_kernels::upptri_blk_solve_warp<double>(const double *, int , int , double *);

template <typename T>
__global__ void matrixdense_kernels::upptri_rect_update_warp(
    const T *U, int m_rows, int diag_offset, T *x_soln
) {

    __shared__ T xs_updating[genmat_gpu_const::WARPSIZE];
    __shared__ T xs_using[genmat_gpu_const::WARPSIZE];
    __shared__ int col;

    int updating_row = (blockIdx.x*blockDim.x) + threadIdx.x;
    int using_row = diag_offset + threadIdx.x;

    xs_updating[threadIdx.x] = x_soln[updating_row];
    xs_using[threadIdx.x] = x_soln[using_row];

    #pragma unroll
    for (int i=0; i<genmat_gpu_const::WARPSIZE; ++i) {
        col = diag_offset + i;
        if (col < m_rows) {
            xs_updating[threadIdx.x] -= U[updating_row+col*m_rows]*xs_using[i];
        }
    }

    x_soln[updating_row] = xs_updating[threadIdx.x];

}

template __global__ void matrixdense_kernels::upptri_rect_update_warp<__half>(const __half *, int , int , __half *);
template __global__ void matrixdense_kernels::upptri_rect_update_warp<float>(const float *, int , int , float *);
template __global__ void matrixdense_kernels::upptri_rect_update_warp<double>(const double *, int , int , double *);

template <typename T>
__global__ void matrixdense_kernels::lowtri_blk_solve_warp(
    const T *L, int m_rows, int diag_offset, T *x_soln
) {

    __shared__ T xs[genmat_gpu_const::WARPSIZE];
    xs[threadIdx.x] = x_soln[diag_offset+threadIdx.x];

    #pragma unroll
    for (int i=0; i<genmat_gpu_const::WARPSIZE; ++i) {

        if ((diag_offset+threadIdx.x < m_rows) && (diag_offset+i < m_rows)) {

            if (threadIdx.x == i) {
                xs[threadIdx.x] /= L[(diag_offset+threadIdx.x)+(diag_offset+threadIdx.x)*m_rows];
            }

            if (threadIdx.x >= i+1) {
                xs[threadIdx.x] -= L[(diag_offset+threadIdx.x)+(diag_offset+i)*m_rows]*xs[i];
            }

        }

    }

    x_soln[diag_offset+threadIdx.x] = xs[threadIdx.x];

}

template __global__ void matrixdense_kernels::lowtri_blk_solve_warp<__half>(const __half *, int , int , __half *);
template __global__ void matrixdense_kernels::lowtri_blk_solve_warp<float>(const float *, int , int , float *);
template __global__ void matrixdense_kernels::lowtri_blk_solve_warp<double>(const double *, int , int , double *);

template <typename T>
__global__ void matrixdense_kernels::lowtri_rect_update_warp(
    const T *L, int m_rows, int diag_offset, T *x_soln
) {

    __shared__ T xs_updating[genmat_gpu_const::WARPSIZE];
    __shared__ T xs_using[genmat_gpu_const::WARPSIZE];
    __shared__ int col;

    int updating_row = diag_offset + (blockIdx.x*blockDim.x) + threadIdx.x + genmat_gpu_const::WARPSIZE;
    int using_row = diag_offset + threadIdx.x;

    xs_using[threadIdx.x] = x_soln[using_row];

    if (updating_row < m_rows) {

        xs_updating[threadIdx.x] = x_soln[updating_row];

        #pragma unroll
        for (int i=0; i<genmat_gpu_const::WARPSIZE; ++i) {
            col = diag_offset + i;
            if (col < m_rows) {
                xs_updating[threadIdx.x] -= L[updating_row+col*m_rows]*xs_using[i];
            }
        }

        x_soln[updating_row] = xs_updating[threadIdx.x];

    }

}

template __global__ void matrixdense_kernels::lowtri_rect_update_warp<__half>(const __half *, int , int , __half *);
template __global__ void matrixdense_kernels::lowtri_rect_update_warp<float>(const float *, int , int , float *);
template __global__ void matrixdense_kernels::lowtri_rect_update_warp<double>(const double *, int , int , double *);

// *** MatrixDense double kernel implementations ***

__global__ void matrixdense_dbl_kernels::solve_pivot_and_find_alpha(double *rhs, double *diag, double *alpha) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    rhs[tid] = rhs[tid]/diag[tid];
    alpha[tid] = -rhs[tid];
}

// *** MatrixDense single kernel implementations ***

__global__ void matrixdense_sgl_kernels::solve_pivot_and_find_alpha(float *rhs, float *diag, float *alpha) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    rhs[tid] = rhs[tid]/diag[tid];
    alpha[tid] = -rhs[tid];
}

// *** MatrixDense half kernel implementations ***

__global__ void matrixdense_hlf_kernels::solve_pivot_and_find_alpha(__half *rhs, __half *diag, float *alpha) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    rhs[tid] = rhs[tid]/diag[tid];
    alpha[tid] = __half2float(-rhs[tid]);
}
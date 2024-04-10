#include "types/MatrixDense/MatrixDense_gpu_kernels.cuh"

template <typename T>
__global__ void matrixdense_kernels::upptri_blk_solve_warp(
    const T *U, int m_rows, int diag_offset, T *x_soln
) {

    __shared__ T xs[WARPSIZE];
    xs[threadIdx.x] = x_soln[diag_offset+threadIdx.x];

    #pragma unroll
    for (int i=WARPSIZE-1; i>=0; --i) {

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

    __shared__ T xs[WARPSIZE];
    xs[threadIdx.x] = x_soln[diag_offset + threadIdx.x];

    int col = diag_offset + threadIdx.x;
    int row = (blockIdx.y*blockDim.y) + threadIdx.y;

    if ((row < m_rows) && (col < m_rows)) {
        atomicAdd(x_soln+row, -U[row+col*m_rows]*xs[threadIdx.x]);
    }

}

template __global__ void matrixdense_kernels::upptri_rect_update_warp<__half>(const __half *, int , int , __half *);
template __global__ void matrixdense_kernels::upptri_rect_update_warp<float>(const float *, int , int , float *);
template __global__ void matrixdense_kernels::upptri_rect_update_warp<double>(const double *, int , int , double *);

template <typename T>
__global__ void matrixdense_kernels::lowtri_blk_solve_warp(
    const T *L, int m_rows, int diag_offset, T *x_soln
) {

    __shared__ T xs[WARPSIZE];
    xs[threadIdx.x] = x_soln[diag_offset+threadIdx.x];

    #pragma unroll
    for (int i=0; i<WARPSIZE; ++i) {

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

    __shared__ T xs[WARPSIZE];
    xs[threadIdx.x] = x_soln[diag_offset + threadIdx.x];

    int col = diag_offset + threadIdx.x;
    int row = diag_offset + WARPSIZE + (blockIdx.y*blockDim.y) + threadIdx.y;

    if (row < m_rows) {
        atomicAdd(x_soln+row, -L[row+col*m_rows]*xs[threadIdx.x]);
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
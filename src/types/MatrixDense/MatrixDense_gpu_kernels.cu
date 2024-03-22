#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "types/MatrixDense/MatrixDense_gpu_kernels.cuh"

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
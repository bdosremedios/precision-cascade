#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "types/MatrixDense/MatrixDense_gpu_kernels.cuh"

// *** MatrixDense double kernel implementations ***

__global__ void matrixdense_dbl_kernels::solve_pivot_and_find_alpha(double *rhs, double *diag, double *alpha) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    rhs[tid] = rhs[tid]/diag[tid];
    alpha[tid] = -rhs[tid];
}

__global__ void matrixdense_dbl_kernels::cast_to_half(double *mat_src, half *mat_dest, int m) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < m) {
        mat_dest[tid] = __double2half(mat_src[tid]);
    }
}

__global__ void matrixdense_dbl_kernels::cast_to_float(double *mat_src, float *mat_dest, int m) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < m) {
        mat_dest[tid] = __double2float_rn(mat_src[tid]);
    }
}

// *** MatrixDense single kernel implementations ***

__global__ void matrixdense_sgl_kernels::solve_pivot_and_find_alpha(float *rhs, float *diag, float *alpha) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    rhs[tid] = rhs[tid]/diag[tid];
    alpha[tid] = -rhs[tid];
}

__global__ void matrixdense_sgl_kernels::cast_to_half(float *mat_src, half *mat_dest, int m) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < m) {
        mat_dest[tid] = __float2half(mat_src[tid]);
    }
}

__global__ void matrixdense_sgl_kernels::cast_to_double(float *mat_src, double *mat_dest, int m) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < m) {
        mat_dest[tid] = static_cast<double>(mat_src[tid]);
    }
}

// *** MatrixDense half kernel implementations ***
__global__ void matrixdense_hlf_kernels::solve_pivot_and_find_alpha(__half *rhs, __half *diag, float *alpha) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    rhs[tid] = rhs[tid]/diag[tid];
    alpha[tid] = __half2float(-rhs[tid]);
}

__global__ void matrixdense_hlf_kernels::cast_to_float(__half *mat_src, float *mat_dest, int m) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < m) {
        mat_dest[tid] = __half2float(mat_src[tid]);
    }
}

__global__ void matrixdense_hlf_kernels::cast_to_double(__half *mat_src, double *mat_dest, int m) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < m) {
        mat_dest[tid] = static_cast<double>(mat_src[tid]);
    }
}
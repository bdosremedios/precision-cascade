#include "types/GeneralMatrix/GeneralMatrix_gpu_kernels.cuh"

__global__ void generalmatrix_dbl_kernels::cast_to_half(
    double *mat_src, __half *mat_dest, int m
) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < m) {
        mat_dest[tid] = __double2half(mat_src[tid]);
    }
}

__global__ void generalmatrix_dbl_kernels::cast_to_float(
    double *mat_src, float *mat_dest, int m
) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < m) {
        mat_dest[tid] = __double2float_rn(mat_src[tid]);
    }
}

__global__ void generalmatrix_sgl_kernels::cast_to_half(
    float *mat_src, __half *mat_dest, int m
) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < m) {
        mat_dest[tid] = __float2half(mat_src[tid]);
    }
}

__global__ void generalmatrix_sgl_kernels::cast_to_double(
    float *mat_src, double *mat_dest, int m
) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < m) {
        mat_dest[tid] = static_cast<double>(mat_src[tid]);
    }
}

__global__ void generalmatrix_hlf_kernels::cast_to_float(
    __half *mat_src, float *mat_dest, int m
) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < m) {
        mat_dest[tid] = __half2float(mat_src[tid]);
    }
}

__global__ void generalmatrix_hlf_kernels::cast_to_double(
    __half *mat_src, double *mat_dest, int m
) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < m) {
        mat_dest[tid] = static_cast<double>(mat_src[tid]);
    }
}
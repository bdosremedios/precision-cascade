#include "types/Scalar/Scalar_gpu_kernels.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>

// *** Scalar double kernel implementations *** 

__global__ void scalar_dbl_kernels::scalar_abs(double *scalar) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    scalar[tid] = fabs(scalar[tid]);
}

__global__ void scalar_dbl_kernels::scalar_sqrt(double *scalar) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    scalar[tid] = sqrt(scalar[tid]);
}

__global__ void scalar_dbl_kernels::scalar_recip(double *scalar) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    scalar[tid] = CUDART_ONE/(scalar[tid]);
}

__global__ void scalar_dbl_kernels::cast_to_half(double *scalar_src, __half *scalar_dest) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    scalar_dest[tid] = __double2half(scalar_src[tid]);
}

__global__ void scalar_dbl_kernels::cast_to_float(double *scalar_src, float *scalar_dest) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    scalar_dest[tid] = __double2float_rn(scalar_src[tid]);
}

// *** Scalar single kernel implementations *** 

__global__ void scalar_sgl_kernels::scalar_abs(float *scalar) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    scalar[tid] = fabsf(scalar[tid]);
}

__global__ void scalar_sgl_kernels::scalar_sqrt(float *scalar) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    scalar[tid] = sqrtf(scalar[tid]);
}

__global__ void scalar_sgl_kernels::scalar_recip(float *scalar) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    scalar[tid] = CUDART_ONE_F/(scalar[tid]);
}

__global__ void scalar_sgl_kernels::cast_to_half(float *scalar_src, __half *scalar_dest) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    scalar_dest[tid] = __float2half(scalar_src[tid]);
}

__global__ void scalar_sgl_kernels::cast_to_double(float *scalar_src, double *scalar_dest) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    scalar_dest[tid] = static_cast<double>(scalar_src[tid]);
}

// *** Scalar half kernel implementations ***

__global__ void scalar_hlf_kernels::scalar_abs(half *scalar) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    scalar[tid] = __habs(scalar[tid]);
}

__global__ void scalar_hlf_kernels::scalar_sqrt(half *scalar) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    scalar[tid] = hsqrt(scalar[tid]);
}

__global__ void scalar_hlf_kernels::scalar_recip(__half *scalar) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    scalar[tid] = CUDART_ONE_FP16/(scalar[tid]);
}

__global__ void scalar_hlf_kernels::cast_to_float(__half *scalar_src, float *scalar_dest) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    scalar_dest[tid] = __half2float(scalar_src[tid]);
}

__global__ void scalar_hlf_kernels::cast_to_double(__half *scalar_src, double *scalar_dest) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    scalar_dest[tid] = static_cast<double>(scalar_src[tid]);
}
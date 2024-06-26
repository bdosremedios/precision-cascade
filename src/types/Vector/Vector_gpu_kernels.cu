#include "types/Vector/Vector_gpu_kernels.cuh"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

namespace cascade {

__global__ void vector_dbl_kernels::cast_to_half(
    double *vec_src, __half *vec_dest, int m
) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < m) {
        vec_dest[tid] = __double2half(vec_src[tid]);
    }
}

__global__ void vector_dbl_kernels::cast_to_float(
    double *vec_src, float *vec_dest, int m
) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < m) {
        vec_dest[tid] = __double2float_rn(vec_src[tid]);
    }
}

__global__ void vector_sgl_kernels::cast_to_half(
    float *vec_src, __half *vec_dest, int m
) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < m) {
        vec_dest[tid] = __float2half(vec_src[tid]);
    }
}

__global__ void vector_sgl_kernels::cast_to_double(
    float *vec_src, double *vec_dest, int m
) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < m) {
        vec_dest[tid] = static_cast<double>(vec_src[tid]);
    }
}

__global__ void vector_hlf_kernels::cast_to_float(
    __half *scalar_src, float *scalar_dest, int m
) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < m) {
        scalar_dest[tid] = __half2float(scalar_src[tid]);
    }
}

__global__ void vector_hlf_kernels::cast_to_double(
    __half *scalar_src, double *scalar_dest, int m
) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < m) {
        scalar_dest[tid] = static_cast<double>(scalar_src[tid]);
    }
}

}
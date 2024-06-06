#ifndef VECTOR_GPU_KERNELS_CUH
#define VECTOR_GPU_KERNELS_CUH

#include <cuda_fp16.h>

namespace vector_dbl_kernels
{
    __global__ void cast_to_half(double *vec_src, __half *vec_dest, int m);
    __global__ void cast_to_float(double *vec_src, float *vec_dest, int m);
}

namespace vector_sgl_kernels
{
    __global__ void cast_to_half(float *vec_src, half *vec_dest, int m);
    __global__ void cast_to_double(float *vec_src, double *vec_dest, int m);
}

namespace vector_hlf_kernels
{
    __global__ void cast_to_float(__half *scalar_src, float *scalar_dest, int m);
    __global__ void cast_to_double(__half *scalar_src, double *scalar_dest, int m);
}

#endif
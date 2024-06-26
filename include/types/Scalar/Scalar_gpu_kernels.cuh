#ifndef SCALAR_GPU_KERNELS_CUH
#define SCALAR_GPU_KERNELS_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace cascade::scalar_dbl_kernels
{
    __global__ void scalar_abs(double *scalar);
    __global__ void scalar_sqrt(double *scalar); 
    __global__ void scalar_recip(double *scalar);
    __global__ void cast_to_half(double *scalar_src, __half *scalar_dest);
    __global__ void cast_to_float(double *scalar_src, float *scalar_dest);

}

namespace cascade::scalar_sgl_kernels
{
    __global__ void scalar_abs(float *scalar);
    __global__ void scalar_sqrt(float *scalar); 
    __global__ void scalar_recip(float *scalar);
    __global__ void cast_to_half(float *scalar_src, __half *scalar_dest);
    __global__ void cast_to_double(float *scalar_src, double *scalar_dest);

}

namespace cascade::scalar_hlf_kernels
{
    __global__ void scalar_abs(__half *scalar);
    __global__ void scalar_sqrt(__half *scalar); 
    __global__ void scalar_recip(__half *scalar);
    __global__ void cast_to_float(__half *scalar_src, float *scalar_dest);
    __global__ void cast_to_double(__half *scalar_src, double *scalar_dest);

}

#endif
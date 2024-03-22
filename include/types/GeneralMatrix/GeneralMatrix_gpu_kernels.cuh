#ifndef GENERALMATRIX_GPU_KERNELS_CUH
#define GENERALMATRIX_GPU_KERNELS_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace generalmatrix_dbl_kernels
{
    __global__ void cast_to_half(double *mat_src, __half *mat_dest, int m);
    __global__ void cast_to_float(double *mat_src, float *mat_dest, int m);
}

namespace generalmatrix_sgl_kernels
{
    __global__ void cast_to_half(float *mat_src, __half *mat_dest, int m);
    __global__ void cast_to_double(float *mat_src, double *mat_dest, int m);
}

namespace generalmatrix_hlf_kernels
{
    __global__ void cast_to_float(__half *mat_src, float *mat_dest, int m);
    __global__ void cast_to_double(__half *mat_src, double *mat_dest, int m);
}

#endif
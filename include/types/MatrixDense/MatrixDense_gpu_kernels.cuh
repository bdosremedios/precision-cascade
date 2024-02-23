#ifndef MATRIXDENSE_GPU_KERNELS_CUH
#define MATRIXDENSE_GPU_KERNELS_CUH

namespace matrixdense_dbl_kernels
{
    __global__ void solve_pivot_and_find_alpha(double *rhs, double *diag, double *alpha);
    __global__ void cast_to_half(double *mat_src, half *mat_dest, int m);
    __global__ void cast_to_float(double *mat_src, float *mat_dest, int m);
}

namespace matrixdense_sgl_kernels
{
    __global__ void solve_pivot_and_find_alpha(float *rhs, float *diag, float *alpha);
    __global__ void cast_to_half(float *mat_src, half *mat_dest, int m);
    __global__ void cast_to_double(float *mat_src, double *mat_dest, int m);
}

namespace matrixdense_hlf_kernels
{
    __global__ void solve_pivot_and_find_alpha(__half *rhs, __half *diag, float *alpha);
    __global__ void cast_to_float(__half *mat_src, float *mat_dest, int m);
    __global__ void cast_to_double(__half *mat_src, double *mat_dest, int m);
}

#endif
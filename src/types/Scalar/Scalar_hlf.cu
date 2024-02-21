#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "types/Scalar.h"
#include "Scalar.cuh"

template Scalar<__half> Scalar<__half>::operator+(const Scalar<__half> &other) const;
template Scalar<__half> Scalar<__half>::operator-(const Scalar<__half> &other) const;

template Scalar<__half> & Scalar<__half>::operator+=(const Scalar<__half> &);
template Scalar<__half> & Scalar<__half>::operator-=(const Scalar<__half> &);

template Scalar<__half> Scalar<__half>::operator*(const Scalar<__half> &) const;
template Scalar<__half> Scalar<__half>::operator/(const Scalar<__half> &) const;

template Scalar<__half> & Scalar<__half>::operator*=(const Scalar<__half> &);
template Scalar<__half> & Scalar<__half>::operator/=(const Scalar<__half> &);

template bool Scalar<__half>::operator==(const Scalar<__half> &) const;

__global__ void scalar_abs(half *scalar) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    scalar[tid] = __habs(scalar[tid]);
}

Scalar<__half> & Scalar<__half>::abs() {
    scalar_abs<<<1, 1>>>(d_scalar);
    return *this;
}

__global__ void scalar_sqrt(half *scalar) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    scalar[tid] = hsqrt(scalar[tid]);
}

Scalar<__half> & Scalar<__half>::sqrt() {
    scalar_sqrt<<<1, 1>>>(d_scalar);
    return *this;
}

__global__ void scalar_recip(__half *scalar) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    scalar[tid] = CUDART_ONE_FP16/(scalar[tid]);
}

Scalar<__half> & Scalar<__half>::reciprocol() {
    scalar_recip<<<1, 1>>>(d_scalar);
    return *this;
}

namespace scal_hlf_kern
{
    __global__ void cast_to_float(__half *scalar_src, float *scalar_dest) {
        int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
        scalar_dest[tid] = __half2float(scalar_src[tid]);
    }

    __global__ void cast_to_double(__half *scalar_src, double *scalar_dest) {
        int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
        scalar_dest[tid] = static_cast<double>(scalar_src[tid]);
    }
}

Scalar<__half> Scalar<__half>::to_half() const { return Scalar<__half>(*this); }

Scalar<float> Scalar<__half>::to_float() const {
    Scalar<float> created_scalar;
    scal_hlf_kern::cast_to_float<<<1, 1>>>(d_scalar, created_scalar.d_scalar);
    return created_scalar;
}

Scalar<double> Scalar<__half>::to_double() const {
    Scalar<double> created_scalar;
    scal_hlf_kern::cast_to_double<<<1, 1>>>(d_scalar, created_scalar.d_scalar);
    return created_scalar;
}
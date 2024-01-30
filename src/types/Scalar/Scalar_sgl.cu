#include "types/Scalar.h"
#include "Scalar.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

template Scalar<float> Scalar<float>::operator+(const Scalar<float> &other) const;
template Scalar<float> Scalar<float>::operator-(const Scalar<float> &other) const;

template void Scalar<float>::operator+=(const Scalar<float> &);
template void Scalar<float>::operator-=(const Scalar<float> &);

template Scalar<float> Scalar<float>::operator*(const Scalar<float> &) const;
template Scalar<float> Scalar<float>::operator/(const Scalar<float> &) const;

template void Scalar<float>::operator*=(const Scalar<float> &);
template void Scalar<float>::operator/=(const Scalar<float> &);

template bool Scalar<float>::operator==(const Scalar<float> &) const;

__global__ void scalar_abs(float *scalar) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    scalar[tid] = fabsf(scalar[tid]);
}

Scalar<float> & Scalar<float>::abs() {
    scalar_abs<<<1, 1>>>(d_scalar);
    return *this;
}

__global__ void scalar_sqrt(float *scalar) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    scalar[tid] = sqrtf(scalar[tid]);
}

Scalar<float> & Scalar<float>::sqrt() {
    scalar_sqrt<<<1, 1>>>(d_scalar);
    return *this;
}

namespace scal_sgl_kern
{
    __global__ void cast_to_half(float *scalar_src, __half *scalar_dest) {
        int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
        scalar_dest[tid] = __float2half(scalar_src[tid]);
    }

    __global__ void cast_to_double(float *scalar_src, double *scalar_dest) {
        int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
        scalar_dest[tid] = static_cast<double>(scalar_src[tid]);
    }
}

Scalar<__half> Scalar<float>::to_half() const {
    Scalar<__half> created_scalar;
    scal_sgl_kern::cast_to_half<<<1, 1>>>(d_scalar, created_scalar.d_scalar);
    return created_scalar;
}

Scalar<float> Scalar<float>::to_float() const { return Scalar<float>(*this); }

Scalar<double> Scalar<float>::to_double() const{
    Scalar<double> created_scalar;
    scal_sgl_kern::cast_to_double<<<1, 1>>>(d_scalar, created_scalar.d_scalar);
    return created_scalar;
}
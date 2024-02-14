#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include "types/Scalar.h"
#include "Scalar.cuh"

template Scalar<double> Scalar<double>::operator+(const Scalar<double> &other) const;
template Scalar<double> Scalar<double>::operator-(const Scalar<double> &other) const;

template void Scalar<double>::operator+=(const Scalar<double> &other);
template void Scalar<double>::operator-=(const Scalar<double> &other);

template Scalar<double> Scalar<double>::operator*(const Scalar<double> &other) const;
template Scalar<double> Scalar<double>::operator/(const Scalar<double> &other) const;

template void Scalar<double>::operator*=(const Scalar<double> &other);
template void Scalar<double>::operator/=(const Scalar<double> &other);

template bool Scalar<double>::operator==(const Scalar<double> &other) const;

__global__ void scalar_abs(double *scalar) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    scalar[tid] = fabs(scalar[tid]);
}

Scalar<double> & Scalar<double>::abs() {
    scalar_abs<<<1, 1>>>(d_scalar);
    return *this;
}

__global__ void scalar_sqrt(double *scalar) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    scalar[tid] = sqrt(scalar[tid]);
}

Scalar<double> & Scalar<double>::sqrt() {
    scalar_sqrt<<<1, 1>>>(d_scalar);
    return *this;
}

__global__ void scalar_recip(double *scalar) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    scalar[tid] = CUDART_ONE/(scalar[tid]);
}

Scalar<double> & Scalar<double>::reciprocol() {
    scalar_recip<<<1, 1>>>(d_scalar);
    return *this;
}

namespace scal_dbl_kern
{
    __global__ void cast_to_half(double *scalar_src, __half *scalar_dest) {
        int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
        scalar_dest[tid] = __double2half(scalar_src[tid]);
    }

    __global__ void cast_to_float(double *scalar_src, float *scalar_dest) {
        int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
        scalar_dest[tid] = __double2float_rn(scalar_src[tid]);
    }
}

Scalar<__half> Scalar<double>::to_half() const {
    Scalar<__half> created_scalar;
    scal_dbl_kern::cast_to_half<<<1, 1>>>(d_scalar, created_scalar.d_scalar);
    return created_scalar;
}

Scalar<float> Scalar<double>::to_float() const {
    Scalar<float> created_scalar;
    scal_dbl_kern::cast_to_float<<<1, 1>>>(d_scalar, created_scalar.d_scalar);
    return created_scalar;}

Scalar<double> Scalar<double>::to_double() const { return Scalar<double>(*this); }
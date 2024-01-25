#include "types/Scalar.h"
#include "./Scalar.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

template Scalar<__half> Scalar<__half>::operator+(const Scalar<__half> &other) const;
template Scalar<__half> Scalar<__half>::operator-(const Scalar<__half> &other) const;

template void Scalar<__half>::operator+=(const Scalar<__half> &);
template void Scalar<__half>::operator-=(const Scalar<__half> &);

template Scalar<__half> Scalar<__half>::operator*(const Scalar<__half> &) const;
template Scalar<__half> Scalar<__half>::operator/(const Scalar<__half> &) const;

template void Scalar<__half>::operator*=(const Scalar<__half> &);
template void Scalar<__half>::operator/=(const Scalar<__half> &);

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
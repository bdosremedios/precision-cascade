#include "types/Scalar.h"
#include "Scalar.cuh"

#include <cuda_runtime.h>

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
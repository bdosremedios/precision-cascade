#include "types/Scalar.h"
#include "Scalar.cuh"

#include <cuda_runtime.h>

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
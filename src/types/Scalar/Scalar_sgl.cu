#include "types/Scalar.h"
#include "Scalar.cuh"

#include <cuda_runtime.h>
#include <cublas_v2.h>

template Scalar<float> Scalar<float>::operator+(const Scalar<float> &other) const;
template Scalar<float> Scalar<float>::operator-(const Scalar<float> &other) const;

template void Scalar<float>::operator+=(const Scalar<float> &);
template void Scalar<float>::operator-=(const Scalar<float> &);

template Scalar<float> Scalar<float>::operator*(const Scalar<float> &) const;
template Scalar<float> Scalar<float>::operator/(const Scalar<float> &) const;

template void Scalar<float>::operator*=(const Scalar<float> &);
template void Scalar<float>::operator/=(const Scalar<float> &);
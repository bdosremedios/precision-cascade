#include "types/Scalar.h"
#include "Scalar.cuh"

#include <cuda_runtime.h>
#include <cublas_v2.h>

template Scalar<double> Scalar<double>::operator+(const Scalar<double> &other) const;
template Scalar<double> Scalar<double>::operator-(const Scalar<double> &other) const;

template void Scalar<double>::operator+=(const Scalar<double> &other);
template void Scalar<double>::operator-=(const Scalar<double> &other);

template Scalar<double> Scalar<double>::operator*(const Scalar<double> &other) const;
template Scalar<double> Scalar<double>::operator/(const Scalar<double> &other) const;

template void Scalar<double>::operator*=(const Scalar<double> &other);
template void Scalar<double>::operator/=(const Scalar<double> &other);
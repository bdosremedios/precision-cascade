#include "types/Scalar.h"
#include "Scalar.cuh"

#include <cuda_runtime.h>
#include <cublas_v2.h>

template Scalar<__half> Scalar<__half>::operator*(const Scalar<__half> &) const;
template Scalar<__half> Scalar<__half>::operator/(const Scalar<__half> &) const;

template void Scalar<__half>::operator*=(const Scalar<__half> &);
template void Scalar<__half>::operator/=(const Scalar<__half> &);
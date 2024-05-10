#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "types/Scalar/Scalar.h"
#include "types/Scalar/Scalar_template_subroutines.cuh"

template Scalar<double> Scalar<double>::operator+(const Scalar<double> &other) const;
template Scalar<double> Scalar<double>::operator-(const Scalar<double> &other) const;

template Scalar<double> & Scalar<double>::operator+=(const Scalar<double> &other);
template Scalar<double> & Scalar<double>::operator-=(const Scalar<double> &other);

template Scalar<double> Scalar<double>::operator*(const Scalar<double> &other) const;
template Scalar<double> Scalar<double>::operator/(const Scalar<double> &other) const;

template Scalar<double> & Scalar<double>::operator*=(const Scalar<double> &other);
template Scalar<double> & Scalar<double>::operator/=(const Scalar<double> &other);

template bool Scalar<double>::operator==(const Scalar<double> &other) const;

Scalar<double> & Scalar<double>::abs() {
    scalar_dbl_kernels::scalar_abs<<<1, 1>>>(d_scalar);
    check_cuda_error(cudaGetLastError());
    return *this;
}

Scalar<double> & Scalar<double>::sqrt() {
    scalar_dbl_kernels::scalar_sqrt<<<1, 1>>>(d_scalar);
    check_cuda_error(cudaGetLastError());
    return *this;
}

Scalar<double> & Scalar<double>::reciprocol() {
    scalar_dbl_kernels::scalar_recip<<<1, 1>>>(d_scalar);
    check_cuda_error(cudaGetLastError());
    return *this;
}

Scalar<__half> Scalar<double>::to_half() const {
    Scalar<__half> created_scalar;
    scalar_dbl_kernels::cast_to_half<<<1, 1>>>(d_scalar, created_scalar.d_scalar);
    check_cuda_error(cudaGetLastError());
    return created_scalar;
}

Scalar<float> Scalar<double>::to_float() const {
    Scalar<float> created_scalar;
    scalar_dbl_kernels::cast_to_float<<<1, 1>>>(d_scalar, created_scalar.d_scalar);
    check_cuda_error(cudaGetLastError());
    return created_scalar;}

Scalar<double> Scalar<double>::to_double() const { return Scalar<double>(*this); }
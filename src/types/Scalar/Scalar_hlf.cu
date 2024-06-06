#include "types/Scalar/Scalar.h"
#include "types/Scalar/Scalar_template_subroutines.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

template Scalar<__half> Scalar<__half>::operator+(const Scalar<__half> &other) const;
template Scalar<__half> Scalar<__half>::operator-(const Scalar<__half> &other) const;

template Scalar<__half> & Scalar<__half>::operator+=(const Scalar<__half> &);
template Scalar<__half> & Scalar<__half>::operator-=(const Scalar<__half> &);

template Scalar<__half> Scalar<__half>::operator*(const Scalar<__half> &) const;
template Scalar<__half> Scalar<__half>::operator/(const Scalar<__half> &) const;

template Scalar<__half> & Scalar<__half>::operator*=(const Scalar<__half> &);
template Scalar<__half> & Scalar<__half>::operator/=(const Scalar<__half> &);

template bool Scalar<__half>::operator==(const Scalar<__half> &) const;

Scalar<__half> & Scalar<__half>::abs() {
    scalar_hlf_kernels::scalar_abs<<<1, 1>>>(d_scalar);
    check_kernel_launch(
        cudaGetLastError(),
        "Scalar<__half>::abs",
        "scalar_hlf_kernels::scalar_abs",
        1, 1
    );
    return *this;
}


Scalar<__half> & Scalar<__half>::sqrt() {
    scalar_hlf_kernels::scalar_sqrt<<<1, 1>>>(d_scalar);
    check_kernel_launch(
        cudaGetLastError(),
        "Scalar<__half>::sqrt",
        "scalar_hlf_kernels::scalar_sqrt",
        1, 1
    );
    return *this;
}


Scalar<__half> & Scalar<__half>::reciprocol() {
    scalar_hlf_kernels::scalar_recip<<<1, 1>>>(d_scalar);
    check_kernel_launch(
        cudaGetLastError(),
        "Scalar<__half>::reciprocol",
        "scalar_hlf_kernels::scalar_recip",
        1, 1
    );
    return *this;
}

Scalar<__half> Scalar<__half>::to_half() const { return Scalar<__half>(*this); }

Scalar<float> Scalar<__half>::to_float() const {
    Scalar<float> created_scalar;
    scalar_hlf_kernels::cast_to_float<<<1, 1>>>(d_scalar, created_scalar.d_scalar);
    check_kernel_launch(
        cudaGetLastError(),
        "Scalar<__half>::to_float",
        "scalar_hlf_kernels::cast_to_float",
        1, 1
    );
    return created_scalar;
}

Scalar<double> Scalar<__half>::to_double() const {
    Scalar<double> created_scalar;
    scalar_hlf_kernels::cast_to_double<<<1, 1>>>(d_scalar, created_scalar.d_scalar);
    check_kernel_launch(
        cudaGetLastError(),
        "Scalar<__half>::to_double",
        "scalar_hlf_kernels::cast_to_double",
        1, 1
    );
    return created_scalar;
}
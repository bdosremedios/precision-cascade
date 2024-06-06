#include "types/Scalar/Scalar.h"
#include "types/Scalar/Scalar_template_subroutines.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

template Scalar<float> Scalar<float>::operator+(const Scalar<float> &other) const;
template Scalar<float> Scalar<float>::operator-(const Scalar<float> &other) const;

template Scalar<float> & Scalar<float>::operator+=(const Scalar<float> &);
template Scalar<float> & Scalar<float>::operator-=(const Scalar<float> &);

template Scalar<float> Scalar<float>::operator*(const Scalar<float> &) const;
template Scalar<float> Scalar<float>::operator/(const Scalar<float> &) const;

template Scalar<float> & Scalar<float>::operator*=(const Scalar<float> &);
template Scalar<float> & Scalar<float>::operator/=(const Scalar<float> &);

template bool Scalar<float>::operator==(const Scalar<float> &) const;

Scalar<float> & Scalar<float>::abs() {
    scalar_sgl_kernels::scalar_abs<<<1, 1>>>(d_scalar);
    check_kernel_launch(
        cudaGetLastError(),
        "Scalar<float>::abs",
        "scalar_sgl_kernels::scalar_abs",
        1, 1
    );
    return *this;
}

Scalar<float> & Scalar<float>::sqrt() {
    scalar_sgl_kernels::scalar_sqrt<<<1, 1>>>(d_scalar);
    check_kernel_launch(
        cudaGetLastError(),
        "Scalar<float>::sqrt",
        "scalar_sgl_kernels::scalar_sqrt",
        1, 1
    );
    return *this;
}

Scalar<float> & Scalar<float>::reciprocol() {
    scalar_sgl_kernels::scalar_recip<<<1, 1>>>(d_scalar);
    check_kernel_launch(
        cudaGetLastError(),
        "Scalar<float>::reciprocol",
        "scalar_sgl_kernels::scalar_recip",
        1, 1
    );
    return *this;
}

Scalar<__half> Scalar<float>::to_half() const {
    Scalar<__half> created_scalar;
    scalar_sgl_kernels::cast_to_half<<<1, 1>>>(d_scalar, created_scalar.d_scalar);
    check_kernel_launch(
        cudaGetLastError(),
        "Scalar<float>::to_half",
        "scalar_sgl_kernels::cast_to_half",
        1, 1
    );
    return created_scalar;
}

Scalar<float> Scalar<float>::to_float() const { return Scalar<float>(*this); }

Scalar<double> Scalar<float>::to_double() const{
    Scalar<double> created_scalar;
    scalar_sgl_kernels::cast_to_double<<<1, 1>>>(d_scalar, created_scalar.d_scalar);
    check_kernel_launch(
        cudaGetLastError(),
        "Scalar<float>::to_double",
        "scalar_sgl_kernels::cast_to_double",
        1, 1
    );
    return created_scalar;
}
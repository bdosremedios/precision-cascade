#include "types/Scalar/Scalar.h"
#include "types/Scalar/Scalar_template_subroutines.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

template Scalar<double> Scalar<double>::operator+(
    const Scalar<double> &
) const;
template Scalar<double> Scalar<double>::operator-(
    const Scalar<double> &
) const;

template Scalar<double> & Scalar<double>::operator+=(
    const Scalar<double> &
);
template Scalar<double> & Scalar<double>::operator-=(
    const Scalar<double> &
);

template Scalar<double> Scalar<double>::operator*(
    const Scalar<double> &
) const;
template Scalar<double> Scalar<double>::operator/(
    const Scalar<double> &
) const;

template Scalar<double> & Scalar<double>::operator*=(
    const Scalar<double> &
);
template Scalar<double> & Scalar<double>::operator/=(
    const Scalar<double> &
);

template bool Scalar<double>::operator==(
    const Scalar<double> &
) const;

Scalar<double> & Scalar<double>::abs() {
    scalar_dbl_kernels::scalar_abs<<<1, 1>>>(d_scalar);
    check_kernel_launch(
        cudaGetLastError(),
        "Scalar<double>::abs",
        "scalar_dbl_kernels::scalar_abs",
        1, 1
    );
    return *this;
}

Scalar<double> & Scalar<double>::sqrt() {
    scalar_dbl_kernels::scalar_sqrt<<<1, 1>>>(d_scalar);
    check_kernel_launch(
        cudaGetLastError(),
        "Scalar<double>::sqrt",
        "scalar_dbl_kernels::scalar_sqrt",
        1, 1
    );
    return *this;
}

Scalar<double> & Scalar<double>::reciprocol() {
    scalar_dbl_kernels::scalar_recip<<<1, 1>>>(d_scalar);
    check_kernel_launch(
        cudaGetLastError(),
        "Scalar<double>::reciprocol",
        "scalar_dbl_kernels::scalar_recip",
        1, 1
    );
    return *this;
}

Scalar<__half> Scalar<double>::to_half() const {
    Scalar<__half> created_scalar;
    scalar_dbl_kernels::cast_to_half<<<1, 1>>>(
        d_scalar, created_scalar.d_scalar
    );
    check_kernel_launch(
        cudaGetLastError(),
        "Scalar<double>::to_half",
        "scalar_dbl_kernels::cast_to_half",
        1, 1
    );
    return created_scalar;
}

Scalar<float> Scalar<double>::to_float() const {
    Scalar<float> created_scalar;
    scalar_dbl_kernels::cast_to_float<<<1, 1>>>(
        d_scalar, created_scalar.d_scalar
    );
    check_kernel_launch(
        cudaGetLastError(),
        "Scalar<double>::to_float",
        "scalar_dbl_kernels::cast_to_float",
        1, 1
    );
    return created_scalar;}

Scalar<double> Scalar<double>::to_double() const {
    return Scalar<double>(*this);
}
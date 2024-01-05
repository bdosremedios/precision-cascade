#include "types/MatrixVector.h"
#include "tools/cublas_check.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

template<>
MatrixVector<double> MatrixVector<double>::operator*(const double &scalar) const {

    MatrixVector<double> c(*this);

    cublasStatus_t status = cublasScalEx(
        handle, m, &scalar, CUDA_R_64F, c.d_vec, CUDA_R_64F, 1, CUDA_R_64F
    );
    check_cublas_status(status);

    return c;

}

template<>
MatrixVector<double>& MatrixVector<double>::operator*=(const double &scalar) {

    cublasStatus_t status = cublasScalEx(
        handle, m, &scalar, CUDA_R_64F, d_vec, CUDA_R_64F, 1, CUDA_R_64F
    );
    check_cublas_status(status);

    return *this;

}
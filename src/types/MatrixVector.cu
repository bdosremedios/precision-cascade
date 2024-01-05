#include "types/MatrixVector.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <stdexcept>
#include <string>

void check_cublas_status(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublas error: " + std::to_string(status));
    }
}

template<>
MatrixVector<double> MatrixVector<double>::operator*(const double &scalar) const {

    MatrixVector<double> c(*this);

    cublasStatus_t status = cublasScalEx(
        this->handle, m, &scalar, CUDA_R_64F, c.d_vec, CUDA_R_64F, 1, CUDA_R_64F
    );
    check_cublas_status(status);

    return c;

}

template<>
MatrixVector<float> MatrixVector<float>::operator*(const float &scalar) const {

    MatrixVector<float> c(*this);

    cublasStatus_t status = cublasScalEx(
        handle, m, &scalar, CUDA_R_32F, c.d_vec, CUDA_R_32F, 1, CUDA_R_32F
    );
    check_cublas_status(status);

    return c;

}

template<>
MatrixVector<__half> MatrixVector<__half>::operator*(const __half &scalar) const {

    MatrixVector<__half> c(*this);
    float *scalar_cast = static_cast<float *>(malloc(sizeof(float)));
    *scalar_cast = static_cast<float>(scalar);

    cublasStatus_t status = cublasScalEx(
        handle, m, scalar_cast, CUDA_R_32F, c.d_vec, CUDA_R_16F, 1, CUDA_R_32F
    );
    check_cublas_status(status);

    free(scalar_cast);

    return c;

}

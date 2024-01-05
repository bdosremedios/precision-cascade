#include "types/MatrixVector.h"
#include "tools/cublas_check.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

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

template<>
MatrixVector<__half>& MatrixVector<__half>::operator*=(const __half &scalar) {

    float *scalar_cast = static_cast<float *>(malloc(sizeof(float)));
    *scalar_cast = static_cast<float>(scalar);

    cublasStatus_t status = cublasScalEx(
        handle, m, scalar_cast, CUDA_R_32F, d_vec, CUDA_R_16F, 1, CUDA_R_32F
    );
    check_cublas_status(status);

    free(scalar_cast);

    return *this;

}
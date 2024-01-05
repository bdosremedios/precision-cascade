#include "types/MatrixVector.h"
#include "tools/cublas_check.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

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
MatrixVector<float>& MatrixVector<float>::operator*=(const float &scalar) {

    cublasStatus_t status = cublasScalEx(
        handle, m, &scalar, CUDA_R_32F, d_vec, CUDA_R_32F, 1, CUDA_R_32F
    );
    check_cublas_status(status);

    return *this;

}
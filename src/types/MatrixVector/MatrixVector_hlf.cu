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

    float scalar_cast = static_cast<float>(scalar);

    cublasStatus_t status = cublasScalEx(
        handle, m, &scalar_cast, CUDA_R_32F, d_vec, CUDA_R_16F, 1, CUDA_R_32F
    );
    check_cublas_status(status);


    return *this;

}

template<>
MatrixVector<__half> MatrixVector<__half>::operator+(const MatrixVector<__half> &vec) const {

    MatrixVector<__half> c(*this);
    float alpha = 1.;

    cublasStatus_t status = cublasAxpyEx(
        handle, m, &alpha, CUDA_R_32F, vec.d_vec, CUDA_R_16F, 1, c.d_vec, CUDA_R_16F, 1, CUDA_R_32F
    );
    check_cublas_status(status);

    return c;

}

template<>
MatrixVector<__half> MatrixVector<__half>::operator-(const MatrixVector<__half> &vec) const {

    MatrixVector<__half> c(*this);
    float alpha = -1.;

    cublasStatus_t status = cublasAxpyEx(
        handle, m, &alpha, CUDA_R_32F, vec.d_vec, CUDA_R_16F, 1, c.d_vec, CUDA_R_16F, 1, CUDA_R_32F
    );
    check_cublas_status(status);

    return c;

}

template<>
MatrixVector<__half>& MatrixVector<__half>::operator+=(const MatrixVector<__half> &vec) {

    float alpha = 1.;

    cublasStatus_t status = cublasAxpyEx(
        handle, m, &alpha, CUDA_R_32F, vec.d_vec, CUDA_R_16F, 1, d_vec, CUDA_R_16F, 1, CUDA_R_32F
    );
    check_cublas_status(status);

    return *this;

}

template<>
MatrixVector<__half>& MatrixVector<__half>::operator-=(const MatrixVector<__half> &vec) {

    float alpha = -1.;

    cublasStatus_t status = cublasAxpyEx(
        handle, m, &alpha, CUDA_R_32F, vec.d_vec, CUDA_R_16F, 1, d_vec, CUDA_R_16F, 1, CUDA_R_32F
    );
    check_cublas_status(status);

    return *this;

}

template<>
__half MatrixVector<__half>::dot(const MatrixVector<__half> &vec) const {
    
    __half result;

    cublasStatus_t status = cublasDotEx(
        handle, m, d_vec, CUDA_R_16F, 1, vec.d_vec, CUDA_R_16F, 1, &result, CUDA_R_16F, CUDA_R_32F
    );
    check_cublas_status(status);

    return result;

}

template<>
__half MatrixVector<__half>::norm() const {

    __half result;

    cublasStatus_t status = cublasNrm2Ex(
        handle, m, d_vec, CUDA_R_16F, 1, &result, CUDA_R_16F, CUDA_R_32F
    );
    check_cublas_status(status);

    return result;

}
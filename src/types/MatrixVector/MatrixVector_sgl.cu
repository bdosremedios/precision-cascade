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

template<>
MatrixVector<float> MatrixVector<float>::operator+(const MatrixVector<float> &vec) const {

    MatrixVector<float> c(*this);
    float alpha = 1.;

    cublasStatus_t status = cublasAxpyEx(
        handle, m, &alpha, CUDA_R_32F, vec.d_vec, CUDA_R_32F, 1, c.d_vec, CUDA_R_32F, 1, CUDA_R_32F
    );
    check_cublas_status(status);

    return c;

}

template<>
MatrixVector<float> MatrixVector<float>::operator-(const MatrixVector<float> &vec) const {

    MatrixVector<float> c(*this);
    float alpha = -1.;

    cublasStatus_t status = cublasAxpyEx(
        handle, m, &alpha, CUDA_R_32F, vec.d_vec, CUDA_R_32F, 1, c.d_vec, CUDA_R_32F, 1, CUDA_R_32F
    );
    check_cublas_status(status);

    return c;

}

template<>
MatrixVector<float>& MatrixVector<float>::operator+=(const MatrixVector<float> &vec) {

    float alpha = 1.;

    cublasStatus_t status = cublasAxpyEx(
        handle, m, &alpha, CUDA_R_32F, vec.d_vec, CUDA_R_32F, 1, d_vec, CUDA_R_32F, 1, CUDA_R_32F
    );
    check_cublas_status(status);

    return *this;

}

template<>
MatrixVector<float>& MatrixVector<float>::operator-=(const MatrixVector<float> &vec) {

    float alpha = -1.;

    cublasStatus_t status = cublasAxpyEx(
        handle, m, &alpha, CUDA_R_32F, vec.d_vec, CUDA_R_32F, 1, d_vec, CUDA_R_32F, 1, CUDA_R_32F
    );
    check_cublas_status(status);

    return *this;

}

template<>
float MatrixVector<float>::dot(const MatrixVector<float> &vec) const {
    
    float result;

    cublasStatus_t status = cublasDotEx(
        handle, m, d_vec, CUDA_R_32F, 1, vec.d_vec, CUDA_R_32F, 1, &result, CUDA_R_32F, CUDA_R_32F
    );
    check_cublas_status(status);

    return result;

}

template<>
float MatrixVector<float>::norm() const {

    float result;

    cublasStatus_t status = cublasNrm2Ex(
        handle, m, d_vec, CUDA_R_32F, 1, &result, CUDA_R_32F, CUDA_R_32F
    );
    check_cublas_status(status);

    return result;

}
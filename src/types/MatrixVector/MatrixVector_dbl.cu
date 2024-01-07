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

template<>
MatrixVector<double> MatrixVector<double>::operator+(const MatrixVector<double> &vec) const {

    MatrixVector<double> c(*this);
    double alpha = 1.;

    cublasStatus_t status = cublasAxpyEx(
        handle, m, &alpha, CUDA_R_64F, vec.d_vec, CUDA_R_64F, 1, c.d_vec, CUDA_R_64F, 1, CUDA_R_64F
    );
    check_cublas_status(status);

    return c;

}

template<>
MatrixVector<double> MatrixVector<double>::operator-(const MatrixVector<double> &vec) const {

    MatrixVector<double> c(*this);
    double alpha = -1.;

    cublasStatus_t status = cublasAxpyEx(
        handle, m, &alpha, CUDA_R_64F, vec.d_vec, CUDA_R_64F, 1, c.d_vec, CUDA_R_64F, 1, CUDA_R_64F
    );
    check_cublas_status(status);

    return c;

}

template<>
MatrixVector<double>& MatrixVector<double>::operator+=(const MatrixVector<double> &vec) {

    double alpha = 1.;

    cublasStatus_t status = cublasAxpyEx(
        handle, m, &alpha, CUDA_R_64F, vec.d_vec, CUDA_R_64F, 1, d_vec, CUDA_R_64F, 1, CUDA_R_64F
    );
    check_cublas_status(status);

    return *this;

}

template<>
MatrixVector<double>& MatrixVector<double>::operator-=(const MatrixVector<double> &vec) {

    double alpha = -1.;

    cublasStatus_t status = cublasAxpyEx(
        handle, m, &alpha, CUDA_R_64F, vec.d_vec, CUDA_R_64F, 1, d_vec, CUDA_R_64F, 1, CUDA_R_64F
    );
    check_cublas_status(status);

    return *this;

}

template<>
double MatrixVector<double>::dot(const MatrixVector<double> &vec) const {
    
    double result;

    cublasStatus_t status = cublasDotEx(
        handle, m, d_vec, CUDA_R_64F, 1, vec.d_vec, CUDA_R_64F, 1, &result, CUDA_R_64F, CUDA_R_64F
    );
    check_cublas_status(status);

    return result;

}

template<>
double MatrixVector<double>::norm() const {

    double result;

    cublasStatus_t status = cublasNrm2Ex(
        handle, m, d_vec, CUDA_R_64F, 1, &result, CUDA_R_64F, CUDA_R_64F
    );
    check_cublas_status(status);

    return result;

}
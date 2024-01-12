#include "types/MatrixDense.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

template<>
MatrixDense<float> MatrixDense<float>::operator*(const float &scalar) const {

    MatrixDense<float> c(*this);

    check_cublas_status(
        cublasScalEx(
            handle, m_rows*n_cols, &scalar, CUDA_R_32F, c.d_mat, CUDA_R_32F, 1, CUDA_R_32F
        )
    );

    return c;

}

template <>
MatrixVector<float> MatrixDense<float>::operator*(const MatrixVector<float> &vec) const {

    if (vec.rows() != n_cols) {
        throw std::runtime_error(
            "MatrixDense: invalid vec in matrix-vector prod (operator*(const MatrixVector<float> &vec))"
        );
    }

    MatrixVector<float> c(MatrixVector<float>::Zero(handle, m_rows));

    float alpha = 1.;
    float beta = 0.;

    check_cublas_status(
        cublasGemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            m_rows, 1, n_cols,
            &alpha,
            d_mat, CUDA_R_32F, m_rows,
            vec.d_vec, CUDA_R_32F, n_cols,
            &beta,
            c.d_vec, CUDA_R_32F, m_rows,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        )
    );

    return c;

}

template <>
MatrixVector<float> MatrixDense<float>::transpose_prod(const MatrixVector<float> &vec) const {

    if (vec.rows() != m_rows) { throw std::runtime_error("MatrixDense: invalid vec in transpose_prod"); }

    MatrixVector<float> c(handle, n_cols);

    float alpha = static_cast<float>(1.);
    float beta = static_cast<float>(0.);

    check_cublas_status(
        cublasGemmEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            n_cols, 1, m_rows,
            &alpha,
            d_mat, CUDA_R_32F, m_rows,
            vec.d_vec, CUDA_R_32F, m_rows,
            &beta,
            c.d_vec, CUDA_R_32F, n_cols,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        )
    );

    return c;

}
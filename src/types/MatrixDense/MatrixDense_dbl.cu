#include "types/MatrixDense.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

template<>
MatrixDense<double> MatrixDense<double>::operator*(const double &scalar) const {

    MatrixDense<double> c(*this);

    check_cublas_status(
        cublasScalEx(
            handle, m_rows*n_cols, &scalar, CUDA_R_64F, c.d_mat, CUDA_R_64F, 1, CUDA_R_64F
        )
    );

    return c;
}

template <>
MatrixVector<double> MatrixDense<double>::operator*(const MatrixVector<double> &vec) const {

    if (vec.rows() != n_cols) {
        throw std::runtime_error(
            "MatrixDense: invalid vec in matrix-vector prod (operator*(const MatrixVector<double> &vec))"
        );
    }

    MatrixVector<double> c(MatrixVector<double>::Zero(handle, m_rows));

    double alpha = 1.;
    double beta = 0.;

    check_cublas_status(
        cublasGemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            m_rows, 1, n_cols,
            &alpha,
            d_mat, CUDA_R_64F, m_rows,
            vec.d_vec, CUDA_R_64F, n_cols,
            &beta,
            c.d_vec, CUDA_R_64F, m_rows,
            CUBLAS_COMPUTE_64F,
            CUBLAS_GEMM_DEFAULT
        )
    );

    return c;

}

template <>
MatrixVector<double> MatrixDense<double>::transpose_prod(const MatrixVector<double> &vec) const {

    if (vec.rows() != m_rows) { throw std::runtime_error("MatrixDense: invalid vec in transpose_prod"); }

    MatrixVector<double> c(MatrixVector<double>::Zero(handle, n_cols));

    double alpha = 1.;
    double beta = 0.;

    check_cublas_status(
        cublasGemmEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            n_cols, 1, m_rows,
            &alpha,
            d_mat, CUDA_R_64F, m_rows,
            vec.d_vec, CUDA_R_64F, m_rows,
            &beta,
            c.d_vec, CUDA_R_64F, n_cols,
            CUBLAS_COMPUTE_64F,
            CUBLAS_GEMM_DEFAULT
        )
    );

    return c;

}

template <>
MatrixDense<double> MatrixDense<double>::operator*(const MatrixDense<double> &mat) const {

    if (mat.rows() != n_cols) {
        throw std::runtime_error(
            "MatrixDense: invalid mat in matrix-matrix prod (operator*(const MatrixDense<double> &mat))"
        );
    }

    MatrixDense<double> c(MatrixDense<double>::Zero(handle, m_rows, mat.cols()));

    double alpha = 1.;
    double beta = 0.;

    check_cublas_status(
        cublasGemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            m_rows, mat.cols(), n_cols,
            &alpha,
            d_mat, CUDA_R_64F, m_rows,
            mat.d_mat, CUDA_R_64F, n_cols,
            &beta,
            c.d_mat, CUDA_R_64F, m_rows,
            CUBLAS_COMPUTE_64F,
            CUBLAS_GEMM_DEFAULT
        )
    );

    return c;

}

template <>
MatrixDense<double> MatrixDense<double>::operator+(const MatrixDense<double> &mat) const {

    if ((mat.rows() != m_rows) || (mat.cols() != n_cols)) {
        throw std::runtime_error(
            "MatrixDense: invalid mat in matrix add (operator+(const MatrixDense<double> &mat))"
        );
    }

    MatrixDense<double> c(*this);

    double alpha = 1.;

    check_cublas_status(
        cublasAxpyEx(
            handle, m_rows*n_cols,
            &alpha, CUDA_R_64F,
            mat.d_mat, CUDA_R_64F, 1,
            c.d_mat, CUDA_R_64F, 1,
            CUDA_R_64F
        )
    );

    return c;

}

template <>
MatrixDense<double> MatrixDense<double>::operator-(const MatrixDense<double> &mat) const {

    if ((mat.rows() != m_rows) || (mat.cols() != n_cols)) {
        throw std::runtime_error(
            "MatrixDense: invalid mat in matrix subtract (operator-(const MatrixDense<double> &mat))"
        );
    }

    MatrixDense<double> c(*this);

    double alpha = -1.;

    check_cublas_status(
        cublasAxpyEx(
            handle, m_rows*n_cols,
            &alpha, CUDA_R_64F,
            mat.d_mat, CUDA_R_64F, 1,
            c.d_mat, CUDA_R_64F, 1,
            CUDA_R_64F
        )
    );

    return c;

}
#include "types/MatrixDense.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

template<>
MatrixDense<__half> MatrixDense<__half>::operator*(const __half &scalar) const {

    MatrixDense<__half> c(*this);

    float scalar_cast = static_cast<float>(scalar);

    check_cublas_status(
        cublasScalEx(
            handle, m_rows*n_cols, &scalar_cast, CUDA_R_32F, c.d_mat, CUDA_R_16F, 1, CUDA_R_32F
        )
    );

    return c;
}

template <>
MatrixVector<__half> MatrixDense<__half>::transpose_prod(const MatrixVector<__half> &vec) const {

    if (vec.rows() != m_rows) { throw std::runtime_error("MatrixDense: invalid vec in transpose_prod"); }

    MatrixVector<__half> c(handle, n_cols);

    __half alpha = static_cast<__half>(1.);
    __half beta = static_cast<__half>(0.);

    check_cublas_status(
        cublasGemmEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            n_cols, 1, m_rows,
            &alpha,
            d_mat, CUDA_R_16F, m_rows,
            vec.d_vec, CUDA_R_16F, m_rows,
            &beta,
            c.d_vec, CUDA_R_16F, n_cols,
            CUBLAS_COMPUTE_16F,
            CUBLAS_GEMM_DEFAULT
        )
    );

    return c;

}
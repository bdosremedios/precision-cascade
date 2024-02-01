#include "types/MatrixDense.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

MatrixDense<__half> MatrixDense<__half>::operator*(const Scalar<__half> &scalar) const {

    MatrixDense<__half> c(*this);

    Scalar<float> temp_cast(scalar.cast<float>());

    check_cublas_status(
        cublasScalEx(
            handle, m_rows*n_cols,
            temp_cast.d_scalar, CUDA_R_32F,
            c.d_mat, CUDA_R_16F, 1,
            CUDA_R_32F
        )
    );

    return c;
}

Vector<__half> MatrixDense<__half>::operator*(const Vector<__half> &vec) const {

    if (vec.rows() != n_cols) {
        throw std::runtime_error(
            "MatrixDense: invalid vec in matrix-vector prod (operator*(const Vector<__half> &vec))"
        );
    }

    Vector<__half> c(Vector<__half>::Zero(handle, m_rows));

    Scalar<__half> alpha(static_cast<__half>(1.));
    Scalar<__half> beta(static_cast<__half>(0.));

    check_cublas_status(
        cublasGemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            m_rows, 1, n_cols,
            alpha.d_scalar,
            d_mat, CUDA_R_16F, m_rows,
            vec.d_vec, CUDA_R_16F, n_cols,
            beta.d_scalar,
            c.d_vec, CUDA_R_16F, m_rows,
            CUBLAS_COMPUTE_16F,
            CUBLAS_GEMM_DEFAULT
        )
    );

    return c;

}

Vector<__half> MatrixDense<__half>::transpose_prod(const Vector<__half> &vec) const {

    if (vec.rows() != m_rows) { throw std::runtime_error("MatrixDense: invalid vec in transpose_prod"); }

    Vector<__half> c(handle, n_cols);

    Scalar<__half> alpha(static_cast<__half>(1.));
    Scalar<__half> beta(static_cast<__half>(0.));

    check_cublas_status(
        cublasGemmEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            n_cols, 1, m_rows,
            alpha.d_scalar,
            d_mat, CUDA_R_16F, m_rows,
            vec.d_vec, CUDA_R_16F, m_rows,
            beta.d_scalar,
            c.d_vec, CUDA_R_16F, n_cols,
            CUBLAS_COMPUTE_16F,
            CUBLAS_GEMM_DEFAULT
        )
    );

    return c;

}

MatrixDense<__half> MatrixDense<__half>::operator*(const MatrixDense<__half> &mat) const {

    if (mat.rows() != n_cols) {
        throw std::runtime_error(
            "MatrixDense: invalid mat in matrix-matrix prod (operator*(const MatrixDense<__half> &mat))"
        );
    }

    MatrixDense<__half> c(MatrixDense<__half>::Zero(handle, m_rows, mat.cols()));

    Scalar<__half> alpha(static_cast<__half>(1.));
    Scalar<__half> beta(static_cast<__half>(0.));

    check_cublas_status(
        cublasGemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            m_rows, mat.cols(), n_cols,
            alpha.d_scalar,
            d_mat, CUDA_R_16F, m_rows,
            mat.d_mat, CUDA_R_16F, n_cols,
            beta.d_scalar,
            c.d_mat, CUDA_R_16F, m_rows,
            CUBLAS_COMPUTE_16F,
            CUBLAS_GEMM_DEFAULT
        )
    );

    return c;

}

MatrixDense<__half> MatrixDense<__half>::operator+(const MatrixDense<__half> &mat) const {

    if ((mat.rows() != m_rows) || (mat.cols() != n_cols)) {
        throw std::runtime_error(
            "MatrixDense: invalid mat in matrix add (operator+(const MatrixDense<double> &mat))"
        );
    }

    MatrixDense<__half> c(*this);

    Scalar<float> alpha(static_cast<__half>(1.));

    check_cublas_status(
        cublasAxpyEx(
            handle, m_rows*n_cols,
            alpha.d_scalar, CUDA_R_32F,
            mat.d_mat, CUDA_R_16F, 1,
            c.d_mat, CUDA_R_16F, 1,
            CUDA_R_32F
        )
    );

    return c;

}

MatrixDense<__half> MatrixDense<__half>::operator-(const MatrixDense<__half> &mat) const {

    if ((mat.rows() != m_rows) || (mat.cols() != n_cols)) {
        throw std::runtime_error(
            "MatrixDense: invalid mat in matrix subtract (operator-(const MatrixDense<double> &mat))"
        );
    }

    MatrixDense<__half> c(*this);

    Scalar<float> alpha(static_cast<__half>(-1.));

    check_cublas_status(
        cublasAxpyEx(
            handle, m_rows*n_cols,
            alpha.d_scalar, CUDA_R_32F,
            mat.d_mat, CUDA_R_16F, 1,
            c.d_mat, CUDA_R_16F, 1,
            CUDA_R_32F
        )
    );

    return c;

}

Scalar<__half> MatrixDense<__half>::norm() const {

    Scalar<__half> result;

    check_cublas_status(
        cublasNrm2Ex(
            handle, m_rows*n_cols,
            d_mat, CUDA_R_16F, 1,
            result.d_scalar, CUDA_R_16F,
            CUDA_R_32F
        )
    );

    return result;

}

namespace matdense_hlf_kern
{
    __global__ void cast_to_float(__half *mat_src, float *mat_dest, int m) {
        int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (tid < m) {
            mat_dest[tid] = __half2float(mat_src[tid]);
        }
    }

    __global__ void cast_to_double(__half *mat_src, double *mat_dest, int m) {
        int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (tid < m) {
            mat_dest[tid] = static_cast<double>(mat_src[tid]);
        }
    }
}

MatrixDense<__half> MatrixDense<__half>::to_half() const { return MatrixDense<__half>(*this); }

MatrixDense<float> MatrixDense<__half>::to_float() const {
    
    MatrixDense<float> created_mat(handle, m_rows, n_cols);

    double NUM_THREADS = 1024; // threads per thread block just 1 warp
    double NUM_BLOCKS = static_cast<double>(
        std::ceil(static_cast<double>(m_rows*n_cols)/static_cast<double>(NUM_THREADS))
    );
    matdense_hlf_kern::cast_to_float<<<NUM_THREADS, NUM_BLOCKS>>>(d_mat, created_mat.d_mat, m_rows*n_cols);

    return created_mat;

}

MatrixDense<double> MatrixDense<__half>::to_double() const {
    
    MatrixDense<double> created_mat(handle, m_rows, n_cols);

    double NUM_THREADS = 1024; // threads per thread block just 1 warp
    double NUM_BLOCKS = static_cast<double>(
        std::ceil(static_cast<double>(m_rows*n_cols)/static_cast<double>(NUM_THREADS))
    );
    matdense_hlf_kern::cast_to_double<<<NUM_THREADS, NUM_BLOCKS>>>(d_mat, created_mat.d_mat, m_rows*n_cols);

    return created_mat;

}
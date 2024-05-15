#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include "types/GeneralMatrix/GeneralMatrix_gpu_constants.cuh"

#include "types/MatrixDense/MatrixDense.h"

MatrixDense<float> MatrixDense<float>::operator*(const Scalar<float> &scalar) const {

    MatrixDense<float> c(*this);

    check_cublas_status(
        cublasScalEx(
            cu_handles.get_cublas_handle(),
            m_rows*n_cols,
            scalar.d_scalar, CUDA_R_32F,
            c.d_mat, CUDA_R_32F, 1,
            CUDA_R_32F
        )
    );

    return c;

}

MatrixDense<float> & MatrixDense<float>::operator*=(const Scalar<float> &scalar) {

    check_cublas_status(
        cublasScalEx(
            cu_handles.get_cublas_handle(),
            m_rows*n_cols,
            scalar.d_scalar, CUDA_R_32F,
            d_mat, CUDA_R_32F, 1,
            CUDA_R_32F
        )
    );

    return *this;

}

Vector<float> MatrixDense<float>::operator*(const Vector<float> &vec) const {

    if (vec.rows() != n_cols) {
        throw std::runtime_error(
            "MatrixDense: invalid vec in matrix-vector prod (operator*(const Vector<float> &vec))"
        );
    }

    Vector<float> c(Vector<float>::Zero(cu_handles, m_rows));

    check_cublas_status(
        cublasGemmEx(
            cu_handles.get_cublas_handle(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            m_rows, 1, n_cols,
            SCALAR_ONE_F.d_scalar,
            d_mat, CUDA_R_32F, m_rows,
            vec.d_vec, CUDA_R_32F, n_cols,
            SCALAR_ZERO_F.d_scalar,
            c.d_vec, CUDA_R_32F, m_rows,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        )
    );

    return c;

}

Vector<float> MatrixDense<float>::mult_subset_cols(int start, int cols, const Vector<float> &vec) const {

    if ((start < 0) || ((start+cols) > n_cols) || (cols <= 0)) {
        throw std::runtime_error("MatrixDense: invalid column values in mult_subset_cols");
    }
    if (vec.rows() != cols) {
        throw std::runtime_error("MatrixDense: invalid vec in mult_subset_cols");
    }

    Vector<float> c(Vector<float>::Zero(cu_handles, m_rows));

    check_cublas_status(
        cublasGemmEx(
            cu_handles.get_cublas_handle(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            m_rows, 1, cols,
            SCALAR_ONE_F.d_scalar,
            d_mat+start*m_rows, CUDA_R_32F, m_rows,
            vec.d_vec, CUDA_R_32F, n_cols,
            SCALAR_ZERO_F.d_scalar,
            c.d_vec, CUDA_R_32F, m_rows,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        )
    );

    return c;

}

Vector<float> MatrixDense<float>::transpose_prod(const Vector<float> &vec) const {

    if (vec.rows() != m_rows) { throw std::runtime_error("MatrixDense: invalid vec in transpose_prod"); }

    Vector<float> c(cu_handles, n_cols);

    check_cublas_status(
        cublasGemmEx(
            cu_handles.get_cublas_handle(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            n_cols, 1, m_rows,
            SCALAR_ONE_F.d_scalar,
            d_mat, CUDA_R_32F, m_rows,
            vec.d_vec, CUDA_R_32F, m_rows,
            SCALAR_ZERO_F.d_scalar,
            c.d_vec, CUDA_R_32F, n_cols,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        )
    );

    return c;

}

Vector<float> MatrixDense<float>::transpose_prod_subset_cols(int start, int cols, const Vector<float> &vec) const {

    if ((start < 0) || ((start+cols) > n_cols) || (cols <= 0)) {
        throw std::runtime_error("MatrixDense: invalid column values in transpose_prod_subset_cols");
    }
    if (vec.rows() != m_rows) {
        throw std::runtime_error("MatrixDense: invalid vec in transpose_prod_subset_cols");
    }

    Vector<float> c(Vector<float>::Zero(cu_handles, cols));

    check_cublas_status(
        cublasGemmEx(
            cu_handles.get_cublas_handle(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            cols, 1, m_rows,
            SCALAR_ONE_F.d_scalar,
            d_mat+start*m_rows, CUDA_R_32F, m_rows,
            vec.d_vec, CUDA_R_32F, m_rows,
            SCALAR_ZERO_F.d_scalar,
            c.d_vec, CUDA_R_32F, cols,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        )
    );

    return c;

}

MatrixDense<float> MatrixDense<float>::operator*(const MatrixDense<float> &mat) const {

    if (mat.rows() != n_cols) {
        throw std::runtime_error(
            "MatrixDense: invalid mat in matrix-matrix prod (operator*(const MatrixDense<float> &mat))"
        );
    }

    MatrixDense<float> c(MatrixDense<float>::Zero(cu_handles, m_rows, mat.cols()));

    check_cublas_status(
        cublasGemmEx(
            cu_handles.get_cublas_handle(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            m_rows, mat.cols(), n_cols,
            SCALAR_ONE_F.d_scalar,
            d_mat, CUDA_R_32F, m_rows,
            mat.d_mat, CUDA_R_32F, n_cols,
            SCALAR_ZERO_F.d_scalar,
            c.d_mat, CUDA_R_32F, m_rows,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        )
    );

    return c;

}

MatrixDense<float> MatrixDense<float>::operator+(const MatrixDense<float> &mat) const {

    if ((mat.rows() != m_rows) || (mat.cols() != n_cols)) {
        throw std::runtime_error(
            "MatrixDense: invalid mat in matrix add (operator+(const MatrixDense<float> &mat))"
        );
    }

    MatrixDense<float> c(*this);

    check_cublas_status(
        cublasAxpyEx(
            cu_handles.get_cublas_handle(),
            m_rows*n_cols,
            SCALAR_ONE_F.d_scalar, CUDA_R_32F,
            mat.d_mat, CUDA_R_32F, 1,
            c.d_mat, CUDA_R_32F, 1,
            CUDA_R_32F
        )
    );

    return c;

}

MatrixDense<float> MatrixDense<float>::operator-(const MatrixDense<float> &mat) const {

    if ((mat.rows() != m_rows) || (mat.cols() != n_cols)) {
        throw std::runtime_error(
            "MatrixDense: invalid mat in matrix subtract (operator-(const MatrixDense<float> &mat))"
        );
    }

    MatrixDense<float> c(*this);

    check_cublas_status(
        cublasAxpyEx(
            cu_handles.get_cublas_handle(),
            m_rows*n_cols,
            SCALAR_MINUS_ONE_F.d_scalar, CUDA_R_32F,
            mat.d_mat, CUDA_R_32F, 1,
            c.d_mat, CUDA_R_32F, 1,
            CUDA_R_32F
        )
    );

    return c;

}

Scalar<float> MatrixDense<float>::norm() const {

    Scalar<float> result;

    check_cublas_status(
        cublasNrm2Ex(
            cu_handles.get_cublas_handle(),
            m_rows*n_cols,
            d_mat, CUDA_R_32F, 1,
            result.d_scalar, CUDA_R_32F,
            CUDA_R_32F
        )
    );

    return result;

}

MatrixDense<__half> MatrixDense<float>::to_half() const {
    
    MatrixDense<__half> created_mat(cu_handles, m_rows, n_cols);

    int NUM_THREADS = genmat_gpu_const::MAXTHREADSPERBLOCK;
    int NUM_BLOCKS = std::ceil(static_cast<double>(m_rows*n_cols)/static_cast<double>(NUM_THREADS));

    if (NUM_BLOCKS > 0) {

        generalmatrix_sgl_kernels::cast_to_half<<<NUM_BLOCKS, NUM_THREADS>>>(
            d_mat, created_mat.d_mat, m_rows*n_cols
        );

        check_kernel_launch(
            cudaGetLastError(),
            "MatrixDense<float>::to_half",
            "generalmatrix_sgl_kernels::cast_to_half",
            NUM_BLOCKS, NUM_THREADS
        );

    }

    return created_mat;

}

MatrixDense<float> MatrixDense<float>::to_float() const {
    return MatrixDense<float>(*this);
}

MatrixDense<double> MatrixDense<float>::to_double() const {
    
    MatrixDense<double> created_mat(cu_handles, m_rows, n_cols);

    int NUM_THREADS = genmat_gpu_const::MAXTHREADSPERBLOCK;
    int NUM_BLOCKS = std::ceil(static_cast<double>(m_rows*n_cols)/static_cast<double>(NUM_THREADS));

    if (NUM_BLOCKS > 0) {

        generalmatrix_sgl_kernels::cast_to_double<<<NUM_BLOCKS, NUM_THREADS>>>(
            d_mat, created_mat.d_mat, m_rows*n_cols
        );

        check_kernel_launch(
            cudaGetLastError(),
            "MatrixDense<float>::to_double",
            "generalmatrix_sgl_kernels::cast_to_double",
            NUM_THREADS, NUM_BLOCKS
        );

    }

    return created_mat;

}